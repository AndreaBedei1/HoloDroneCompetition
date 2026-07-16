"""Leader-follower team coordination controller.

This controller wraps a legal onboard-only gate controller (the "base"
controller) and adds a thin, distributed coordination layer on top of it. A
staggered team of rovers uses it to run the same gate sequence as a loose
convoy that avoids rear-ending a slower vehicle ahead.

Coordination protocol (fully distributed, no central controller):

* Every rover broadcasts a small heartbeat over the optional inter-rover
  acoustic channel carrying only its *locally estimated* progress, read from
  its own base controller's :class:`LocalCourseTracker`:
  ``{"local_beacon_index": ..., "local_lap": ..., "local_status": ...}``.
  No referee value ever enters a payload; the sender does not have one.
* Each rover knows its predecessor from the static fleet information assigned
  before the race (release order). The frontmost rover has no predecessor and
  races freely, so the first rover naturally acts as the leader.
* A follower advances (runs its base controller unchanged) only while its
  predecessor's *reported local progress* is at least ``min_gate_gap`` gates
  ahead of the follower's own local estimate. Otherwise it holds station.
  Teammate reports are estimates, not ground truth: a wrong local estimate on
  either side simply produces conservative or optimistic yielding and is
  scored by the referee like any other behavior.

The channel keeps its physical limitations (range-dependent latency, maximum
range, payload budget, half-duplex send interval, seeded packet loss), and
stale reports are discarded after ``STALE_AFTER_S``. When comms are disabled
the controller degrades to the uncoordinated base controller.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional

from marine_race_arena.participants.controller_interface import BaseController

_LOCAL_STATUSES = {"RUNNING", "FINISHED"}

_DEFAULT_BASE_ALIAS = "rule_gate_baseline"
_COORDINATION_ALIASES = {"leader_follower"}


class LeaderFollowerController(BaseController):
    """Coordinate a staggered team as a leader-follower convoy around a base controller."""

    debug_only = False
    uses_ground_truth = False

    #: Required predecessor lead, in locally-estimated gates, before a follower advances.
    MIN_GATE_GAP = 2
    #: Minimum local-clock spacing between this rover's own heartbeats (seconds).
    HEARTBEAT_INTERVAL_S = 0.5
    #: Drop teammate information that has not been refreshed within this window.
    STALE_AFTER_S = 2.5
    def __init__(
        self,
        base_controller: Optional[BaseController] = None,
        *,
        base_alias: Optional[str] = None,
        min_gate_gap: Optional[int] = None,
    ) -> None:
        self.base = self._resolve_base(base_controller, base_alias)
        # Mirror the base controller's honesty flag so the official-mode gate keeps
        # rejecting a coordinator that wraps a ground-truth base.
        self.uses_ground_truth = bool(getattr(self.base, "uses_ground_truth", False))
        if min_gate_gap is not None:
            self._min_gate_gap = max(1, int(min_gate_gap))
        else:
            self._min_gate_gap = max(1, int(os.environ.get("MARINE_RACE_COORDINATION_MIN_GAP", self.MIN_GATE_GAP)))

    def reset(self, mission_info: Dict[str, Any]) -> None:
        self.base.reset(mission_info)
        if getattr(self.base, "tracker", None) is None:
            raise ValueError(
                "Leader-follower coordination requires a base controller with a "
                "LocalCourseTracker (attribute 'tracker'); the coordinator only "
                "ever broadcasts locally estimated progress."
            )
        self._participant_id = str(mission_info.get("participant_id", ""))
        self._total_beacons = max(0, int(mission_info.get("total_beacons", 0)))
        self._laps = max(1, int(mission_info.get("laps", 1)))
        self._total_gates = self._total_beacons * self._laps
        fleet = mission_info.get("fleet")
        self._predecessor_id: Optional[str] = None
        if isinstance(fleet, Mapping):
            predecessor = fleet.get("predecessor_id")
            self._predecessor_id = str(predecessor) if predecessor else None
        self._predecessor_report: Optional[Dict[str, Any]] = None
        self._last_heartbeat_time: Optional[float] = None
        # Diagnostics (read by tests and harnesses; never transmitted).
        self.hold_steps = 0
        self.advance_steps = 0
        self.is_holding = False
        self.coordination_diagnostics: Dict[str, Any] = {
            "participant_id": self._participant_id,
            "predecessor_id": self._predecessor_id,
            "min_gate_gap": self._min_gate_gap,
            "is_holding": False,
            "decision": "advance",
            "decision_reason": "not_evaluated",
            "hold_reason": None,
            "hold_steps": 0,
            "advance_steps": 0,
        }

    def step(self, observation: Dict[str, Any]) -> Dict[str, float]:
        local_time_s = _safe_float(observation.get("local_time_s"), 0.0)
        self._ingest_inbox(observation, local_time_s)

        decision = self._coordination_decision(local_time_s)
        hold = bool(decision["is_holding"])
        self.is_holding = hold
        if hold:
            self.hold_steps += 1
            # The base controller still observes (its tracker must keep its
            # local estimate honest even while yielding).
            self.base.step(observation)
            command = self._hold_command(observation)
        else:
            self.advance_steps += 1
            command = dict(self.base.step(observation))
            command.pop("message", None)  # base controllers must not author team messages

        # This snapshot is controller-local analysis output.  It is deliberately
        # assembled after updating the counters, and is never copied into the
        # controller-authored acoustic heartbeat.
        self.coordination_diagnostics = {
            **decision,
            "hold_steps": self.hold_steps,
            "advance_steps": self.advance_steps,
        }

        if self._due_for_heartbeat(local_time_s):
            self._last_heartbeat_time = local_time_s
            command["message"] = self._local_heartbeat()
        return command

    def close(self) -> None:
        self.base.close()

    # -- coordination internals ---------------------------------------------

    def _local_heartbeat(self) -> Dict[str, Any]:
        """Heartbeat payload from this rover's own local tracker, nothing else."""
        tracker = self.base.tracker
        return {
            "local_beacon_index": int(getattr(tracker, "local_beacon_index", 0)),
            "local_lap": int(getattr(tracker, "local_lap", 1)),
            "local_status": str(getattr(tracker, "status", "RUNNING")),
        }

    def _local_progress(self, tracker: Any) -> int:
        return int(getattr(tracker, "local_completed", 0))

    def _reported_progress(self, report: Mapping[str, Any]) -> int:
        """Cumulative locally-estimated gates from a teammate heartbeat."""
        if report["status"] == "FINISHED":
            return self._total_gates or (report["lap"] * max(1, self._total_beacons))
        lap = max(1, int(report["lap"]))
        index = max(0, int(report["beacon_index"]))
        return (lap - 1) * max(1, self._total_beacons) + index

    def _ingest_inbox(self, observation: Mapping[str, Any], local_time_s: float) -> None:
        comms = observation.get("comms")
        if not isinstance(comms, Mapping):
            return
        for message in comms.get("inbox", []) or []:
            if not isinstance(message, Mapping):
                continue
            sender = message.get("from")
            payload = message.get("payload")
            if sender is None or not isinstance(payload, Mapping):
                continue
            if self._predecessor_id is None or str(sender) != self._predecessor_id:
                continue  # only the assigned predecessor matters for yielding
            received_at = _safe_float(message.get("received_at_s"), local_time_s)
            existing = self._predecessor_report
            # Keep only the freshest report (messages may arrive out of order
            # after range-dependent acoustic latency).
            if existing is not None and existing["last_time"] > received_at:
                continue
            status = str(payload.get("local_status", "RUNNING"))
            if status not in _LOCAL_STATUSES:
                status = "RUNNING"
            self._predecessor_report = {
                "beacon_index": _safe_int(payload.get("local_beacon_index"), 0),
                "lap": _safe_int(payload.get("local_lap"), 1),
                "status": status,
                "last_time": received_at,
            }

    def _coordination_decision(self, local_time_s: float) -> Dict[str, Any]:
        """Explain a hold/advance decision using controller-local estimates only.

        The returned values are safe for offline diagnostics: local tracker
        state, a controller-assigned predecessor id, and fields from the most
        recently delivered predecessor heartbeat.  No referee state or hidden
        simulator state is read here.
        """
        tracker = self.base.tracker
        my_completed = self._local_progress(tracker)
        decision: Dict[str, Any] = {
            "participant_id": self._participant_id,
            "predecessor_id": self._predecessor_id,
            "min_gate_gap": self._min_gate_gap,
            "local_beacon_index": int(getattr(tracker, "local_beacon_index", 0)),
            "local_lap": int(getattr(tracker, "local_lap", 1)),
            "local_status": str(getattr(tracker, "status", "RUNNING")),
            "local_completed_at_decision": my_completed,
            "predecessor_local_beacon_index": None,
            "predecessor_local_lap": None,
            "predecessor_local_status": None,
            "predecessor_local_completed": None,
            "predecessor_report_age_s": None,
            "local_gate_gap": None,
            "is_holding": False,
            "decision": "advance",
            "decision_reason": "",
            "hold_reason": None,
        }

        def advance(reason: str) -> Dict[str, Any]:
            decision["decision_reason"] = reason
            return decision

        if self._predecessor_id is None:
            return advance("leader_no_predecessor")
        report = self._predecessor_report
        if report is None:
            return advance("no_predecessor_report")

        report_age_s = local_time_s - report["last_time"]
        predecessor_progress = self._reported_progress(report)
        decision.update(
            {
                "predecessor_local_beacon_index": int(report["beacon_index"]),
                "predecessor_local_lap": int(report["lap"]),
                "predecessor_local_status": str(report["status"]),
                "predecessor_local_completed": predecessor_progress,
                "predecessor_report_age_s": report_age_s,
                "local_gate_gap": predecessor_progress - my_completed,
            }
        )

        if report_age_s > self.STALE_AFTER_S:
            return advance("stale_predecessor_report")
        if report["status"] == "FINISHED":
            return advance("predecessor_reported_finished")
        if self._total_gates and predecessor_progress >= self._total_gates:
            return advance("predecessor_reported_sequence_complete")

        local_gate_gap = predecessor_progress - my_completed
        if local_gate_gap <= 0:
            reason = "predecessor_not_ahead_in_local_sequence"
        elif local_gate_gap < self._min_gate_gap:
            reason = "local_gate_gap_below_minimum"
        else:
            return advance("local_gate_gap_satisfied")

        decision.update(
            {
                "is_holding": True,
                "decision": "hold",
                "decision_reason": reason,
                "hold_reason": reason,
            }
        )
        return decision

    def _hold_command(self, observation: Mapping[str, Any]) -> Dict[str, float]:
        """Conservative zero-command yield.

        The coordinator does not manufacture motion to influence referee
        stuck scoring.  A long intentional wait may therefore be penalized by
        the independent referee, and that outcome remains visible in results.
        """
        del observation
        return {"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}

    def _due_for_heartbeat(self, local_time_s: float) -> bool:
        if self._last_heartbeat_time is None:
            return True
        return (local_time_s - self._last_heartbeat_time) >= self.HEARTBEAT_INTERVAL_S

    def _resolve_base(
        self,
        base_controller: Optional[BaseController],
        base_alias: Optional[str],
    ) -> BaseController:
        if base_controller is not None:
            return base_controller
        alias = base_alias or os.environ.get("MARINE_RACE_COORDINATION_BASE", _DEFAULT_BASE_ALIAS)
        if alias in _COORDINATION_ALIASES:
            raise ValueError(f"Coordination base controller may not be '{alias}' (self-reference).")
        # Imported lazily to avoid an import cycle (the loader imports controllers).
        from marine_race_arena.participants.controller_loader import ControllerLoader

        return ControllerLoader().load(alias)  # type: ignore[return-value]


def _safe_float(value: Any, default: float) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted if converted == converted else default  # reject NaN


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
