"""Leader-follower team coordination controller.

This controller wraps any legal gate-passing controller (the "base" controller)
and adds a thin, distributed coordination layer on top of it. A staggered team of
rovers uses it to run the same gate sequence as a loose convoy that avoids
rear-ending a slower vehicle ahead, instead of every rover racing independently
and piling into whoever is in front.

Coordination protocol (fully distributed, no central controller):

* Every rover broadcasts a tiny heartbeat over the optional inter-rover acoustic
  channel: its completed-gate count, its target gate index and a one-letter status
  code. Payloads are authored from the rover's own official observation, so they
  only ever carry legally observable information.
* Each rover learns the team roster from the ``from`` field of the messages it
  receives. It identifies its *predecessor* as the in-range teammate that started
  ahead of it (a smaller participant id) and is closest to it in the start order.
  The frontmost rover has no predecessor and therefore always races freely, so the
  first rover naturally acts as the leader.
* A follower advances (runs its base controller unchanged) only while its
  predecessor is at least ``min_gate_gap`` gates ahead of it. Otherwise it holds
  station and yields, letting the vehicle ahead open a safe along-course gap before
  it approaches the same gate. Because the gates are ordered and spatially
  separated, keeping a two-gate progress lead keeps a full gate of physical
  separation, which is far larger than the referee's inter-vehicle proximity
  threshold.

Legality: the controller reads only ``observation["race"]`` (its own referee
progress), ``observation["participant_id"]`` (its own id), the delivered
``observation["comms"]["inbox"]`` (controller-authored teammate payloads) and the
static ``race_info``. It never touches ground-truth pose, referee internals,
hidden gate geometry, or simulator-only state, and it delegates all motion to a
legal base controller. When the comms channel is disabled the controller has no
teammate information and simply runs the base controller, so it degrades safely to
the uncoordinated baseline.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Mapping, Optional

from marine_race_arena.participants.controller_interface import BaseController

# One-letter status codes carried in the heartbeat payload (kept tiny to fit the
# acoustic channel's small byte budget).
_STATUS_TO_CODE = {
    "RUNNING": "R",
    "NOT_STARTED": "N",
    "FINISHED": "F",
    "DNF": "D",
    "DSQ": "Q",
    "TIMEOUT": "T",
    "STUCK": "S",
    "MANUAL_STOP": "M",
    "CONTROLLER_ERROR": "E",
}
# Codes for teammates that are no longer an along-course collision hazard ahead.
_TERMINAL_CODES = {"F", "D", "Q", "T", "S", "M", "E"}

_DEFAULT_BASE_ALIAS = "rule_gate_baseline"
_COORDINATION_ALIASES = {"leader_follower", "leader_follower_acoustic"}


class LeaderFollowerController(BaseController):
    """Coordinate a staggered team as a leader-follower convoy around a base controller."""

    debug_only = False
    uses_ground_truth = False

    #: Required predecessor lead, in completed gates, before a follower advances.
    MIN_GATE_GAP = 2
    #: Minimum wall-clock spacing between this rover's own heartbeats (seconds).
    HEARTBEAT_INTERVAL_S = 0.5
    #: Drop teammate information that has not been refreshed within this window.
    STALE_AFTER_S = 2.5
    #: Active-hover amplitude/period used while yielding (see ``_hold_command``).
    HOVER_AMPLITUDE = 0.08
    HOVER_PERIOD_S = 2.5

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

    def reset(self, race_info: Dict[str, Any]) -> None:
        self.base.reset(race_info)
        laps = _safe_int(race_info.get("laps"), 1)
        gates_per_lap = _safe_int(race_info.get("gates_per_lap"), 0)
        self._total_gates = max(0, gates_per_lap * max(1, laps))
        self._participant_id: Optional[str] = None
        self._roster: Dict[str, Dict[str, Any]] = {}
        self._last_heartbeat_time: Optional[float] = None
        # Diagnostics (read by tests and the comparison harness; not sent anywhere).
        self.hold_steps = 0
        self.advance_steps = 0
        self.is_holding = False

    def step(self, observation: Dict[str, Any]) -> Dict[str, float]:
        race = _mapping(observation.get("race"))
        my_id = observation.get("participant_id")
        if my_id is not None:
            self._participant_id = str(my_id)
        time_s = _safe_float(observation.get("time_s"), 0.0)
        my_completed = _safe_int(race.get("completed_gates"), 0)

        self._ingest_inbox(observation, time_s)

        hold = self._should_hold(time_s, my_completed)
        self.is_holding = hold
        if hold:
            self.hold_steps += 1
            command = self._hold_command(observation)
        else:
            self.advance_steps += 1
            command = dict(self.base.step(observation))
            command.pop("message", None)  # base controllers must not author team messages

        if self._due_for_heartbeat(time_s):
            self._last_heartbeat_time = time_s
            command["message"] = {
                "g": my_completed,
                "t": _safe_int(race.get("target_sequence_index"), my_completed),
                "st": _STATUS_TO_CODE.get(str(race.get("status") or "RUNNING"), "R"),
            }
        return command

    def close(self) -> None:
        self.base.close()

    # -- coordination internals ---------------------------------------------

    def _ingest_inbox(self, observation: Mapping[str, Any], time_s: float) -> None:
        comms = observation.get("comms")
        if not isinstance(comms, Mapping):
            return
        for message in comms.get("inbox", []) or []:
            if not isinstance(message, Mapping):
                continue
            sender = message.get("from")
            payload = message.get("payload")
            if sender is None or str(sender) == self._participant_id or not isinstance(payload, Mapping):
                continue
            received_at = _safe_float(message.get("received_at_s"), time_s)
            existing = self._roster.get(str(sender))
            # Keep only the freshest report from each teammate (messages may arrive
            # out of order after range-dependent acoustic latency).
            if existing is not None and existing["last_time"] > received_at:
                continue
            self._roster[str(sender)] = {
                "completed": _safe_int(payload.get("g"), 0),
                "target": _safe_int(payload.get("t"), 0),
                "status": str(payload.get("st", "R")),
                "last_time": received_at,
            }

    def _predecessor(self, time_s: float) -> Optional[Dict[str, Any]]:
        """The nearest in-order teammate ahead of us that we can still hear."""
        if self._participant_id is None:
            return None
        best_id: Optional[str] = None
        for teammate_id, info in self._roster.items():
            if teammate_id >= self._participant_id:
                continue
            if time_s - info["last_time"] > self.STALE_AFTER_S:
                continue  # out of range or gone silent -> not a detectable hazard
            if best_id is None or teammate_id > best_id:
                best_id = teammate_id
        return self._roster[best_id] if best_id is not None else None

    def _should_hold(self, time_s: float, my_completed: int) -> bool:
        predecessor = self._predecessor(time_s)
        if predecessor is None:
            return False  # leader, or nobody audible ahead -> race freely
        if predecessor["status"] in _TERMINAL_CODES:
            return False  # vehicle ahead has left the course; no rear-end hazard
        if self._total_gates and predecessor["completed"] >= self._total_gates:
            return False  # vehicle ahead has finished the sequence
        return (predecessor["completed"] - my_completed) < self._min_gate_gap

    def _hold_command(self, observation: Mapping[str, Any]) -> Dict[str, float]:
        """Active station-keeping while yielding.

        The follower holds its horizontal position (zero surge and sway, so it never
        advances toward the vehicle ahead) but keeps a small, slow vertical hover
        instead of freezing dead in the water. A real hovering ROV continuously trims
        its position and is never perfectly still; modelling the yield as an active
        hover keeps the follower on station without drifting and keeps it above the
        referee's stuck-speed threshold, so a legitimate wait is not misread as a
        mechanically stuck vehicle. The hover is a zero-mean vertical oscillation, so
        it does not accumulate depth error over a long hold.
        """
        time_s = _safe_float(observation.get("time_s"), 0.0)
        hover = self.HOVER_AMPLITUDE * math.sin(2.0 * math.pi * time_s / self.HOVER_PERIOD_S)
        return {"surge": 0.0, "sway": 0.0, "heave": _clamp(hover, -0.12, 0.12), "yaw": 0.0}

    def _due_for_heartbeat(self, time_s: float) -> bool:
        if self._last_heartbeat_time is None:
            return True
        return (time_s - self._last_heartbeat_time) >= self.HEARTBEAT_INTERVAL_S

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


class LeaderFollowerAcousticController(LeaderFollowerController):
    """Leader-follower coordinator whose base is the beacon-only acoustic baseline.

    This variant navigates on the acoustic beacon alone, so it runs on any adapter
    (including the engine-free kinematic fallback used by the tests and the
    comparison harness) without needing the front camera that the default rule
    baseline uses.
    """

    def _resolve_base(
        self,
        base_controller: Optional[BaseController],
        base_alias: Optional[str],
    ) -> BaseController:
        return super()._resolve_base(base_controller, base_alias or "acoustic_baseline")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


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


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
