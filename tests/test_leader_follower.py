from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from marine_race_arena.controllers.leader_follower import LeaderFollowerController
from marine_race_arena.controllers.official_baselines import RuleGateBaselineController
from marine_race_arena.participants.controller_interface import BaseController
from marine_race_arena.participants.controller_loader import ControllerLoader


class _StubBase(BaseController):
    """Fixed-command onboard base with an inspectable local tracker."""

    uses_ground_truth = False

    def __init__(self, uses_ground_truth: bool = False) -> None:
        self.uses_ground_truth = uses_ground_truth
        self.reset_calls = 0
        self.step_calls = 0
        self.closed = False
        self.reset_info: Optional[Dict[str, Any]] = None
        self.observations: List[Dict[str, Any]] = []
        self.tracker = SimpleNamespace(
            local_beacon_index=0,
            local_lap=1,
            local_completed=0,
            status="RUNNING",
        )

    def reset(self, mission_info: Dict[str, Any]) -> None:
        self.reset_calls += 1
        self.reset_info = mission_info
        self.tracker = SimpleNamespace(
            local_beacon_index=0,
            local_lap=1,
            local_completed=0,
            status="RUNNING",
        )

    def step(self, observation: Dict[str, Any]) -> Dict[str, float]:
        self.step_calls += 1
        self.observations.append(observation)
        return {"surge": 0.5, "sway": 0.1, "heave": -0.2, "yaw": 0.3}

    def close(self) -> None:
        self.closed = True


def _mission_info(
    participant_id: str = "bluerov2_02",
    predecessor_id: Optional[str] = "bluerov2_01",
) -> Dict[str, Any]:
    return {
        "participant_id": participant_id,
        "initial_beacon_id": "B01",
        "total_beacons": 12,
        "laps": 1,
        "command_limits": {
            axis: [-0.95, 0.95] for axis in ("surge", "sway", "heave", "yaw")
        },
        "fleet": {
            "participant_order": ["bluerov2_01", "bluerov2_02", "bluerov2_03"],
            "release_index": int(participant_id.rsplit("_", 1)[-1]) - 1,
            "predecessor_id": predecessor_id,
        },
    }


def _inbox(
    sender: str,
    beacon_index: int,
    status: str = "RUNNING",
    received_at_s: float = 0.9,
    lap: int = 1,
) -> List[Dict[str, Any]]:
    return [
        {
            "from": sender,
            "payload": {
                "local_beacon_index": beacon_index,
                "local_lap": lap,
                "local_status": status,
            },
            "received_at_s": received_at_s,
        }
    ]


def _observation(
    time_s: float,
    inbox: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    observation: Dict[str, Any] = {
        "local_time_s": time_s,
        "sensors": {
            "DepthSensor": [-4.0],
            "DVLSensor": [0.0, 0.0, 0.0],
            "IMUSensor": [0.0, 0.0, 0.0, 1.0],
        },
        "beacons": [
            {
                "beacon_id": "B01",
                "bearing_deg": 0.0,
                "elevation_deg": 0.0,
                "range_m": 6.0,
                "signal_strength": 0.8,
                "received_at_s": time_s,
            }
        ],
    }
    if inbox is not None:
        observation["comms"] = {"inbox": inbox}
    return observation


def _coordinator(
    base: BaseController,
    min_gate_gap: int = 2,
    *,
    participant_id: str = "bluerov2_02",
    predecessor_id: Optional[str] = "bluerov2_01",
) -> LeaderFollowerController:
    controller = LeaderFollowerController(base_controller=base, min_gate_gap=min_gate_gap)
    controller.reset(_mission_info(participant_id, predecessor_id))
    return controller


def _set_progress(base: _StubBase, completed: int, *, lap: int = 1, status: str = "RUNNING") -> None:
    base.tracker.local_beacon_index = completed % 12
    base.tracker.local_lap = lap
    base.tracker.local_completed = completed
    base.tracker.status = status


# -- loading / legality ------------------------------------------------------


def test_alias_loads_with_the_current_onboard_base() -> None:
    controller = ControllerLoader().load("leader_follower")
    assert isinstance(controller, LeaderFollowerController)
    assert isinstance(controller.base, RuleGateBaselineController)
    assert controller.uses_ground_truth is False


def test_reflects_the_base_controllers_ground_truth_flag() -> None:
    honest = LeaderFollowerController(base_controller=_StubBase(uses_ground_truth=False))
    cheating = LeaderFollowerController(base_controller=_StubBase(uses_ground_truth=True))
    assert honest.uses_ground_truth is False
    assert cheating.uses_ground_truth is True


def test_base_may_not_be_a_coordination_alias() -> None:
    with pytest.raises(ValueError):
        LeaderFollowerController(base_alias="leader_follower")


def test_reset_forwards_only_static_mission_info_to_base() -> None:
    base = _StubBase()
    mission_info = _mission_info()
    controller = LeaderFollowerController(base_controller=base)
    controller.reset(mission_info)

    assert base.reset_calls == 1
    assert base.reset_info is mission_info
    assert "race" not in mission_info
    assert "target_gate_id" not in mission_info
    assert "completed_gates" not in mission_info


def test_close_forwards_to_base() -> None:
    base = _StubBase()
    controller = _coordinator(base)
    controller.close()
    assert base.closed is True


# -- hold / advance decisions ------------------------------------------------


def test_leader_without_predecessor_runs_the_base() -> None:
    base = _StubBase()
    controller = _coordinator(
        base,
        participant_id="bluerov2_01",
        predecessor_id=None,
    )
    command = controller.step(_observation(1.0, inbox=_inbox("bluerov2_02", 0)))
    assert base.step_calls == 1
    assert command["surge"] == 0.5
    assert controller.is_holding is False
    assert controller.coordination_diagnostics["decision_reason"] == "leader_no_predecessor"


def test_follower_holds_when_predecessor_local_estimate_is_not_far_enough_ahead() -> None:
    base = _StubBase()
    controller = _coordinator(base)
    command = controller.step(_observation(1.0, inbox=_inbox("bluerov2_01", 1)))

    assert controller.is_holding is True
    # The base still observes while holding so its LocalCourseTracker stays current.
    assert base.step_calls == 1
    assert command["surge"] == 0.0
    assert command["sway"] == 0.0
    assert controller.coordination_diagnostics["hold_reason"] == "local_gate_gap_below_minimum"
    assert controller.coordination_diagnostics["local_gate_gap"] == 1
    assert controller.coordination_diagnostics["hold_steps"] == 1
    assert controller.coordination_diagnostics["advance_steps"] == 0


def test_follower_advances_when_predecessor_local_estimate_is_far_enough_ahead() -> None:
    base = _StubBase()
    controller = _coordinator(base)
    command = controller.step(_observation(1.0, inbox=_inbox("bluerov2_01", 2)))
    assert controller.is_holding is False
    assert base.step_calls == 1
    assert command["surge"] == 0.5
    assert controller.coordination_diagnostics["decision_reason"] == "local_gate_gap_satisfied"


def test_hold_decision_uses_the_followers_own_local_tracker_progress() -> None:
    base = _StubBase()
    controller = _coordinator(base)
    _set_progress(base, 3)

    controller.step(_observation(1.0, inbox=_inbox("bluerov2_01", 4)))
    assert controller.is_holding is True

    controller.step(_observation(1.1, inbox=_inbox("bluerov2_01", 5, received_at_s=1.1)))
    assert controller.is_holding is False


def test_follower_advances_once_predecessor_reports_local_finish() -> None:
    base = _StubBase()
    controller = _coordinator(base)
    command = controller.step(
        _observation(1.0, inbox=_inbox("bluerov2_01", 0, status="FINISHED"))
    )
    assert controller.is_holding is False
    assert command["surge"] == 0.5


def test_follower_advances_when_predecessor_reports_the_full_local_sequence() -> None:
    base = _StubBase()
    controller = _coordinator(base)
    command = controller.step(_observation(1.0, inbox=_inbox("bluerov2_01", 12)))
    assert controller.is_holding is False
    assert command["surge"] == 0.5


def test_uses_only_the_statically_assigned_predecessor() -> None:
    base = _StubBase()
    controller = _coordinator(
        base,
        participant_id="bluerov2_03",
        predecessor_id="bluerov2_02",
    )
    inbox = _inbox("bluerov2_01", 10) + _inbox("bluerov2_02", 0)
    controller.step(_observation(1.0, inbox=inbox))
    assert controller.is_holding is True


def test_ignores_reports_from_non_predecessors() -> None:
    base = _StubBase()
    controller = _coordinator(base)
    controller.step(_observation(1.0, inbox=_inbox("bluerov2_03", 0)))
    assert controller.is_holding is False


def test_stale_predecessor_information_is_ignored() -> None:
    base = _StubBase()
    controller = _coordinator(base)
    controller.step(
        _observation(0.0, inbox=_inbox("bluerov2_01", 0, received_at_s=0.0))
    )
    assert controller.is_holding is True

    controller.step(_observation(5.0, inbox=[]))
    assert controller.is_holding is False
    assert controller.coordination_diagnostics["decision_reason"] == "stale_predecessor_report"
    assert controller.coordination_diagnostics["predecessor_report_age_s"] == 5.0


def test_without_a_comms_channel_it_degrades_to_the_base() -> None:
    base = _StubBase()
    controller = _coordinator(base)
    command = controller.step(_observation(1.0, inbox=None))
    assert controller.is_holding is False
    assert command["surge"] == 0.5
    assert controller.coordination_diagnostics["decision_reason"] == "no_predecessor_report"


@pytest.mark.parametrize(
    ("beacon_index", "status", "expected_reason"),
    [
        (0, "FINISHED", "predecessor_reported_finished"),
        (12, "RUNNING", "predecessor_reported_sequence_complete"),
        (0, "RUNNING", "predecessor_not_ahead_in_local_sequence"),
        (1, "RUNNING", "local_gate_gap_below_minimum"),
        (2, "RUNNING", "local_gate_gap_satisfied"),
    ],
)
def test_coordination_diagnostics_explain_sequence_and_gap_decisions(
    beacon_index: int,
    status: str,
    expected_reason: str,
) -> None:
    base = _StubBase()
    controller = _coordinator(base)

    controller.step(
        _observation(
            1.0,
            inbox=_inbox("bluerov2_01", beacon_index, status=status),
        )
    )

    diagnostics = controller.coordination_diagnostics
    assert diagnostics["decision_reason"] == expected_reason
    assert diagnostics["predecessor_id"] == "bluerov2_01"
    assert diagnostics["predecessor_local_beacon_index"] == beacon_index
    assert diagnostics["predecessor_local_status"] == status
    assert diagnostics["min_gate_gap"] == 2
    assert diagnostics["hold_reason"] == (
        expected_reason if diagnostics["is_holding"] else None
    )
    # The diagnostics are structured controller-local data, not privileged
    # referee fields and not an extension of the acoustic heartbeat.
    json.dumps(diagnostics)
    assert not {
        "completed_gates",
        "target_sequence_index",
        "official_status",
        "referee_finish_state",
    } & set(diagnostics)


def test_coordination_diagnostic_counters_follow_the_unchanged_commands() -> None:
    base = _StubBase()
    controller = _coordinator(base)

    held = controller.step(_observation(1.0, inbox=_inbox("bluerov2_01", 1)))
    advanced = controller.step(
        _observation(1.1, inbox=_inbox("bluerov2_01", 2, received_at_s=1.1))
    )

    assert held["surge"] == 0.0
    assert advanced["surge"] == 0.5
    assert controller.coordination_diagnostics["is_holding"] is False
    assert controller.coordination_diagnostics["hold_steps"] == 1
    assert controller.coordination_diagnostics["advance_steps"] == 1


# -- heartbeat ---------------------------------------------------------------


def test_heartbeat_carries_only_local_tracker_estimates() -> None:
    base = _StubBase()
    controller = _coordinator(
        base,
        participant_id="bluerov2_01",
        predecessor_id=None,
    )
    _set_progress(base, 3)
    command = controller.step(_observation(0.0, inbox=[]))
    message = command.get("message")

    assert message == {
        "local_beacon_index": 3,
        "local_lap": 1,
        "local_status": "RUNNING",
    }
    assert len(json.dumps(message).encode("utf-8")) <= 128
    assert not {"completed_gates", "target_sequence_index", "status"} & set(message)


def test_heartbeat_is_rate_limited_by_local_time() -> None:
    controller = _coordinator(
        _StubBase(),
        participant_id="bluerov2_01",
        predecessor_id=None,
    )
    first = controller.step(_observation(0.0, inbox=[]))
    soon = controller.step(_observation(0.1, inbox=[]))
    later = controller.step(_observation(0.7, inbox=[]))
    assert "message" in first
    assert "message" not in soon
    assert "message" in later


def test_does_not_read_referee_or_ground_truth_fields() -> None:
    controller = _coordinator(_StubBase())
    observation = _ForbiddenFieldGuard(
        _observation(1.0, inbox=_inbox("bluerov2_01", 0))
    )
    controller.step(observation)


def test_default_onboard_base_runs_with_the_official_contract() -> None:
    controller = LeaderFollowerController()
    controller.reset(_mission_info("bluerov2_01", None))
    observation = _ForbiddenFieldGuard(_observation(0.0, inbox=None))

    command = controller.step(observation)

    assert set(command) == {"surge", "sway", "heave", "yaw", "message"}
    assert controller.is_holding is False
    assert controller.base.tracker.expected_beacon_id == "B01"


class _ForbiddenFieldGuard(dict):
    _forbidden = {"race", "beacon", "debug_ground_truth"}

    def get(self, key, default=None):  # type: ignore[no-untyped-def]
        if key in self._forbidden:
            raise AssertionError(f"coordinator accessed forbidden field {key}")
        return super().get(key, default)

    def __getitem__(self, key):  # type: ignore[no-untyped-def]
        if key in self._forbidden:
            raise AssertionError(f"coordinator accessed forbidden field {key}")
        return super().__getitem__(key)
