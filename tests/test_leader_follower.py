from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from marine_race_arena.controllers.leader_follower import (
    LeaderFollowerAcousticController,
    LeaderFollowerController,
)
from marine_race_arena.controllers.official_baselines import (
    AcousticBaselineController,
    SmoothGateBaselineController,
)
from marine_race_arena.participants.controller_interface import BaseController
from marine_race_arena.participants.controller_loader import ControllerLoader
import marine_race_arena.scripts.run_marine_race as run_marine_race
from marine_race_arena.scripts.run_algorithm_comparison import simulate_fleet

TRACK = "marine_race_arena/tracks/marine_race_horseshoe_bay.json"
RACE_INFO = {"laps": 1, "gates_per_lap": 12, "max_command": 0.95}


class _StubBase(BaseController):
    """A base controller that returns a fixed non-zero command and records calls."""

    uses_ground_truth = False

    def __init__(self, uses_ground_truth: bool = False) -> None:
        self.uses_ground_truth = uses_ground_truth
        self.reset_calls = 0
        self.step_calls = 0
        self.closed = False

    def reset(self, race_info: Dict[str, Any]) -> None:
        self.reset_calls += 1

    def step(self, observation: Dict[str, Any]) -> Dict[str, float]:
        self.step_calls += 1
        return {"surge": 0.5, "sway": 0.1, "heave": -0.2, "yaw": 0.3}

    def close(self) -> None:
        self.closed = True


def _inbox(sender: str, completed: int, status: str = "R", received_at_s: float = 0.9) -> List[Dict[str, Any]]:
    return [{"from": sender, "payload": {"g": completed, "t": completed, "st": status}, "received_at_s": received_at_s}]


def _observation(
    participant_id: str,
    time_s: float,
    completed: int,
    inbox: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    observation: Dict[str, Any] = {
        "participant_id": participant_id,
        "time_s": time_s,
        "beacon": {"valid": True, "bearing_deg": 0.0, "elevation_deg": 0.0, "range_m": 6.0},
        "sensors": {"depth_m": 4.0},
        "race": {"status": "RUNNING", "completed_gates": completed, "target_sequence_index": completed},
    }
    if inbox is not None:
        observation["comms"] = {"inbox": inbox}
    return observation


def _coordinator(base: BaseController, min_gate_gap: int = 2) -> LeaderFollowerController:
    controller = LeaderFollowerController(base_controller=base, min_gate_gap=min_gate_gap)
    controller.reset(RACE_INFO)
    return controller


# -- loading / legality ------------------------------------------------------


@pytest.mark.parametrize(
    ("alias", "controller_type", "base_type"),
    [
        ("leader_follower", LeaderFollowerController, None),
        ("leader_follower_acoustic", LeaderFollowerAcousticController, AcousticBaselineController),
    ],
)
def test_aliases_load_and_are_not_ground_truth(alias, controller_type, base_type) -> None:
    controller = ControllerLoader().load(alias)
    assert isinstance(controller, controller_type)
    assert controller.uses_ground_truth is False
    if base_type is not None:
        assert isinstance(controller.base, base_type)


def test_reflects_the_base_controllers_ground_truth_flag() -> None:
    honest = LeaderFollowerController(base_controller=_StubBase(uses_ground_truth=False))
    cheating = LeaderFollowerController(base_controller=_StubBase(uses_ground_truth=True))
    assert honest.uses_ground_truth is False
    assert cheating.uses_ground_truth is True


def test_base_may_not_be_a_coordination_alias() -> None:
    with pytest.raises(ValueError):
        LeaderFollowerController(base_alias="leader_follower")


def test_close_forwards_to_base() -> None:
    base = _StubBase()
    controller = _coordinator(base)
    controller.close()
    assert base.closed is True


# -- hold / advance decisions ------------------------------------------------


def test_leader_without_predecessor_runs_the_base() -> None:
    base = _StubBase()
    controller = _coordinator(base)
    # bluerov2_01 is the smallest id; the only teammate heard started behind it.
    command = controller.step(_observation("bluerov2_01", 1.0, completed=0, inbox=_inbox("bluerov2_02", 0)))
    assert base.step_calls == 1
    assert command["surge"] == 0.5
    assert controller.is_holding is False


def test_follower_holds_when_predecessor_is_not_far_enough_ahead() -> None:
    base = _StubBase()
    controller = _coordinator(base)
    command = controller.step(_observation("bluerov2_02", 1.0, completed=0, inbox=_inbox("bluerov2_01", 1)))
    # gap of 1 < min_gate_gap (2): yield, do not run the base motion.
    assert controller.is_holding is True
    assert base.step_calls == 0
    assert command["surge"] == 0.0
    assert command["sway"] == 0.0


def test_follower_advances_when_predecessor_is_far_enough_ahead() -> None:
    base = _StubBase()
    controller = _coordinator(base)
    command = controller.step(_observation("bluerov2_02", 1.0, completed=0, inbox=_inbox("bluerov2_01", 2)))
    assert controller.is_holding is False
    assert base.step_calls == 1
    assert command["surge"] == 0.5


def test_follower_advances_once_predecessor_has_left_the_course() -> None:
    base = _StubBase()
    controller = _coordinator(base)
    command = controller.step(
        _observation("bluerov2_02", 1.0, completed=0, inbox=_inbox("bluerov2_01", 0, status="F"))
    )
    assert controller.is_holding is False
    assert command["surge"] == 0.5


def test_follower_advances_when_predecessor_has_finished_all_gates() -> None:
    base = _StubBase()
    controller = _coordinator(base)
    command = controller.step(_observation("bluerov2_02", 1.0, completed=0, inbox=_inbox("bluerov2_01", 12)))
    assert controller.is_holding is False
    assert command["surge"] == 0.5


def test_yields_to_the_immediate_predecessor_not_a_distant_leader() -> None:
    base = _StubBase()
    controller = _coordinator(base)
    inbox = _inbox("bluerov2_01", 10) + _inbox("bluerov2_02", 0)
    # bluerov2_03 must yield to its immediate predecessor bluerov2_02 (close),
    # even though the distant leader bluerov2_01 is far ahead.
    controller.step(_observation("bluerov2_03", 1.0, completed=0, inbox=inbox))
    assert controller.is_holding is True


def test_ignores_teammates_that_started_behind_in_the_order() -> None:
    base = _StubBase()
    controller = _coordinator(base)
    # bluerov2_02 hears only bluerov2_03, which is behind it -> no predecessor.
    controller.step(_observation("bluerov2_02", 1.0, completed=0, inbox=_inbox("bluerov2_03", 0)))
    assert controller.is_holding is False


def test_stale_predecessor_information_is_ignored() -> None:
    base = _StubBase()
    controller = _coordinator(base)
    # Hear the predecessor once early...
    controller.step(_observation("bluerov2_02", 0.0, completed=0, inbox=_inbox("bluerov2_01", 0, received_at_s=0.0)))
    assert controller.is_holding is True
    # ...then nothing for longer than STALE_AFTER_S: the predecessor is out of
    # range, so there is no detectable hazard and the follower proceeds.
    controller.step(_observation("bluerov2_02", 5.0, completed=0, inbox=[]))
    assert controller.is_holding is False


def test_without_a_comms_channel_it_degrades_to_the_base() -> None:
    base = _StubBase()
    controller = _coordinator(base)
    # No "comms" key at all (channel disabled): a follower id still just runs base.
    command = controller.step(_observation("bluerov2_02", 1.0, completed=0, inbox=None))
    assert controller.is_holding is False
    assert command["surge"] == 0.5


# -- heartbeat ---------------------------------------------------------------


def test_heartbeat_is_emitted_and_carries_only_legal_tiny_payload() -> None:
    controller = _coordinator(_StubBase())
    command = controller.step(_observation("bluerov2_01", 0.0, completed=3, inbox=_inbox("bluerov2_02", 0)))
    message = command.get("message")
    assert message is not None
    assert set(message) == {"g", "t", "st"}
    assert message["g"] == 3
    assert len(json.dumps(message).encode("utf-8")) <= 128


def test_heartbeat_is_rate_limited() -> None:
    controller = _coordinator(_StubBase())
    first = controller.step(_observation("bluerov2_01", 0.0, completed=0, inbox=[]))
    soon = controller.step(_observation("bluerov2_01", 0.1, completed=0, inbox=[]))
    later = controller.step(_observation("bluerov2_01", 0.7, completed=0, inbox=[]))
    assert "message" in first
    assert "message" not in soon
    assert "message" in later


def test_does_not_read_ground_truth() -> None:
    controller = _coordinator(_StubBase())
    observation = _GroundTruthGuard(_observation("bluerov2_02", 1.0, 0, inbox=_inbox("bluerov2_01", 0)))
    observation["debug_ground_truth"] = _Poison()
    controller.step(observation)


# -- end to end on the kinematic fallback ------------------------------------


def test_coordination_removes_inter_vehicle_events_while_the_team_finishes(monkeypatch) -> None:
    monkeypatch.setattr(run_marine_race, "_print_multi_agent_diagnostics", lambda *a, **k: None)

    def team(coordinated: bool) -> List[BaseController]:
        bases: List[BaseController] = [SmoothGateBaselineController()] + [
            AcousticBaselineController() for _ in range(3)
        ]
        if not coordinated:
            return bases
        return [LeaderFollowerController(base_controller=base) for base in bases]

    uncoordinated = simulate_fleet(
        TRACK, team(False), duration_s=400.0, inter_vehicle_collision_mode="penalize"
    )
    coordinated = simulate_fleet(
        TRACK, team(True), duration_s=400.0, comms_enabled=True, inter_vehicle_collision_mode="penalize"
    )

    raw_team = uncoordinated["team_summary"]
    coord_team = coordinated["team_summary"]

    # Without coordination the faster followers overtake the slower leader and trip
    # the inter-vehicle proximity detector.
    assert raw_team["total_inter_vehicle_collisions"] > 0
    # With coordination the team stays a spaced convoy: no inter-vehicle events, no
    # spurious stuck penalties, and every rover still finishes the gate sequence.
    assert coord_team["total_inter_vehicle_collisions"] == 0
    assert coord_team["all_rovers_finished"] is True
    assert coord_team["total_completed_gates"] == coord_team["expected_total_gates"]
    assert sum(p["stuck_events"] for p in coordinated["participants"]) == 0


def test_the_leader_is_never_slowed_by_coordination(monkeypatch) -> None:
    monkeypatch.setattr(run_marine_race, "_print_multi_agent_diagnostics", lambda *a, **k: None)
    leader = LeaderFollowerController(base_controller=SmoothGateBaselineController())
    followers = [LeaderFollowerController(base_controller=AcousticBaselineController()) for _ in range(2)]
    simulate_fleet(TRACK, [leader] + followers, duration_s=400.0, comms_enabled=True)
    # The frontmost rover has no predecessor, so it never yields.
    assert leader.hold_steps == 0
    assert all(follower.hold_steps > 0 for follower in followers)


class _GroundTruthGuard(dict):
    def get(self, key, default=None):  # type: ignore[no-untyped-def]
        if key == "debug_ground_truth":
            raise AssertionError("coordinator accessed debug_ground_truth")
        return super().get(key, default)

    def __getitem__(self, key):  # type: ignore[no-untyped-def]
        if key == "debug_ground_truth":
            raise AssertionError("coordinator accessed debug_ground_truth")
        return super().__getitem__(key)


class _Poison:
    def __getattribute__(self, name):  # type: ignore[no-untyped-def]
        raise AssertionError("coordinator accessed debug_ground_truth")
