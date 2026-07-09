from __future__ import annotations

from pathlib import Path

import pytest

from marine_race_arena.controllers.official_baselines import (
    SMOOTH_PHASE_APPROACH,
    SMOOTH_PHASE_EXIT,
    SMOOTH_PHASE_TRANSIT,
    AcousticBaselineController,
    SmoothGateBaselineController,
)
from marine_race_arena.participants.controller_loader import ControllerLoader
from marine_race_arena.scripts.run_algorithm_comparison import simulate_fleet

TRACK = "marine_race_arena/tracks/marine_race_horseshoe_bay.json"
COMMAND_KEYS = ("surge", "sway", "heave", "yaw")


def _observation(**beacon_overrides) -> dict:
    beacon = {"valid": True, "bearing_deg": 10.0, "elevation_deg": -3.0, "range_m": 6.0}
    beacon.update(beacon_overrides)
    return {
        "participant_id": "bluerov2_01",
        "time_s": 1.0,
        "beacon": beacon,
        "sensors": {"depth_m": 4.0, "DVLSensor": [0.0, 0.0, 0.0]},
        "race": {"status": "RUNNING", "completed_gates": 0, "target_sequence_index": 0},
    }


def _controller() -> SmoothGateBaselineController:
    controller = SmoothGateBaselineController()
    controller.reset({"max_command": 0.95})
    return controller


def test_alias_loads_and_is_not_ground_truth() -> None:
    controller = ControllerLoader().load("smooth_gate_baseline")
    assert isinstance(controller, SmoothGateBaselineController)
    assert controller.debug_only is False
    assert controller.uses_ground_truth is False


def test_commands_stay_within_the_conservative_envelope() -> None:
    controller = _controller()
    for bearing in (-90.0, -30.0, 0.0, 30.0, 90.0):
        for range_m in (0.5, 3.0, 8.0, 15.0):
            command = controller.step(_observation(bearing_deg=bearing, range_m=range_m))
            assert set(command) >= set(COMMAND_KEYS)
            assert abs(command["surge"]) <= controller.max_surge + 1e-9
            assert abs(command["sway"]) <= controller.max_sway + 1e-9
            assert abs(command["heave"]) <= controller.max_heave + 1e-9
            assert abs(command["yaw"]) <= controller.max_yaw + 1e-9


def test_invalid_beacon_still_creeps_forward_to_search() -> None:
    controller = _controller()
    command = controller.step(_observation(valid=False))
    assert command["surge"] > 0.0  # keeps looking rather than stalling


def test_yaw_turns_toward_the_beacon_bearing() -> None:
    left = _controller().step(_observation(bearing_deg=40.0))
    right = _controller().step(_observation(bearing_deg=-40.0))
    assert left["yaw"] > 0.0
    assert right["yaw"] < 0.0


def test_phase_progresses_approach_transit_exit() -> None:
    controller = _controller()
    assert controller.phase == SMOOTH_PHASE_APPROACH

    # Close and roughly aligned -> commit to the transit phase.
    controller.step(_observation(range_m=2.0, bearing_deg=5.0))
    assert controller.phase == SMOOTH_PHASE_TRANSIT

    # A newly completed gate switches to the exit settle phase.
    observation = _observation(range_m=8.0, bearing_deg=5.0)
    observation["race"]["completed_gates"] = 1
    controller.step(observation)
    assert controller.phase == SMOOTH_PHASE_EXIT


def test_command_changes_are_rate_limited_for_smoothness() -> None:
    controller = _controller()
    previous = controller.step(_observation(bearing_deg=0.0, range_m=10.0))
    # A large step change in the target must not produce a large jump in the output.
    command = controller.step(_observation(bearing_deg=90.0, range_m=1.0))
    assert abs(command["surge"] - previous["surge"]) <= 0.06 + 1e-9
    assert abs(command["yaw"] - previous["yaw"]) <= 0.02 + 1e-9


def test_does_not_read_ground_truth() -> None:
    controller = _controller()
    observation = _GroundTruthGuard(_observation())
    observation["debug_ground_truth"] = _Poison()
    controller.step(observation)  # must not touch debug_ground_truth


@pytest.mark.parametrize("duration_s", [400.0])
def test_completes_horseshoe_more_slowly_than_the_acoustic_baseline(duration_s: float) -> None:
    smooth = simulate_fleet(
        TRACK, [SmoothGateBaselineController()], duration_s=duration_s, inter_vehicle_collision_mode="off"
    )["participants"][0]
    acoustic = simulate_fleet(
        TRACK, [AcousticBaselineController()], duration_s=duration_s, inter_vehicle_collision_mode="off"
    )["participants"][0]

    assert smooth["status"] == "FINISHED"
    assert acoustic["status"] == "FINISHED"
    assert smooth["completed_gates"] == acoustic["completed_gates"]
    # The conservative controller is a genuinely different algorithm: it finishes
    # the same gates but takes meaningfully longer.
    assert smooth["official_time_s"] > acoustic["official_time_s"] * 1.25


def test_fallback_comparison_is_deterministic() -> None:
    # The comparison harness seeds the arena (and thus the beacon-noise RNG), so a
    # repeated run must reproduce the same time and gate count bit-for-bit.
    first = simulate_fleet(
        TRACK, [AcousticBaselineController()], duration_s=200.0, inter_vehicle_collision_mode="off"
    )["participants"][0]
    second = simulate_fleet(
        TRACK, [AcousticBaselineController()], duration_s=200.0, inter_vehicle_collision_mode="off"
    )["participants"][0]
    assert first["official_time_s"] == second["official_time_s"]
    assert first["completed_gates"] == second["completed_gates"]


class _GroundTruthGuard(dict):
    def get(self, key, default=None):  # type: ignore[no-untyped-def]
        if key == "debug_ground_truth":
            raise AssertionError("smooth baseline accessed debug_ground_truth")
        return super().get(key, default)

    def __getitem__(self, key):  # type: ignore[no-untyped-def]
        if key == "debug_ground_truth":
            raise AssertionError("smooth baseline accessed debug_ground_truth")
        return super().__getitem__(key)


class _Poison:
    def __getattribute__(self, name):  # type: ignore[no-untyped-def]
        raise AssertionError("smooth baseline accessed debug_ground_truth")
