from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from marine_race_arena.config.loader import load_track_config
from marine_race_arena.controllers.official_baselines import (
    AcousticBaselineController,
    AcousticVisionBaselineController,
    PHASE_APPROACH_GATE,
    PHASE_EXIT_GATE,
    PHASE_TRANSIT_GATE,
)
from marine_race_arena.participants.controller_loader import ControllerLoader
from marine_race_arena.participants.participant import RaceParticipant
from marine_race_arena.scripts.run_benchmark import main as run_benchmark_main
from marine_race_arena.scripts.run_marine_race import _reject_invalid_official_controllers


TRACK_DIR = Path(__file__).resolve().parents[1] / "marine_race_arena" / "tracks"


@pytest.mark.parametrize(
    ("alias", "controller_type"),
    [
        ("acoustic_baseline", AcousticBaselineController),
        ("acoustic_vision_baseline", AcousticVisionBaselineController),
    ],
)
def test_official_baseline_aliases_load_and_are_not_ground_truth(
    alias: str,
    controller_type: type,
) -> None:
    controller = ControllerLoader().load(alias)

    assert isinstance(controller, controller_type)
    assert controller.debug_only is False
    assert controller.uses_ground_truth is False


@pytest.mark.parametrize("alias", ["acoustic_baseline", "acoustic_vision_baseline"])
def test_official_mode_accepts_reproducible_baselines(alias: str) -> None:
    config = load_track_config(TRACK_DIR / "marine_race_horseshoe_bay.json")
    config = replace(config, race=replace(config.race, official_mode=True))
    participant_config = replace(config.participants[0], controller=alias)
    controller = ControllerLoader().load(alias)
    participant = RaceParticipant(
        config=participant_config,
        controller=controller,
        position=tuple(participant_config.spawn["position"]),
        rotation_rpy_deg=tuple(participant_config.spawn["rotation_rpy_deg"]),
    )

    _reject_invalid_official_controllers(config, {participant.id: participant})


@pytest.mark.parametrize(
    "controller",
    [AcousticBaselineController(), AcousticVisionBaselineController()],
)
def test_official_baselines_return_valid_commands_for_synthetic_observation(controller: object) -> None:
    controller.reset({"max_command": 0.95})

    command = controller.step(_guarded_observation(_synthetic_observation()))

    assert set(command) == {"surge", "sway", "heave", "yaw"}
    assert all(-1.0 <= value <= 1.0 for value in command.values())
    assert command["surge"] > 0.0


@pytest.mark.parametrize(
    "controller",
    [AcousticBaselineController(), AcousticVisionBaselineController()],
)
def test_official_baselines_handle_missing_inputs_safely(controller: object) -> None:
    controller.reset({"max_command": 0.95})

    command = controller.step(_guarded_observation({"beacon": {}, "sensors": {}, "race": {}}))

    assert set(command) == {"surge", "sway", "heave", "yaw"}
    assert all(-1.0 <= value <= 1.0 for value in command.values())


def test_acoustic_baseline_yaw_deadband() -> None:
    controller = AcousticBaselineController()
    controller.reset({"max_command": 0.95})
    observation = _synthetic_observation()
    observation["beacon"]["bearing_deg"] = 1.5
    observation["beacon"]["range_m"] = 8.0

    command = controller.step(_guarded_observation(observation))

    assert command["yaw"] == 0.0


def test_acoustic_baseline_reduces_speed_near_gate() -> None:
    far_controller = AcousticBaselineController()
    near_controller = AcousticBaselineController()
    far_controller.reset({"max_command": 0.95})
    near_controller.reset({"max_command": 0.95})
    far_observation = _synthetic_observation()
    far_observation["beacon"]["range_m"] = 12.0
    far_observation["beacon"]["bearing_deg"] = 0.0
    near_observation = _synthetic_observation()
    near_observation["beacon"]["range_m"] = 1.4
    near_observation["beacon"]["bearing_deg"] = 0.0

    far_command = far_controller.step(_guarded_observation(far_observation))
    near_command = near_controller.step(_guarded_observation(near_observation))

    assert far_controller.phase == PHASE_APPROACH_GATE
    assert near_controller.phase == PHASE_TRANSIT_GATE
    assert near_command["surge"] < far_command["surge"]


def test_acoustic_baseline_phase_transitions() -> None:
    controller = AcousticBaselineController()
    controller.reset({"max_command": 0.95})
    far_observation = _synthetic_observation()
    far_observation["race"]["completed_gates"] = 0
    far_observation["beacon"]["range_m"] = 8.0
    near_observation = _synthetic_observation()
    near_observation["race"]["completed_gates"] = 0
    near_observation["beacon"]["range_m"] = 1.8
    crossed_observation = _synthetic_observation()
    crossed_observation["race"]["completed_gates"] = 1
    crossed_observation["beacon"]["range_m"] = 7.0

    controller.step(_guarded_observation(far_observation))
    assert controller.phase == PHASE_APPROACH_GATE
    controller.step(_guarded_observation(near_observation))
    assert controller.phase == PHASE_TRANSIT_GATE
    controller.step(_guarded_observation(crossed_observation))
    assert controller.phase == PHASE_EXIT_GATE


def test_acoustic_baseline_missing_beacon_fallback_safety() -> None:
    controller = AcousticBaselineController()
    controller.reset({"max_command": 0.95})

    command = controller.step(_guarded_observation({"beacon": {"valid": False}, "sensors": {}, "race": {}}))

    assert command["surge"] > 0.0
    assert all(-1.0 <= value <= 1.0 for value in command.values())


def test_vision_baseline_falls_back_without_camera() -> None:
    observation = _synthetic_observation()
    observation["sensors"].pop("FrontCamera", None)
    acoustic = AcousticBaselineController()
    vision = AcousticVisionBaselineController()
    acoustic.reset({"max_command": 0.95})
    vision.reset({"max_command": 0.95})

    assert vision.step(_guarded_observation(observation)) == pytest.approx(
        acoustic.step(_guarded_observation(observation))
    )


def test_vision_baseline_uses_front_camera_for_local_alignment() -> None:
    controller = AcousticVisionBaselineController()
    controller.reset({"max_command": 0.95})
    observation = _synthetic_observation()
    observation["beacon"]["bearing_deg"] = 0.0
    observation["beacon"]["elevation_deg"] = 0.0
    observation["sensors"]["FrontCamera"] = _camera_with_colored_gate_blob(x_start=65, x_stop=92)

    command = controller.step(_guarded_observation(observation))

    assert command["yaw"] < 0.0
    assert command["sway"] < 0.0


def test_acoustic_baseline_smoke_benchmark_with_fallback(tmp_path: Path) -> None:
    output_dir = tmp_path / "benchmark"

    exit_code = run_benchmark_main(
        [
            "--benchmark-task",
            "clean_gate",
            "--track",
            str(TRACK_DIR / "marine_race_horseshoe_bay.json"),
            "--controller",
            "acoustic_baseline",
            "--adapter",
            "fallback",
            "--seeds",
            "0",
            "--duration",
            "0.2",
            "--dt",
            "0.1",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "benchmark_summary.csv").exists()
    assert (output_dir / "benchmark_summary.json").exists()


def _synthetic_observation() -> dict:
    return {
        "beacon": {
            "valid": True,
            "bearing_deg": 12.0,
            "elevation_deg": -4.0,
            "range_m": 6.0,
            "signal_strength": 0.8,
        },
        "sensors": {
            "depth_m": 4.0,
            "DVLSensor": [0.0, 0.0, 0.0],
            "FrontCamera": _camera_with_colored_gate_blob(x_start=35, x_stop=62),
        },
        "race": {
            "status": "RUNNING",
            "completed_gates": 0,
            "target_gate_id": "G01",
        },
    }


def _camera_with_colored_gate_blob(x_start: int, x_stop: int) -> list[list[list[int]]]:
    width = 100
    height = 80
    image = [[[8, 14, 22] for _ in range(width)] for _ in range(height)]
    for y in range(22, 58):
        for x in range(x_start, x_stop):
            if x in (x_start, x_stop - 1) or y in (22, 57):
                image[y][x] = [20, 235, 80]
    return image


def _guarded_observation(observation: dict) -> "_GroundTruthGuard":
    guarded = _GroundTruthGuard(observation)
    guarded["debug_ground_truth"] = _GroundTruthAccessError()
    return guarded


class _GroundTruthGuard(dict):
    def get(self, key, default=None):  # type: ignore[no-untyped-def]
        if key == "debug_ground_truth":
            raise AssertionError("official baseline accessed debug_ground_truth")
        return super().get(key, default)

    def __getitem__(self, key):  # type: ignore[no-untyped-def]
        if key == "debug_ground_truth":
            raise AssertionError("official baseline accessed debug_ground_truth")
        return super().__getitem__(key)


class _GroundTruthAccessError:
    def __getattribute__(self, name):  # type: ignore[no-untyped-def]
        raise AssertionError("official baseline accessed debug_ground_truth")
