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
    RuleGateBaselineController,
    VISION_PHASE_ACOUSTIC_APPROACH,
    VISION_PHASE_EXIT_GATE,
    VISION_PHASE_TURN_TO_BEACON,
    VISION_PHASE_VISUAL_ALIGN,
    VISION_PHASE_VISUAL_TRANSIT,
    VisionGateBaselineController,
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
        ("rule_gate_baseline", RuleGateBaselineController),
        ("vision_gate_baseline", VisionGateBaselineController),
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


@pytest.mark.parametrize(
    "alias",
    ["acoustic_baseline", "acoustic_vision_baseline", "rule_gate_baseline", "vision_gate_baseline"],
)
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
    [
        AcousticBaselineController(),
        AcousticVisionBaselineController(),
        RuleGateBaselineController(),
        VisionGateBaselineController(),
    ],
)
def test_official_baselines_return_valid_commands_for_synthetic_observation(controller: object) -> None:
    controller.reset({"max_command": 0.95})

    command = controller.step(_guarded_observation(_synthetic_observation()))

    assert set(command) == {"surge", "sway", "heave", "yaw"}
    assert all(-1.0 <= value <= 1.0 for value in command.values())
    assert command["surge"] > 0.0


@pytest.mark.parametrize(
    "controller",
    [
        AcousticBaselineController(),
        AcousticVisionBaselineController(),
        RuleGateBaselineController(),
        VisionGateBaselineController(),
    ],
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


def test_rule_gate_baseline_large_positive_bearing_gives_small_positive_yaw() -> None:
    controller = RuleGateBaselineController()
    controller.reset({"max_command": 0.95})
    observation = _synthetic_observation()
    observation["sensors"].pop("FrontCamera", None)
    observation["beacon"]["bearing_deg"] = 60.0

    command = controller.step(_guarded_observation(observation))

    assert 0.0 < command["yaw"] <= 0.12
    assert command["surge"] < 0.20


def test_rule_gate_baseline_large_negative_bearing_gives_small_negative_yaw() -> None:
    controller = RuleGateBaselineController()
    controller.reset({"max_command": 0.95})
    observation = _synthetic_observation()
    observation["sensors"].pop("FrontCamera", None)
    observation["beacon"]["bearing_deg"] = -60.0

    command = controller.step(_guarded_observation(observation))

    assert -0.12 <= command["yaw"] < 0.0
    assert command["surge"] < 0.20


def test_rule_gate_baseline_centered_gate_surges_forward() -> None:
    controller = RuleGateBaselineController()
    controller.reset({"max_command": 0.95})
    observation = _synthetic_observation()
    observation["beacon"]["bearing_deg"] = 0.0
    observation["beacon"]["elevation_deg"] = 0.0
    observation["sensors"]["FrontCamera"] = _camera_with_colored_gate_blob(x_start=35, x_stop=62)

    for _ in range(5):
        command = controller.step(_guarded_observation(observation))

    assert command["surge"] > 0.25
    assert abs(command["yaw"]) < 0.04


def test_rule_gate_baseline_off_center_gate_uses_gentle_yaw() -> None:
    controller = RuleGateBaselineController()
    controller.reset({"max_command": 0.95})
    observation = _synthetic_observation()
    observation["beacon"]["bearing_deg"] = 0.0
    observation["beacon"]["elevation_deg"] = 0.0
    observation["sensors"]["FrontCamera"] = _camera_with_colored_gate_blob(x_start=65, x_stop=92)

    command = controller.step(_guarded_observation(observation))

    assert 0.0 < command["yaw"] <= controller.max_yaw
    assert 0.0 <= command["sway"] <= controller.max_sway
    assert command["surge"] < 0.20


def test_rule_gate_baseline_missing_camera_uses_beacon_fallback() -> None:
    controller = RuleGateBaselineController()
    controller.reset({"max_command": 0.95})
    observation = _synthetic_observation()
    observation["sensors"].pop("FrontCamera", None)
    observation["beacon"]["bearing_deg"] = 30.0

    command = controller.step(_guarded_observation(observation))

    assert command["surge"] > 0.0
    assert command["yaw"] > 0.0


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


def test_vision_gate_baseline_handles_missing_camera_safely() -> None:
    controller = VisionGateBaselineController()
    controller.reset({"max_command": 0.95})
    observation = _synthetic_observation()
    observation["sensors"].pop("FrontCamera", None)

    command = controller.step(_guarded_observation(observation))

    assert controller.phase == VISION_PHASE_ACOUSTIC_APPROACH
    assert set(command) == {"surge", "sway", "heave", "yaw"}
    assert all(-1.0 <= value <= 1.0 for value in command.values())


def test_vision_gate_baseline_handles_invalid_image_safely() -> None:
    controller = VisionGateBaselineController()
    controller.reset({"max_command": 0.95})
    observation = _synthetic_observation()
    observation["sensors"]["FrontCamera"] = "not an image"

    command = controller.step(_guarded_observation(observation))

    assert controller.phase == VISION_PHASE_ACOUSTIC_APPROACH
    assert all(-1.0 <= value <= 1.0 for value in command.values())


def test_vision_gate_baseline_low_confidence_falls_back_to_acoustic_phase() -> None:
    controller = VisionGateBaselineController()
    controller.reset({"max_command": 0.95})
    observation = _synthetic_observation()
    observation["beacon"]["bearing_deg"] = 0.0
    observation["sensors"]["FrontCamera"] = [[[5, 5, 5] for _ in range(80)] for _ in range(60)]

    command = controller.step(_guarded_observation(observation))

    assert controller.phase == VISION_PHASE_ACOUSTIC_APPROACH
    assert command["surge"] > 0.0
    assert abs(command["yaw"]) <= controller.max_yaw


def test_vision_gate_baseline_does_not_acquire_visual_when_beacon_is_off_axis() -> None:
    controller = VisionGateBaselineController()
    controller.reset({"max_command": 0.95})
    observation = _synthetic_observation()
    observation["beacon"]["range_m"] = 7.0
    observation["beacon"]["bearing_deg"] = 42.0
    observation["sensors"]["FrontCamera"] = _camera_with_colored_gate_blob(x_start=35, x_stop=62)

    command = controller.step(_guarded_observation(observation))

    assert controller.phase == VISION_PHASE_TURN_TO_BEACON
    assert command["surge"] > 0.0


def test_vision_gate_baseline_rejects_centered_visual_when_close_beacon_bearing_is_large() -> None:
    controller = VisionGateBaselineController()
    controller.reset({"max_command": 0.95})
    observation = _synthetic_observation()
    observation["beacon"]["range_m"] = 4.0
    observation["beacon"]["bearing_deg"] = 60.0
    observation["sensors"]["FrontCamera"] = _camera_with_colored_gate_blob(x_start=35, x_stop=62)

    controller.step(_guarded_observation(observation))

    assert controller.phase == VISION_PHASE_TURN_TO_BEACON


def test_vision_gate_baseline_allows_strong_close_visual_when_beacon_is_noisy() -> None:
    controller = VisionGateBaselineController()
    controller.reset({"max_command": 0.95})
    observation = _synthetic_observation()
    observation["beacon"]["range_m"] = 2.5
    observation["beacon"]["bearing_deg"] = 95.0
    observation["sensors"]["FrontCamera"] = _camera_with_colored_gate_blob(x_start=35, x_stop=62)

    controller.step(_guarded_observation(observation))

    assert controller.phase in {VISION_PHASE_VISUAL_ALIGN, VISION_PHASE_VISUAL_TRANSIT}


def test_vision_gate_baseline_large_positive_beacon_bearing_gives_negative_holoocean_yaw() -> None:
    controller = VisionGateBaselineController()
    controller.reset({"max_command": 0.95})
    observation = _synthetic_observation()
    observation["sensors"].pop("FrontCamera", None)
    observation["beacon"]["range_m"] = 7.0
    observation["beacon"]["bearing_deg"] = 45.0

    command = controller.step(_guarded_observation(observation))

    assert controller.phase == VISION_PHASE_TURN_TO_BEACON
    assert command["yaw"] < 0.0


def test_vision_gate_baseline_large_negative_beacon_bearing_gives_positive_holoocean_yaw() -> None:
    controller = VisionGateBaselineController()
    controller.reset({"max_command": 0.95})
    observation = _synthetic_observation()
    observation["sensors"].pop("FrontCamera", None)
    observation["beacon"]["range_m"] = 7.0
    observation["beacon"]["bearing_deg"] = -45.0

    command = controller.step(_guarded_observation(observation))

    assert controller.phase == VISION_PHASE_TURN_TO_BEACON
    assert command["yaw"] > 0.0


def test_vision_gate_baseline_large_beacon_bearing_reduces_surge() -> None:
    straight = VisionGateBaselineController()
    turning = VisionGateBaselineController()
    straight.reset({"max_command": 0.95})
    turning.reset({"max_command": 0.95})
    straight_observation = _synthetic_observation()
    turning_observation = _synthetic_observation()
    straight_observation["sensors"].pop("FrontCamera", None)
    turning_observation["sensors"].pop("FrontCamera", None)
    straight_observation["beacon"]["range_m"] = 7.0
    straight_observation["beacon"]["bearing_deg"] = 0.0
    turning_observation["beacon"]["range_m"] = 7.0
    turning_observation["beacon"]["bearing_deg"] = 70.0

    for _ in range(5):
        straight_command = straight.step(_guarded_observation(straight_observation))
        turning_command = turning.step(_guarded_observation(turning_observation))

    assert turning_command["surge"] < straight_command["surge"]


def test_vision_gate_baseline_turn_direction_follows_wrapped_bearing() -> None:
    controller = VisionGateBaselineController()
    controller.reset({"max_command": 0.95})
    positive = _synthetic_observation()
    wrapped = _synthetic_observation()
    positive["sensors"].pop("FrontCamera", None)
    wrapped["sensors"].pop("FrontCamera", None)
    positive["beacon"]["range_m"] = 4.0
    positive["beacon"]["bearing_deg"] = 80.0
    wrapped["beacon"]["range_m"] = 4.0
    wrapped["beacon"]["bearing_deg"] = -170.0

    controller.step(_guarded_observation(positive))
    for _ in range(4):
        command = controller.step(_guarded_observation(wrapped))

    assert controller.phase == VISION_PHASE_TURN_TO_BEACON
    assert command["yaw"] > 0.0


def test_vision_gate_baseline_turn_direction_updates_when_bearing_really_changes_side() -> None:
    controller = VisionGateBaselineController()
    controller.reset({"max_command": 0.95})
    positive = _synthetic_observation()
    negative = _synthetic_observation()
    positive["sensors"].pop("FrontCamera", None)
    negative["sensors"].pop("FrontCamera", None)
    positive["beacon"]["range_m"] = 4.0
    positive["beacon"]["bearing_deg"] = 80.0
    negative["beacon"]["range_m"] = 4.0
    negative["beacon"]["bearing_deg"] = -60.0

    controller.step(_guarded_observation(positive))
    for _ in range(4):
        command = controller.step(_guarded_observation(negative))

    assert controller.phase == VISION_PHASE_TURN_TO_BEACON
    assert command["yaw"] > 0.0


def test_vision_gate_baseline_phase_transitions() -> None:
    controller = VisionGateBaselineController()
    controller.reset({"max_command": 0.95})
    no_camera = _synthetic_observation()
    no_camera["sensors"].pop("FrontCamera", None)
    align = _synthetic_observation()
    align["sensors"]["FrontCamera"] = _camera_with_colored_gate_blob(x_start=65, x_stop=92)
    transit = _synthetic_observation()
    transit["sensors"]["FrontCamera"] = _camera_with_colored_gate_blob(x_start=30, x_stop=70)
    crossed = _synthetic_observation()
    crossed["race"]["completed_gates"] = 1
    crossed["sensors"]["FrontCamera"] = _camera_with_colored_gate_blob(x_start=30, x_stop=70)

    controller.step(_guarded_observation(no_camera))
    assert controller.phase == VISION_PHASE_ACOUSTIC_APPROACH
    controller.step(_guarded_observation(align))
    assert controller.phase == VISION_PHASE_VISUAL_ALIGN
    controller.step(_guarded_observation(transit))
    assert controller.phase == VISION_PHASE_VISUAL_TRANSIT
    controller.step(_guarded_observation(crossed))
    assert controller.phase == VISION_PHASE_EXIT_GATE


def test_vision_gate_baseline_yaw_is_bounded() -> None:
    controller = VisionGateBaselineController()
    controller.reset({"max_command": 0.95})
    observation = _synthetic_observation()
    observation["sensors"]["FrontCamera"] = _camera_with_colored_gate_blob(x_start=78, x_stop=99)

    for _ in range(8):
        command = controller.step(_guarded_observation(observation))

    assert abs(command["yaw"]) <= controller.max_yaw


def test_vision_gate_baseline_visual_align_yaws_toward_image_error() -> None:
    left = VisionGateBaselineController()
    right = VisionGateBaselineController()
    left.reset({"max_command": 0.95})
    right.reset({"max_command": 0.95})
    left_observation = _synthetic_observation()
    right_observation = _synthetic_observation()
    left_observation["beacon"]["bearing_deg"] = 0.0
    right_observation["beacon"]["bearing_deg"] = 0.0
    left_observation["sensors"]["FrontCamera"] = _camera_with_colored_gate_blob(x_start=8, x_stop=35)
    right_observation["sensors"]["FrontCamera"] = _camera_with_colored_gate_blob(x_start=65, x_stop=92)

    left_command = left.step(_guarded_observation(left_observation))
    right_command = right.step(_guarded_observation(right_observation))

    assert left.phase == VISION_PHASE_VISUAL_ALIGN
    assert right.phase == VISION_PHASE_VISUAL_ALIGN
    assert left_command["yaw"] < 0.0
    assert right_command["yaw"] > 0.0


def test_vision_gate_baseline_persistent_post_gate_collision_backs_off() -> None:
    controller = VisionGateBaselineController()
    controller.reset({"max_command": 0.95})
    observation = _synthetic_observation()
    observation["beacon"]["range_m"] = 4.0
    observation["race"]["target_sequence_index"] = 2
    observation["sensors"].pop("FrontCamera", None)
    observation["sensors"]["CollisionSensor"] = True

    for _ in range(10):
        command = controller.step(_guarded_observation(observation))

    assert command["surge"] < 0.0


def test_vision_gate_baseline_collision_near_current_gate_does_not_back_off() -> None:
    controller = VisionGateBaselineController()
    controller.reset({"max_command": 0.95})
    observation = _synthetic_observation()
    observation["beacon"]["range_m"] = 1.0
    observation["race"]["target_sequence_index"] = 2
    observation["sensors"].pop("FrontCamera", None)
    observation["sensors"]["CollisionSensor"] = True

    for _ in range(10):
        command = controller.step(_guarded_observation(observation))

    assert command["surge"] > 0.0


def test_vision_gate_baseline_exit_gate_collision_does_not_back_off() -> None:
    controller = VisionGateBaselineController()
    controller.reset({"max_command": 0.95})
    controller._phase = VISION_PHASE_EXIT_GATE
    controller._exit_steps_remaining = 24
    observation = _synthetic_observation()
    observation["beacon"]["range_m"] = 5.0
    observation["race"]["target_sequence_index"] = 3
    observation["sensors"].pop("FrontCamera", None)
    observation["sensors"]["CollisionSensor"] = True

    for _ in range(10):
        command = controller.step(_guarded_observation(observation))

    assert command["surge"] > 0.0


def test_vision_gate_baseline_collision_before_third_gate_does_not_back_off() -> None:
    controller = VisionGateBaselineController()
    controller.reset({"max_command": 0.95})
    observation = _synthetic_observation()
    observation["beacon"]["range_m"] = 1.0
    observation["race"]["target_sequence_index"] = 1
    observation["sensors"].pop("FrontCamera", None)
    observation["sensors"]["CollisionSensor"] = True

    for _ in range(10):
        command = controller.step(_guarded_observation(observation))

    assert command["surge"] > 0.0


def test_vision_gate_baseline_visual_heave_uses_camera_y_sign() -> None:
    controller = VisionGateBaselineController()
    controller.reset({"max_command": 0.95})
    observation = _synthetic_observation()
    observation["beacon"]["elevation_deg"] = 0.0
    observation["sensors"]["FrontCamera"] = _camera_with_colored_gate_blob(
        x_start=35,
        x_stop=62,
        y_start=42,
        y_stop=76,
    )

    command = controller.step(_guarded_observation(observation))

    assert controller.phase == VISION_PHASE_VISUAL_ALIGN
    assert command["heave"] < 0.0


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


def _camera_with_colored_gate_blob(
    x_start: int,
    x_stop: int,
    y_start: int = 22,
    y_stop: int = 58,
) -> list[list[list[int]]]:
    width = 100
    height = 80
    image = [[[8, 14, 22] for _ in range(width)] for _ in range(height)]
    for y in range(y_start, y_stop):
        for x in range(x_start, x_stop):
            if x in (x_start, x_stop - 1) or y in (y_start, y_stop - 1):
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
