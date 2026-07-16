from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from marine_race_arena.config.loader import load_track_config
from marine_race_arena.controllers.local_course_tracker import (
    PHASE_APPROACH,
    PHASE_COMMIT,
    PHASE_FINISHED,
    PHASE_VISUAL_ALIGN,
    LocalCourseTracker,
)
from marine_race_arena.controllers.official_baselines import (
    RuleGateBaselineController,
    RuleGateCenterThenCommitController,
    _yaw_rate_from_sensors,
)
from marine_race_arena.participants.controller_loader import ControllerLoader
from marine_race_arena.participants.participant import RaceParticipant
from marine_race_arena.scripts.run_benchmark import main as run_benchmark_main
from marine_race_arena.scripts.run_marine_race import _reject_invalid_official_controllers


TRACK_DIR = Path(__file__).resolve().parents[1] / "marine_race_arena" / "tracks"
CONTROLLER_TYPES = (RuleGateBaselineController, RuleGateCenterThenCommitController)


def _mission_info(total_beacons: int = 3) -> dict:
    return {
        "participant_id": "bluerov2_01",
        "initial_beacon_id": "B01",
        "total_beacons": total_beacons,
        "laps": 1,
        "command_limits": {
            axis: [-0.95, 0.95] for axis in ("surge", "sway", "heave", "yaw")
        },
    }


@pytest.mark.parametrize(
    ("alias", "controller_type"),
    [
        ("rule_gate_baseline", RuleGateBaselineController),
        ("rule_gate_center_then_commit", RuleGateCenterThenCommitController),
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
    controller.reset(_mission_info())
    assert isinstance(controller.tracker, LocalCourseTracker)


@pytest.mark.parametrize(
    "alias",
    ["rule_gate_baseline", "rule_gate_center_then_commit"],
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


@pytest.mark.parametrize("controller_type", CONTROLLER_TYPES)
def test_official_baselines_return_valid_commands_for_onboard_observation(
    controller_type: type,
) -> None:
    controller = controller_type()
    controller.reset(_mission_info())

    command = controller.step(_guarded_observation(_synthetic_observation()))

    assert set(command) == {"surge", "sway", "heave", "yaw"}
    assert all(-1.0 <= value <= 1.0 for value in command.values())
    assert command["surge"] > 0.0


@pytest.mark.parametrize("controller_type", CONTROLLER_TYPES)
def test_official_baselines_handle_missing_inputs_safely(controller_type: type) -> None:
    controller = controller_type()
    controller.reset(_mission_info())

    command = controller.step(
        _guarded_observation(
            {"local_time_s": 0.0, "sensors": {}, "beacons": []}
        )
    )

    assert set(command) == {"surge", "sway", "heave", "yaw"}
    assert all(-1.0 <= value <= 1.0 for value in command.values())


@pytest.mark.parametrize("controller_type", CONTROLLER_TYPES)
def test_official_baselines_ignore_packets_for_non_expected_beacons(controller_type: type) -> None:
    expected_only = controller_type()
    mixed_packets = controller_type()
    expected_only.reset(_mission_info())
    mixed_packets.reset(_mission_info())
    base_observation = _synthetic_observation(camera=False, bearing_deg=12.0)
    mixed_observation = _synthetic_observation(camera=False, bearing_deg=12.0)
    mixed_observation["beacons"].insert(
        0,
        _beacon_packet(
            "B02",
            bearing_deg=-90.0,
            elevation_deg=25.0,
            range_m=1.0,
            received_at_s=0.0,
        ),
    )

    expected_command = expected_only.step(_guarded_observation(base_observation))
    mixed_command = mixed_packets.step(_guarded_observation(mixed_observation))

    assert mixed_command == pytest.approx(expected_command)
    assert mixed_packets.tracker.expected_beacon_id == "B01"


@pytest.mark.parametrize("controller_type", CONTROLLER_TYPES)
def test_official_baseline_yaw_deadband(controller_type: type) -> None:
    controller = controller_type()
    controller.reset(_mission_info())
    observation = _synthetic_observation(
        camera=False,
        bearing_deg=1.5,
        range_m=8.0,
    )

    command = controller.step(_guarded_observation(observation))

    assert command["yaw"] == 0.0


@pytest.mark.parametrize("controller_type", CONTROLLER_TYPES)
def test_official_baseline_reduces_speed_when_close_without_visual_lock(
    controller_type: type,
) -> None:
    far_controller = controller_type()
    near_controller = controller_type()
    far_controller.reset(_mission_info())
    near_controller.reset(_mission_info())

    far_command = far_controller.step(
        _guarded_observation(
            _synthetic_observation(camera=False, bearing_deg=0.0, range_m=8.0)
        )
    )
    near_command = near_controller.step(
        _guarded_observation(
            _synthetic_observation(camera=False, bearing_deg=0.0, range_m=2.4)
        )
    )

    assert far_controller.tracker.phase == PHASE_APPROACH
    assert near_controller.tracker.phase == PHASE_APPROACH
    assert near_command["surge"] < far_command["surge"]


@pytest.mark.parametrize("controller_type", CONTROLLER_TYPES)
def test_official_baseline_missing_expected_beacon_searches_safely(controller_type: type) -> None:
    controller = controller_type()
    controller.reset(_mission_info())
    observation = _synthetic_observation(camera=False)
    observation["beacons"] = [
        _beacon_packet("B02", bearing_deg=0.0, range_m=4.0, received_at_s=0.0)
    ]

    command = controller.step(_guarded_observation(observation))

    assert command["surge"] > 0.0
    assert command["yaw"] > 0.0
    assert all(-1.0 <= value <= 1.0 for value in command.values())


@pytest.mark.parametrize(
    ("bearing_deg", "expected_sign"),
    [(60.0, 1.0), (-60.0, -1.0)],
)
def test_rule_gate_baseline_large_bearing_turns_in_the_expected_direction(
    bearing_deg: float,
    expected_sign: float,
) -> None:
    controller = RuleGateBaselineController()
    controller.reset(_mission_info())
    observation = _synthetic_observation(
        camera=False,
        bearing_deg=bearing_deg,
        range_m=8.0,
    )

    command = controller.step(_guarded_observation(observation))

    assert command["yaw"] * expected_sign > 0.0
    assert abs(command["yaw"]) <= controller.max_yaw
    assert command["surge"] == 0.0


def test_rule_gate_baseline_centered_gate_surges_forward() -> None:
    controller = RuleGateBaselineController()
    controller.reset(_mission_info())
    observation = _synthetic_observation(bearing_deg=0.0, range_m=5.0)

    command = _step_repeated(controller, observation, count=5)

    assert controller.tracker.phase == PHASE_VISUAL_ALIGN
    assert command["surge"] > 0.0
    assert abs(command["yaw"]) < 0.02


def test_rule_gate_baseline_visual_alignment_yaws_toward_image_error() -> None:
    left = RuleGateBaselineController()
    right = RuleGateBaselineController()
    left.reset(_mission_info())
    right.reset(_mission_info())
    left_observation = _synthetic_observation(
        bearing_deg=0.0,
        x_start=8,
        x_stop=35,
    )
    right_observation = _synthetic_observation(
        bearing_deg=0.0,
        x_start=65,
        x_stop=92,
    )

    left_command = left.step(_guarded_observation(left_observation))
    right_command = right.step(_guarded_observation(right_observation))

    assert left_command["yaw"] > 0.0
    assert right_command["yaw"] < 0.0
    assert abs(left_command["yaw"]) <= left.max_yaw
    assert abs(right_command["yaw"]) <= right.max_yaw


def test_rule_gate_baseline_large_beacon_bearing_rejects_visual_surge() -> None:
    controller = RuleGateBaselineController()
    controller.reset(_mission_info())
    observation = _synthetic_observation(bearing_deg=60.0, range_m=7.0)

    command = _step_repeated(controller, observation, count=5)

    assert controller.tracker.phase == PHASE_APPROACH
    assert command["surge"] == 0.0
    assert command["yaw"] > 0.0


def test_rule_gate_baseline_latches_large_wrapped_bearing_direction() -> None:
    controller = RuleGateBaselineController()
    controller.reset(_mission_info())
    positive = _synthetic_observation(camera=False, bearing_deg=80.0, range_m=4.0)
    wrapped = _synthetic_observation(camera=False, bearing_deg=-170.0, range_m=4.0)

    controller.step(_guarded_observation(positive))
    command = _step_repeated(controller, wrapped, count=4, start_time_s=0.1)

    assert command["yaw"] > 0.0
    assert abs(command["yaw"]) <= controller.max_yaw


def test_rule_gate_baseline_turn_direction_updates_when_bearing_changes_side() -> None:
    controller = RuleGateBaselineController()
    controller.reset(_mission_info())
    positive = _synthetic_observation(camera=False, bearing_deg=60.0, range_m=4.0)
    negative = _synthetic_observation(camera=False, bearing_deg=-60.0, range_m=4.0)

    controller.step(_guarded_observation(positive))
    command = _step_repeated(controller, negative, count=5, start_time_s=0.1)

    assert command["yaw"] < 0.0


def test_rule_gate_baseline_brakes_when_close_and_not_aligned() -> None:
    controller = RuleGateBaselineController()
    controller.reset(_mission_info())
    observation = _synthetic_observation(
        camera=False,
        bearing_deg=45.0,
        range_m=2.4,
    )

    command = _step_repeated(controller, observation, count=5)

    assert command["surge"] < 0.0
    assert command["yaw"] > 0.0


def test_rule_gate_baseline_missing_camera_uses_beacon_fallback() -> None:
    controller = RuleGateBaselineController()
    controller.reset(_mission_info())
    observation = _synthetic_observation(
        camera=False,
        bearing_deg=10.0,
        range_m=7.0,
    )

    command = controller.step(_guarded_observation(observation))

    assert controller.tracker.phase == PHASE_APPROACH
    assert command["surge"] > 0.0
    assert command["yaw"] > 0.0


def test_rule_gate_baseline_handles_invalid_camera_image_safely() -> None:
    controller = RuleGateBaselineController()
    controller.reset(_mission_info())
    observation = _synthetic_observation(bearing_deg=0.0, range_m=6.0)
    observation["sensors"]["FrontCamera"] = "not an image"

    command = controller.step(_guarded_observation(observation))

    assert controller.tracker.phase == PHASE_APPROACH
    assert set(command) == {"surge", "sway", "heave", "yaw"}
    assert all(-1.0 <= value <= 1.0 for value in command.values())


def test_rule_gate_baseline_visual_heave_uses_camera_y_sign() -> None:
    high = RuleGateBaselineController()
    low = RuleGateBaselineController()
    high.reset(_mission_info())
    low.reset(_mission_info())
    high_observation = _synthetic_observation(
        bearing_deg=0.0,
        elevation_deg=0.0,
        range_m=4.0,
        y_start=4,
        y_stop=24,
    )
    low_observation = _synthetic_observation(
        bearing_deg=0.0,
        elevation_deg=0.0,
        range_m=4.0,
        y_start=56,
        y_stop=76,
    )

    high_command = high.step(_guarded_observation(high_observation))
    low_command = low.step(_guarded_observation(low_observation))

    assert high_command["heave"] > 0.0
    assert low_command["heave"] < 0.0


def test_commit_yaw_damping_uses_real_imu_angular_velocity_row() -> None:
    sensors = {
        "IMUSensor": [
            [0.0, 0.0, -9.8],
            [0.0, 0.0, -0.4],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    }
    controller = RuleGateBaselineController()
    controller.reset(_mission_info())

    assert _yaw_rate_from_sensors(sensors) == pytest.approx(-0.4)
    assert controller._yaw_damping_command(sensors) < 0.0


@pytest.mark.parametrize("controller_type", CONTROLLER_TYPES)
def test_official_baseline_uses_real_depth_sensor_for_hold(controller_type: type) -> None:
    controller = controller_type()
    controller.reset(_mission_info())
    nominal = _synthetic_observation(
        camera=False,
        bearing_deg=0.0,
        elevation_deg=0.0,
        range_m=8.0,
        depth_z=-4.0,
    )
    deeper = _synthetic_observation(
        camera=False,
        bearing_deg=0.0,
        elevation_deg=0.0,
        range_m=8.0,
        depth_z=-5.0,
        local_time_s=0.1,
    )

    controller.step(_guarded_observation(nominal))
    command = controller.step(_guarded_observation(deeper))

    assert "depth_m" not in deeper["sensors"]
    assert command["heave"] > 0.0


def test_center_then_commit_changes_only_the_gate_passage_strategy() -> None:
    servo = RuleGateBaselineController()
    commit = RuleGateCenterThenCommitController()
    servo.reset(_mission_info())
    commit.reset(_mission_info())
    centered = _synthetic_observation(bearing_deg=0.0, range_m=2.4)

    _step_repeated(servo, centered, count=5)
    _step_repeated(commit, centered, count=5)
    assert servo.tracker.phase == PHASE_COMMIT
    assert commit.tracker.phase == PHASE_COMMIT
    assert commit.commit_active is True

    off_center = _synthetic_observation(
        bearing_deg=0.0,
        range_m=2.4,
        x_start=65,
        x_stop=92,
        local_time_s=0.5,
    )
    servo_command = servo.step(_guarded_observation(off_center))
    commit_command = commit.step(_guarded_observation(off_center))

    assert servo_command["yaw"] == 0.0
    assert commit_command["yaw"] == 0.0
    assert abs(servo_command["sway"]) > abs(commit_command["sway"])


@pytest.mark.parametrize("controller_type", CONTROLLER_TYPES)
def test_camera_dropout_recovers_without_local_advancement(controller_type: type) -> None:
    controller = controller_type()
    controller.reset(_mission_info())
    centered = _synthetic_observation(bearing_deg=0.0, range_m=2.0)
    _step_repeated(controller, centered, count=2)
    missing_camera = _synthetic_observation(
        camera=False,
        bearing_deg=0.0,
        range_m=2.0,
        local_time_s=0.2,
    )

    controller.step(_guarded_observation(missing_camera))

    assert controller.tracker.local_completed == 0
    assert controller.tracker.expected_beacon_id == "B01"
    _step_repeated(controller, centered, count=4, start_time_s=0.3)
    assert controller.tracker.phase == PHASE_COMMIT
    assert controller.tracker.local_completed == 0


@pytest.mark.parametrize("controller_type", CONTROLLER_TYPES)
def test_both_official_controllers_finish_from_onboard_evidence_and_stop(
    controller_type: type,
) -> None:
    controller = controller_type()
    controller.reset(_mission_info(total_beacons=1))

    _drive_single_gate_to_local_finish(controller)

    assert controller.tracker.phase == PHASE_FINISHED
    assert controller.tracker.local_completed == 1
    assert controller.tracker.expected_beacon_id == "B01"
    stopped = controller.step(
        _guarded_observation(
            {"local_time_s": 4.0, "sensors": {}, "beacons": []}
        )
    )
    assert stopped == {"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}


def test_rule_gate_baseline_respects_static_command_limits() -> None:
    controller = RuleGateBaselineController()
    mission_info = _mission_info()
    mission_info["command_limits"] = {
        "surge": [-0.2, 0.2],
        "sway": [-0.15, 0.15],
        "heave": [-0.12, 0.12],
        "yaw": [-0.1, 0.1],
    }
    controller.reset(mission_info)
    observation = _synthetic_observation(
        camera=False,
        bearing_deg=90.0,
        elevation_deg=35.0,
        range_m=8.0,
    )

    command = _step_repeated(controller, observation, count=8)

    assert abs(command["surge"]) <= 0.2
    assert abs(command["sway"]) <= 0.15
    assert abs(command["heave"]) <= 0.12
    assert abs(command["yaw"]) <= 0.1


def test_rule_gate_baseline_smoke_benchmark_with_fallback(tmp_path: Path) -> None:
    output_dir = tmp_path / "benchmark"

    exit_code = run_benchmark_main(
        [
            "--benchmark-task",
            "clean_gate",
            "--track",
            str(TRACK_DIR / "marine_race_horseshoe_bay.json"),
            "--controller",
            "rule_gate_baseline",
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
    metadata_path = next((output_dir / "runs").rglob("benchmark_metadata.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert len(metadata["source_tree_sha256"]) == 64


def _drive_single_gate_to_local_finish(controller: object) -> None:
    centered = _synthetic_observation(bearing_deg=0.0, range_m=2.0)
    _step_repeated(controller, centered, count=5)
    assert controller.tracker.phase == PHASE_COMMIT

    # Four close, forward-sector packets establish the local passage envelope;
    # only the later range rise and persistent visual disappearance may finish.
    passage_ranges = (1.25, 1.20, 1.15, 1.10, 1.6, 2.2, 3.0, 3.0, 3.0)
    for index, range_m in enumerate(passage_ranges, start=1):
        time_s = 0.4 + 0.5 * index
        observation = _synthetic_observation(
            bearing_deg=170.0 if index >= 5 else 0.0,
            range_m=range_m,
            camera=True,
            colored_gate=False,
            dvl_surge=0.8,
            local_time_s=time_s,
        )
        controller.step(_guarded_observation(observation))


def _step_repeated(
    controller: object,
    observation: dict,
    *,
    count: int,
    start_time_s: float = 0.0,
    dt: float = 0.1,
) -> dict:
    command = {}
    for index in range(count):
        time_s = start_time_s + index * dt
        observation["local_time_s"] = time_s
        for packet in observation["beacons"]:
            packet["received_at_s"] = time_s
        command = controller.step(_guarded_observation(observation))
    return command


def _synthetic_observation(
    *,
    bearing_deg: float = 12.0,
    elevation_deg: float = -4.0,
    range_m: float = 6.0,
    camera: bool = True,
    colored_gate: bool = True,
    x_start: int = 35,
    x_stop: int = 62,
    y_start: int = 22,
    y_stop: int = 58,
    depth_z: float = -4.0,
    dvl_surge: float = 0.0,
    local_time_s: float = 0.0,
) -> dict:
    sensors = {
        "DepthSensor": [depth_z],
        "DVLSensor": [dvl_surge, 0.0, 0.0],
        "IMUSensor": [0.0, 0.0, 0.0, 1.0],
    }
    if camera:
        sensors["FrontCamera"] = _camera_with_colored_gate_blob(
            x_start=x_start,
            x_stop=x_stop,
            y_start=y_start,
            y_stop=y_stop,
            colored_gate=colored_gate,
        )
    return {
        "local_time_s": local_time_s,
        "sensors": sensors,
        "beacons": [
            _beacon_packet(
                "B01",
                bearing_deg=bearing_deg,
                elevation_deg=elevation_deg,
                range_m=range_m,
                received_at_s=local_time_s,
            )
        ],
    }


def _beacon_packet(
    beacon_id: str,
    *,
    bearing_deg: float,
    range_m: float,
    received_at_s: float,
    elevation_deg: float = 0.0,
) -> dict:
    return {
        "beacon_id": beacon_id,
        "bearing_deg": bearing_deg,
        "elevation_deg": elevation_deg,
        "range_m": range_m,
        "signal_strength": 0.8,
        "received_at_s": received_at_s,
    }


def _camera_with_colored_gate_blob(
    x_start: int,
    x_stop: int,
    y_start: int = 22,
    y_stop: int = 58,
    *,
    colored_gate: bool = True,
) -> list[list[list[int]]]:
    width = 100
    height = 80
    image = [[[8, 14, 22] for _ in range(width)] for _ in range(height)]
    if not colored_gate:
        return image
    for y in range(y_start, y_stop):
        for x in range(x_start, x_stop):
            if x in (x_start, x_stop - 1) or y in (y_start, y_stop - 1):
                image[y][x] = [20, 235, 80]
    return image


def _guarded_observation(observation: dict) -> "_OfficialContractGuard":
    return _OfficialContractGuard(observation)


class _OfficialContractGuard(dict):
    _forbidden = {"race", "beacon", "debug_ground_truth"}

    def get(self, key, default=None):  # type: ignore[no-untyped-def]
        if key in self._forbidden:
            raise AssertionError(f"official baseline accessed forbidden field {key}")
        return super().get(key, default)

    def __getitem__(self, key):  # type: ignore[no-untyped-def]
        if key in self._forbidden:
            raise AssertionError(f"official baseline accessed forbidden field {key}")
        return super().__getitem__(key)
