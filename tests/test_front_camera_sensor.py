from __future__ import annotations

from pathlib import Path

from marine_race_arena.adapters.base import AdapterParticipantState
from marine_race_arena.adapters.holoocean_adapter import HoloOceanRaceAdapter
from marine_race_arena.arena.arena_builder import ArenaBuilder
from marine_race_arena.config.loader import load_track_config
from marine_race_arena.participants.participant import RaceParticipant
from marine_race_arena.scripts.run_marine_race import (
    _copy_observation_for_controller,
    _strip_front_camera_from_sensors,
)


TRACK_DIR = Path(__file__).resolve().parents[1] / "marine_race_arena" / "tracks"


def test_holoocean_agent_config_includes_official_front_camera() -> None:
    adapter, participant = _adapter_and_participant()

    sensors = adapter._build_sensor_configs(participant)
    front_camera = next(sensor for sensor in sensors if sensor.get("sensor_name") == "FrontCamera")

    assert front_camera["sensor_type"] == "RGBCamera"
    assert front_camera["socket"] == "CameraSocket"
    assert front_camera["rotation"] == [0.0, 0.0, 0.0]
    assert front_camera["Hz"] == 30
    assert front_camera["configuration"]["CaptureWidth"] == 640
    assert front_camera["configuration"]["CaptureHeight"] == 480
    assert front_camera["configuration"]["FovAngle"] == 90.0


def test_rgbcamera_raw_key_is_exposed_as_front_camera_without_ground_truth() -> None:
    adapter, participant = _adapter_and_participant()
    image = [[[1, 2, 3, 255]]]
    adapter._participants = {participant.id: participant}
    adapter._states = {
        participant.id: AdapterParticipantState(
            participant_id=participant.id,
            position=participant.position,
            rotation_rpy_deg=participant.rotation_rpy_deg,
            raw_sensors={},
        )
    }
    adapter._raw_state = {
        participant.id: {
            "RGBCamera": image,
            "PoseSensor": [[1.0, 0.0, 0.0, 99.0]],
            "DepthSensor": [4.0],
        }
    }

    sensors = adapter.get_allowed_sensor_data(participant.id, participant.config.sensors)

    assert sensors["FrontCamera"] == image
    assert "RGBCamera" not in sensors
    assert "PoseSensor" not in sensors
    assert sensors["DepthSensor"] == [4.0]


def test_benchmark_tracks_enable_front_camera_in_sensor_profile() -> None:
    for track_name in (
        "marine_race_horseshoe_bay.json",
        "marine_race_vertical_serpent.json",
        "marine_race_mixed_endurance.json",
    ):
        config = load_track_config(TRACK_DIR / track_name)
        sensors = config.participants[0].sensors

        assert "FrontCamera" in sensors["allowed_sensors"]
        assert any(
            sensor.get("sensor_type") == "RGBCamera" and sensor.get("sensor_name") == "FrontCamera"
            for sensor in sensors["holoocean_sensors"]
        )


def test_front_camera_can_be_stripped_for_non_official_debug_runs() -> None:
    config = load_track_config(TRACK_DIR / "marine_race_horseshoe_bay.json")
    stripped = _strip_front_camera_from_sensors(config.participants[0].sensors)

    assert "FrontCamera" not in stripped["allowed_sensors"]
    assert not any(
        sensor.get("sensor_name") == "FrontCamera" or sensor.get("sensor_type") == "RGBCamera"
        for sensor in stripped["holoocean_sensors"]
    )


def test_observation_copy_keeps_large_images_by_reference() -> None:
    class ImageLike:
        shape = (480, 640, 4)

    image = ImageLike()
    observation = {"sensors": {"FrontCamera": image, "DepthSensor": [4.0]}}

    copied = _copy_observation_for_controller(observation)

    assert copied is not observation
    assert copied["sensors"] is not observation["sensors"]
    assert copied["sensors"]["FrontCamera"] is image
    assert copied["sensors"]["DepthSensor"] == [4.0]
    assert copied["sensors"]["DepthSensor"] is not observation["sensors"]["DepthSensor"]


def _adapter_and_participant() -> tuple[HoloOceanRaceAdapter, RaceParticipant]:
    config = load_track_config(TRACK_DIR / "marine_race_horseshoe_bay.json")
    arena = ArenaBuilder(config).build()
    participant_config = config.participants[0]
    participant = RaceParticipant(
        config=participant_config,
        controller=object(),
        position=tuple(participant_config.spawn["position"]),
        rotation_rpy_deg=tuple(participant_config.spawn["rotation_rpy_deg"]),
    )
    return HoloOceanRaceAdapter(config, arena), participant
