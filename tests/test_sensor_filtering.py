from __future__ import annotations

from pathlib import Path

from marine_race_arena.adapters.fallback_adapter import FallbackRaceAdapter
from marine_race_arena.arena.arena_builder import ArenaBuilder
from marine_race_arena.config.loader import load_track_config
from marine_race_arena.participants.sensor_profile import build_observation


TRACK_DIR = Path(__file__).resolve().parents[1] / "marine_race_arena" / "tracks"


def _adapter():
    config = load_track_config(TRACK_DIR / "marine_race_horseshoe_bay.json")
    arena = ArenaBuilder(config).build()
    return FallbackRaceAdapter(config, arena)


def test_official_sensor_filter_removes_ground_truth_sensors() -> None:
    adapter = _adapter()
    raw = {
        "PoseSensor": [[1, 0, 0, 0]],
        "LocationSensor": [0, 0, -4],
        "RotationSensor": [0, 0, 0],
        "DynamicsSensor": [0] * 18,
        "DepthSensor": [4],
    }
    filtered = adapter.filter_sensor_data(
        raw,
        {"allowed_sensors": ["PoseSensor", "LocationSensor", "RotationSensor", "DynamicsSensor", "DepthSensor"]},
        official_mode=True,
    )
    assert filtered == {"DepthSensor": [4]}


def test_official_sensor_filter_removes_current_vector() -> None:
    adapter = _adapter()
    raw = {
        "DVLSensor": {"x": 0.1, "y": 0.2},
        "VelocitySensor": [0.1, 0.2, 0.0],
        "environment_current_m_s": [0.75, 1.05, 0.0],
        "current_physical_coupling_active": True,
        "DepthSensor": [4],
    }
    # Even when explicitly allow-listed, the current vector must not reach an
    # official controller observation.
    profile = {
        "allowed_sensors": [
            "DVLSensor",
            "VelocitySensor",
            "environment_current_m_s",
            "current_physical_coupling_active",
            "DepthSensor",
        ]
    }
    official = adapter.filter_sensor_data(raw, profile, official_mode=True)
    assert "environment_current_m_s" not in official
    # Legal onboard sensors are preserved.
    assert "DVLSensor" in official
    assert "VelocitySensor" in official
    assert "DepthSensor" in official

    # In non-official (diagnostic) runs the field remains available.
    unofficial = adapter.filter_sensor_data(raw, profile, official_mode=False)
    assert "environment_current_m_s" in unofficial


def test_current_vector_stripped_even_without_allow_list() -> None:
    adapter = _adapter()
    raw = {"environment_current_m_s": [1.0, 1.0, 0.0], "DepthSensor": [4]}
    # No allow-list configured: ground-truth-style strips must still apply.
    official = adapter.filter_sensor_data(raw, {}, official_mode=True)
    assert "environment_current_m_s" not in official


def test_debug_ground_truth_only_added_when_not_official() -> None:
    debug = {"own_position": (0.0, 0.0, -4.0)}
    unofficial = build_observation("p1", 0.0, {}, {}, {}, official_mode=False, debug_ground_truth=debug)
    official = build_observation("p1", 0.0, {}, {}, {}, official_mode=True, debug_ground_truth=debug)
    assert "debug_ground_truth" in unofficial
    assert "debug_ground_truth" not in official
