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


def test_debug_ground_truth_only_added_when_not_official() -> None:
    debug = {"own_position": (0.0, 0.0, -4.0)}
    unofficial = build_observation("p1", 0.0, {}, {}, {}, official_mode=False, debug_ground_truth=debug)
    official = build_observation("p1", 0.0, {}, {}, {}, official_mode=True, debug_ground_truth=debug)
    assert "debug_ground_truth" in unofficial
    assert "debug_ground_truth" not in official
