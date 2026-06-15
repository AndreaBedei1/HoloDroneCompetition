from __future__ import annotations

import tempfile
from pathlib import Path

from marine_race_arena.adapters.visual_spawner import HoloOceanVisualSpawner
from marine_race_arena.arena.arena_builder import ArenaBuilder
from marine_race_arena.config.loader import load_track_config


TRACK_DIR = Path(__file__).resolve().parents[1] / "marine_race_arena" / "tracks"
TRACK_CANDIDATES = (
    "marine_race_horseshoe_bay.json",
    "marine_race_mixed_endurance.json",
    "marine_race_vertical_serpent.json",
)


class FakeSpawnPropEnv:
    def __init__(self) -> None:
        self.calls = []

    def spawn_prop(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def _visual_bars():
    config = load_track_config(_track_path())
    arena = ArenaBuilder(config).build()
    return [bar for visual_gate in arena.visual_gates for bar in visual_gate.bars]


def _track_path() -> Path:
    for track_name in TRACK_CANDIDATES:
        path = TRACK_DIR / track_name
        if path.exists():
            return path
    raise AssertionError(f"No test track found in {TRACK_DIR}")


def test_gate_factory_creates_four_bars_per_gate() -> None:
    config = load_track_config(_track_path())
    arena = ArenaBuilder(config).build()
    assert all(len(visual_gate.bars) == 4 for visual_gate in arena.visual_gates)


def test_visual_spawner_reports_runtime_spawn_prop() -> None:
    bars = _visual_bars()
    env = FakeSpawnPropEnv()
    spawner = HoloOceanVisualSpawner(env)
    spawner.spawn_gate_bars(bars)
    assert spawner.report.physically_spawned
    assert spawner.report.method == "runtime_spawn_prop_hybrid_micro_top_bottom"
    assert spawner.report.spawned_bar_count == len(env.calls)
    assert len(env.calls) > len(bars)


def test_visual_spawner_uses_uniform_vertical_pillars() -> None:
    bars = _visual_bars()
    env = FakeSpawnPropEnv()
    spawner = HoloOceanVisualSpawner(env)
    spawner.spawn_gate_bars(bars)

    pillars = [prop for prop in spawner.spawned_props if prop["part"] in {"left", "right"}]
    assert pillars
    assert all(prop["dimensions_m"][2] == 1.5 for prop in pillars)
    assert all(prop["method"] == "uniform_four_bar_box" for prop in pillars)


def test_default_visual_spawner_uses_dense_micro_blocks_for_top_and_bottom() -> None:
    bars = _visual_bars()
    env = FakeSpawnPropEnv()
    spawner = HoloOceanVisualSpawner(env)
    spawner.spawn_gate_bars(bars)

    top_segments = [
        prop
        for prop in spawner.spawned_props
        if prop["source_bar_id"] == "G03_top"
    ]
    bottom_segments = [
        prop
        for prop in spawner.spawned_props
        if prop["source_bar_id"] == "G03_bottom"
    ]

    assert len(top_segments) >= 50
    assert len(bottom_segments) == len(top_segments)
    assert all(prop["method"] == "hybrid_micro_top_bottom_block" for prop in top_segments)
    assert all(prop["spawn_rotation_deg"] == (0.0, 0.0, 0.0) for prop in top_segments)
    assert all(prop["dimensions_m"] == (0.09, 0.09, 0.09) for prop in top_segments)
    assert all(prop["segment_count"] == len(top_segments) for prop in top_segments)


def test_spawn_prop_rotation_uses_verified_holoocean_box_mapping() -> None:
    bars = _visual_bars()
    env = FakeSpawnPropEnv()
    spawner = HoloOceanVisualSpawner(env)
    spawner.spawn_gate_bars(bars)

    rotated = next(
        prop
        for prop in spawner.spawned_props
        if prop["part"] == "left" and abs(prop["rotation_rpy_deg"][2]) > 1.0
    )
    roll, pitch, yaw = rotated["rotation_rpy_deg"]
    assert rotated["spawn_rotation_deg"] == (yaw, pitch, roll)
    assert rotated["spawn_rotation_order"] == "holoocean_spawn_prop_yaw_pitch_roll"

    matching_call = next(call for call in env.calls if call[1]["tag"] == rotated["source_bar_id"])
    assert matching_call[1]["rotation"] == [yaw, pitch, roll]


def test_visual_spawner_can_use_uniform_long_bar_mode() -> None:
    bars = _visual_bars()
    env = FakeSpawnPropEnv()
    spawner = HoloOceanVisualSpawner(env, mode="uniform")
    spawner.spawn_gate_bars(bars)

    assert spawner.report.method == "runtime_spawn_prop_uniform_box"
    assert len(env.calls) == len(bars)
    assert all(prop["method"] == "uniform_four_bar_box" for prop in spawner.spawned_props)


def test_visual_spawner_can_use_segmented_fallback_mode() -> None:
    bars = _visual_bars()
    env = FakeSpawnPropEnv()
    spawner = HoloOceanVisualSpawner(env, mode="segmented")
    spawner.spawn_gate_bars(bars)

    assert spawner.report.method == "runtime_spawn_prop_segmented_cubes"
    assert len(env.calls) > len(bars)
    assert all(prop["method"] == "segmented_axis_aligned_cube" for prop in spawner.spawned_props)
    assert all(prop["spawn_rotation_deg"] == (0.0, 0.0, 0.0) for prop in spawner.spawned_props)


def test_visual_spawner_reports_export_only_without_env() -> None:
    bars = _visual_bars()
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "gate_bars.json"
        spawner = HoloOceanVisualSpawner(None, export_path=export_path)
        spawner.spawn_gate_bars(bars)
        assert not spawner.report.physically_spawned
        assert spawner.report.method == "export_only"
        assert export_path.exists()
