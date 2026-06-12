from __future__ import annotations

import tempfile
from pathlib import Path

from marine_race_arena.adapters.visual_spawner import HoloOceanVisualSpawner
from marine_race_arena.arena.arena_builder import ArenaBuilder
from marine_race_arena.config.loader import load_track_config


TRACK_DIR = Path(__file__).resolve().parents[1] / "marine_race_arena" / "tracks"


class FakeSpawnPropEnv:
    def __init__(self) -> None:
        self.calls = []

    def spawn_prop(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def _visual_bars():
    config = load_track_config(TRACK_DIR / "abu_dhabi_marine_easy.json")
    arena = ArenaBuilder(config).build()
    return [bar for visual_gate in arena.visual_gates for bar in visual_gate.bars]


def test_gate_factory_creates_four_bars_per_gate() -> None:
    config = load_track_config(TRACK_DIR / "abu_dhabi_marine_easy.json")
    arena = ArenaBuilder(config).build()
    assert all(len(visual_gate.bars) == 4 for visual_gate in arena.visual_gates)


def test_visual_spawner_reports_runtime_spawn_prop() -> None:
    bars = _visual_bars()
    env = FakeSpawnPropEnv()
    spawner = HoloOceanVisualSpawner(env)
    spawner.spawn_gate_bars(bars)
    assert spawner.report.physically_spawned
    assert spawner.report.method == "runtime_spawn_prop_uniform_box"
    assert spawner.report.spawned_bar_count == len(env.calls)
    assert len(env.calls) == len(bars)


def test_visual_spawner_uses_uniform_vertical_pillars() -> None:
    bars = _visual_bars()
    env = FakeSpawnPropEnv()
    spawner = HoloOceanVisualSpawner(env)
    spawner.spawn_gate_bars(bars)

    pillars = [prop for prop in spawner.spawned_props if prop["part"] in {"left", "right"}]
    assert pillars
    assert all(prop["dimensions_m"][2] == 1.5 for prop in pillars)
    assert all(prop["method"] == "uniform_four_bar_box" for prop in pillars)


def test_spawn_prop_rotation_uses_verified_holoocean_box_mapping() -> None:
    bars = _visual_bars()
    env = FakeSpawnPropEnv()
    spawner = HoloOceanVisualSpawner(env)
    spawner.spawn_gate_bars(bars)

    rotated = next(prop for prop in spawner.spawned_props if prop["source_bar_id"] == "G03_left")
    assert rotated["rotation_rpy_deg"] == (0.0, 0.0, 45.0)
    assert rotated["spawn_rotation_deg"] == (45.0, 0.0, 0.0)
    assert rotated["spawn_rotation_order"] == "holoocean_spawn_prop_yaw_pitch_roll"

    matching_call = next(call for call in env.calls if call[1]["tag"] == "G03_left")
    assert matching_call[1]["rotation"] == [45.0, 0.0, 0.0]


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
