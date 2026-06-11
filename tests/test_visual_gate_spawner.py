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
    assert spawner.report.method == "runtime_spawn_prop"
    assert spawner.report.spawned_bar_count == len(bars)
    assert len(env.calls) == len(bars)


def test_visual_spawner_reports_export_only_without_env() -> None:
    bars = _visual_bars()
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "gate_bars.json"
        spawner = HoloOceanVisualSpawner(None, export_path=export_path)
        spawner.spawn_gate_bars(bars)
        assert not spawner.report.physically_spawned
        assert spawner.report.method == "export_only"
        assert export_path.exists()

