from __future__ import annotations

from pathlib import Path

from marine_race_arena.arena.arena_builder import ArenaBuilder
from marine_race_arena.config.loader import load_track_config
from marine_race_arena.referee.referee import Referee


TRACK_DIR = Path(__file__).resolve().parents[1] / "marine_race_arena" / "tracks"


def test_referee_advances_expected_gate_after_valid_crossing() -> None:
    config = load_track_config(TRACK_DIR / "abu_dhabi_marine_easy.json")
    arena = ArenaBuilder(config).build()
    referee = Referee(config, arena.gate_map, arena.bounds)
    referee.register_participants(["p1"])
    referee.start_race(0.0)

    events = referee.update("p1", (-3.0, 0.0, -4.0), (-1.0, 0.0, -4.0), 1.0)

    assert any(event["event"] == "gate_passed" for event in events)
    assert referee.expected_gate_id("p1") == "G02"
    assert referee.states["p1"].official_start_time == 1.0
    assert referee.states["p1"].valid_gate_crossings == 1


def test_referee_counts_wrong_direction() -> None:
    config = load_track_config(TRACK_DIR / "abu_dhabi_marine_easy.json")
    arena = ArenaBuilder(config).build()
    referee = Referee(config, arena.gate_map, arena.bounds)
    referee.register_participants(["p1"])
    referee.start_race(0.0)

    events = referee.update("p1", (-1.0, 0.0, -4.0), (-3.0, 0.0, -4.0), 1.0)

    assert any(event["event"] == "wrong_direction" for event in events)
    assert referee.states["p1"].wrong_direction_crossings == 1
    assert referee.expected_gate_id("p1") == "G01"

