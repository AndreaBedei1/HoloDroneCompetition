from __future__ import annotations

from pathlib import Path

from marine_race_arena.arena.arena_builder import ArenaBuilder
from marine_race_arena.config.loader import load_track_config
from marine_race_arena.referee.referee import Referee


TRACK_DIR = Path(__file__).resolve().parents[1] / "marine_race_arena" / "tracks"


def test_referee_advances_expected_gate_after_valid_crossing() -> None:
    config = load_track_config(TRACK_DIR / "marine_race_horseshoe_bay.json")
    arena = ArenaBuilder(config).build()
    referee = Referee(config, arena.gate_map, arena.bounds)
    referee.register_participants(["p1"])
    referee.start_race(0.0)
    gate = arena.gate_map["G01"]
    previous_position = _add(gate.center, _scale(gate.normal_vector, -1.0))
    current_position = _add(gate.center, _scale(gate.normal_vector, 1.0))

    events = referee.update("p1", previous_position, current_position, 1.0)

    assert any(event["event"] == "gate_passed" for event in events)
    assert referee.expected_gate_id("p1") == "G02"
    assert referee.states["p1"].official_start_time == 1.0
    assert referee.states["p1"].valid_gate_crossings == 1


def test_referee_counts_wrong_direction() -> None:
    config = load_track_config(TRACK_DIR / "marine_race_horseshoe_bay.json")
    arena = ArenaBuilder(config).build()
    referee = Referee(config, arena.gate_map, arena.bounds)
    referee.register_participants(["p1"])
    referee.start_race(0.0)
    gate = arena.gate_map["G01"]
    previous_position = _add(gate.center, _scale(gate.normal_vector, 1.0))
    current_position = _add(gate.center, _scale(gate.normal_vector, -1.0))

    events = referee.update("p1", previous_position, current_position, 1.0)

    assert any(event["event"] == "wrong_direction" for event in events)
    assert referee.states["p1"].wrong_direction_crossings == 1
    assert referee.expected_gate_id("p1") == "G01"


def _add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(vector: tuple[float, float, float], scalar: float) -> tuple[float, float, float]:
    return (vector[0] * scalar, vector[1] * scalar, vector[2] * scalar)
