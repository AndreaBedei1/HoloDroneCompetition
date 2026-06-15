from __future__ import annotations

from pathlib import Path

from marine_race_arena.arena.arena_builder import ArenaBuilder
from marine_race_arena.config.loader import load_track_config
from marine_race_arena.referee.race_state import ParticipantRaceState, ParticipantStatus
from marine_race_arena.referee.referee import Referee
from marine_race_arena.referee.scoring import penalized_time_s


TRACK_DIR = Path(__file__).resolve().parents[1] / "marine_race_arena" / "tracks"


def test_collision_adds_penalty_but_does_not_dnf() -> None:
    referee = _referee()
    previous_position = (-32.0, -12.0, -4.0)
    current_position = (-32.0, -12.0, -4.0)

    events = referee.update("p1", previous_position, current_position, 1.0, collision=True)
    state = referee.states["p1"]

    assert any(event["event"] == "collision" for event in events)
    assert state.status == ParticipantStatus.RUNNING
    assert state.collision_events == 1
    assert state.penalties_s == 5.0


def test_out_of_bounds_adds_penalty_but_does_not_dnf() -> None:
    referee = _referee()
    previous_position = (50.0, 0.0, -4.0)
    current_position = (51.0, 0.0, -4.0)

    events = referee.update("p1", previous_position, current_position, 1.0)
    state = referee.states["p1"]

    assert any(event["event"] == "out_of_bounds" for event in events)
    assert state.status == ParticipantStatus.RUNNING
    assert state.out_of_bounds_events == 1
    assert state.penalties_s == 10.0


def test_stuck_adds_penalty_once_but_does_not_dnf() -> None:
    referee = _referee()
    referee.config.referee.gate_validation["stuck_timeout_s"] = 0.1
    position = (-32.0, -12.0, -4.0)

    events = referee.update("p1", position, position, 0.2)
    second_events = referee.update("p1", position, position, 0.3)
    state = referee.states["p1"]

    assert any(event["event"] == "stuck" for event in events)
    assert not any(event["event"] == "stuck" for event in second_events)
    assert state.status == ParticipantStatus.RUNNING
    assert state.stuck_events == 1
    assert state.penalties_s == 15.0


def test_wrong_direction_is_event_only() -> None:
    config = load_track_config(TRACK_DIR / "marine_race_horseshoe_bay.json")
    arena = ArenaBuilder(config).build()
    referee = Referee(config, arena.gate_map, arena.bounds)
    referee.register_participants(["p1"])
    referee.start_race(0.0)
    gate = arena.gate_map["G01"]
    previous_position = _add(gate.center, _scale(gate.normal_vector, 1.0))
    current_position = _add(gate.center, _scale(gate.normal_vector, -1.0))

    events = referee.update("p1", previous_position, current_position, 1.0)
    state = referee.states["p1"]

    assert any(event["event"] == "wrong_direction" for event in events)
    assert state.status == ParticipantStatus.RUNNING
    assert state.wrong_direction_crossings == 1
    assert state.penalties_s == 0.0


def test_missed_gate_still_causes_dnf() -> None:
    config = load_track_config(TRACK_DIR / "marine_race_horseshoe_bay.json")
    arena = ArenaBuilder(config).build()
    referee = Referee(config, arena.gate_map, arena.bounds)
    referee.register_participants(["p1"])
    referee.start_race(0.0)
    gate = arena.gate_map["G02"]
    previous_position = _add(gate.center, _scale(gate.normal_vector, -1.0))
    current_position = _add(gate.center, _scale(gate.normal_vector, 1.0))

    events = referee.update("p1", previous_position, current_position, 1.0)
    state = referee.states["p1"]

    assert any(event["event"] == "missed_gate" for event in events)
    assert state.status == ParticipantStatus.DNF
    assert state.missed_gate_attempts == 1


def test_timeout_is_disabled_by_default() -> None:
    config = load_track_config(TRACK_DIR / "marine_race_horseshoe_bay.json")
    arena = ArenaBuilder(config).build()
    referee = Referee(config, arena.gate_map, arena.bounds)
    referee.register_participants(["p1"])
    referee.start_race(0.0)
    position = (-32.0, -12.0, -4.0)

    referee.update("p1", position, position, config.race.max_duration_s + 100.0)

    assert referee.states["p1"].status == ParticipantStatus.RUNNING


def test_controller_error_remains_terminal() -> None:
    referee = _referee()

    referee.update(
        "p1",
        (-32.0, -12.0, -4.0),
        (-32.0, -12.0, -4.0),
        1.0,
        controller_error="boom",
    )

    assert referee.states["p1"].status == ParticipantStatus.CONTROLLER_ERROR


def test_penalized_time_adds_official_time_and_penalties() -> None:
    state = ParticipantRaceState("p1")
    state.official_start_time = 10.0
    state.official_finish_time = 25.0
    state.penalties_s = 7.5

    assert penalized_time_s(state) == 22.5


def _referee() -> Referee:
    config = load_track_config(TRACK_DIR / "marine_race_horseshoe_bay.json")
    arena = ArenaBuilder(config).build()
    referee = Referee(config, arena.gate_map, arena.bounds)
    referee.register_participants(["p1"])
    referee.start_race(0.0)
    return referee


def _add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(vector: tuple[float, float, float], scalar: float) -> tuple[float, float, float]:
    return (vector[0] * scalar, vector[1] * scalar, vector[2] * scalar)
