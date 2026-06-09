from __future__ import annotations

from marine_race_arena.arena.gate import Gate
from marine_race_arena.referee.race_state import ParticipantRaceState, ParticipantStatus
from marine_race_arena.referee.scoring import penalized_time_s, rank_participants


def test_finished_rank_by_penalized_time() -> None:
    a = ParticipantRaceState("a", status=ParticipantStatus.FINISHED)
    a.official_start_time = 0.0
    a.official_finish_time = 12.0
    a.penalties_s = 5.0

    b = ParticipantRaceState("b", status=ParticipantStatus.FINISHED)
    b.official_start_time = 0.0
    b.official_finish_time = 13.0
    b.penalties_s = 0.0

    gate = Gate("G01", "single", (0.0, 0.0, -4.0), (0.0, 0.0, 0.0), 1.5, 1.5, 0.18, "#fff", (1.0, 0.0, 0.0))
    ranked = rank_participants([a, b], ["G01"], {"G01": gate}, {"a": (0.0, 0.0, -4.0), "b": (0.0, 0.0, -4.0)})

    assert penalized_time_s(a) == 17.0
    assert penalized_time_s(b) == 13.0
    assert [state.participant_id for state in ranked] == ["b", "a"]


def test_unfinished_rank_by_completed_gates_then_collisions() -> None:
    a = ParticipantRaceState("a", status=ParticipantStatus.DNF, valid_gate_crossings=3, collision_events=1)
    b = ParticipantRaceState("b", status=ParticipantStatus.DNF, valid_gate_crossings=4, collision_events=3)
    gate = Gate("G01", "single", (0.0, 0.0, -4.0), (0.0, 0.0, 0.0), 1.5, 1.5, 0.18, "#fff", (1.0, 0.0, 0.0))

    ranked = rank_participants([a, b], ["G01"], {"G01": gate}, {"a": (0.0, 0.0, -4.0), "b": (0.0, 0.0, -4.0)})

    assert [state.participant_id for state in ranked] == ["b", "a"]

