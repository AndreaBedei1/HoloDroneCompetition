"""Scoring and ranking helpers."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List

from marine_race_arena.arena.gate import Gate
from marine_race_arena.config.schema import Vector3
from marine_race_arena.referee.race_state import ParticipantRaceState, ParticipantStatus


def penalized_time_s(state: ParticipantRaceState) -> float | None:
    if state.official_time_s is None:
        return None
    return state.official_time_s + state.penalties_s


def rank_participants(
    states: Iterable[ParticipantRaceState],
    gate_sequence: List[str],
    gate_map: Dict[str, Gate],
    latest_positions: Dict[str, Vector3],
) -> List[ParticipantRaceState]:
    return sorted(
        states,
        key=lambda state: _ranking_key(state, gate_sequence, gate_map, latest_positions),
    )


def _ranking_key(
    state: ParticipantRaceState,
    gate_sequence: List[str],
    gate_map: Dict[str, Gate],
    latest_positions: Dict[str, Vector3],
) -> tuple[float, float, float, float, str]:
    if state.status == ParticipantStatus.FINISHED:
        official = penalized_time_s(state)
        return (0.0, official if official is not None else math.inf, 0.0, 0.0, state.participant_id)

    next_distance = distance_to_next_gate(state, gate_sequence, gate_map, latest_positions)
    return (
        1.0,
        -float(state.valid_gate_crossings),
        float(state.collision_events),
        next_distance,
        state.participant_id,
    )


def distance_to_next_gate(
    state: ParticipantRaceState,
    gate_sequence: List[str],
    gate_map: Dict[str, Gate],
    latest_positions: Dict[str, Vector3],
) -> float:
    position = latest_positions.get(state.participant_id)
    if position is None or not gate_sequence:
        return math.inf
    gate_id = gate_sequence[min(state.expected_gate_index, len(gate_sequence) - 1)]
    gate = gate_map[gate_id]
    return math.sqrt(
        (position[0] - gate.center[0]) ** 2
        + (position[1] - gate.center[1]) ** 2
        + (position[2] - gate.center[2]) ** 2
    )

