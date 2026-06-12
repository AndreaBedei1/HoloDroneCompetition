"""Gate validation helpers used by the referee."""

from __future__ import annotations

from marine_race_arena.arena.gate import Gate, GateCrossingResult
from marine_race_arena.config.schema import Vector3


def validate_gate_crossing(
    gate: Gate,
    previous_position: Vector3,
    current_position: Vector3,
    clearance_margin_m: float = 0.0,
) -> GateCrossingResult:
    return gate.validate_crossing(previous_position, current_position, clearance_margin_m=clearance_margin_m)
