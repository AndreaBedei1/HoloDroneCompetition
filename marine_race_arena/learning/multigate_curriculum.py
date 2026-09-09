"""Curriculum definition for the observation-v3 learned controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

SINGLE_GATE = "marine_race_arena/tracks/training/stage1_single_gate.json"
TWO_GATE_STRAIGHT = "marine_race_arena/tracks/tests/two_gate_straight.json"
TWO_GATE_LEFT = "marine_race_arena/tracks/tests/two_gate_left_curve.json"
TWO_GATE_RIGHT = "marine_race_arena/tracks/tests/two_gate_right_curve.json"
THREE_GATE_S = "marine_race_arena/tracks/tests/three_gate_s_curve.json"
THREE_GATE_TRAINING = "marine_race_arena/tracks/training/stage3_three_gates.json"
SIX_GATE = "marine_race_arena/tracks/training/stage4_six_gates.json"
HORSESHOE = "marine_race_arena/tracks/marine_race_horseshoe_bay.json"
VERTICAL = "marine_race_arena/tracks/marine_race_vertical_serpent.json"
MIXED = "marine_race_arena/tracks/marine_race_mixed_endurance.json"


@dataclass(frozen=True)
class MultiGateStage:
    key: str
    name: str
    tracks: Tuple[str, ...]
    objective: str
    eval_episodes: int
    min_completion_rate: float
    official: bool = False


STAGES = (
    MultiGateStage(
        "R0",
        "Transferred-policy regression",
        (SINGLE_GATE,),
        "Preserve BC-v1 single-gate competence and neutral-feature parity before PPO.",
        10,
        0.8,
    ),
    MultiGateStage(
        "R1",
        "Two aligned gates",
        (TWO_GATE_STRAIGHT,),
        "Cross B01, clear it under learned control, acquire B02, and finish.",
        10,
        0.8,
    ),
    MultiGateStage(
        "R2",
        "Balanced two-gate turns",
        (TWO_GATE_LEFT, TWO_GATE_RIGHT),
        "Learn both left and right inter-gate turns without returning to B01.",
        10,
        0.8,
    ),
    MultiGateStage(
        "R3",
        "Three-gate S curve",
        (THREE_GATE_S, THREE_GATE_TRAINING),
        "Progress B01-B02-B03 with learned reacquisition at both transitions.",
        10,
        0.8,
    ),
    MultiGateStage(
        "R4",
        "Six-gate sequence",
        (SIX_GATE,),
        "Scale repeated transitions before any full official circuit.",
        10,
        0.8,
    ),
    MultiGateStage(
        "R5",
        "Official current-free circuits",
        (HORSESHOE, VERTICAL, MIXED),
        "Complete unchanged official circuits in real HoloOcean with currents disabled.",
        5,
        0.6,
        official=True,
    ),
)

_BY_KEY: Dict[str, MultiGateStage] = {stage.key: stage for stage in STAGES}


def stage(key: str) -> MultiGateStage:
    try:
        return _BY_KEY[str(key).upper()]
    except KeyError as exc:
        raise KeyError(f"unknown multi-gate curriculum stage {key!r}") from exc


def meets_progression_criterion(key: str, per_track_completion_rates) -> bool:
    """Require the threshold on every track, not only in aggregate."""
    rates = list(per_track_completion_rates)
    return bool(rates) and all(
        float(rate) >= stage(key).min_completion_rate for rate in rates
    )
