"""Training curriculum: staged, increasingly hard tasks with success criteria.

Do not start on full tracks. Each stage must meet its completion criterion on
held-out evaluation seeds before advancing. Stages 0-3 use the training-only
tracks under ``marine_race_arena/tracks/training/`` (which preserve the official
gate geometry, beacon model, observation, action mapping and referee); later
stages reference the official tracks *unchanged* for evaluation and eventual
fine-tuning, with training and evaluation seeds kept separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

STAGE1_TRACK = "marine_race_arena/tracks/training/stage1_single_gate.json"
STAGE3_TRACK = "marine_race_arena/tracks/training/stage3_three_gates.json"
HORSESHOE = "marine_race_arena/tracks/marine_race_horseshoe_bay.json"
VERTICAL = "marine_race_arena/tracks/marine_race_vertical_serpent.json"
MIXED = "marine_race_arena/tracks/marine_race_mixed_endurance.json"


@dataclass(frozen=True)
class CurriculumStage:
    key: str
    name: str
    track: Optional[str]           # None for the no-training API sanity stage
    description: str
    eval_episodes: int
    min_completion_rate: float
    randomize_start: bool = False
    beacon_noise: bool = False
    current_profile: Optional[str] = None
    official_track: bool = False   # True stages evaluate on unchanged official tracks


STAGES: List[CurriculumStage] = [
    CurriculumStage(
        key="stage0",
        name="API and actuation sanity",
        track=STAGE1_TRACK,
        description=(
            "No training. Deterministic fixed commands verify reset/step/close, reward "
            "and termination, and that the policy observation contains no referee/pose leakage."
        ),
        eval_episodes=0,
        min_completion_rate=0.0,
    ),
    CurriculumStage(
        key="stage1",
        name="Single gate, easy initial pose",
        track=STAGE1_TRACK,
        description="One gate, clean water, no obstacles/current, aligned start close enough to perceive the gate.",
        eval_episodes=20,
        min_completion_rate=0.90,
    ),
    CurriculumStage(
        key="stage2",
        name="Single gate, randomized start",
        track=STAGE1_TRACK,
        description="Random lateral/yaw/depth offset and limited beacon noise; recover alignment and pass.",
        eval_episodes=20,
        min_completion_rate=0.90,
        randomize_start=True,
        beacon_noise=True,
    ),
    CurriculumStage(
        key="stage3",
        name="Three ordered gates",
        track=STAGE3_TRACK,
        description="Shallow S-curve, three gates, local tracker progression; complete without false advancement.",
        eval_episodes=20,
        min_completion_rate=0.80,
        beacon_noise=True,
    ),
    CurriculumStage(
        key="stage5_horseshoe",
        name="Full Horseshoe Bay, clean",
        track=HORSESHOE,
        description="Official track unchanged. Training seeds must be separated from evaluation seeds.",
        eval_episodes=20,
        min_completion_rate=0.80,
        official_track=True,
    ),
    CurriculumStage(
        key="stage6_generalization",
        name="Vertical Serpent and Mixed Endurance generalization",
        track=VERTICAL,
        description="Evaluate zero-shot before any fine-tuning; record whether a result is zero-shot or fine-tuned.",
        eval_episodes=20,
        min_completion_rate=0.80,
        official_track=True,
    ),
    CurriculumStage(
        key="stage7_disturbance",
        name="Disturbance robustness",
        track=HORSESHOE,
        description="Weak current first, then medium; strong current only as a stress test, not an initial target.",
        eval_episodes=20,
        min_completion_rate=0.80,
        official_track=True,
        current_profile="medium",
    ),
]

_BY_KEY = {stage.key: stage for stage in STAGES}


def stage(key: str) -> CurriculumStage:
    if key not in _BY_KEY:
        raise KeyError(f"unknown curriculum stage '{key}'; known: {list(_BY_KEY)}")
    return _BY_KEY[key]


def next_stage(key: str) -> Optional[CurriculumStage]:
    keys = [s.key for s in STAGES]
    idx = keys.index(key)
    return STAGES[idx + 1] if idx + 1 < len(STAGES) else None


def meets_criterion(stage_key: str, completion_rate: float) -> bool:
    """A stage passes when its held-out completion rate meets the criterion."""
    return completion_rate >= stage(stage_key).min_completion_rate
