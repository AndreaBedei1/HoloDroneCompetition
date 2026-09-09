"""Training curriculum: staged tasks, seeded randomization and an executable runner.

Do not start on full tracks. Each stage must meet its completion criterion on
held-out evaluation seeds before advancing; the runner evaluates a controller
through the unchanged referee and refuses to advance automatically when the
criterion is not met. Stages 0-4 use the training-only tracks under
``marine_race_arena/tracks/training/`` (which preserve the official gate geometry,
beacon model, observation, action mapping and referee); later stages reference the
official tracks *unchanged*.

Training, validation and evaluation seeds are kept disjoint (see
:func:`seed_split`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from marine_race_arena.learning.randomization import StartRandomization

STAGE1_TRACK = "marine_race_arena/tracks/training/stage1_single_gate.json"
STAGE3_TRACK = "marine_race_arena/tracks/training/stage3_three_gates.json"
STAGE4_TRACK = "marine_race_arena/tracks/training/stage4_six_gates.json"
HORSESHOE = "marine_race_arena/tracks/marine_race_horseshoe_bay.json"
VERTICAL = "marine_race_arena/tracks/marine_race_vertical_serpent.json"
MIXED = "marine_race_arena/tracks/marine_race_mixed_endurance.json"

# Stage 2 randomization: modest, bounded, deterministic-per-seed.
STAGE2_RANDOMIZATION = StartRandomization(
    lateral_offset_m=1.0,
    depth_offset_m=0.5,
    yaw_offset_deg=15.0,
    longitudinal_offset_m=0.5,
    beacon_angular_noise_std_deg=0.2,
    beacon_range_noise_std_m=0.2,
    beacon_dropout_probability=0.02,
)


@dataclass(frozen=True)
class CurriculumStage:
    key: str
    name: str
    tracks: Tuple[str, ...]
    description: str
    eval_episodes: int
    min_completion_rate: float
    randomization: Optional[StartRandomization] = None
    # For disturbance stages: profiles to escalate through, gentlest first. "none"
    # means the disturbance-free baseline; the benchmark tracks define none/medium/
    # strong, so the intended weak->medium->strong ramp is mapped onto none->medium->
    # strong (a finer weak profile is a documented future refinement).
    current_progression: Tuple[str, ...] = ()
    official_track: bool = False

    @property
    def track(self) -> str:
        """Primary track (first of ``tracks``)."""
        return self.tracks[0]


STAGES: List[CurriculumStage] = [
    CurriculumStage(
        key="stage0",
        name="API and actuation sanity",
        tracks=(STAGE1_TRACK,),
        description=(
            "No training. Deterministic fixed commands verify reset/step/close, reward and "
            "termination, and that the policy observation contains no referee/pose leakage."
        ),
        eval_episodes=0,
        min_completion_rate=0.0,
    ),
    CurriculumStage(
        key="stage1",
        name="Single gate, easy initial pose",
        tracks=(STAGE1_TRACK,),
        description="One gate, clean water, no obstacles/current, aligned start close enough to perceive the gate.",
        eval_episodes=20,
        min_completion_rate=0.90,
    ),
    CurriculumStage(
        key="stage2",
        name="Single gate, randomized start",
        tracks=(STAGE1_TRACK,),
        description="Seeded random lateral/yaw/depth start offset and limited beacon noise; recover alignment and pass.",
        eval_episodes=20,
        min_completion_rate=0.90,
        randomization=STAGE2_RANDOMIZATION,
    ),
    CurriculumStage(
        key="stage3",
        name="Three ordered gates",
        tracks=(STAGE3_TRACK,),
        description="Shallow S-curve, three gates, local tracker progression; complete without false advancement.",
        eval_episodes=20,
        min_completion_rate=0.80,
    ),
    CurriculumStage(
        key="stage4",
        name="Five or six gates",
        tracks=(STAGE4_TRACK,),
        description="Six gates with moderate turns, depth changes and varied gate orientations; no current.",
        eval_episodes=20,
        min_completion_rate=0.80,
    ),
    CurriculumStage(
        key="stage5_horseshoe",
        name="Full Horseshoe Bay, clean",
        tracks=(HORSESHOE,),
        description="Official track unchanged. Training seeds must be separated from evaluation seeds.",
        eval_episodes=20,
        min_completion_rate=0.80,
        official_track=True,
    ),
    CurriculumStage(
        key="stage6_generalization",
        name="Vertical Serpent and Mixed Endurance generalization",
        tracks=(VERTICAL, MIXED),
        description="Evaluate BOTH official tracks. Record separately whether a result is zero-shot or fine-tuned.",
        eval_episodes=20,
        min_completion_rate=0.80,
        official_track=True,
    ),
    CurriculumStage(
        key="stage7_disturbance",
        name="Disturbance robustness",
        tracks=(HORSESHOE,),
        description="Escalate gentlest-first: none (baseline) -> medium (primary disturbance) -> strong (stress test).",
        eval_episodes=20,
        min_completion_rate=0.80,
        official_track=True,
        current_progression=("none", "medium", "strong"),
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


def seed_split(n_train: int = 200, n_val: int = 40, n_eval: int = 40, base: int = 0) -> Dict[str, List[int]]:
    """Disjoint seed ranges for training, validation and final evaluation."""
    t0 = base
    v0 = t0 + n_train
    e0 = v0 + n_val
    return {
        "train": list(range(t0, v0)),
        "val": list(range(v0, e0)),
        "eval": list(range(e0, e0 + n_eval)),
    }


@dataclass
class StageResult:
    stage_key: str
    completion_rate: float
    n_episodes: int
    passed: bool
    decision: str
    per_evaluation: Dict[str, Dict[str, float]] = field(default_factory=dict)


def evaluate_stage(
    stage_key: str,
    controller_factory: Callable[[], object],
    *,
    seeds: Sequence[int],
    adapter: str = "fallback",
    allow_fallback: bool = True,
    duration_s: Optional[float] = None,
    dt: float = 0.1,
) -> StageResult:
    """Evaluate a controller on a stage over held-out seeds through the referee.

    A stage with multiple tracks and/or a current progression is evaluated on every
    (track, profile) combination; the stage completion rate is the WORST of them, so
    a stage passes only if it meets the criterion everywhere. This does not advance
    the curriculum by itself — use :func:`advance_decision`.
    """
    from marine_race_arena.learning.evaluate_policy import evaluate_controller

    s = stage(stage_key)
    profiles: Tuple[Optional[str], ...] = s.current_progression or (None,)
    per_evaluation: Dict[str, Dict[str, float]] = {}
    rates: List[float] = []
    for track in s.tracks:
        for profile in profiles:
            current_profile = None if profile in (None, "none") else profile
            report = evaluate_controller(
                track,
                controller_factory,
                seeds=seeds,
                label=stage_key,
                adapter=adapter,
                allow_fallback=allow_fallback,
                duration_s=duration_s,
                dt=dt,
                current_profile=current_profile,
                start_randomization=s.randomization,
            )
            key = track if profile is None else f"{track}::{profile}"
            per_evaluation[key] = report.summary()
            rates.append(report.completion_rate)

    completion_rate = min(rates) if rates else 0.0
    passed = completion_rate >= s.min_completion_rate
    decision = (
        f"{'ADVANCE' if passed else 'HOLD'} at {stage_key}: "
        f"completion {completion_rate:.0%} vs criterion {s.min_completion_rate:.0%} "
        f"over {len(seeds)} held-out seeds"
    )
    return StageResult(stage_key, completion_rate, len(seeds), passed, decision, per_evaluation)


def advance_decision(stage_key: str, completion_rate: float) -> Tuple[Optional[str], str]:
    """Return (next stage key or None, human-readable decision).

    Refuses to advance (returns ``None``) when the criterion is not met.
    """
    if meets_criterion(stage_key, completion_rate):
        nxt = next_stage(stage_key)
        return (nxt.key if nxt else None), f"advance from {stage_key} (criterion met)"
    return None, (
        f"HOLD at {stage_key}: completion {completion_rate:.0%} below "
        f"criterion {stage(stage_key).min_completion_rate:.0%} -- do not advance"
    )
