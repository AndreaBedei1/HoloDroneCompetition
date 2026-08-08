"""Generic procedural sequence curriculum.

Local transitions are already learned reasonably well; the open problem is
*chaining* them without accumulating error.  That must be learned from broad
procedural variation, never from the sealed final circuits, so this module owns
only the episode-length mixture and its promotion rule.  Geometry itself stays
with the existing procedural sampler.

Promotion is driven by measured generalization on the unseen validation band, not
by elapsed transitions: a run that has merely survived a lot of steps has not
earned harder work.  Short transitions are never removed, because they remain the
highest-signal-per-second source of crossing and target-switch learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

CURRICULUM_VERSION = "generic_sequence_curriculum_v1"

# Length buckets. "focus" is the two-gate transition episode.
BUCKETS = ("focus", "short", "medium", "long")
BUCKET_GATE_RANGES: Dict[str, Tuple[int, int]] = {
    "focus": (2, 2),
    "short": (3, 5),
    "medium": (6, 12),
    "long": (13, 24),
}

STAGES: Tuple[Dict[str, Any], ...] = (
    {
        "name": "S0_transition_heavy",
        "mixture": {"focus": 0.30, "short": 0.30, "medium": 0.25, "long": 0.15},
    },
    {
        "name": "S1_balanced",
        "mixture": {"focus": 0.22, "short": 0.28, "medium": 0.28, "long": 0.22},
    },
    {
        "name": "S2_sequence_heavy",
        "mixture": {"focus": 0.15, "short": 0.25, "medium": 0.30, "long": 0.30},
    },
)

# Promotion needs generalization on unseen procedural courses, measured twice.
DEFAULT_PROMOTION = {
    "min_universal_transition_success_rate": 0.70,
    "min_long_sequence_completion_score": 0.45,
    "max_collision_episode_rate": 0.25,
    "consecutive_qualifying_evaluations": 2,
}
# Demotion protects a run that has been promoted too early.
DEFAULT_DEMOTION = {
    "min_universal_transition_success_rate": 0.40,
    "consecutive_failing_evaluations": 2,
}


def normalized_mixture(mixture: Mapping[str, float]) -> Dict[str, float]:
    unknown = sorted(set(mixture) - set(BUCKETS))
    if unknown:
        raise ValueError(f"unknown curriculum buckets {unknown}")
    if any(float(value) < 0.0 for value in mixture.values()):
        raise ValueError("curriculum weights cannot be negative")
    total = float(sum(mixture.values()))
    if total <= 0.0:
        raise ValueError("curriculum mixture must have positive mass")
    out = {bucket: float(mixture.get(bucket, 0.0)) / total for bucket in BUCKETS}
    if out["focus"] <= 0.0:
        raise ValueError(
            "short transitions must retain non-zero mass: they are the "
            "highest-signal source of crossing and target-switch learning"
        )
    return out


@dataclass
class SequenceCurriculumState:
    stage_index: int = 0
    consecutive_qualifying: int = 0
    consecutive_failing: int = 0
    total_environment_transitions: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)


class GenericSequenceCurriculum:
    """Length mixture over procedurally generated courses only."""

    def __init__(
        self,
        *,
        stages: Sequence[Mapping[str, Any]] = STAGES,
        promotion: Optional[Mapping[str, Any]] = None,
        demotion: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not stages:
            raise ValueError("at least one curriculum stage is required")
        self.stages = [
            {"name": str(stage["name"]),
             "mixture": normalized_mixture(stage["mixture"])}
            for stage in stages
        ]
        self.promotion = {**DEFAULT_PROMOTION, **dict(promotion or {})}
        self.demotion = {**DEFAULT_DEMOTION, **dict(demotion or {})}
        self.state = SequenceCurriculumState()

    @property
    def stage(self) -> Dict[str, Any]:
        return self.stages[self.state.stage_index]

    @property
    def mixture(self) -> Dict[str, float]:
        return dict(self.stage["mixture"])

    def bucket_for(self, value: float) -> str:
        """Map a uniform draw in [0,1) to a length bucket."""

        cumulative = 0.0
        for bucket in BUCKETS:
            cumulative += self.mixture[bucket]
            if float(value) < cumulative:
                return bucket
        return BUCKETS[-1]

    def sample_gate_count(self, rng: Any) -> int:
        """Draw an episode length from the current mixture."""

        bucket = self.bucket_for(float(rng.random()))
        low, high = BUCKET_GATE_RANGES[bucket]
        return int(low if low == high else rng.integers(low, high + 1))

    def _qualifies(self, metrics: Mapping[str, Any]) -> bool:
        n_eval = max(1, int(metrics.get("n_eval", 0) or 0))
        success = float(metrics.get("universal_transition_success_rate", 0.0) or 0.0)
        long_score = float(metrics.get("long_sequence_completion_score", 0.0) or 0.0)
        collisions = float(metrics.get("collision_episodes", 0) or 0) / n_eval
        return (
            success >= float(self.promotion["min_universal_transition_success_rate"])
            and long_score >= float(self.promotion["min_long_sequence_completion_score"])
            and collisions <= float(self.promotion["max_collision_episode_rate"])
        )

    def _fails(self, metrics: Mapping[str, Any]) -> bool:
        success = float(metrics.get("universal_transition_success_rate", 0.0) or 0.0)
        return success < float(self.demotion["min_universal_transition_success_rate"])

    def observe_validation(
        self, metrics: Mapping[str, Any], total_transitions: int
    ) -> Dict[str, Any]:
        """Update the stage from one *validation-band* evaluation."""

        self.state.total_environment_transitions = max(
            self.state.total_environment_transitions, int(total_transitions)
        )
        qualifies = self._qualifies(metrics)
        fails = self._fails(metrics)
        self.state.consecutive_qualifying = (
            self.state.consecutive_qualifying + 1 if qualifies else 0
        )
        self.state.consecutive_failing = (
            self.state.consecutive_failing + 1 if fails else 0
        )
        previous = self.state.stage_index
        promoted = demoted = False
        if (
            self.state.consecutive_qualifying
            >= int(self.promotion["consecutive_qualifying_evaluations"])
            and self.state.stage_index < len(self.stages) - 1
        ):
            self.state.stage_index += 1
            self.state.consecutive_qualifying = 0
            promoted = True
        elif (
            self.state.consecutive_failing
            >= int(self.demotion["consecutive_failing_evaluations"])
            and self.state.stage_index > 0
        ):
            self.state.stage_index -= 1
            self.state.consecutive_failing = 0
            demoted = True
        record = {
            "schema_version": CURRICULUM_VERSION,
            "total_environment_transitions": int(total_transitions),
            "from_stage": self.stages[previous]["name"],
            "to_stage": self.stage["name"],
            "promoted": promoted,
            "demoted": demoted,
            "qualifies": qualifies,
            "consecutive_qualifying": self.state.consecutive_qualifying,
            "consecutive_failing": self.state.consecutive_failing,
            "mixture": self.mixture,
        }
        self.state.history.append(record)
        return record

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": CURRICULUM_VERSION,
            "stage_index": self.state.stage_index,
            "consecutive_qualifying": self.state.consecutive_qualifying,
            "consecutive_failing": self.state.consecutive_failing,
            "total_environment_transitions": self.state.total_environment_transitions,
            "history": list(self.state.history[-100:]),
            "stages": [dict(stage) for stage in self.stages],
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if value.get("schema_version") != CURRICULUM_VERSION:
            raise ValueError("unsupported generic sequence curriculum state")
        self.state = SequenceCurriculumState(
            stage_index=int(value.get("stage_index", 0)),
            consecutive_qualifying=int(value.get("consecutive_qualifying", 0)),
            consecutive_failing=int(value.get("consecutive_failing", 0)),
            total_environment_transitions=int(
                value.get("total_environment_transitions", 0)
            ),
            history=list(value.get("history") or []),
        )
        self.state.stage_index = max(
            0, min(self.state.stage_index, len(self.stages) - 1)
        )


def empirical_mixture(gate_counts: Sequence[int]) -> Dict[str, float]:
    """Observed bucket shares, for verifying a sampler against its mixture."""

    counts = {bucket: 0 for bucket in BUCKETS}
    for value in gate_counts:
        for bucket, (low, high) in BUCKET_GATE_RANGES.items():
            if low <= int(value) <= high:
                counts[bucket] += 1
                break
    total = max(1, len(gate_counts))
    return {bucket: counts[bucket] / total for bucket in BUCKETS}
