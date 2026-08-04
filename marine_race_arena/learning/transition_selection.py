"""Competence qualification, safety ranking and final policy selection.

Selecting a universal-transition policy is split into three explicitly separated
stages so that a policy which avoids safety events by refusing to move can never
win:

``competence qualification``
    A mandatory gate.  A policy must demonstrate that it actually performs the
    task (crossing the first gate, switching to the new beacon, completing whole
    transitions) *and* that it actually participates (it moves, it commands
    non-trivial actions, it reaches gates).  Policies that fail are classified
    ``degenerate_inactive_policy`` or ``insufficient_task_competence`` and are
    excluded from ranking entirely.

``safety ranking``
    Applied *only* among policies that passed the gate, in the fixed priority
    collisions, missed gates, wrong direction, previous-gate returns,
    acquisition timeouts, transition success, long-sequence completion, jerk.

``final selection``
    The deterministic maximum of the combined key over candidates visited in
    sorted name order.

Collision ranking deliberately uses *episodes* first and *entries* second.
Sustained contact produces many repeated referee events from a single physical
collision, so raw frame or event counts alone would over-punish one long contact
relative to several distinct impacts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

COMPETENCE_GATE_VERSION = "universal_transition_competence_gate_v1"
SELECTION_SCHEMA_VERSION = "universal_transition_selection_v1"

CLASSIFICATION_COMPETENT = "competent"
CLASSIFICATION_DEGENERATE_INACTIVE = "degenerate_inactive_policy"
CLASSIFICATION_INSUFFICIENT_COMPETENCE = "insufficient_task_competence"
CLASSIFICATION_INSUFFICIENT_EVIDENCE = "insufficient_evaluation_evidence"

SAFETY_PRIORITY = (
    "collision_episode_rate",
    "collision_entries_per_episode",
    "missed_gate_dnf_rate",
    "wrong_direction_rate",
    "previous_gate_return_rate",
    "acquisition_timeout_rate",
    "universal_transition_success_rate",
    "long_sequence_completion_score",
    "mean_action_jerk",
)


@dataclass(frozen=True)
class CompetenceThresholds:
    """Minimum evidence a policy must show before safety ranking applies.

    The task-competence minimums are the mandated contract values.  The
    anti-inactivity minimums are conservative values chosen from the measured
    warm and scratch A/B evidence: the scratch policy crossed 15 gates in 1,012
    episodes with a mean absolute action of 0.0034, while the warm policy
    crossed 946 gates with a mean absolute action of 0.0848 and commanded a
    non-trivial action on 97.9% of steps.  Every threshold below therefore sits
    far above the inactive policy and far below the competent one.
    """

    # -- task competence (mandated) -------------------------------------
    min_first_gate_crossing_rate: float = 0.80
    min_target_switch_rate: float = 0.70
    min_universal_transition_success_rate: float = 0.20
    # -- task participation ---------------------------------------------
    min_completed_gate_count: int = 10
    min_completed_gates_per_episode: float = 0.50
    min_fraction_of_episodes_reaching_first_gate: float = 0.80
    # -- raw activity (anti-inactivity) ----------------------------------
    min_mean_distance_travelled_m: float = 1.0
    min_mean_absolute_action: float = 0.02
    min_nontrivial_action_fraction: float = 0.25
    # -- evidence sufficiency --------------------------------------------
    min_transition_cases: int = 50

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": COMPETENCE_GATE_VERSION,
            "min_first_gate_crossing_rate": self.min_first_gate_crossing_rate,
            "min_target_switch_rate": self.min_target_switch_rate,
            "min_universal_transition_success_rate": (
                self.min_universal_transition_success_rate
            ),
            "min_completed_gate_count": self.min_completed_gate_count,
            "min_completed_gates_per_episode": self.min_completed_gates_per_episode,
            "min_fraction_of_episodes_reaching_first_gate": (
                self.min_fraction_of_episodes_reaching_first_gate
            ),
            "min_mean_distance_travelled_m": self.min_mean_distance_travelled_m,
            "min_mean_absolute_action": self.min_mean_absolute_action,
            "min_nontrivial_action_fraction": self.min_nontrivial_action_fraction,
            "min_transition_cases": self.min_transition_cases,
        }


DEFAULT_COMPETENCE_THRESHOLDS = CompetenceThresholds()

# Rollback / collapse detection reuses exactly the mandated task-competence
# minimums so that "no longer competent" and "must not be promoted" agree.
BASELINE_COLLAPSE_THRESHOLDS = CompetenceThresholds()


@dataclass(frozen=True)
class Criterion:
    name: str
    metric: str
    minimum: float
    group: str
    observed: Optional[float]

    @property
    def evaluated(self) -> bool:
        return self.observed is not None

    @property
    def satisfied(self) -> bool:
        return self.observed is not None and float(self.observed) >= float(self.minimum)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric,
            "group": self.group,
            "minimum": float(self.minimum),
            "observed": None if self.observed is None else float(self.observed),
            "evaluated": self.evaluated,
            "satisfied": self.satisfied,
        }


@dataclass(frozen=True)
class CompetenceVerdict:
    passed: bool
    classification: str
    criteria: Tuple[Criterion, ...]
    thresholds: CompetenceThresholds

    @property
    def failed_criteria(self) -> Tuple[str, ...]:
        return tuple(
            item.name for item in self.criteria if item.evaluated and not item.satisfied
        )

    @property
    def unavailable_metrics(self) -> Tuple[str, ...]:
        return tuple(item.metric for item in self.criteria if not item.evaluated)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": COMPETENCE_GATE_VERSION,
            "passed": self.passed,
            "classification": self.classification,
            "failed_criteria": list(self.failed_criteria),
            "unavailable_metrics": list(self.unavailable_metrics),
            "thresholds": self.thresholds.as_dict(),
            "criteria": [item.as_dict() for item in self.criteria],
        }


def _number(metrics: Mapping[str, Any], key: str) -> Optional[float]:
    if key not in metrics:
        return None
    value = metrics[key]
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _count(metrics: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = _number(metrics, key)
    return default if value is None else value


def _episode_count(metrics: Mapping[str, Any]) -> int:
    """Episodes the counter metrics were accumulated over."""

    value = _number(metrics, "n_eval")
    return max(1, int(value)) if value is not None else 1


def _derived_activity(metrics: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    """Fill participation metrics that older reports only carry per episode."""

    n_eval = _episode_count(metrics)
    completed = _number(metrics, "completed_gate_count")
    per_episode = _number(metrics, "mean_completed_gates_per_episode")
    if per_episode is None and completed is not None:
        per_episode = completed / n_eval
    reaching = _number(metrics, "fraction_of_episodes_reaching_first_gate")
    if reaching is None:
        # The first-gate crossing rate is measured over transition cases only;
        # it is a valid participation signal when the whole-suite fraction was
        # not recorded.
        reaching = _number(metrics, "first_gate_crossing_rate")
    return {
        "completed_gate_count": completed,
        "mean_completed_gates_per_episode": per_episode,
        "fraction_of_episodes_reaching_first_gate": reaching,
    }


def evaluate_competence_gate(
    metrics: Mapping[str, Any],
    thresholds: CompetenceThresholds = DEFAULT_COMPETENCE_THRESHOLDS,
) -> CompetenceVerdict:
    """Qualify a policy for safety ranking.

    A criterion whose metric is absent from the report is recorded as
    ``evaluated=False`` and cannot fail the gate; this keeps frozen historical
    reports rankable while newly produced evaluations are checked in full.
    """

    activity = _derived_activity(metrics)
    criteria = (
        Criterion(
            "first_gate_crossing", "first_gate_crossing_rate",
            thresholds.min_first_gate_crossing_rate, "task_competence",
            _number(metrics, "first_gate_crossing_rate"),
        ),
        Criterion(
            "target_switch", "target_switch_rate",
            thresholds.min_target_switch_rate, "task_competence",
            _number(metrics, "target_switch_rate"),
        ),
        Criterion(
            "universal_transition_success", "universal_transition_success_rate",
            thresholds.min_universal_transition_success_rate, "task_competence",
            _number(metrics, "universal_transition_success_rate"),
        ),
        Criterion(
            "completed_gates", "completed_gate_count",
            thresholds.min_completed_gate_count, "task_participation",
            activity["completed_gate_count"],
        ),
        Criterion(
            "completed_gates_per_episode", "mean_completed_gates_per_episode",
            thresholds.min_completed_gates_per_episode, "task_participation",
            activity["mean_completed_gates_per_episode"],
        ),
        Criterion(
            "episodes_reaching_first_gate",
            "fraction_of_episodes_reaching_first_gate",
            thresholds.min_fraction_of_episodes_reaching_first_gate,
            "task_participation",
            activity["fraction_of_episodes_reaching_first_gate"],
        ),
        Criterion(
            "distance_travelled", "mean_distance_travelled_m",
            thresholds.min_mean_distance_travelled_m, "activity",
            _number(metrics, "mean_distance_travelled_m"),
        ),
        Criterion(
            "mean_absolute_action", "mean_absolute_action",
            thresholds.min_mean_absolute_action, "activity",
            _number(metrics, "mean_absolute_action"),
        ),
        Criterion(
            "nontrivial_action", "nontrivial_action_fraction",
            thresholds.min_nontrivial_action_fraction, "activity",
            _number(metrics, "nontrivial_action_fraction"),
        ),
        Criterion(
            "evaluated_transition_cases", "transition_n",
            thresholds.min_transition_cases, "evidence",
            _number(metrics, "transition_n"),
        ),
    )
    failed_groups = {
        item.group for item in criteria if item.evaluated and not item.satisfied
    }
    if not failed_groups:
        classification = CLASSIFICATION_COMPETENT
        passed = True
    elif "evidence" in failed_groups and failed_groups == {"evidence"}:
        classification = CLASSIFICATION_INSUFFICIENT_EVIDENCE
        passed = False
    elif "activity" in failed_groups:
        classification = CLASSIFICATION_DEGENERATE_INACTIVE
        passed = False
    else:
        classification = CLASSIFICATION_INSUFFICIENT_COMPETENCE
        passed = False
    return CompetenceVerdict(
        passed=passed,
        classification=classification,
        criteria=criteria,
        thresholds=thresholds,
    )


def is_competent(
    metrics: Mapping[str, Any],
    thresholds: CompetenceThresholds = DEFAULT_COMPETENCE_THRESHOLDS,
) -> bool:
    return evaluate_competence_gate(metrics, thresholds).passed


def collision_entry_count(metrics: Mapping[str, Any]) -> float:
    """Distinct collision entries, falling back to referee collision events.

    ``collision_events`` is the referee's cooldown-limited counter, so sustained
    contact still inflates it.  It is used only as the secondary collision key
    for reports produced before ``collision_entries`` existed.
    """

    entries = _number(metrics, "collision_entries")
    if entries is not None:
        return entries
    return _count(metrics, "collision_events")


def safety_rank_key(metrics: Mapping[str, Any]) -> Tuple[float, ...]:
    """Safety-first ordering among already-qualified policies (higher is better).

    Counters are normalized per evaluated episode so evaluations of different
    sizes remain comparable.  Sustained-contact frame counts are deliberately
    excluded.
    """

    n_eval = _episode_count(metrics)
    jerk = _number(metrics, "mean_action_jerk")
    return (
        -_count(metrics, "collision_episodes") / n_eval,
        -collision_entry_count(metrics) / n_eval,
        -_count(metrics, "missed_gate_dnf") / n_eval,
        -_count(metrics, "wrong_direction_events") / n_eval,
        -_count(metrics, "previous_gate_returns") / n_eval,
        -_count(metrics, "acquisition_timeouts") / n_eval,
        _count(metrics, "universal_transition_success_rate"),
        _count(metrics, "long_sequence_completion_score"),
        -float("inf") if jerk is None else -jerk,
    )


def transition_policy_rank_key(
    metrics: Mapping[str, Any],
    thresholds: CompetenceThresholds = DEFAULT_COMPETENCE_THRESHOLDS,
) -> Tuple[float, ...]:
    """Competence qualification first, then safety ranking.

    No number of avoided safety events can lift a policy that failed the
    competence gate above one that passed it.
    """

    qualified = 1.0 if evaluate_competence_gate(metrics, thresholds).passed else 0.0
    return (qualified,) + safety_rank_key(metrics)


def select_policy(
    candidates: Mapping[str, Mapping[str, Any]],
    thresholds: CompetenceThresholds = DEFAULT_COMPETENCE_THRESHOLDS,
) -> Dict[str, Any]:
    """Deterministically qualify, rank and select among named candidates.

    Candidates are visited in sorted name order and ``max`` keeps the first
    maximal element, so equal keys always resolve to the same name.
    """

    if not candidates:
        raise ValueError("selection requires at least one candidate")
    names = sorted(candidates)
    qualification = {
        name: evaluate_competence_gate(candidates[name], thresholds) for name in names
    }
    competent = [name for name in names if qualification[name].passed]
    ranked = sorted(
        competent,
        key=lambda name: safety_rank_key(candidates[name]),
        reverse=True,
    )
    selected = ranked[0] if ranked else None
    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "competence_gate": thresholds.as_dict(),
        "safety_priority": list(SAFETY_PRIORITY),
        "qualification": {
            name: qualification[name].as_dict() for name in names
        },
        "competent_candidates": list(ranked),
        "rejected_candidates": {
            name: qualification[name].classification
            for name in names
            if not qualification[name].passed
        },
        "safety_rank_keys": {
            name: list(safety_rank_key(candidates[name])) for name in names
        },
        "selected": selected,
        "selection_reason": (
            "no candidate passed the competence gate"
            if selected is None
            else "highest safety rank among competent candidates"
        ),
    }


def baseline_collapse_criteria(
    metrics: Mapping[str, Any],
    thresholds: CompetenceThresholds = BASELINE_COLLAPSE_THRESHOLDS,
) -> Tuple[str, ...]:
    """Task-competence minimums a policy has fallen below (empty when healthy)."""

    verdict = evaluate_competence_gate(metrics, thresholds)
    return tuple(
        item.name
        for item in verdict.criteria
        if item.group == "task_competence" and item.evaluated and not item.satisfied
    )


def has_collapsed_below_baseline(
    metrics: Mapping[str, Any],
    thresholds: CompetenceThresholds = BASELINE_COLLAPSE_THRESHOLDS,
) -> bool:
    return bool(baseline_collapse_criteria(metrics, thresholds))


def thresholds_from_mapping(
    value: Optional[Mapping[str, Any]],
    base: CompetenceThresholds = DEFAULT_COMPETENCE_THRESHOLDS,
) -> CompetenceThresholds:
    """Build thresholds from config, rejecting unknown keys."""

    if not value:
        return base
    known = set(base.as_dict()) - {"schema_version"}
    unknown = sorted(set(value) - known)
    if unknown:
        raise ValueError(f"unknown competence threshold keys {unknown}")
    return replace(base, **{key: type(getattr(base, key))(value[key]) for key in value})


def qualification_table(
    candidates: Mapping[str, Mapping[str, Any]],
    thresholds: CompetenceThresholds = DEFAULT_COMPETENCE_THRESHOLDS,
) -> Sequence[Dict[str, Any]]:
    """Rows describing why each candidate did or did not qualify."""

    rows = []
    for name in sorted(candidates):
        verdict = evaluate_competence_gate(candidates[name], thresholds)
        rows.append({
            "candidate": name,
            "classification": verdict.classification,
            "passed": verdict.passed,
            "failed_criteria": list(verdict.failed_criteria),
            "unavailable_metrics": list(verdict.unavailable_metrics),
        })
    return rows
