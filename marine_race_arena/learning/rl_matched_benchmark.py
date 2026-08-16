"""Paired validation benchmark for choosing between transition checkpoints.

Unpaired n=112 evaluations cannot discriminate between PPO checkpoints: a policy
whose weights had moved by 0.006% scored anywhere from 0.51 to 0.87 across
separate small samples, so almost the entire observed spread was case-difficulty
noise rather than policy quality.  Two design decisions remove that noise:

``matched cases``
    Every candidate is evaluated on the *identical* fixed VALIDATION seed set,
    which makes the generated geometry list byte-for-byte the same for each
    candidate.  Comparisons are then made per case, so the difficulty of an
    individual course cancels out of the difference instead of inflating its
    variance.

``a rule fixed before the numbers exist``
    :data:`SELECTION_RULE` is data-independent and serialisable, and the plan
    records its fingerprint.  The ranking hierarchy is therefore logged before
    any candidate is run and cannot be reshaped afterwards to fit a favourite.

Statistics are implemented here rather than pulled from scipy so that the
selection path has no dependency the evaluation environment does not already
carry, and so every interval is unit-testable without an engine.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from marine_race_arena.learning.longrun_checkpoint import canonical_hash, sha256_file
from marine_race_arena.learning.provenance import now_utc
from marine_race_arena.learning.rl_holdout_policy import (
    evaluation_seeds,
    role_of_seed,
    seed_group,
)

MATCHED_BENCHMARK_VERSION = "rl_matched_benchmark_v1"
SELECTION_RULE_VERSION = "matched_selection_rule_v1"

#: The benchmark is a model-selection instrument, so it lives on VALIDATION.
#: Train seeds would measure memorisation and TEST is never spent.
BENCHMARK_ROLE = "validation"

#: Fixed so two runs of the same comparison produce the same intervals.
DEFAULT_BOOTSTRAP_SEED = 20260808

#: Which per-episode field decides success for each benchmark episode type.
#: Pooling both types gives the primary paired metric every evaluated case.
CASE_SUCCESS_FIELDS: Dict[str, str] = {
    "transition_focus": "universal_transition_success",
    "full_sequence": "full_sequence_completion",
}
PRIMARY_PAIRED_METRIC = "case_success"

RANKING_CRITERIA: Tuple[str, ...] = (
    "sequence_reliability",
    "universal_transition_success",
    "safety",
    "switch_alignment_quality",
    "smoothness",
)

_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# Stands in for -inf inside sort keys: an unmeasured criterion must lose to every
# measured one, but the score itself has to stay JSON-serialisable.
_MISSING_SCORE = -1e308


# --------------------------------------------------------------- selection rule

@dataclass(frozen=True)
class HealthGates:
    """Hard rejection thresholds applied before any ranking happens.

    A policy that barely moves produces very few safety events, so without these
    gates the safest-looking candidate is the one that refuses to fly.  The
    activity minimums are the values the competence gate already uses; the event
    ceilings are deliberately loose, because they exist to reject broken
    candidates, not to do the ranking.
    """

    min_nontrivial_action_fraction: float = 0.25
    min_mean_absolute_action: float = 0.02
    max_out_of_bounds_rate: float = 0.05
    max_wrong_direction_rate: float = 0.10
    max_collision_episode_rate: float = 0.35

    def as_dict(self) -> Dict[str, float]:
        return {
            "min_nontrivial_action_fraction": float(self.min_nontrivial_action_fraction),
            "min_mean_absolute_action": float(self.min_mean_absolute_action),
            "max_out_of_bounds_rate": float(self.max_out_of_bounds_rate),
            "max_wrong_direction_rate": float(self.max_wrong_direction_rate),
            "max_collision_episode_rate": float(self.max_collision_episode_rate),
        }


@dataclass(frozen=True)
class SelectionRule:
    """The complete, data-independent hierarchy used to pick a parent.

    ``criteria`` is applied lexicographically, higher is better at every level.
    Smoothness sits last on purpose: a smoother policy that completes fewer
    sequences is not a better parent, so jerk may only break ties.
    """

    version: str = SELECTION_RULE_VERSION
    gates: HealthGates = HealthGates()
    criteria: Tuple[str, ...] = RANKING_CRITERIA
    #: Completion of an L-gate sequence is weighted by ``L ** exponent`` so that
    #: chaining twenty gates counts for more than chaining three.
    length_weight_exponent: float = 1.0
    primary_paired_metric: str = PRIMARY_PAIRED_METRIC
    confidence: float = 0.95
    bootstrap_resamples: int = 10000
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED

    def as_dict(self) -> Dict[str, Any]:
        return {
            "version": str(self.version),
            "step_0_reject": self.gates.as_dict(),
            "criteria": list(self.criteria),
            "length_weight_exponent": float(self.length_weight_exponent),
            "primary_paired_metric": str(self.primary_paired_metric),
            "confidence": float(self.confidence),
            "bootstrap_resamples": int(self.bootstrap_resamples),
            "bootstrap_seed": int(self.bootstrap_seed),
        }

    def fingerprint(self) -> str:
        """Hash of the rule, so a plan can prove which rule it was written with."""

        return canonical_hash(self.as_dict())


SELECTION_RULE = SelectionRule()


def selection_rule_from_mapping(value: Mapping[str, Any]) -> SelectionRule:
    """Rebuild a rule from its serialised form, rejecting unknown keys."""

    known = set(SELECTION_RULE.as_dict())
    unknown = sorted(set(value) - known)
    if unknown:
        raise ValueError(f"unknown selection rule keys {unknown}")
    gates_value = dict(value.get("step_0_reject") or {})
    gate_keys = set(HealthGates().as_dict())
    unknown_gates = sorted(set(gates_value) - gate_keys)
    if unknown_gates:
        raise ValueError(f"unknown selection gate keys {unknown_gates}")
    return SelectionRule(
        version=str(value.get("version", SELECTION_RULE_VERSION)),
        gates=HealthGates(**{key: float(val) for key, val in gates_value.items()}),
        criteria=tuple(value.get("criteria", RANKING_CRITERIA)),
        length_weight_exponent=float(value.get("length_weight_exponent", 1.0)),
        primary_paired_metric=str(
            value.get("primary_paired_metric", PRIMARY_PAIRED_METRIC)
        ),
        confidence=float(value.get("confidence", 0.95)),
        bootstrap_resamples=int(value.get("bootstrap_resamples", 10000)),
        bootstrap_seed=int(value.get("bootstrap_seed", DEFAULT_BOOTSTRAP_SEED)),
    )


# ------------------------------------------------------------------ statistics

def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF (Acklam) with one Halley refinement step.

    The refinement uses :func:`math.erfc`, which brings the approximation to
    near machine precision; scipy is deliberately not a dependency here.
    """

    if not 0.0 < float(p) < 1.0:
        raise ValueError(f"normal quantile requires 0 < p < 1, got {p}")
    p = float(p)
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    low = 0.02425
    if p < low:
        q = math.sqrt(-2.0 * math.log(p))
        x = ((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
             / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0))
    elif p <= 1.0 - low:
        q = p - 0.5
        r = q * q
        x = ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
             / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0))
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
              / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0))
    error = 0.5 * math.erfc(-x / math.sqrt(2.0)) - p
    step = error * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return x - step / (1.0 + x * step / 2.0)


def _z_for(confidence: float) -> float:
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    return _normal_quantile(0.5 + float(confidence) / 2.0)


def _validate_counts(successes: int, n: int) -> Tuple[int, int]:
    successes, n = int(successes), int(n)
    if n <= 0:
        raise ValueError("an interval needs at least one observation")
    if not 0 <= successes <= n:
        raise ValueError(f"successes {successes} outside [0, {n}]")
    return successes, n


def wilson_interval(
    successes: int, n: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """Wilson score interval for a proportion -- the primary CI.

    Preferred over the normal approximation because the success rates that
    matter here sit near 0 or 1, where the normal interval leaves the unit
    interval and reports zero width at the extremes.
    """

    successes, n = _validate_counts(successes, n)
    z = _z_for(confidence)
    rate = successes / n
    denominator = 1.0 + z * z / n
    centre = (rate + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(rate * (1.0 - rate) / n + z * z / (4.0 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_interval(
    successes: int,
    n: int,
    *,
    confidence: float = 0.95,
    resamples: int = 10000,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for a proportion.

    Resampling a 0/1 vector with replacement is exactly a Binomial(n, p_hat)
    draw, so the resamples are generated directly instead of materialising and
    indexing the outcome vector ten thousand times.
    """

    successes, n = _validate_counts(successes, n)
    if int(resamples) < 2:
        raise ValueError("bootstrap requires at least two resamples")
    alpha = 1.0 - float(confidence)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    rng = np.random.default_rng(int(seed))
    draws = rng.binomial(n, successes / n, size=int(resamples)) / n
    low, high = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return (float(low), float(high))


def _binomial_two_sided_p(n_positive: int, n_negative: int) -> float:
    """Exact two-sided sign-test p-value under p=0.5."""

    trials = int(n_positive) + int(n_negative)
    if trials <= 0:
        # No discordant pair carries any evidence about the direction.
        return 1.0
    extreme = min(int(n_positive), int(n_negative))
    if trials <= 4096:
        tail = sum(math.comb(trials, index) for index in range(extreme + 1))
        # Both operands are exact integers and the ratio is <= 1, so the float
        # division is correctly rounded and cannot overflow.
        return min(1.0, 2.0 * (tail / (1 << trials)))
    # Beyond a few thousand pairs the exact integer sum costs more than it is
    # worth; the same binomial mass is summed in log space instead.
    log_half = math.log(0.5)
    tail = math.fsum(
        math.exp(
            math.lgamma(trials + 1)
            - math.lgamma(index + 1)
            - math.lgamma(trials - index + 1)
            + trials * log_half
        )
        for index in range(extreme + 1)
    )
    return min(1.0, 2.0 * tail)


def _outcome_array(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite outcomes")
    return array


def align_outcomes(
    a_outcomes: Mapping[str, Any], b_outcomes: Mapping[str, Any]
) -> Tuple[List[float], List[float], List[str]]:
    """Align two case->outcome maps on the cases both candidates actually ran."""

    keys = sorted(set(a_outcomes) & set(b_outcomes))
    if not keys:
        raise ValueError("candidates share no evaluated case; they are not paired")
    return (
        [float(a_outcomes[key]) for key in keys],
        [float(b_outcomes[key]) for key in keys],
        keys,
    )


def paired_difference(
    a_outcomes: Any,
    b_outcomes: Any,
    *,
    confidence: float = 0.95,
    resamples: int = 10000,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Difference between two candidates measured case by case.

    Bootstrapping over *pairs* rather than over each candidate separately is the
    whole point of the module: the difficulty of a case enters both outcomes and
    cancels in the difference, so a small real advantage becomes detectable even
    when the two marginal intervals overlap almost completely.

    Accepts either aligned sequences or two case->outcome mappings, in which
    case only the cases present on both sides are compared.
    """

    aligned_on: Optional[List[str]] = None
    if isinstance(a_outcomes, Mapping) and isinstance(b_outcomes, Mapping):
        a_values, b_values, aligned_on = align_outcomes(a_outcomes, b_outcomes)
    else:
        a_values, b_values = a_outcomes, b_outcomes
    a = _outcome_array(a_values, name="a_outcomes")
    b = _outcome_array(b_values, name="b_outcomes")
    if a.size != b.size:
        raise ValueError(
            f"paired comparison needs equal-length outcomes, got {a.size} and {b.size}"
        )
    alpha = 1.0 - float(confidence)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    if int(resamples) < 2:
        raise ValueError("bootstrap requires at least two resamples")

    difference = a - b
    n_pairs = int(difference.size)
    n_a_better = int(np.count_nonzero(difference > 0.0))
    n_b_better = int(np.count_nonzero(difference < 0.0))

    rng = np.random.default_rng(int(seed))
    means = np.empty(int(resamples), dtype=np.float64)
    # Chunked so a large case count cannot allocate a resamples x n index matrix.
    chunk = max(1, min(int(resamples), max(1, 2_000_000 // n_pairs)))
    filled = 0
    while filled < int(resamples):
        size = min(chunk, int(resamples) - filled)
        indices = rng.integers(0, n_pairs, size=(size, n_pairs))
        means[filled:filled + size] = difference[indices].mean(axis=1)
        filled += size
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    interval = (float(low), float(high))

    return {
        "schema_version": MATCHED_BENCHMARK_VERSION,
        "n_pairs": n_pairs,
        "aligned_on_cases": None if aligned_on is None else len(aligned_on),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_difference": float(difference.mean()),
        "difference_ci": [interval[0], interval[1]],
        "confidence": float(confidence),
        "n_discordant": n_a_better + n_b_better,
        "n_a_better": n_a_better,
        "n_b_better": n_b_better,
        "sign_test_p_value": _binomial_two_sided_p(n_a_better, n_b_better),
        "equivalent_within_noise": equivalent_within_noise(interval),
        "bootstrap_resamples": int(resamples),
        "bootstrap_seed": int(seed),
    }


def equivalent_within_noise(diff_ci: Any) -> bool:
    """Whether a paired difference interval straddles zero.

    Accepts the interval itself or the whole :func:`paired_difference` result.
    """

    if isinstance(diff_ci, Mapping):
        diff_ci = diff_ci["difference_ci"]
    low, high = (float(diff_ci[0]), float(diff_ci[1]))
    if low > high:
        raise ValueError(f"interval bounds are inverted: {(low, high)}")
    return low <= 0.0 <= high


# ------------------------------------------------------------------- case data

def case_outcomes(report: Mapping[str, Any]) -> Dict[str, int]:
    """Per-case 0/1 success for one evaluation report.

    Keys are built from the case identity rather than its position, so two
    reports stay aligned even if one was produced by a resumed run that wrote
    its rows in a different order.  The identity is namespaced by the benchmark
    seed because episode seeds are drawn *with replacement* from a 50k band:
    without that prefix, two benchmark seeds that happened to draw the same
    episode seed would collapse two distinct cases into one and silently drop a
    paired unit.

    That same replacement draw also makes a handful of cases per thousand share
    an identity within one report; those are separated by order of appearance,
    which is well defined because the evaluator rebuilds its rows in case order
    even after a resume.  Reordering is therefore tolerated for distinct
    identities only, which is why the identity is made as specific as the row
    allows.
    """

    rows = report.get("episodes") or []
    benchmark_seed = report.get("seed")
    seen: Dict[str, int] = {}
    outcomes: Dict[str, int] = {}
    for row in rows:
        episode_type = str(row.get("episode_type"))
        field = CASE_SUCCESS_FIELDS.get(episode_type)
        if field is None:
            continue
        base = (
            f"{benchmark_seed}:{row.get('seed')}:{episode_type}:"
            f"{row.get('gate_count')}:{row.get('geometry_group')}"
        )
        occurrence = seen.get(base, 0)
        seen[base] = occurrence + 1
        outcomes[f"{base}:{occurrence}"] = int(bool(row.get(field)))
    return outcomes


# ---------------------------------------------------------------------- scores

def _number(metrics: Mapping[str, Any], key: str) -> Optional[float]:
    value = metrics.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _episode_count(metrics: Mapping[str, Any]) -> Optional[int]:
    value = _number(metrics, "n_eval")
    return None if value is None or value <= 0 else int(value)


def _rate(metrics: Mapping[str, Any], key: str) -> Optional[float]:
    """Counter normalised per evaluated episode, ``None`` when unmeasurable."""

    count = _number(metrics, key)
    n_eval = _episode_count(metrics)
    return None if count is None or n_eval is None else count / n_eval


def sequence_reliability_score(
    metrics: Mapping[str, Any], *, exponent: float = 1.0
) -> Optional[float]:
    """Mean sequence completion across measured lengths, weighted toward longer.

    Two-gate transition success is already the second ranking criterion; this
    one exists to reward policies that chain gates without accumulating error,
    which is the actual open problem.
    """

    by_length = metrics.get("full_sequence_success_by_length") or {}
    numerator = denominator = 0.0
    for key, entry in by_length.items():
        try:
            length = int(key)
        except (TypeError, ValueError):
            continue
        if not isinstance(entry, Mapping) or int(entry.get("n", 0) or 0) <= 0:
            continue
        rate = entry.get("completion_rate")
        if rate is None:
            continue
        weight = float(length) ** float(exponent)
        numerator += weight * float(rate)
        denominator += weight
    return None if denominator <= 0.0 else numerator / denominator


def _switch_alignment_score(metrics: Mapping[str, Any]) -> Optional[float]:
    values = [
        _number(metrics, key)
        for key in (
            "target_switch_rate",
            "new_target_alignment_rate",
            "new_target_range_decrease_rate",
        )
    ]
    present = [value for value in values if value is not None]
    return None if not present else sum(present) / len(present)


def _safety_score(metrics: Mapping[str, Any]) -> Optional[float]:
    rates = [
        _rate(metrics, key)
        for key in ("collision_episodes", "missed_gate_dnf", "out_of_bounds_episodes")
    ]
    present = [value for value in rates if value is not None]
    return None if not present else -sum(present)


def candidate_scores(
    metrics: Mapping[str, Any], rule: SelectionRule = SELECTION_RULE
) -> Dict[str, Optional[float]]:
    """Higher-is-better score per ranking criterion, ``None`` when unmeasured."""

    jerk = _number(metrics, "mean_action_jerk")
    return {
        "sequence_reliability": sequence_reliability_score(
            metrics, exponent=rule.length_weight_exponent
        ),
        "universal_transition_success": _number(
            metrics, "universal_transition_success_rate"
        ),
        "safety": _safety_score(metrics),
        "switch_alignment_quality": _switch_alignment_score(metrics),
        "smoothness": None if jerk is None else -jerk,
    }


def health_gate_failures(
    metrics: Mapping[str, Any], gates: HealthGates = SELECTION_RULE.gates
) -> List[Dict[str, Any]]:
    """Step 0 of the rule: reasons this candidate must not be ranked at all.

    A metric that the report does not carry is recorded as unevaluated and
    cannot reject: silently failing a candidate because an older report predates
    a field would be indistinguishable from failing it on evidence.  A *counter*
    without its denominator is a different case and refuses to pass quietly:
    three of the five gates would be disabled at once, so the missing episode
    count is raised instead of being read as clean flying.
    """

    if _episode_count(metrics) is None and any(
        _number(metrics, key) is not None
        for key in ("out_of_bounds_episodes", "wrong_direction_events",
                    "collision_episodes")
    ):
        raise ValueError(
            "metrics carry safety counters but no usable n_eval; the event "
            "rates the step-0 gates test cannot be formed"
        )
    checks = (
        ("nontrivial_action", "nontrivial_action_fraction",
         _number(metrics, "nontrivial_action_fraction"),
         gates.min_nontrivial_action_fraction, "min"),
        ("mean_absolute_action", "mean_absolute_action",
         _number(metrics, "mean_absolute_action"),
         gates.min_mean_absolute_action, "min"),
        ("out_of_bounds", "out_of_bounds_episode_rate",
         _rate(metrics, "out_of_bounds_episodes"),
         gates.max_out_of_bounds_rate, "max"),
        ("wrong_direction", "wrong_direction_rate",
         _rate(metrics, "wrong_direction_events"),
         gates.max_wrong_direction_rate, "max"),
        ("collision", "collision_episode_rate",
         _rate(metrics, "collision_episodes"),
         gates.max_collision_episode_rate, "max"),
    )
    failures = []
    for name, metric, observed, limit, direction in checks:
        if observed is None:
            continue
        failed = observed < limit if direction == "min" else observed > limit
        if failed:
            failures.append({
                "gate": name,
                "metric": metric,
                "observed": float(observed),
                "limit": float(limit),
                "direction": direction,
            })
    return failures


# --------------------------------------------------------------------- ranking

def _metrics_of(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept either a full result record or a bare metrics mapping."""

    inner = value.get("metrics")
    return inner if isinstance(inner, Mapping) else value


def _as_result_map(results: Any) -> Dict[str, Mapping[str, Any]]:
    if isinstance(results, Mapping):
        return {str(name): value for name, value in results.items()}
    mapped: Dict[str, Mapping[str, Any]] = {}
    for entry in results:
        name = str(entry.get("candidate") or entry.get("name"))
        mapped[name] = entry
    return mapped


def rank_candidates(
    results: Any, *, rule: SelectionRule = SELECTION_RULE
) -> List[Dict[str, Any]]:
    """Apply the pre-registered hierarchy: reject first, then rank survivors.

    Rejected candidates are kept in the output so the report explains why a
    candidate is absent from the contest, but they always sort last and can
    never be selected.
    """

    mapped = _as_result_map(results)
    if not mapped:
        raise ValueError("ranking requires at least one candidate")
    rows = []
    for name in sorted(mapped):
        metrics = _metrics_of(mapped[name])
        failures = health_gate_failures(metrics, rule.gates)
        rows.append({
            "candidate": name,
            "rejected": bool(failures),
            "gate_failures": failures,
            "scores": candidate_scores(metrics, rule),
            "criteria": list(rule.criteria),
        })

    def order_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
        scores = row["scores"]
        values = tuple(
            -(_MISSING_SCORE if scores.get(name) is None else float(scores[name]))
            for name in rule.criteria
        )
        return (1 if row["rejected"] else 0,) + values + (row["candidate"],)

    ordered = sorted(rows, key=order_key)
    rank = 0
    for row in ordered:
        if row["rejected"]:
            row["rank"] = None
            row["rejection_reason"] = "; ".join(
                f"{item['metric']}={item['observed']:.4g} violates "
                f"{item['direction']} {item['limit']:.4g}"
                for item in row["gate_failures"]
            )
        else:
            rank += 1
            row["rank"] = rank
            row["rejection_reason"] = None
    return ordered


def _score_or_none(ranking_row: Mapping[str, Any], name: str) -> Optional[float]:
    value = ranking_row["scores"].get(name)
    return None if value is None else float(value)


def select_final_parent(
    results: Any, rule: SelectionRule = SELECTION_RULE
) -> Dict[str, Any]:
    """Pick the parent checkpoint and record exactly why it won.

    When the top two are statistically indistinguishable on the primary paired
    metric, the ranking order is noise and is not used to break the tie; the
    safer candidate is taken instead and the reason says so.
    """

    mapped = _as_result_map(results)
    ranking = rank_candidates(mapped, rule=rule)
    accepted = [row for row in ranking if not row["rejected"]]
    record: Dict[str, Any] = {
        "schema_version": MATCHED_BENCHMARK_VERSION,
        "selection_rule": rule.as_dict(),
        "selection_rule_fingerprint": rule.fingerprint(),
        "ranking": ranking,
        "paired_comparison": None,
        "utc": now_utc(),
    }
    if not accepted:
        record.update({
            "selected": None,
            "checkpoint": None,
            "sha256": None,
            "reason": (
                "every candidate failed a step-0 health gate; "
                "no parent may be promoted"
            ),
        })
        return record

    leader = accepted[0]
    reason = (
        f"ranked first on {rule.criteria[0]} and the criteria below it "
        f"among {len(accepted)} candidates passing the health gates"
    )
    if len(accepted) >= 2:
        runner_up = accepted[1]
        comparison = _compare_top_two(mapped, leader, runner_up, rule)
        record["paired_comparison"] = comparison
        if comparison is not None and comparison["equivalent_within_noise"]:
            leader_safety = _score_or_none(leader, "safety")
            runner_safety = _score_or_none(runner_up, "safety")
            tied = (
                f"{leader['candidate']} and {runner_up['candidate']} are "
                f"statistically equivalent on {rule.primary_paired_metric} "
                f"(paired mean difference {comparison['mean_difference']:+.4f}, "
                f"{int(rule.confidence * 100)}% CI "
                f"[{comparison['difference_ci'][0]:+.4f}, "
                f"{comparison['difference_ci'][1]:+.4f}] straddles 0)"
            )
            if (
                runner_safety is not None
                and leader_safety is not None
                and runner_safety > leader_safety
            ):
                leader = runner_up
                reason = (
                    f"{tied}; the tie was broken on safety, which favours "
                    f"{runner_up['candidate']} "
                    f"(safety score {runner_safety:.4f} vs {leader_safety:.4f})"
                )
            else:
                reason = (
                    f"{tied}; safety does not separate them, so the "
                    f"pre-registered rank order stands"
                )

    source = mapped.get(leader["candidate"], {})
    record.update({
        "selected": leader["candidate"],
        "checkpoint": source.get("checkpoint"),
        "sha256": source.get("sha256"),
        "reason": reason,
    })
    return record


def _compare_top_two(
    mapped: Mapping[str, Mapping[str, Any]],
    leader: Mapping[str, Any],
    runner_up: Mapping[str, Any],
    rule: SelectionRule,
) -> Optional[Dict[str, Any]]:
    a = dict(mapped.get(leader["candidate"], {}).get("case_outcomes") or {})
    b = dict(mapped.get(runner_up["candidate"], {}).get("case_outcomes") or {})
    if not (set(a) & set(b)):
        return None
    comparison = paired_difference(
        a, b,
        confidence=rule.confidence,
        resamples=rule.bootstrap_resamples,
        seed=rule.bootstrap_seed,
    )
    comparison["metric"] = rule.primary_paired_metric
    comparison["a"] = leader["candidate"]
    comparison["b"] = runner_up["candidate"]
    return comparison


# ------------------------------------------------------------------- benchmark

def _normalize_candidates(candidates: Any) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if isinstance(candidates, Mapping):
        items = [{"name": name, "checkpoint": value}
                 for name, value in candidates.items()]
    else:
        items = []
        for entry in candidates:
            if isinstance(entry, Mapping):
                items.append(dict(entry))
            elif isinstance(entry, (str, Path)):
                items.append({"name": Path(entry).stem, "checkpoint": entry})
            else:
                name, checkpoint = entry
                items.append({"name": name, "checkpoint": checkpoint})
    for item in items:
        name = str(item.get("name") or Path(str(item["checkpoint"])).stem)
        if not _NAME_PATTERN.match(name):
            raise ValueError(
                f"candidate name {name!r} is not a safe output directory name"
            )
        checkpoint = Path(str(item["checkpoint"]))
        sha = item.get("sha256")
        if sha is None and checkpoint.is_file():
            sha = sha256_file(checkpoint)
        entries.append({
            "name": name,
            "checkpoint": str(checkpoint),
            "sha256": sha,
            "algorithm": str(item.get("algorithm", "ppo")),
        })
    names = [entry["name"] for entry in entries]
    if not names:
        raise ValueError("a matched benchmark needs at least one candidate")
    if len(set(names)) != len(names):
        raise ValueError(f"candidate names must be unique, got {sorted(names)}")
    return entries


def plan_matched_benchmark(
    candidates: Any,
    *,
    transition_cases: int,
    sequences_per_length: int,
    validation_offset: int = 0,
    seeds: int = 1,
    difficulty: str = "G6",
    rule: SelectionRule = SELECTION_RULE,
) -> Dict[str, Any]:
    """Fix the seed set, the workload and the selection rule before running.

    Two different seeds matter here and conflating them silently invalidates the
    whole comparison.  The benchmark seed is only the *sampler* seed: it decides
    which cases are drawn, and giving every candidate the same list is what makes
    the comparison paired case by case.  The band the individual episode seeds
    come from is decided by ``dataset_split``, which is pinned to VALIDATION so a
    checkpoint can never be selected on the courses it trained on, nor spend the
    sealed TEST band.  The plan records both, and ``assert_plan_uses_validation``
    verifies the realised geometry rather than trusting the declaration.
    """

    if int(transition_cases) < 1:
        raise ValueError("a matched benchmark needs at least one transition case")
    if int(sequences_per_length) < 0:
        raise ValueError("sequences_per_length cannot be negative")
    if int(seeds) < 1:
        raise ValueError("a matched benchmark needs at least one benchmark seed")

    group = seed_group(BENCHMARK_ROLE)
    benchmark_seeds = [
        int(value)
        for value in evaluation_seeds(
            BENCHMARK_ROLE, int(seeds), offset=int(validation_offset)
        )
    ]
    for seed in benchmark_seeds:
        role = role_of_seed(seed)
        if role != BENCHMARK_ROLE:
            raise ValueError(
                f"benchmark seed {seed} resolves to role {role!r}; model selection "
                f"may only use the {BENCHMARK_ROLE} band "
                f"[{group.start}, {group.end})"
            )
    if len(set(benchmark_seeds)) != len(benchmark_seeds):
        raise ValueError("benchmark seeds must be distinct to add evidence")

    entries = _normalize_candidates(candidates)
    plan = {
        "schema_version": MATCHED_BENCHMARK_VERSION,
        "paired": True,
        "seed_role": BENCHMARK_ROLE,
        "seed_band": group.as_dict(),
        "seeds": benchmark_seeds,
        "validation_offset": int(validation_offset),
        "difficulty": str(difficulty),
        "dataset_split": BENCHMARK_ROLE,
        "transition_cases": int(transition_cases),
        "full_cases_per_length": int(sequences_per_length),
        "cases_per_candidate": None,  # filled by the evaluator, geometry-dependent
        "candidates": [
            # The seed list is repeated per candidate so the pairing property is
            # explicit in the artifact rather than implied by a shared field.
            {**entry, "seeds": list(benchmark_seeds)}
            for entry in entries
        ],
        "selection_rule": rule.as_dict(),
        "selection_rule_fingerprint": rule.fingerprint(),
        "utc": now_utc(),
    }
    plan["plan_fingerprint"] = canonical_hash(
        {key: value for key, value in plan.items() if key != "utc"}
    )
    return plan


def assert_plan_uses_validation(plan: Mapping[str, Any]) -> Dict[str, Any]:
    """Materialise the plan's cases and check the band they actually landed in.

    The declared ``dataset_split`` is only a request.  This generates the real
    case list the evaluator will run and inspects every episode seed, so a
    regression that dropped the split on the way to the sampler is caught before
    hours of engine time are spent, not after.  It also proves the pairing
    property directly: identical arguments must yield identical geometry.
    """

    from marine_race_arena.learning.transition_evaluation import _benchmark_cases

    split = str(plan.get("dataset_split", BENCHMARK_ROLE))
    checked: Dict[str, Any] = {"dataset_split": split, "seeds": {}}
    for seed in plan["seeds"]:
        cases = _benchmark_cases(
            output=Path("unused"),
            seed=int(seed),
            difficulty=str(plan["difficulty"]),
            transition_cases=int(plan["transition_cases"]),
            full_cases_per_length=int(plan["full_cases_per_length"]),
            dataset_split=split,
        )
        episode_seeds = [geometry.seed for _, geometry, _, _ in cases]
        roles = sorted({role_of_seed(value) for value in episode_seeds})
        if roles != [BENCHMARK_ROLE]:
            raise ValueError(
                f"benchmark seed {seed} produced cases in roles {roles}; model "
                f"selection may only run {BENCHMARK_ROLE} geometry"
            )
        repeat = _benchmark_cases(
            output=Path("unused"),
            seed=int(seed),
            difficulty=str(plan["difficulty"]),
            transition_cases=int(plan["transition_cases"]),
            full_cases_per_length=int(plan["full_cases_per_length"]),
            dataset_split=split,
        )
        if [geometry for _, geometry, _, _ in repeat] != [
            geometry for _, geometry, _, _ in cases
        ]:
            raise ValueError(
                f"benchmark seed {seed} is not reproducible; candidates would "
                f"face different geometry and the comparison would not be paired"
            )
        checked["seeds"][str(seed)] = {
            "cases": len(cases),
            "distinct_episode_seeds": len(set(episode_seeds)),
            "episode_seed_range": [min(episode_seeds), max(episode_seeds)],
        }
    return checked


def rule_from_plan(plan: Mapping[str, Any]) -> SelectionRule:
    """Recover the rule the plan was written with, refusing a tampered copy."""

    rule = selection_rule_from_mapping(plan["selection_rule"])
    recorded = plan.get("selection_rule_fingerprint")
    # Absent is treated as tampered rather than as "unverifiable": accepting a
    # plan with no fingerprint would let anyone bypass the check by deleting the
    # field, which is the same edit the check exists to catch.
    if not recorded:
        raise ValueError(
            "plan carries no selection rule fingerprint, so the ranking "
            "hierarchy cannot be shown to predate the results"
        )
    if rule.fingerprint() != recorded:
        raise ValueError(
            "selection rule fingerprint does not match the plan; the ranking "
            "hierarchy was changed after the plan was recorded"
        )
    return rule


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=False), encoding="utf-8")
    os.replace(tmp, path)


#: Everything that has to match before a report on disk may stand in for a run.
#: ``difficulty`` belongs here: a G1 report and a G6 report describe completely
#: different courses, so reusing one for the other would compare two candidates
#: on different geometry -- exactly the unpaired comparison this module exists to
#: replace.  ``dataset_split`` belongs here for the opposite reason: a report
#: produced before the split was threaded through carries train-band geometry and
#: would quietly select a checkpoint on its own training courses.  Both are
#: matched strictly, so a report that is merely *silent* about one is recomputed.
_RESUME_IDENTITY = (
    "seed", "difficulty", "transition_cases", "full_cases_per_length",
    "dataset_split", "checkpoint",
)


def _expected_identity(
    plan: Mapping[str, Any], *, seed: int, checkpoint: str
) -> Dict[str, Any]:
    return {
        "seed": int(seed),
        "difficulty": str(plan["difficulty"]),
        "transition_cases": int(plan["transition_cases"]),
        "full_cases_per_length": int(plan["full_cases_per_length"]),
        "dataset_split": str(plan.get("dataset_split", BENCHMARK_ROLE)),
        "checkpoint": str(checkpoint),
    }


def _stamp_identity(
    report: Mapping[str, Any], expected: Mapping[str, Any], *, algorithm: str
) -> Dict[str, Any]:
    """Record on a fresh report what the run was actually asked to do.

    The module knows the arguments it passed to the runner, so it may record
    them; that is what lets a resumed sweep recognise the file later even when
    the injected runner does not describe its own work.  A runner that reports
    something *different* from what it was asked for is a bug that would break
    the pairing silently, so it raises instead of being overwritten.
    """

    stamped = dict(report)
    for key, value in expected.items():
        observed = stamped.get(key)
        if observed is not None and str(observed) != str(value):
            raise ValueError(
                f"runner returned a report with {key}={observed!r} but was asked "
                f"for {value!r}; candidates would not face the same cases"
            )
        stamped[key] = value
    stamped.setdefault("algorithm", str(algorithm))
    return stamped


def _reusable_report(
    path: Path, expected: Mapping[str, Any], *, algorithm: str
) -> Optional[Dict[str, Any]]:
    """An existing report only counts if it is the same work on the same policy."""

    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(report, Mapping) or not isinstance(report.get("metrics"), Mapping):
        return None
    for key in _RESUME_IDENTITY:
        if str(report.get(key)) != str(expected[key]):
            return None
    recorded = report.get("algorithm")
    if recorded is not None and str(recorded) != str(algorithm):
        return None
    return dict(report)


def _combined_metrics(reports: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(reports) == 1:
        return dict(reports[0]["metrics"])
    rows: List[Mapping[str, Any]] = []
    for report in reports:
        episodes = report.get("episodes")
        if not episodes:
            raise ValueError(
                "combining several benchmark seeds needs per-episode rows; "
                "one report carries only aggregated metrics"
            )
        rows.extend(episodes)
    from marine_race_arena.learning.transition_evaluation import (
        aggregate_transition_benchmark,
    )

    return aggregate_transition_benchmark(rows)


def _default_runner(checkpoint: Any, **kwargs: Any) -> Dict[str, Any]:
    # The checkpoint arrives positionally, matching the evaluator's own
    # signature.  Accepting only keywords here made every injected-runner test
    # pass while the production path raised TypeError on its first call.
    # Imported lazily so planning, ranking and the statistics stay importable in
    # environments that have no simulator installed.
    from marine_race_arena.learning.transition_evaluation import (
        evaluate_checkpoint_universal_transition_benchmark,
    )

    return evaluate_checkpoint_universal_transition_benchmark(checkpoint, **kwargs)


def run_matched_benchmark(
    plan: Mapping[str, Any],
    *,
    output_root: str | Path,
    parallel_workers: int,
    runner: Optional[Callable[..., Mapping[str, Any]]] = None,
    adapter: str = "holoocean",
    max_steps: int = 3600,
) -> Dict[str, Any]:
    """Evaluate every candidate on the identical planned cases and report.

    Resume-safe at candidate/seed granularity: a candidate whose report already
    exists for a seed is not re-run, which matters because a full matched sweep
    costs hours of engine time and is routinely interrupted.  The returned
    report is written to ``output_root/matched_benchmark.json``.
    """

    rule = rule_from_plan(plan)
    root = Path(output_root)
    run = runner if runner is not None else _default_runner
    if runner is None:
        # Only meaningful against the production evaluator; an injected runner in
        # a test has no obligation to consult the sampler at all.
        assert_plan_uses_validation(plan)
    results: Dict[str, Dict[str, Any]] = {}

    for candidate in plan["candidates"]:
        name = str(candidate["name"])
        checkpoint = str(candidate["checkpoint"])
        seeds = [int(value) for value in candidate.get("seeds", plan["seeds"])]
        if seeds != [int(value) for value in plan["seeds"]]:
            raise ValueError(
                f"candidate {name!r} carries a different seed set than the plan; "
                f"the benchmark would no longer be paired"
            )
        reports: List[Dict[str, Any]] = []
        reused: List[int] = []
        evaluated: List[int] = []
        paths: Dict[str, str] = {}
        algorithm = str(candidate.get("algorithm", "ppo"))
        for seed in seeds:
            output_dir = root / name / f"seed_{seed}"
            report_path = output_dir / "evaluation.json"
            expected = _expected_identity(plan, seed=seed, checkpoint=checkpoint)
            report = _reusable_report(report_path, expected, algorithm=algorithm)
            if report is None:
                report = _stamp_identity(
                    run(
                        checkpoint,
                        output_dir=output_dir,
                        seed=seed,
                        difficulty=str(plan["difficulty"]),
                        transition_cases=int(plan["transition_cases"]),
                        full_cases_per_length=int(plan["full_cases_per_length"]),
                        adapter=adapter,
                        max_steps=int(max_steps),
                        parallel_workers=int(parallel_workers),
                        algorithm=algorithm,
                        dataset_split=expected["dataset_split"],
                    ),
                    expected,
                    algorithm=algorithm,
                )
                # Persisted here as well as by the evaluator so that resume works
                # for any injected runner, not just the production one.
                _atomic_write_json(report_path, report)
                evaluated.append(seed)
            else:
                reused.append(seed)
            reports.append(report)
            paths[str(seed)] = str(report_path)

        outcomes: Dict[str, int] = {}
        for report in reports:
            outcomes.update(case_outcomes(report))
        results[name] = {
            "candidate": name,
            "checkpoint": checkpoint,
            "sha256": candidate.get("sha256"),
            "algorithm": algorithm,
            "seeds": seeds,
            "reused_seeds": reused,
            "evaluated_seeds": evaluated,
            "reused": bool(reused) and not evaluated,
            "report_paths": paths,
            "metrics": _combined_metrics(reports),
            "case_outcomes": outcomes,
            "n_cases": len(outcomes),
        }

    selection = select_final_parent(results, rule)
    report = {
        "schema_version": MATCHED_BENCHMARK_VERSION,
        "utc": now_utc(),
        "plan": dict(plan),
        "results": results,
        "ranking": selection["ranking"],
        "selection": selection,
    }
    _atomic_write_json(root / "matched_benchmark.json", report)
    return report
