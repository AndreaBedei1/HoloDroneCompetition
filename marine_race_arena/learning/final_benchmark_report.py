"""Aggregate, compare and report the final common benchmark.

Reads the episode table produced by :mod:`final_benchmark` and produces

* an aggregate comparison table (per controller and per test group),
* paired controller comparisons computed only on episodes that share the same
  case *and* the same seed, so no time is ever compared across suites,
* trajectory plots for representative successes and failures,
* a concise Markdown report and the recommended checkpoint.

Time is never aggregated across test groups: every timing figure belongs to one
group, and every timing comparison is a paired per-seed difference inside one
group.  The recommendation follows the stated priority order --- full-circuit
reliability, then zero safety events, then three-gate and vertical-transition
reliability, then completion time, then smoothness --- and is computed from this
benchmark alone, never from the training metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPORT_SCHEMA_VERSION = "final_benchmark_report_v1"

OFFICIAL_GROUPS = (
    "official_horseshoe_bay",
    "official_vertical_serpent",
    "official_mixed_endurance",
)
THREE_GATE_GROUPS = ("three_gate_sequence", "three_gate_s_shape")
VERTICAL_GROUPS = ("vertical_low_to_high", "vertical_high_to_low")
GROUP_ORDER = (
    "single_gate_retention",
    "two_gate_straight",
    "two_gate_left",
    "two_gate_right",
    "vertical_low_to_high",
    "vertical_high_to_low",
    "three_gate_sequence",
    "three_gate_s_shape",
) + OFFICIAL_GROUPS

# The official clock runs from the first gate to the last one, so a single-gate
# case has a definitionally zero completion time. Timing is neither reported nor
# compared for such groups; only their reliability and safety count.
UNTIMED_GROUPS = ("single_gate_retention",)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[Optional[float], Optional[float]]:
    """Wilson score interval; the right choice for small-n success counts."""
    if n <= 0:
        return (None, None)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


# Two-sided 95% t quantiles for small samples; 1.96 beyond the table.
_T95 = {
    2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
    9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145,
    16: 2.131, 17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093, 21: 2.086, 22: 2.080,
    23: 2.074, 24: 2.069, 25: 2.064, 26: 2.060, 27: 2.056, 28: 2.052, 29: 2.048,
    30: 2.045,
}


def _t95(n: int) -> float:
    if n <= 1:
        return float("nan")
    return _T95.get(n, 1.96)


def mean_ci(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """Mean with a Student-t 95% interval; ``None`` bounds when n < 2."""
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"n": 0, "mean": None, "median": None, "ci_low": None, "ci_high": None, "sd": None}
    n = len(clean)
    mean = statistics.fmean(clean)
    median = statistics.median(clean)
    if n < 2:
        return {"n": n, "mean": round(mean, 4), "median": round(median, 4),
                "ci_low": None, "ci_high": None, "sd": None}
    sd = statistics.stdev(clean)
    half = _t95(n) * sd / math.sqrt(n)
    return {
        "n": n,
        "mean": round(mean, 4),
        "median": round(median, 4),
        "sd": round(sd, 4),
        "ci_low": round(mean - half, 4),
        "ci_high": round(mean + half, 4),
    }


def sign_test_p(wins: int, losses: int) -> Optional[float]:
    """Exact two-sided binomial sign test on paired wins vs losses."""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return round(min(1.0, 2.0 * tail), 5)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _rate(successes: int, n: int) -> Optional[float]:
    return round(successes / n, 4) if n else None


def aggregate_rows(all_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """All benchmark-facing metrics for one homogeneous set of episodes.

    Episodes that never ran (a simulator launch failure recorded as
    ``HARNESS_ERROR``) are excluded from every rate: they are infrastructure
    failures, not controller results. Their count is reported as
    ``harness_errors`` so an incomplete cell is visible rather than silently
    depressing a success rate.
    """
    rows = [r for r in all_rows if r.get("referee_status") != "HARNESS_ERROR"]
    n = len(rows)
    successes = sum(1 for r in rows if r.get("finished"))
    full = sum(1 for r in rows if r.get("full_completion"))
    finished_rows = [r for r in rows if r.get("finished")]
    timed_rows = [
        r for r in finished_rows if str(r.get("group")) not in UNTIMED_GROUPS
    ]
    official_times = [r.get("official_time_s") for r in timed_rows]
    penalized_times = [r.get("penalized_time_s") for r in timed_rows]
    per_gate = [r.get("time_per_gate_s") for r in timed_rows if r.get("time_per_gate_s") is not None]
    success_ci = wilson_interval(successes, n)
    full_ci = wilson_interval(full, n)
    return {
        "episodes": n,
        "successes": successes,
        "success_rate": _rate(successes, n),
        "success_rate_wilson95_low": success_ci[0],
        "success_rate_wilson95_high": success_ci[1],
        "full_completions": full,
        "full_completion_rate": _rate(full, n),
        "full_completion_wilson95_low": full_ci[0],
        "full_completion_wilson95_high": full_ci[1],
        "gates_completed_total": int(sum(int(r.get("completed_gates", 0)) for r in rows)),
        "gates_expected_total": int(sum(int(r.get("expected_gates", 0)) for r in rows)),
        "mean_gates": round(
            sum(int(r.get("completed_gates", 0)) for r in rows) / n, 4
        ) if n else None,
        "completion_time_s": mean_ci([t for t in official_times if t is not None]),
        "penalized_time_s": mean_ci([t for t in penalized_times if t is not None]),
        "time_per_gate_s": mean_ci(per_gate),
        "collision_events": int(sum(int(r.get("collision_events", 0)) for r in rows)),
        "collision_frames": int(sum(int(r.get("collision_frames", 0)) for r in rows)),
        "collision_episodes": int(sum(1 for r in rows if r.get("collision_episode"))),
        "out_of_bounds_events": int(sum(int(r.get("out_of_bounds_events", 0)) for r in rows)),
        "out_of_bounds_frames": int(sum(int(r.get("out_of_bounds_frames", 0)) for r in rows)),
        "out_of_bounds_episodes": int(sum(1 for r in rows if r.get("out_of_bounds_episode"))),
        "wrong_direction_events": int(
            sum(int(r.get("wrong_direction_crossings", 0)) for r in rows)
        ),
        "wrong_direction_episodes": int(
            sum(1 for r in rows if r.get("wrong_direction_episode"))
        ),
        "previous_gate_returns": int(
            sum(int(r.get("previous_gate_returns", 0) or 0) for r in rows)
        ),
        "missed_gate_attempts": int(sum(int(r.get("missed_gate_attempts", 0)) for r in rows)),
        "stuck_events": int(sum(int(r.get("stuck_events", 0)) for r in rows)),
        "safety_event_episodes": int(sum(1 for r in rows if r.get("any_safety_event"))),
        "safety_events_total": int(
            sum(
                int(r.get("collision_events", 0))
                + int(r.get("out_of_bounds_events", 0))
                + int(r.get("wrong_direction_crossings", 0))
                for r in rows
            )
        ),
        "mean_action_jerk": mean_ci([r.get("mean_action_jerk") for r in rows]),
        "mean_action_saturation": mean_ci([r.get("action_saturation") for r in rows]),
        "path_length_m": mean_ci([r.get("path_length_m") for r in rows]),
        "timeouts": int(sum(1 for r in rows if r.get("timeout"))),
        "timeout_rate": _rate(sum(1 for r in rows if r.get("timeout")), n),
        "timing_meaningful": bool(timed_rows),
        "harness_errors": int(
            sum(1 for r in all_rows if r.get("referee_status") == "HARNESS_ERROR")
        ),
        "mean_inference_ms": mean_ci([r.get("mean_inference_ms") for r in rows]),
        "referee_status_counts": _counts(r.get("referee_status") for r in rows),
        "end_reason_counts": _counts(r.get("evaluation_end_reason") for r in rows),
        "failure_breakdown": failure_breakdown(rows),
        "seeds": sorted({int(r.get("seed", 0)) for r in rows}),
    }


def failure_mode(row: Mapping[str, Any]) -> Optional[str]:
    """Classify why one episode did not finish, from the referee's own record.

    ``missed_gate`` is the dominant mode on the official circuits: the referee
    DNFs a participant that bypasses its expected gate, so the episode ends where
    the policy lost the sequence rather than at the deadline.
    """
    if row.get("finished"):
        return None
    if row.get("referee_status") == "HARNESS_ERROR":
        return "harness_error"
    if int(row.get("missed_gate_attempts", 0) or 0) > 0:
        return "missed_gate_dnf"
    if int(row.get("stuck_events", 0) or 0) > 0:
        return "stuck"
    if row.get("referee_status") == "DSQ":
        return "disqualified"
    if row.get("timeout"):
        return "timeout_without_dnf"
    if row.get("referee_status") == "DNF":
        return "dnf_other"
    return f"unfinished_{str(row.get('referee_status', 'unknown')).lower()}"


def failure_breakdown(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Failure modes plus where in the gate sequence each episode stopped."""
    failures = [r for r in rows if not r.get("finished")]
    modes = _counts(failure_mode(r) for r in failures)
    stop_points = _counts(
        f"{int(r.get('completed_gates', 0))}/{int(r.get('expected_gates', 0))}"
        for r in failures
    )
    return {
        "failures": len(failures),
        "modes": modes,
        "stopped_after_gates": stop_points,
        "median_gates_at_failure": (
            statistics.median([int(r.get("completed_gates", 0)) for r in failures])
            if failures
            else None
        ),
    }


def _counts(values: Iterable[Any]) -> Dict[str, int]:
    out: Dict[str, int] = defaultdict(int)
    for value in values:
        out[str(value)] += 1
    return dict(sorted(out.items()))


def build_aggregates(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Per-controller aggregates: overall, per group, and per seed role."""
    controllers = sorted({str(r["controller"]) for r in rows})
    groups = [g for g in GROUP_ORDER if any(r.get("group") == g for r in rows)]
    groups += sorted({str(r["group"]) for r in rows} - set(groups))
    per_controller: Dict[str, Any] = {}
    for controller in controllers:
        subset = [r for r in rows if r.get("controller") == controller]
        non_official = [r for r in subset if not r.get("official_circuit")]
        official = [r for r in subset if r.get("official_circuit")]
        per_controller[controller] = {
            "overall_excluding_official": aggregate_rows(non_official),
            "official_circuits": aggregate_rows(official),
            "all_episodes": aggregate_rows(subset),
            "holdout_only": aggregate_rows(
                [r for r in subset if r.get("seed_role") == "holdout"]
            ),
            "reused_only": aggregate_rows(
                [r for r in subset if r.get("seed_role") == "reused"]
            ),
            "by_group": {
                group: aggregate_rows([r for r in subset if r.get("group") == group])
                for group in groups
                if any(r.get("group") == group for r in subset)
            },
            "by_case": {
                case: aggregate_rows([r for r in subset if r.get("case_uid") == case])
                for case in sorted({str(r["case_uid"]) for r in subset})
            },
        }
    return {"controllers": per_controller, "groups": groups}


# --------------------------------------------------------------------------- #
# Paired comparison
# --------------------------------------------------------------------------- #
def paired_comparison(
    rows: Sequence[Mapping[str, Any]],
    controller_a: str,
    controller_b: str,
    *,
    groups: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compare two controllers only on episodes sharing case *and* seed.

    Times are differenced within a pair, never pooled across cases, so a timing
    result never mixes two different geometries or two different suites.
    """
    def index(controller: str) -> Dict[Tuple[str, int], Mapping[str, Any]]:
        return {
            (str(r["case_uid"]), int(r["seed"])): r
            for r in rows
            if r.get("controller") == controller
            and (groups is None or r.get("group") in groups)
        }

    left, right = index(controller_a), index(controller_b)
    keys = sorted(
        key
        for key in set(left) & set(right)
        # A pair is only comparable when both episodes actually ran.
        if left[key].get("referee_status") != "HARNESS_ERROR"
        and right[key].get("referee_status") != "HARNESS_ERROR"
    )
    both_finished = [k for k in keys if left[k].get("finished") and right[k].get("finished")]
    a_only = [k for k in keys if left[k].get("finished") and not right[k].get("finished")]
    b_only = [k for k in keys if right[k].get("finished") and not left[k].get("finished")]
    time_diffs = [
        float(left[k]["official_time_s"]) - float(right[k]["official_time_s"])
        for k in both_finished
        if left[k].get("official_time_s") is not None
        and right[k].get("official_time_s") is not None
        and str(left[k].get("group")) not in UNTIMED_GROUPS
    ]
    jerk_diffs = [
        float(left[k].get("mean_action_jerk", 0.0)) - float(right[k].get("mean_action_jerk", 0.0))
        for k in keys
    ]
    path_diffs = [
        float(left[k].get("path_length_m", 0.0)) - float(right[k].get("path_length_m", 0.0))
        for k in both_finished
    ]
    faster = sum(1 for d in time_diffs if d < 0)
    slower = sum(1 for d in time_diffs if d > 0)
    return {
        "controller_a": controller_a,
        "controller_b": controller_b,
        "groups": list(groups) if groups else "all",
        "paired_episodes": len(keys),
        "a_successes": sum(1 for k in keys if left[k].get("finished")),
        "b_successes": sum(1 for k in keys if right[k].get("finished")),
        "a_full_completions": sum(1 for k in keys if left[k].get("full_completion")),
        "b_full_completions": sum(1 for k in keys if right[k].get("full_completion")),
        "both_finished": len(both_finished),
        "only_a_finished": len(a_only),
        "only_b_finished": len(b_only),
        "only_a_finished_cases": [f"{c}@{s}" for c, s in a_only],
        "only_b_finished_cases": [f"{c}@{s}" for c, s in b_only],
        "reliability_sign_test_p": sign_test_p(len(a_only), len(b_only)),
        "time_delta_s": mean_ci(time_diffs),
        "a_faster_episodes": faster,
        "b_faster_episodes": slower,
        "time_sign_test_p": sign_test_p(faster, slower),
        "jerk_delta": mean_ci(jerk_diffs),
        "path_length_delta_m": mean_ci(path_diffs),
        "a_safety_events": int(
            sum(
                int(left[k].get("collision_events", 0))
                + int(left[k].get("out_of_bounds_events", 0))
                + int(left[k].get("wrong_direction_crossings", 0))
                for k in keys
            )
        ),
        "b_safety_events": int(
            sum(
                int(right[k].get("collision_events", 0))
                + int(right[k].get("out_of_bounds_events", 0))
                + int(right[k].get("wrong_direction_crossings", 0))
                for k in keys
            )
        ),
    }


# --------------------------------------------------------------------------- #
# Recommendation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SelectionEntry:
    controller: str
    key: Tuple[float, ...]
    evidence: Dict[str, Any]


def group_mean_ranks(
    aggregates: Mapping[str, Any], metric: str, *, timed_only: bool
) -> Dict[str, Optional[float]]:
    """Average within-group rank of each controller on ``metric`` (1 = best).

    Ranking inside each group and then averaging the *ranks* keeps the ordering
    faithful to the rule that times are never compared across test suites: a rank
    carries no unit, so no second is ever weighed against a second from a
    different geometry. Groups where a controller finished nothing are skipped
    for that controller rather than imputed.
    """
    controllers = list(aggregates["controllers"])
    per_controller: Dict[str, List[float]] = {c: [] for c in controllers}
    for group in aggregates["groups"]:
        if timed_only and group in UNTIMED_GROUPS:
            continue
        values: List[Tuple[str, float]] = []
        for controller in controllers:
            entry = aggregates["controllers"][controller]["by_group"].get(group)
            if entry is None:
                continue
            value = entry[metric]["mean"] if isinstance(entry[metric], Mapping) else entry[metric]
            if value is None:
                continue
            values.append((controller, float(value)))
        if len(values) < 2:
            continue
        values.sort(key=lambda item: item[1])
        for position, (controller, _) in enumerate(values, start=1):
            per_controller[controller].append(float(position))
    return {
        controller: (round(statistics.fmean(ranks), 3) if ranks else None)
        for controller, ranks in per_controller.items()
    }


def selection_key(
    aggregate: Mapping[str, Any],
    *,
    time_rank: Optional[float],
    jerk_rank: Optional[float],
) -> Tuple[Tuple[float, ...], Dict[str, Any]]:
    """Lexicographic selection key in the stated priority order.

    1. full-circuit reliability (official circuits)
    2. zero safety events
    3. three-gate and vertical-transition reliability
    4. completion time (mean within-group rank, never a cross-suite time)
    5. smoothness (mean within-group jerk rank)
    """
    official = aggregate["official_circuits"]
    by_group = aggregate["by_group"]

    def group_rate(names: Sequence[str]) -> float:
        episodes = sum(by_group[g]["episodes"] for g in names if g in by_group)
        successes = sum(by_group[g]["successes"] for g in names if g in by_group)
        return successes / episodes if episodes else 0.0

    all_eps = aggregate["all_episodes"]
    official_full = official["full_completion_rate"] or 0.0
    safety_episodes = all_eps["safety_event_episodes"]
    three_gate = group_rate(THREE_GATE_GROUPS)
    vertical = group_rate(VERTICAL_GROUPS)
    non_official = aggregate["overall_excluding_official"]
    evidence = {
        "official_full_completion_rate": official_full,
        "official_episodes": official["episodes"],
        "safety_event_episodes": safety_episodes,
        "safety_events_total": all_eps["safety_events_total"],
        "three_gate_success_rate": round(three_gate, 4),
        "vertical_transition_success_rate": round(vertical, 4),
        "non_official_success_rate": non_official["success_rate"],
        "mean_within_group_time_rank": time_rank,
        "mean_within_group_jerk_rank": jerk_rank,
    }
    key = (
        official_full,
        -float(safety_episodes),
        min(three_gate, vertical),
        three_gate + vertical,
        non_official["success_rate"] or 0.0,
        -(time_rank if time_rank is not None else 1e9),
        -(jerk_rank if jerk_rank is not None else 1e9),
    )
    return key, evidence


def recommend(aggregates: Mapping[str, Any], controller_meta: Mapping[str, Any]) -> Dict[str, Any]:
    """Rank the fully policy-based controllers; hybrid is scored but never selected."""
    time_ranks = group_mean_ranks(aggregates, "completion_time_s", timed_only=True)
    jerk_ranks = group_mean_ranks(aggregates, "mean_action_jerk", timed_only=False)
    ranked: List[SelectionEntry] = []
    excluded: List[Dict[str, Any]] = []
    for controller, aggregate in aggregates["controllers"].items():
        meta = controller_meta.get(controller, {})
        key, evidence = selection_key(
            aggregate,
            time_rank=time_ranks.get(controller),
            jerk_rank=jerk_ranks.get(controller),
        )
        if meta.get("separate_reporting"):
            excluded.append(
                {"controller": controller, "reason": "not fully policy-based", **evidence}
            )
            continue
        ranked.append(SelectionEntry(controller, key, evidence))
    ranked.sort(key=lambda e: e.key, reverse=True)
    ppo = [e for e in ranked if str(controller_meta.get(e.controller, {}).get("kind")) == "ppo"]
    return {
        "priority": [
            "full-circuit reliability",
            "zero safety events",
            "three-gate and vertical-transition reliability",
            "completion time",
            "smoothness",
        ],
        "ranking": [
            {"rank": i + 1, "controller": e.controller, **e.evidence}
            for i, e in enumerate(ranked)
        ],
        "best_overall_policy_controller": ranked[0].controller if ranked else None,
        "best_ppo_checkpoint": ppo[0].controller if ppo else None,
        "ppo_ranking": [e.controller for e in ppo],
        "excluded_from_selection": excluded,
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def _ci(entry: Mapping[str, Any], digits: int = 2) -> str:
    if entry.get("mean") is None:
        return "n/a"
    if entry.get("ci_low") is None:
        return f"{entry['mean']:.{digits}f}"
    return f"{entry['mean']:.{digits}f} [{entry['ci_low']:.{digits}f}, {entry['ci_high']:.{digits}f}]"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def group_table(aggregates: Mapping[str, Any], group: str) -> str:
    headers = [
        "controller", "succ/tot", "success", "95% CI", "full-circuit",
        "gates", "mean t (s)", "median t (s)", "t/gate (s)", "pen. t (s)",
        "coll ev/fr/ep", "OOB ev/ep", "wrong-dir", "prev-gate ret",
        "jerk", "path (m)", "timeout",
    ]
    rows = []
    untimed = group in UNTIMED_GROUPS
    for controller, aggregate in aggregates["controllers"].items():
        entry = aggregate["by_group"].get(group)
        if entry is None:
            continue
        time_entry = entry["completion_time_s"]
        rows.append([
            controller,
            f"{entry['successes']}/{entry['episodes']}",
            _pct(entry["success_rate"]),
            f"[{_fmt(entry['success_rate_wilson95_low'],2)}, {_fmt(entry['success_rate_wilson95_high'],2)}]",
            _pct(entry["full_completion_rate"]),
            f"{entry['gates_completed_total']}/{entry['gates_expected_total']}",
            "n/a" if untimed else _fmt(time_entry["mean"], 2),
            "n/a" if untimed else _fmt(time_entry["median"], 2),
            "n/a" if untimed else _fmt(entry["time_per_gate_s"]["mean"], 2),
            "n/a" if untimed else _fmt(entry["penalized_time_s"]["mean"], 2),
            f"{entry['collision_events']}/{entry['collision_frames']}/{entry['collision_episodes']}",
            f"{entry['out_of_bounds_events']}/{entry['out_of_bounds_episodes']}",
            str(entry["wrong_direction_events"]),
            str(entry["previous_gate_returns"]),
            _fmt(entry["mean_action_jerk"]["mean"], 4),
            _fmt(entry["path_length_m"]["mean"], 1),
            _pct(entry["timeout_rate"]),
        ])
    return markdown_table(headers, rows)


def overall_table(aggregates: Mapping[str, Any], scope: str) -> str:
    headers = [
        "controller", "episodes", "success", "95% CI", "full-circuit",
        "safety ep", "coll ev", "OOB ev", "wrong-dir", "prev-gate ret",
        "timeout", "mean jerk", "mean pen. t (s)",
    ]
    rows = []
    for controller, aggregate in aggregates["controllers"].items():
        entry = aggregate[scope]
        rows.append([
            controller,
            str(entry["episodes"]),
            _pct(entry["success_rate"]),
            f"[{_fmt(entry['success_rate_wilson95_low'],2)}, {_fmt(entry['success_rate_wilson95_high'],2)}]",
            _pct(entry["full_completion_rate"]),
            str(entry["safety_event_episodes"]),
            str(entry["collision_events"]),
            str(entry["out_of_bounds_events"]),
            str(entry["wrong_direction_events"]),
            str(entry["previous_gate_returns"]),
            _pct(entry["timeout_rate"]),
            _fmt(entry["mean_action_jerk"]["mean"], 4),
            _fmt(entry["penalized_time_s"]["mean"], 2),
        ])
    return markdown_table(headers, rows)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_trajectories(
    rows: Sequence[Mapping[str, Any]], out_dir: Path, *, max_plots: int = 24
) -> List[str]:
    """Top-down and depth plots for representative successes and failures."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - plotting is optional
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = _representative_episodes(rows, max_plots=max_plots)
    written: List[str] = []
    for row in chosen:
        path = row.get("trajectory_path")
        if not path or not Path(path).exists():
            continue
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        points = data.get("trajectory") or []
        if not points:
            continue
        xs = [p["position"][0] for p in points]
        ys = [p["position"][1] for p in points]
        zs = [p["position"][2] for p in points]
        ts = [p["time_s"] for p in points]
        figure, (top, side) = plt.subplots(2, 1, figsize=(9, 8), height_ratios=[2, 1])
        top.plot(xs, ys, linewidth=1.4, color="#1f77b4", label="path")
        top.scatter([xs[0]], [ys[0]], marker="o", color="#2ca02c", zorder=5, label="start")
        top.scatter([xs[-1]], [ys[-1]], marker="X", color="#d62728", zorder=5, label="end")
        for gate in data.get("gates", []):
            centre = gate["center"]
            normal = gate["normal"]
            tangent = (-normal[1], normal[0])
            half = 1.0
            top.plot(
                [centre[0] - half * tangent[0], centre[0] + half * tangent[0]],
                [centre[1] - half * tangent[1], centre[1] + half * tangent[1]],
                color="#ff7f0e", linewidth=2.5,
            )
            top.annotate(gate["gate_id"], (centre[0], centre[1]), fontsize=7,
                         textcoords="offset points", xytext=(4, 4))
        status = data.get("referee_status")
        top.set_title(
            f"{row['controller']} | {row['group']}/{row['case_id']} | seed {row['seed']} | "
            f"{status} | gates {data.get('completed_gates')}/{data.get('expected_gates')}"
        )
        top.set_xlabel("x (m)")
        top.set_ylabel("y (m)")
        top.axis("equal")
        top.grid(alpha=0.3)
        top.legend(loc="best", fontsize=8)
        side.plot(ts, zs, linewidth=1.4, color="#9467bd")
        for gate in data.get("gates", []):
            side.axhline(gate["center"][2], color="#ff7f0e", linewidth=0.8, alpha=0.6)
        side.set_xlabel("time (s)")
        side.set_ylabel("z (m, negative = deeper)")
        side.grid(alpha=0.3)
        figure.tight_layout()
        name = (
            f"{row['group']}__{row['case_id']}__{row['controller']}__seed{row['seed']}"
            f"__{'success' if row.get('finished') else 'failure'}.png"
        )
        figure.savefig(out_dir / name, dpi=130)
        plt.close(figure)
        written.append(name)
    return written


def _representative_episodes(
    rows: Sequence[Mapping[str, Any]], *, max_plots: int
) -> List[Mapping[str, Any]]:
    """One success and one failure per group, preferring the official circuits."""
    with_traj = [r for r in rows if r.get("trajectory_path")]
    chosen: List[Mapping[str, Any]] = []
    groups = [g for g in GROUP_ORDER if any(r.get("group") == g for r in with_traj)]
    for group in reversed(groups):  # official circuits first
        in_group = [r for r in with_traj if r.get("group") == group]
        failures = sorted(
            (r for r in in_group if not r.get("finished")),
            key=lambda r: (int(r.get("completed_gates", 0)), str(r.get("controller"))),
        )
        successes = sorted(
            (r for r in in_group if r.get("finished")),
            key=lambda r: (float(r.get("official_time_s") or 1e9), str(r.get("controller"))),
        )
        chosen.extend(successes[:1])
        chosen.extend(failures[:2])
    return chosen[:max_plots]


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def build_report(out_dir: str | Path, *, plots: bool = True) -> Dict[str, Any]:
    out = Path(out_dir)
    rows = json.loads((out / "episodes.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "suite_manifest.json").read_text(encoding="utf-8"))
    controller_meta = {c["key"]: c for c in manifest["controllers"]}
    aggregates = build_aggregates(rows)

    policy_controllers = [
        c for c in aggregates["controllers"] if not controller_meta.get(c, {}).get("separate_reporting")
    ]
    reference = "rule_gate_center_then_commit"
    comparisons: List[Dict[str, Any]] = []
    for controller in aggregates["controllers"]:
        if controller == reference:
            continue
        comparisons.append(paired_comparison(rows, controller, reference))
    ppo_keys = [c for c in policy_controllers if controller_meta.get(c, {}).get("kind") == "ppo"]
    for i in range(len(ppo_keys)):
        for j in range(i + 1, len(ppo_keys)):
            comparisons.append(paired_comparison(rows, ppo_keys[i], ppo_keys[j]))
    if "hybrid" in aggregates["controllers"]:
        for controller in ppo_keys:
            comparisons.append(paired_comparison(rows, controller, "hybrid"))

    per_group_comparisons = [
        paired_comparison(rows, controller, reference, groups=[group])
        for controller in aggregates["controllers"]
        if controller != reference
        for group in aggregates["groups"]
    ]

    recommendation = recommend(aggregates, controller_meta)
    plot_files = (
        plot_trajectories(rows, out / "plots") if plots else []
    )
    videos = sorted(
        str(Path(r["video_path"]).name) for r in rows if r.get("video_path")
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "suite": {
            "git_sha": manifest.get("git_sha"),
            "created_utc": manifest.get("created_utc"),
            "adapter": manifest.get("adapter"),
            "current_profile": manifest.get("current_profile"),
            "dt": manifest.get("dt"),
            "total_episodes_planned": manifest.get("total_episodes"),
            "total_episodes_run": len(rows),
            "all_seeds": manifest.get("all_seeds"),
            "reused_seeds": manifest.get("reused_seeds"),
            "holdout_seeds": manifest.get("holdout_seeds"),
        },
        "controllers": controller_meta,
        "aggregates": aggregates,
        "paired_comparisons": comparisons,
        "paired_comparisons_by_group": per_group_comparisons,
        "recommendation": recommendation,
        "plots": plot_files,
        "videos": videos,
    }
    _atomic_write_json(out / "aggregate_report.json", report)
    (out / "final_benchmark_report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    _write_aggregate_csv(out / "aggregate_by_group.csv", aggregates)
    _write_paired_csv(
        out / "paired_comparisons.csv", comparisons + per_group_comparisons
    )
    return report


def _write_paired_csv(path: Path, comparisons: Sequence[Mapping[str, Any]]) -> None:
    import csv

    fields = [
        "controller_a", "controller_b", "groups", "paired_episodes",
        "a_successes", "b_successes", "a_full_completions", "b_full_completions",
        "only_a_finished", "only_b_finished", "reliability_sign_test_p",
        "time_delta_mean_s", "time_delta_ci_low", "time_delta_ci_high",
        "a_faster_episodes", "b_faster_episodes", "time_sign_test_p",
        "jerk_delta_mean", "path_length_delta_mean_m",
        "a_safety_events", "b_safety_events",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for c in comparisons:
            writer.writerow({
                "controller_a": c["controller_a"],
                "controller_b": c["controller_b"],
                "groups": c["groups"] if isinstance(c["groups"], str) else "|".join(c["groups"]),
                "paired_episodes": c["paired_episodes"],
                "a_successes": c["a_successes"],
                "b_successes": c["b_successes"],
                "a_full_completions": c["a_full_completions"],
                "b_full_completions": c["b_full_completions"],
                "only_a_finished": c["only_a_finished"],
                "only_b_finished": c["only_b_finished"],
                "reliability_sign_test_p": c["reliability_sign_test_p"],
                "time_delta_mean_s": c["time_delta_s"]["mean"],
                "time_delta_ci_low": c["time_delta_s"].get("ci_low"),
                "time_delta_ci_high": c["time_delta_s"].get("ci_high"),
                "a_faster_episodes": c["a_faster_episodes"],
                "b_faster_episodes": c["b_faster_episodes"],
                "time_sign_test_p": c["time_sign_test_p"],
                "jerk_delta_mean": c["jerk_delta"]["mean"],
                "path_length_delta_mean_m": c["path_length_delta_m"]["mean"],
                "a_safety_events": c["a_safety_events"],
                "b_safety_events": c["b_safety_events"],
            })
    tmp.replace(path)


def _write_aggregate_csv(path: Path, aggregates: Mapping[str, Any]) -> None:
    import csv

    fields = [
        "controller", "group", "episodes", "successes", "success_rate",
        "success_rate_wilson95_low", "success_rate_wilson95_high",
        "full_completions", "full_completion_rate",
        "gates_completed_total", "gates_expected_total", "mean_gates",
        "mean_completion_time_s", "median_completion_time_s",
        "completion_time_ci_low", "completion_time_ci_high",
        "mean_time_per_gate_s", "mean_penalized_time_s",
        "collision_events", "collision_frames", "collision_episodes",
        "out_of_bounds_events", "out_of_bounds_frames", "out_of_bounds_episodes",
        "wrong_direction_events", "wrong_direction_episodes",
        "previous_gate_returns", "missed_gate_attempts", "stuck_events",
        "safety_event_episodes", "mean_action_jerk", "mean_path_length_m",
        "timeouts", "timeout_rate", "mean_inference_ms",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for controller, aggregate in aggregates["controllers"].items():
            scopes = dict(aggregate["by_group"])
            scopes["ALL_non_official"] = aggregate["overall_excluding_official"]
            scopes["ALL_official"] = aggregate["official_circuits"]
            scopes["ALL_episodes"] = aggregate["all_episodes"]
            for group, entry in scopes.items():
                writer.writerow({
                    "controller": controller,
                    "group": group,
                    "episodes": entry["episodes"],
                    "successes": entry["successes"],
                    "success_rate": entry["success_rate"],
                    "success_rate_wilson95_low": entry["success_rate_wilson95_low"],
                    "success_rate_wilson95_high": entry["success_rate_wilson95_high"],
                    "full_completions": entry["full_completions"],
                    "full_completion_rate": entry["full_completion_rate"],
                    "gates_completed_total": entry["gates_completed_total"],
                    "gates_expected_total": entry["gates_expected_total"],
                    "mean_gates": entry["mean_gates"],
                    "mean_completion_time_s": entry["completion_time_s"]["mean"],
                    "median_completion_time_s": entry["completion_time_s"]["median"],
                    "completion_time_ci_low": entry["completion_time_s"].get("ci_low"),
                    "completion_time_ci_high": entry["completion_time_s"].get("ci_high"),
                    "mean_time_per_gate_s": entry["time_per_gate_s"]["mean"],
                    "mean_penalized_time_s": entry["penalized_time_s"]["mean"],
                    "collision_events": entry["collision_events"],
                    "collision_frames": entry["collision_frames"],
                    "collision_episodes": entry["collision_episodes"],
                    "out_of_bounds_events": entry["out_of_bounds_events"],
                    "out_of_bounds_frames": entry["out_of_bounds_frames"],
                    "out_of_bounds_episodes": entry["out_of_bounds_episodes"],
                    "wrong_direction_events": entry["wrong_direction_events"],
                    "wrong_direction_episodes": entry["wrong_direction_episodes"],
                    "previous_gate_returns": entry["previous_gate_returns"],
                    "missed_gate_attempts": entry["missed_gate_attempts"],
                    "stuck_events": entry["stuck_events"],
                    "safety_event_episodes": entry["safety_event_episodes"],
                    "mean_action_jerk": entry["mean_action_jerk"]["mean"],
                    "mean_path_length_m": entry["path_length_m"]["mean"],
                    "timeouts": entry["timeouts"],
                    "timeout_rate": entry["timeout_rate"],
                    "mean_inference_ms": entry["mean_inference_ms"]["mean"],
                })
    tmp.replace(path)


def _group_paired_lines(report: Mapping[str, Any], group: str) -> str:
    """Paired-vs-rule table for one group; every pair shares case and seed."""
    entries = [
        c
        for c in report.get("paired_comparisons_by_group", [])
        if c.get("groups") == [group] and c.get("paired_episodes")
    ]
    if not entries:
        return "_No paired episodes in this group._"
    return "Paired vs the rule baseline on identical seeds:\n\n" + markdown_table(
        ["A", "paired eps", "only A finished", "only rule finished",
         "mean dt A-rule (s)", "A faster", "rule faster", "time sign p"],
        [
            [
                c["controller_a"], str(c["paired_episodes"]),
                str(c["only_a_finished"]), str(c["only_b_finished"]),
                _ci(c["time_delta_s"]),
                str(c["a_faster_episodes"]), str(c["b_faster_episodes"]),
                _fmt(c["time_sign_test_p"], 4),
            ]
            for c in entries
        ],
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    aggregates = report["aggregates"]
    suite = report["suite"]
    recommendation = report["recommendation"]
    lines: List[str] = []
    lines.append("# Final common benchmark: reliability-first PPO vs rule, hybrid and BC-v3")
    lines.append("")
    lines.append(
        f"Commit `{suite.get('git_sha')}` | adapter `{suite.get('adapter')}` | "
        f"currents `{suite.get('current_profile')}` | dt {suite.get('dt')} s | "
        f"{suite.get('total_episodes_run')}/{suite.get('total_episodes_planned')} episodes."
    )
    lines.append("")
    lines.append(
        "Every controller ran the identical suite: same test groups, same generated and "
        "official geometries, same seeds, same adapter and the same current-free simulator "
        "configuration. Timing figures are reported per test group only, and every timing "
        "comparison is a paired per-seed difference within one group."
    )
    lines.append("")

    lines.append("## Controllers under test")
    lines.append("")
    lines.append(markdown_table(
        ["key", "kind", "policy-only", "model", "note"],
        [
            [
                key,
                str(meta.get("kind")),
                "yes" if meta.get("policy_only") else "no",
                f"`{meta.get('model')}`" if meta.get("model") else "-",
                str(meta.get("note", "")),
            ]
            for key, meta in report["controllers"].items()
        ],
    ))
    lines.append("")

    lines.append("## Aggregate comparison (all non-official test groups)")
    lines.append("")
    lines.append(overall_table(aggregates, "overall_excluding_official"))
    lines.append("")
    lines.append("## Aggregate comparison (three official current-free circuits)")
    lines.append("")
    lines.append(overall_table(aggregates, "official_circuits"))
    lines.append("")

    lines.append("## Per-group results")
    lines.append("")
    for group in aggregates["groups"]:
        lines.append(f"### {group}")
        lines.append("")
        if group in UNTIMED_GROUPS:
            lines.append(
                "_Timing columns are `n/a`: the official clock is first-gate-to-last-gate, "
                "so a single-gate case has a definitionally zero completion time._"
            )
            lines.append("")
        lines.append(group_table(aggregates, group))
        lines.append("")
        lines.append(_group_paired_lines(report, group))
        lines.append("")

    lines.append("## Paired controller comparisons (identical seeds and geometries)")
    lines.append("")
    lines.append(markdown_table(
        ["A", "B", "paired eps", "A succ", "B succ", "only A", "only B",
         "reliability sign p", "mean dt A-B (s)", "A faster", "B faster",
         "time sign p", "mean jerk delta"],
        [
            [
                c["controller_a"], c["controller_b"], str(c["paired_episodes"]),
                str(c["a_successes"]), str(c["b_successes"]),
                str(c["only_a_finished"]), str(c["only_b_finished"]),
                _fmt(c["reliability_sign_test_p"], 4),
                _ci(c["time_delta_s"]),
                str(c["a_faster_episodes"]), str(c["b_faster_episodes"]),
                _fmt(c["time_sign_test_p"], 4),
                _ci(c["jerk_delta"], 4),
            ]
            for c in report["paired_comparisons"]
        ],
    ))
    lines.append("")

    lines.append("## Failure modes")
    lines.append("")
    lines.append(
        "Where each unfinished episode stopped, and why. `missed_gate_dnf` means the "
        "referee ended the run because the participant bypassed its expected gate; the "
        "episode therefore stops at the point where the policy lost the sequence, not at "
        "the deadline."
    )
    lines.append("")
    for scope, title in (
        ("official_circuits", "Official circuits"),
        ("overall_excluding_official", "All non-official groups"),
    ):
        rows_out = []
        for controller, aggregate in aggregates["controllers"].items():
            breakdown = aggregate[scope]["failure_breakdown"]
            if not breakdown["failures"]:
                rows_out.append([controller, "0", "-", "-", "-"])
                continue
            rows_out.append([
                controller,
                str(breakdown["failures"]),
                ", ".join(f"{k}={v}" for k, v in breakdown["modes"].items()),
                ", ".join(f"{k}x{v}" for k, v in breakdown["stopped_after_gates"].items()),
                _fmt(breakdown["median_gates_at_failure"], 1),
            ])
        lines.append(f"### {title}")
        lines.append("")
        lines.append(markdown_table(
            ["controller", "failures", "modes", "stopped after gates", "median gates"],
            rows_out,
        ))
        lines.append("")

    lines.append("## Recommendation")
    lines.append("")
    lines.append(
        "Selection priority: " + "; ".join(recommendation["priority"]) + ". "
        "Computed from this benchmark only, not from the training metadata."
    )
    lines.append("")
    lines.append(markdown_table(
        ["rank", "controller", "official full-circuit", "safety ep", "3-gate",
         "vertical", "non-official success", "time rank", "jerk rank"],
        [
            [
                str(entry["rank"]), entry["controller"],
                _pct(entry["official_full_completion_rate"]),
                str(entry["safety_event_episodes"]),
                _pct(entry["three_gate_success_rate"]),
                _pct(entry["vertical_transition_success_rate"]),
                _pct(entry["non_official_success_rate"]),
                _fmt(entry["mean_within_group_time_rank"], 2),
                _fmt(entry["mean_within_group_jerk_rank"], 2),
            ]
            for entry in recommendation["ranking"]
        ],
    ))
    lines.append("")
    lines.append(
        "`time rank` and `jerk rank` are mean within-group ranks (1 = best in that group). "
        "Ranking inside each group and averaging the ranks keeps every timing judgement "
        "inside a single suite; no second from one geometry is ever weighed against a "
        f"second from another. Groups excluded from timing: {', '.join(UNTIMED_GROUPS)} "
        "(the official clock runs first-gate-to-last-gate, so a single-gate case is "
        "definitionally 0 s)."
    )
    lines.append("")
    lines.append(
        f"**Best PPO checkpoint on this benchmark: `{recommendation['best_ppo_checkpoint']}`** "
        f"(PPO order: {', '.join(recommendation['ppo_ranking'])})."
    )
    lines.append("")
    if recommendation["excluded_from_selection"]:
        lines.append(
            "Excluded from selection (reported separately): "
            + ", ".join(
                f"`{e['controller']}` ({e['reason']})"
                for e in recommendation["excluded_from_selection"]
            )
        )
        lines.append("")
    if report.get("plots"):
        lines.append("## Trajectory plots")
        lines.append("")
        for name in report["plots"]:
            lines.append(f"- `plots/{name}`")
        lines.append("")
    if report.get("videos"):
        lines.append("## Official circuit videos")
        lines.append("")
        for name in report["videos"]:
            lines.append(f"- `artifacts/{name}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    report = build_report(args.out, plots=not args.no_plots)
    print(
        f"[report] {report['suite']['total_episodes_run']} episodes -> "
        f"{Path(args.out) / 'final_benchmark_report.md'}"
    )
    print(f"[report] best PPO checkpoint: {report['recommendation']['best_ppo_checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
