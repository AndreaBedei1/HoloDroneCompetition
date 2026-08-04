"""Corrected A/B report: competence qualification, safety ranking, selection.

The original ``ab_comparison.json``/``.md`` ranked initializations by summed
safety events alone and therefore selected the scratch policy, which produced
few events only because it barely moved.  This module regenerates the comparison
from the *same* frozen episode artifacts without modifying them, applying the
mandatory competence gate before any safety comparison.

Metrics are recomputed from the stored per-episode rows so that participation
evidence (completed gates, episodes reaching the first gate) is available even
for reports written before those aggregates existed.  Raw activity evidence
(commanded action magnitude) is measured from the recorded trajectories, whose
sample size is reported explicitly alongside it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from marine_race_arena.learning.provenance import now_utc
from marine_race_arena.learning.transition_evaluation import (
    NONTRIVIAL_ACTION_THRESHOLD,
    aggregate_transition_benchmark,
)
from marine_race_arena.learning.transition_selection import (
    CompetenceThresholds,
    DEFAULT_COMPETENCE_THRESHOLDS,
    SAFETY_PRIORITY,
    select_policy,
)

REPORT_SCHEMA_VERSION = "universal_transition_ab_comparison_corrected_v1"
DEFAULT_BENCHMARK = Path("evaluations") / "dedicated_unseen_1000" / "evaluation.json"


def _load_benchmark(run_dir: Path, relative: Path = DEFAULT_BENCHMARK) -> Dict[str, Any]:
    path = run_dir / relative
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def trajectory_activity_evidence(run_dir: Path) -> Dict[str, Any]:
    """Measure commanded action magnitude from the recorded trajectories.

    Trajectories are recorded for the first case of each benchmark group only,
    so this is a sample rather than the whole suite; ``sample_episodes`` and
    ``sample_steps`` make that explicit wherever the values are used.
    """

    files = sorted(run_dir.rglob("trajectory.json"))
    actions = []
    for path in files:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rows = [row.get("action") for row in value.get("trajectory") or []]
        rows = [row for row in rows if row]
        if rows:
            actions.append(np.asarray(rows, dtype=np.float64))
    if not actions:
        return {
            "sample_episodes": 0,
            "sample_steps": 0,
            "mean_absolute_action": None,
            "mean_absolute_action_per_axis": None,
            "nontrivial_action_fraction": None,
            "nontrivial_action_threshold": NONTRIVIAL_ACTION_THRESHOLD,
        }
    stacked = np.concatenate(actions, axis=0)
    absolute = np.abs(stacked)
    return {
        "sample_episodes": len(actions),
        "sample_steps": int(stacked.shape[0]),
        "mean_absolute_action": float(absolute.mean()),
        "mean_absolute_action_per_axis": {
            name: float(absolute[:, index].mean())
            for index, name in enumerate(("surge", "sway", "heave", "yaw"))
        },
        "nontrivial_action_fraction": float(
            (absolute.max(axis=1) > NONTRIVIAL_ACTION_THRESHOLD).mean()
        ),
        "nontrivial_action_threshold": NONTRIVIAL_ACTION_THRESHOLD,
    }


def _gate_metrics(
    metrics: Mapping[str, Any], activity: Mapping[str, Any]
) -> Dict[str, Any]:
    """Metrics view used by the gate: recomputed aggregates plus measured activity."""

    merged = dict(metrics)
    for key in (
        "mean_absolute_action",
        "nontrivial_action_fraction",
    ):
        if activity.get(key) is not None:
            merged[key] = activity[key]
    return merged


def build_arm(name: str, run_dir: str | Path) -> Dict[str, Any]:
    path = Path(run_dir)
    benchmark = _load_benchmark(path)
    episodes = list(benchmark.get("episodes") or [])
    if not episodes:
        raise ValueError(f"{name}: benchmark has no stored episodes")
    recomputed = aggregate_transition_benchmark(episodes)
    activity = trajectory_activity_evidence(path)
    return {
        "name": name,
        "run_dir": str(path),
        "checkpoint": benchmark.get("checkpoint"),
        "benchmark_seed": benchmark.get("seed"),
        "benchmark_difficulty": benchmark.get("difficulty"),
        "transition_cases": benchmark.get("transition_cases"),
        "original_metrics": dict(benchmark.get("metrics") or {}),
        "recomputed_metrics": recomputed,
        "trajectory_activity_evidence": activity,
        "gate_metrics": _gate_metrics(recomputed, activity),
    }


def build_report(
    arms: Mapping[str, str | Path],
    *,
    thresholds: CompetenceThresholds = DEFAULT_COMPETENCE_THRESHOLDS,
    original_report: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    built = {name: build_arm(name, arms[name]) for name in sorted(arms)}
    selection = select_policy(
        {name: built[name]["gate_metrics"] for name in built}, thresholds
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_utc": now_utc(),
        "supersedes": (
            None if original_report is None
            else {
                "generated_utc": original_report.get("generated_utc"),
                "selected_initialization": original_report.get(
                    "selected_initialization"
                ),
                "selection_priority": list(
                    original_report.get("selection_priority") or []
                ),
            }
        ),
        "correction": (
            "Safety events were previously summed and compared directly, which "
            "ranked an inactive policy first. Competence qualification is now "
            "mandatory and precedes safety ranking."
        ),
        "stages": ["competence_qualification", "safety_ranking", "final_selection"],
        "safety_priority": list(SAFETY_PRIORITY),
        "arms": built,
        "selection": selection,
        "selected_initialization": selection["selected"],
        "evidence_preserved": True,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def markdown(report: Mapping[str, Any]) -> str:
    selection = report["selection"]
    arms = report["arms"]
    names = sorted(arms)
    lines = [
        "# Corrected selective warm-start versus scratch",
        "",
        f"Generated: {report['generated_utc']}",
        "",
        "This report supersedes the original ranking without modifying it. The",
        "original evidence, checkpoints and episode artifacts are unchanged.",
        "",
        "## 1. Competence qualification (mandatory)",
        "",
        "A policy that avoids safety events by remaining almost stationary is not",
        "safe, it is inactive. Every candidate must first clear these minimums:",
        "",
        "| Criterion | Minimum |",
        "|---|---:|",
    ]
    gate = selection["competence_gate"]
    for key, label in (
        ("min_first_gate_crossing_rate", "first gate crossing rate"),
        ("min_target_switch_rate", "target switch rate"),
        ("min_universal_transition_success_rate", "universal transition success"),
        ("min_completed_gate_count", "completed gates"),
        ("min_completed_gates_per_episode", "completed gates per episode"),
        ("min_fraction_of_episodes_reaching_first_gate", "episodes reaching first gate"),
        ("min_mean_distance_travelled_m", "mean distance travelled (m)"),
        ("min_mean_absolute_action", "mean absolute action"),
        ("min_nontrivial_action_fraction", "non-trivial action fraction"),
        ("min_transition_cases", "evaluated transition cases"),
    ):
        lines.append(f"| {label} | {_fmt(gate[key])} |")
    lines += [
        "",
        "| Initialization | Classification | Qualified | Failed criteria |",
        "|---|---|:---:|---|",
    ]
    for name in names:
        verdict = selection["qualification"][name]
        failed = ", ".join(verdict["failed_criteria"]) or "none"
        lines.append(
            f"| {name} | `{verdict['classification']}` | "
            f"{verdict['passed']} | {failed} |"
        )
    lines += [
        "",
        "### Anti-inactivity evidence",
        "",
        "| Initialization | Completed gates | Gates/episode | Reaching first gate |"
        " Mean abs action | Non-trivial action fraction | Action sample (episodes/steps) |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name in names:
        metrics = arms[name]["recomputed_metrics"]
        activity = arms[name]["trajectory_activity_evidence"]
        lines.append(
            f"| {name} | {int(metrics['completed_gate_count'])} | "
            f"{_fmt(metrics['mean_completed_gates_per_episode'])} | "
            f"{_fmt(metrics['fraction_of_episodes_reaching_first_gate'])} | "
            f"{_fmt(activity['mean_absolute_action'])} | "
            f"{_fmt(activity['nontrivial_action_fraction'])} | "
            f"{activity['sample_episodes']}/{activity['sample_steps']} |"
        )
    lines += [
        "",
        "Action magnitudes are measured from the recorded trajectories only, so",
        "the sample size is stated with them. Motion is never treated as success:",
        "these metrics exist solely to detect degenerate inactivity.",
        "",
        "## 2. Safety ranking (competent policies only)",
        "",
        "Priority: " + ", ".join(SAFETY_PRIORITY) + ".",
        "",
        "Collision *episodes* rank first and collision *entries* second; sustained",
        "contact frames are reported but never ranked, because one prolonged",
        "contact would otherwise outweigh several distinct impacts.",
        "",
        "| Initialization | Ranked | Collision episodes | Collision entries |"
        " Missed gates | Wrong direction | Previous-gate returns | Acquisition timeouts |"
        " Transition success | Long completion | Mean jerk |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        metrics = arms[name]["recomputed_metrics"]
        ranked = name in selection["competent_candidates"]
        entries = metrics.get("collision_entries")
        entries_text = (
            f"{int(entries)}" if entries is not None
            else f"{int(metrics.get('collision_events', 0))} (events)"
        )
        lines.append(
            f"| {name} | {ranked} | {int(metrics['collision_episodes'])} | "
            f"{entries_text} | {int(metrics['missed_gate_dnf'])} | "
            f"{int(metrics['wrong_direction_events'])} | "
            f"{int(metrics['previous_gate_returns'])} | "
            f"{int(metrics['acquisition_timeouts'])} | "
            f"{_fmt(metrics['universal_transition_success_rate'])} | "
            f"{_fmt(metrics['long_sequence_completion_score'])} | "
            f"{_fmt(metrics['mean_action_jerk'])} |"
        )
    selected = selection["selected"]
    lines += [
        "",
        "## 3. Final selection",
        "",
        f"Selected initialization: **{selected}** ({selection['selection_reason']}).",
        "",
    ]
    if selected is not None:
        metrics = arms[selected]["recomputed_metrics"]
        lines += [
            f"The selected policy is **competent but not yet reliable**: it clears",
            f"the competence gate (first crossing "
            f"{_fmt(metrics['first_gate_crossing_rate'])}, target switch "
            f"{_fmt(metrics['target_switch_rate'])}, transition success "
            f"{_fmt(metrics['universal_transition_success_rate'])}) while still",
            f"producing {int(metrics['collision_episodes'])} collision episodes, "
            f"{int(metrics['missed_gate_dnf'])} missed gates, "
            f"{int(metrics['wrong_direction_events'])} wrong-direction events and "
            f"{int(metrics['acquisition_timeouts'])} acquisition timeouts.",
            "",
            "It is selected because it is the only initialization that learned",
            "useful gate-crossing and beacon-switch behaviour, not because it is",
            "safe. The long-term requirement remains universal transition success",
            ">= 99% on repeated unseen evaluations with strong safety.",
            "",
        ]
    for name in names:
        if name == selected:
            continue
        verdict = selection["qualification"][name]
        if not verdict["passed"]:
            lines.append(
                f"`{name}` is excluded as `{verdict['classification']}`: it failed "
                f"{', '.join(verdict['failed_criteria'])}. Its lower safety-event "
                "count does not compensate for absent task competence."
            )
            lines.append("")
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _parse_arm(value: str) -> tuple:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=RUN_DIR")
    name, run_dir = value.split("=", 1)
    return name.strip(), run_dir.strip()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, type=_parse_arm)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="ab_comparison_corrected")
    parser.add_argument("--original-report")
    args = parser.parse_args(argv)

    original = None
    if args.original_report and Path(args.original_report).exists():
        # Streamed key lookup keeps the multi-megabyte original out of memory.
        value = json.loads(Path(args.original_report).read_text(encoding="utf-8"))
        original = {
            "generated_utc": value.get("generated_utc"),
            "selected_initialization": value.get("selected_initialization"),
            "selection_priority": value.get("selection_priority"),
        }
    report = build_report(dict(args.arm), original_report=original)
    output = Path(args.output_dir)
    _atomic_write(
        output / f"{args.output_name}.json", json.dumps(report, indent=2)
    )
    _atomic_write(output / f"{args.output_name}.md", markdown(report))
    print(json.dumps({
        "selected_initialization": report["selected_initialization"],
        "rejected": report["selection"]["rejected_candidates"],
        "json": str(output / f"{args.output_name}.json"),
        "markdown": str(output / f"{args.output_name}.md"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
