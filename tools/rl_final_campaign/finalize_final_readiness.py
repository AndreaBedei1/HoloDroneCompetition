"""Materialise the frozen PPO readiness verdict and reporting evidence.

This consumes only the pre-registered procedural VALIDATION benchmark and the
pre-registered difficulty ladder.  It applies the production readiness gate
without an override, then writes a compact, reproducible report with Wilson
intervals and unconditional survival from episode start.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from marine_race_arena.learning.longrun_checkpoint import atomic_write_json
from marine_race_arena.learning.provenance import now_utc
from marine_race_arena.learning.rl_matched_benchmark import wilson_interval
from marine_race_arena.learning.rl_readiness_gate import (
    LADDER_KEY,
    READINESS_THRESHOLDS,
    evaluate_readiness,
)

REPO = Path(__file__).resolve().parents[2]
READINESS_ROOT = REPO / "results/rl_public/ppo_final_readiness_929792"
DEFAULT_EVALUATION = (
    READINESS_ROOT
    / "ppo_final_generic_929792/seed_5000000/evaluation.json"
)
DEFAULT_LADDER = (
    REPO
    / "results/rl_public/ppo_difficulty_ladder"
    / "ppo_final_generic_929792_ladder.json"
)
DEFAULT_OUTPUT = READINESS_ROOT / "readiness_verdict.json"
SURVIVAL_GATES = (1, 2, 3, 5, 8, 12, 17, 22)


def _read(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{path} is not a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rate_row(rows: Iterable[Mapping[str, Any]], key: str) -> Dict[str, Any]:
    selected = list(rows)
    hits = sum(bool(row.get(key)) for row in selected)
    low, high = wilson_interval(hits, len(selected))
    return {
        "successes": hits,
        "n": len(selected),
        "rate": hits / len(selected),
        "wilson95": [low, high],
    }


def _sequence_completion(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    selected = list(rows)
    for length in (3, 5, 8, 12, 17, 22):
        bucket = [row for row in selected if int(row.get("gate_count", 0)) == length]
        completed = sum(bool(row.get("full_sequence_completion")) for row in bucket)
        low, high = wilson_interval(completed, len(bucket))
        result[str(length)] = {
            "completed": completed,
            "n": len(bucket),
            "rate": completed / len(bucket),
            "wilson95": [low, high],
        }
    return result


def _unconditional_survival(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """P(cross gate k from start), among courses that contain gate k.

    Shorter courses cannot possibly expose a later gate, so they are excluded
    from that gate's denominator.  No episode is conditioned on surviving any
    earlier gate: every eligible episode starts in the denominator.
    """

    selected = list(rows)
    result: Dict[str, Any] = {}
    for gate in SURVIVAL_GATES:
        eligible = [
            row for row in selected if int(row.get("gate_count", 0)) >= gate
        ]
        crossed = sum(int(row.get("gates_completed", 0)) >= gate for row in eligible)
        low, high = wilson_interval(crossed, len(eligible))
        result[str(gate)] = {
            "crossed": crossed,
            "eligible_episodes_from_start": len(eligible),
            "probability": crossed / len(eligible),
            "wilson95": [low, high],
        }
    return result


def build_report(
    evaluation: Mapping[str, Any],
    ladder: Mapping[str, Any],
    *,
    evaluation_path: Path,
    ladder_path: Path,
) -> Dict[str, Any]:
    episodes = list(evaluation.get("episodes") or [])
    transitions = [
        row for row in episodes if row.get("episode_type") == "transition_focus"
    ]
    sequences = [
        row for row in episodes if row.get("episode_type") == "full_sequence"
    ]
    if len(transitions) != 500 or len(sequences) != 120:
        raise SystemExit(
            f"readiness evidence is incomplete: {len(transitions)} transitions, "
            f"{len(sequences)} sequences"
        )
    counts_by_length = {
        length: sum(int(row.get("gate_count", 0)) == length for row in sequences)
        for length in (3, 5, 8, 12, 17, 22)
    }
    if any(value != 20 for value in counts_by_length.values()):
        raise SystemExit(f"sequence evidence is not 20/length: {counts_by_length}")

    evidence = dict(evaluation)
    evidence[LADDER_KEY] = dict(ladder)
    verdict = evaluate_readiness(evidence, READINESS_THRESHOLDS, override=None)
    metrics = dict(evaluation.get("metrics") or {})
    position_one = dict(
        (metrics.get("transition_success_by_position") or {}).get("1") or {}
    )

    transition_metrics = {
        "universal_transition_success": _rate_row(
            transitions, "universal_transition_success"
        ),
        "first_gate_crossing": _rate_row(transitions, "first_gate_crossed"),
        "target_switch": _rate_row(transitions, "correct_target_switch"),
        "new_target_alignment": _rate_row(
            transitions, "new_target_acquired_and_aligned"
        ),
        "new_target_range_decrease": _rate_row(
            transitions, "new_target_range_decreasing"
        ),
        "collision_episodes": int(metrics.get("collision_episodes", 0)),
        "collision_events": int(metrics.get("collision_events", 0)),
        "missed_gate_dnf": int(metrics.get("missed_gate_dnf", 0)),
        "wrong_direction_events": int(metrics.get("wrong_direction_events", 0)),
        "out_of_bounds_episodes": int(metrics.get("out_of_bounds_episodes", 0)),
    }

    report: Dict[str, Any] = {
        "schema_version": "ppo_final_readiness_evidence_v1",
        "utc": now_utc(),
        "policy": "ppo_final_generic_929792",
        "dataset_split": evaluation.get("dataset_split"),
        "difficulty": evaluation.get("difficulty"),
        "sources": {
            "evaluation": str(evaluation_path),
            "evaluation_sha256": _sha256(evaluation_path),
            "ladder": str(ladder_path),
            "ladder_sha256": _sha256(ladder_path),
        },
        "sample_contract": {
            "transition_cases": len(transitions),
            "full_sequence_cases": len(sequences),
            "full_sequence_cases_by_length": {
                str(key): value for key, value in counts_by_length.items()
            },
        },
        "transition_metrics": transition_metrics,
        "sequence_completion": _sequence_completion(sequences),
        "unconditional_survival_from_episode_start": _unconditional_survival(
            sequences
        ),
        "metric_clarification": {
            "first_gate_crossing": (
                "Crossing gate 1. It does not assert that the gate-1 to gate-2 "
                "switch, reacquisition, alignment and range-decrease sequence "
                "also succeeded."
            ),
            "first_transition": (
                "Full gate-1 to gate-2 transition at sequence position 1: gate 1 "
                "crossed, target switched, gate 2 aligned, and range decreased."
            ),
            "first_transition_successes": int(position_one.get("successes", 0)),
            "first_transition_n": int(position_one.get("n", 0)),
            "first_transition_rate": position_one.get("success_rate"),
        },
        "difficulty_ladder": dict(ladder),
        "readiness": verdict.as_dict(),
        "manual_override_applied": False,
        "training_closed_regardless_of_verdict": True,
    }
    return report


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Final PPO procedural readiness",
        "",
        f"Overall: **{'PASS' if report['readiness']['ready'] else 'FAIL'}**",
        "",
        "## Criteria",
        "",
        "| Criterion | Observed | Requirement | Result |",
        "|---|---:|---:|---|",
    ]
    for criterion in report["readiness"]["criteria"]:
        direction = ">=" if criterion["direction"] == "min" else "<="
        observed = criterion["observed"]
        lines.append(
            f"| {criterion['name']} | {observed} | {direction} "
            f"{criterion['bound']} | {'PASS' if criterion['satisfied'] else 'FAIL'} |"
        )
    lines.extend((
        "",
        "No readiness override was requested or applied. Training remains closed.",
        "",
    ))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", default=str(DEFAULT_EVALUATION))
    parser.add_argument("--ladder", default=str(DEFAULT_LADDER))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    evaluation_path = Path(args.evaluation).resolve()
    ladder_path = Path(args.ladder).resolve()
    output = Path(args.output).resolve()
    report = build_report(
        _read(evaluation_path),
        _read(ladder_path),
        evaluation_path=evaluation_path,
        ladder_path=ladder_path,
    )
    atomic_write_json(output, report)
    output.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    print(
        f"readiness={'PASS' if report['readiness']['ready'] else 'FAIL'} -> {output}",
        flush=True,
    )
    print("failures:", report["readiness"]["failures"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
