"""Measure how far a checkpoint's competence reaches up the geometry ladder.

Every PPO run to date trained exclusively at G1 (max turn 8 deg, max vertical step
0.20 m, lateral offset 0.35 m), because promotion requires >=0.99 success with zero
safety events and no evaluation ever cleared that.  A headline success rate at G1
therefore says nothing about a real circuit, and a readiness gate built on G1
evidence alone would be self-congratulatory.

This runs the same matched VALIDATION protocol at each rung and reports the curve.
Its two jobs: give the readiness gate evidence that the easy regime was not the
only regime tested, and locate empirically where competence breaks down so the
hard-family mixture targets real weaknesses instead of guesses.

Usage (marine_race_rl env, from the repository root)::

    python -m tools.rl_final_campaign.run_difficulty_ladder \
        --checkpoint <path> --name ppo_628736 --workers 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from marine_race_arena.learning.longrun_checkpoint import sha256_file
from marine_race_arena.learning.rl_matched_benchmark import (
    BENCHMARK_ROLE,
    wilson_interval,
)
from marine_race_arena.learning.transition_curriculum import DIFFICULTY_LEVELS
from marine_race_arena.learning.transition_evaluation import (
    evaluate_checkpoint_universal_transition_benchmark,
)

REPO = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO / "results/rl_public/ppo_difficulty_ladder"

#: The ladder run is a stress probe, not the selection benchmark, so it trades
#: per-rung precision for covering every rung.  Wilson intervals are reported so
#: the coarser sample size is visible rather than implied.
DEFAULT_TRANSITION_CASES = 60
DEFAULT_SEQUENCES_PER_LENGTH = 2
LADDER_SEED = 5_000_000


def run_ladder(
    checkpoint: Path,
    *,
    name: str,
    workers: int,
    transition_cases: int,
    sequences_per_length: int,
    rungs: List[str],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for difficulty in rungs:
        output_dir = OUTPUT_ROOT / name / difficulty
        report = evaluate_checkpoint_universal_transition_benchmark(
            checkpoint,
            output_dir=output_dir,
            seed=LADDER_SEED,
            difficulty=difficulty,
            transition_cases=transition_cases,
            full_cases_per_length=sequences_per_length,
            adapter="holoocean",
            max_steps=3600,
            parallel_workers=workers,
            algorithm="ppo",
            dataset_split=BENCHMARK_ROLE,
        )
        metrics = report["metrics"]
        episodes = report["episodes"]
        transition_rows = [
            row for row in episodes if row.get("episode_type") == "transition_focus"
        ]
        successes = sum(
            1 for row in transition_rows if row.get("universal_transition_success")
        )
        low, high = wilson_interval(successes, len(transition_rows))
        row = {
            "difficulty": difficulty,
            "n": len(transition_rows),
            "success": metrics.get("universal_transition_success_rate"),
            "success_ci": [low, high],
            "first_gate": metrics.get("first_gate_crossing_rate"),
            "switch": metrics.get("target_switch_rate"),
            "alignment": metrics.get("new_target_alignment_rate"),
            "collisions": metrics.get("collision_episodes"),
            "out_of_bounds": metrics.get("out_of_bounds_episodes"),
            "missed_gate_dnf": metrics.get("missed_gate_dnf"),
            "long_sequence": metrics.get("long_sequence_completion_score"),
        }
        rows.append(row)
        print(
            f"  {difficulty}  success={row['success']} "
            f"CI=[{low:.3f},{high:.3f}]  first_gate={row['first_gate']}  "
            f"switch={row['switch']}  coll={row['collisions']}  "
            f"seq={row['long_sequence']}",
            flush=True,
        )
    return {
        "schema_version": "ppo_difficulty_ladder_v1",
        "checkpoint": str(checkpoint),
        "sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
        "dataset_split": BENCHMARK_ROLE,
        "seed": LADDER_SEED,
        "transition_cases": transition_cases,
        "full_cases_per_length": sequences_per_length,
        "rungs": rows,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--transition-cases", type=int,
                        default=DEFAULT_TRANSITION_CASES)
    parser.add_argument("--sequences-per-length", type=int,
                        default=DEFAULT_SEQUENCES_PER_LENGTH)
    parser.add_argument("--rungs", default=",".join(DIFFICULTY_LEVELS))
    args = parser.parse_args(argv)

    rungs = [value.strip() for value in args.rungs.split(",") if value.strip()]
    unknown = [value for value in rungs if value not in DIFFICULTY_LEVELS]
    if unknown:
        raise SystemExit(f"unknown difficulty rungs {unknown}")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    print(f"ladder for {args.name}: {rungs}", flush=True)
    report = run_ladder(
        checkpoint,
        name=args.name,
        workers=args.workers,
        transition_cases=args.transition_cases,
        sequences_per_length=args.sequences_per_length,
        rungs=rungs,
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / f"{args.name}_ladder.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"written -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
