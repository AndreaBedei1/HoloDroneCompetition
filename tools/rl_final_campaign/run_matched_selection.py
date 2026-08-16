"""Run the matched VALIDATION benchmark that selects the final PPO parent.

Ordering is the point of this script.  The selection rule, the thresholds, the
case plan and the git SHA are written to ``pre_registration.json`` *before* the
first engine starts, so the hierarchy that picks the winner cannot be adjusted
once the numbers are visible.  The benchmark itself is paired: every candidate
faces byte-identical geometry, which is what makes a 6-point difference between
two checkpoints interpretable at all.

Difficulty is G1 deliberately.  Every candidate trained exclusively at G1 -- the
curriculum never promoted, because promotion demands 99% success with zero safety
events -- so G1 is the regime they share, it matches the 17 historical
evaluations, and at 0.63-0.87 success it is far from saturated and therefore
discriminates.  It is *not* evidence of readiness for a real circuit; the
difficulty-ladder stress run is what speaks to that.

Usage (marine_race_rl env, from the repository root)::

    python -m tools.rl_final_campaign.run_matched_selection --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from concurrent.futures.process import BrokenProcessPool

from marine_race_arena.learning.longrun_checkpoint import sha256_file
from marine_race_arena.learning.rl_matched_benchmark import (
    SELECTION_RULE,
    assert_plan_uses_validation,
    plan_matched_benchmark,
    run_matched_benchmark,
)
from marine_race_arena.learning.rl_readiness_gate import READINESS_THRESHOLDS

REPO = Path(__file__).resolve().parents[2]
LONGRUN = REPO / "results/rl/universal_transition/longrun"
V2 = LONGRUN / "universal_transition_ppo_generic_sequence_v2_seed23001"
V5 = LONGRUN / "universal_transition_ppo_generic_sequence_v5_rewarm_fixed_seed23001"
OUTPUT_ROOT = REPO / "results/rl_public/ppo_final_matched_benchmark"

#: The shortlist, fixed in advance.  ``ppo_528384`` is the shared ancestor and the
#: only checkpoint with large-sample evidence; ``ppo_935936`` is the final state
#: at the graceful stop; the rest bracket the observed decline after ~630k.
CANDIDATES: List[Dict[str, Any]] = [
    {"name": "ppo_528384", "checkpoint": V2 / "checkpoints/ppo_528384_steps.zip",
     "note": "shared parent of v3/v4/v5; dedicated n=412 evaluation scored 0.8075"},
    {"name": "ppo_628736", "checkpoint": V5 / "checkpoints/ppo_628736_steps.zip",
     "note": "best in-training evaluation observed (0.83 on n=112 train-band)"},
    {"name": "ppo_729088", "checkpoint": V5 / "checkpoints/ppo_729088_steps.zip",
     "note": "alias best_universal_transition"},
    {"name": "ppo_901120", "checkpoint": V5 / "checkpoints/ppo_901120_steps.zip",
     "note": "alias latest_competent"},
    {"name": "ppo_935936", "checkpoint": V5 / "checkpoints/ppo_935936_steps.zip",
     "note": "final checkpoint written by the graceful stop"},
]


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_candidates() -> List[Dict[str, Any]]:
    entries = []
    for entry in CANDIDATES:
        checkpoint = Path(entry["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"shortlist checkpoint missing: {checkpoint}")
        entries.append({
            "name": entry["name"],
            "checkpoint": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "algorithm": "ppo",
            "note": entry["note"],
        })
    return entries


def run_with_resume(
    plan: Dict[str, Any], *, workers: int, attempts: int
) -> Dict[str, Any]:
    """Drive the benchmark across engine failures instead of dying on the first.

    A single HoloOcean handshake failure kills one pool worker, which raises
    ``BrokenProcessPool`` and aborts the whole sweep -- observed during
    calibration, and fatal for an unattended multi-hour run.  Completed cases are
    persisted per case, so re-entering the benchmark resumes rather than repeats.
    Progress between attempts is required: if an attempt adds no new episodes the
    failure is not transient and retrying would spin forever.
    """

    def completed() -> int:
        return len(list(OUTPUT_ROOT.rglob("episode.json")))

    last = -1
    for attempt in range(1, int(attempts) + 1):
        before = completed()
        if before == last:
            raise SystemExit(
                f"attempt {attempt - 1} completed no new cases ({before} total); "
                f"the failure is not transient -- stopping rather than spinning"
            )
        last = before
        try:
            return run_matched_benchmark(
                plan, output_root=OUTPUT_ROOT, parallel_workers=workers,
            )
        except (BrokenProcessPool, RuntimeError, OSError) as exc:
            after = completed()
            print(
                f"attempt {attempt}/{attempts} failed after {after - before} new "
                f"cases ({after} total): {type(exc).__name__}: {exc}",
                flush=True,
            )
            if attempt == int(attempts):
                raise
    raise SystemExit("exhausted benchmark attempts")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--transition-cases", type=int, default=500)
    parser.add_argument("--sequences-per-length", type=int, default=10)
    parser.add_argument("--difficulty", default="G1")
    # Each case builds and tears down its own engine, so with a dozen workers the
    # global start slot -- one concurrent start, 3 s stagger by default -- becomes
    # the throughput ceiling: measured 68 s/case at the defaults against 7.7 s/case
    # with four slots and a 1 s stagger.  The evaluator retries a failed handshake
    # with a fresh UUID, and the benchmark resumes per case, so a start that loses
    # its handshake under the looser setting costs one case, not the run.
    parser.add_argument("--engine-starts", type=int, default=4,
                        help="HOLODRONE_MAX_ENGINE_STARTS for this run")
    parser.add_argument("--engine-stagger", type=float, default=1.0)
    parser.add_argument("--attempts", type=int, default=40,
                        help="resume attempts after a broken engine pool")
    parser.add_argument("--plan-only", action="store_true",
                        help="write the pre-registration and exit without evaluating")
    args = parser.parse_args(argv)

    if args.engine_starts is not None:
        os.environ["HOLODRONE_MAX_ENGINE_STARTS"] = str(args.engine_starts)
    if args.engine_stagger is not None:
        os.environ["HOLODRONE_ENGINE_START_STAGGER_SECONDS"] = str(args.engine_stagger)

    candidates = build_candidates()
    plan = plan_matched_benchmark(
        candidates,
        transition_cases=args.transition_cases,
        sequences_per_length=args.sequences_per_length,
        difficulty=args.difficulty,
    )
    verified = assert_plan_uses_validation(plan)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pre_registration = {
        "schema_version": "ppo_final_campaign_pre_registration_v1",
        "git_sha": git_sha(),
        "purpose": "select FINAL_PPO_PARENT from a fixed shortlist",
        "plan": plan,
        "verified_case_bands": verified,
        "selection_rule": SELECTION_RULE.as_dict(),
        "selection_rule_fingerprint": SELECTION_RULE.fingerprint(),
        "readiness_thresholds": READINESS_THRESHOLDS.as_dict(),
        "readiness_thresholds_sha256": READINESS_THRESHOLDS.sha256(),
        "difficulty_note": (
            "G1 is the only difficulty any candidate ever trained at; the "
            "curriculum never promoted because promotion requires >=0.99 success "
            "with zero safety events. G1 results measure the shared regime and "
            "must not be read as readiness for a real circuit."
        ),
        "committed_before_results": True,
    }
    path = OUTPUT_ROOT / "pre_registration.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("plan", {}).get("plan_fingerprint") != plan["plan_fingerprint"]:
            raise SystemExit(
                f"a different pre-registration already exists at {path}; refusing "
                f"to silently replace the recorded rule"
            )
    else:
        path.write_text(json.dumps(pre_registration, indent=2), encoding="utf-8")
    print(f"pre-registration -> {path}")
    print(f"  git_sha            {pre_registration['git_sha']}")
    print(f"  rule fingerprint   {pre_registration['selection_rule_fingerprint']}")
    print(f"  plan fingerprint   {plan['plan_fingerprint']}")
    print(f"  candidates         {[c['name'] for c in candidates]}")
    for seed, facts in verified["seeds"].items():
        print(f"  seed {seed}: {facts['cases']} cases, "
              f"{facts['distinct_episode_seeds']} distinct validation seeds, "
              f"range {facts['episode_seed_range']}")
    if args.plan_only:
        return 0

    report = run_with_resume(
        plan, workers=args.workers, attempts=args.attempts,
    )
    selection = report["selection"]
    print("\n=== RANKING ===")
    for row in report["ranking"]:
        print(f"  {row}")
    print(f"\nFINAL_PPO_PARENT = {selection.get('selected')}")
    print(f"  checkpoint: {selection.get('checkpoint')}")
    print(f"  sha256    : {selection.get('sha256')}")
    print(f"  reason    : {selection.get('reason')}")
    if selection.get("paired_comparison"):
        print(f"  paired    : {json.dumps(selection['paired_comparison'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
