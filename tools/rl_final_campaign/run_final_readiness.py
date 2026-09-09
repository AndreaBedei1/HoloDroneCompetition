"""Run the one-policy preregistered FINAL PPO readiness benchmark.

This is deliberately separate from ``run_matched_selection``: selection used
ten sequences per length to compare several parents, while the absolute
readiness gate requires 500 VALIDATION transitions and exactly twenty
independent VALIDATION sequences at each configured length.  The immutable
checkpoint and complete case plan are written before the first engine starts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, Dict, List

from marine_race_arena.learning.longrun_checkpoint import (
    atomic_write_json,
    sha256_file,
)
from marine_race_arena.learning.rl_matched_benchmark import (
    assert_plan_uses_validation,
    plan_matched_benchmark,
    run_matched_benchmark,
)
from marine_race_arena.learning.rl_readiness_gate import READINESS_THRESHOLDS


REPO = Path(__file__).resolve().parents[2]
ARTIFACT_FREEZE = (
    REPO / "results/rl_public/ppo_final_generic_929792/artifact_freeze.json"
)
OUTPUT_ROOT = REPO / "results/rl_public/ppo_final_readiness_929792"
PRE_REGISTRATION = OUTPUT_ROOT / "pre_registration.json"
FINAL_NAME = "ppo_final_generic_929792"
TRANSITION_CASES = 500
SEQUENCES_PER_LENGTH = 20
DIFFICULTY = "G1"


def _frozen_candidate() -> Dict[str, Any]:
    freeze = json.loads(ARTIFACT_FREEZE.read_text(encoding="utf-8"))
    artifact = freeze["artifact"]
    checkpoint = Path(artifact["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"frozen checkpoint missing: {checkpoint}")
    actual = sha256_file(checkpoint)
    expected = str(artifact["checkpoint_sha256"])
    if actual != expected:
        raise ValueError(
            f"frozen checkpoint changed: expected {expected}, observed {actual}"
        )
    if not bool(freeze.get("closure", {}).get("training_permanently_closed")):
        raise ValueError("artifact freeze does not permanently close training")
    if bool(freeze.get("final_circuit_access_authorized")):
        raise ValueError("readiness must run while final circuits remain sealed")
    return {
        "name": FINAL_NAME,
        "checkpoint": str(checkpoint),
        "sha256": actual,
        "algorithm": "ppo",
    }


def _planned_evidence() -> Dict[str, Any]:
    candidate = _frozen_candidate()
    plan = plan_matched_benchmark(
        [candidate],
        transition_cases=TRANSITION_CASES,
        sequences_per_length=SEQUENCES_PER_LENGTH,
        difficulty=DIFFICULTY,
    )
    verified = assert_plan_uses_validation(plan)
    value = {
        "schema_version": "ppo_final_readiness_pre_registration_v1",
        "purpose": "judge the immutable final PPO against the unchanged readiness gate",
        "artifact_freeze": str(ARTIFACT_FREEZE),
        "checkpoint_sha256": candidate["sha256"],
        "plan": plan,
        "verified_case_bands": verified,
        "readiness_thresholds": READINESS_THRESHOLDS.as_dict(),
        "readiness_thresholds_sha256": READINESS_THRESHOLDS.sha256(),
        "final_circuits_sealed": True,
        "training_permanently_closed": True,
        "result_dependent_changes_permitted": False,
    }
    if PRE_REGISTRATION.exists():
        existing = json.loads(PRE_REGISTRATION.read_text(encoding="utf-8"))
        checks = (
            ("checkpoint_sha256", existing.get("checkpoint_sha256"), value["checkpoint_sha256"]),
            (
                "plan_fingerprint",
                existing.get("plan", {}).get("plan_fingerprint"),
                plan["plan_fingerprint"],
            ),
            (
                "thresholds_sha256",
                existing.get("readiness_thresholds_sha256"),
                value["readiness_thresholds_sha256"],
            ),
        )
        mismatches = [
            f"{name}: {observed!r} != {expected!r}"
            for name, observed, expected in checks
            if observed != expected
        ]
        if mismatches:
            raise ValueError(
                "existing readiness pre-registration conflicts with this run: "
                + "; ".join(mismatches)
            )
        return existing
    atomic_write_json(PRE_REGISTRATION, value)
    return value


def _run_with_resume(
    plan: Dict[str, Any], *, workers: int, attempts: int
) -> Dict[str, Any]:
    def completed() -> int:
        candidate_root = OUTPUT_ROOT / FINAL_NAME
        return len(list(candidate_root.rglob("episode.json")))

    previous = -1
    for attempt in range(1, int(attempts) + 1):
        before = completed()
        if before == previous:
            raise SystemExit(
                f"attempt {attempt - 1} made no progress ({before} cases); "
                "stopping instead of retrying indefinitely"
            )
        previous = before
        try:
            return run_matched_benchmark(
                plan,
                output_root=OUTPUT_ROOT,
                parallel_workers=int(workers),
            )
        except (BrokenProcessPool, RuntimeError, OSError) as exc:
            after = completed()
            print(
                f"attempt {attempt}/{attempts} failed after {after - before} "
                f"new cases ({after} total): {type(exc).__name__}: {exc}",
                flush=True,
            )
            if attempt == int(attempts):
                raise
    raise SystemExit("readiness benchmark exhausted its retry budget")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--engine-starts", type=int, default=4)
    parser.add_argument("--engine-stagger", type=float, default=1.0)
    parser.add_argument("--attempts", type=int, default=40)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)

    os.environ["HOLODRONE_MAX_ENGINE_STARTS"] = str(args.engine_starts)
    os.environ["HOLODRONE_ENGINE_START_STAGGER_SECONDS"] = str(
        args.engine_stagger
    )
    preregistration = _planned_evidence()
    plan = preregistration["plan"]
    print(
        f"readiness plan {plan['plan_fingerprint']} -> "
        f"{TRANSITION_CASES} transitions + "
        f"{SEQUENCES_PER_LENGTH} sequences/length on VALIDATION",
        flush=True,
    )
    if args.plan_only:
        print(f"pre-registered -> {PRE_REGISTRATION}")
        return 0

    report = _run_with_resume(
        plan, workers=args.workers, attempts=args.attempts
    )
    result = report["results"][FINAL_NAME]
    print(json.dumps(result["metrics"], indent=2), flush=True)
    print(f"written -> {OUTPUT_ROOT / 'matched_benchmark.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
