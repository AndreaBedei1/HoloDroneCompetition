"""Commit the one-shot exploratory holdout decision before circuit access."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from marine_race_arena.learning.rl_readiness_gate import (
    LADDER_KEY,
    evaluate_readiness,
    record_exploratory_holdout_decision,
)

REPO = Path(__file__).resolve().parents[2]
FREEZE_MANIFEST = (
    REPO / "results/rl_public/ppo_final_generic_929792/artifact_freeze.json"
)
READINESS = REPO / "results/rl_public/ppo_final_readiness_929792"
EVALUATION = (
    READINESS / "ppo_final_generic_929792/seed_5000000/evaluation.json"
)
LADDER = (
    REPO
    / "results/rl_public/ppo_difficulty_ladder"
    / "ppo_final_generic_929792_ladder.json"
)
VERDICT = READINESS / "readiness_verdict.json"
OUTPUT = (
    REPO
    / "results/rl_public/ppo_final_holdout_929792"
    / "exploratory_holdout_decision.json"
)


def _read(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{path} is not a JSON object")
    return value


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def main() -> int:
    freeze = _read(FREEZE_MANIFEST)
    readiness = _read(VERDICT)
    evidence = _read(EVALUATION)
    evidence[LADDER_KEY] = _read(LADDER)
    verdict = evaluate_readiness(evidence, override=None)

    if verdict.ready:
        raise SystemExit("readiness passed; an exploratory failure decision is invalid")
    if verdict.as_dict() != readiness.get("readiness"):
        raise SystemExit("materialised readiness verdict does not match source evidence")
    if freeze.get("immutable") is not True:
        raise SystemExit("final PPO artifact is not marked immutable")
    closure = freeze.get("closure") or {}
    if closure.get("training_permanently_closed") is not True:
        raise SystemExit("training is not marked permanently closed")

    artifact = freeze.get("artifact") or {}
    checkpoint = Path(str(artifact.get("checkpoint", "")))
    contracts = dict(freeze.get("contracts") or {})
    provenance = freeze.get("provenance") or {}
    record = record_exploratory_holdout_decision(
        checkpoint,
        run_dir=READINESS,
        metrics=evidence,
        verdict=verdict,
        git_sha=_git_sha(),
        contracts=contracts,
        training_transitions=int(provenance["total_environment_transitions"]),
        curriculum_version=str(contracts["curriculum"]),
        path=OUTPUT,
        note=(
            "Procedural readiness failed. Training and the final PPO checkpoint "
            "remain permanently frozen. The project nevertheless pre-registers "
            "one exploratory evaluation on the three still-unseen official "
            "circuits, with no subsequent parameter, reward, curriculum or "
            "controller tuning."
        ),
    )
    print(f"decision {record['freeze_id']} -> {OUTPUT}", flush=True)
    print(f"readiness failures: {verdict.failures}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
