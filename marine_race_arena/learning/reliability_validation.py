"""Bounded, seed-controlled validation for reliability-first learned policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from marine_race_arena.learning.longrun_checkpoint import (
    atomic_write_json,
    sha256_file,
)
from marine_race_arena.learning.longrun_evaluation import evaluate_longrun_policy
from marine_race_arena.learning.provenance import git_sha, now_utc
from marine_race_arena.learning.reward_v3 import MultiGateRewardConfig
from marine_race_arena.learning.seed_registry import (
    MULTIGATE_RELIABILITY_VALIDATION_SEEDS,
)


class BCPredictor:
    """Minimal SB3-compatible prediction surface over a frozen BC-v3 policy."""

    def __init__(self, checkpoint: str | Path) -> None:
        from marine_race_arena.learning.bc_v3_transfer import load_v3_policy

        self.policy = load_v3_policy(checkpoint)
        self.policy.eval()
        self.num_timesteps = 0

    def predict(self, observation: Any, deterministic: bool = True):
        import torch

        array = np.asarray(observation, dtype=np.float32)
        batched = array.ndim > 1
        tensor = torch.as_tensor(
            array if batched else array.reshape(1, -1), dtype=torch.float32
        )
        with torch.no_grad():
            action = self.policy(tensor).cpu().numpy()
        return (action if batched else action[0]), None


def load_predictor(kind: str, checkpoint: str | Path) -> Any:
    if kind == "bc":
        return BCPredictor(checkpoint)
    if kind != "ppo":
        raise ValueError("model kind must be bc or ppo")
    from stable_baselines3 import PPO

    return PPO.load(str(checkpoint), device="cpu")


def evaluate_model(
    *,
    kind: str,
    checkpoint: str | Path,
    output_dir: str | Path,
    stage: str = "C0",
    mode: str = "full",
    cases: int = 20,
    adapter: str = "holoocean",
    max_steps: int = 1800,
) -> Dict[str, Any]:
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(path)
    if cases <= 0 or cases > len(MULTIGATE_RELIABILITY_VALIDATION_SEEDS):
        raise ValueError("invalid validation case count")
    if mode == "full" and cases < 10:
        raise ValueError("full validation requires at least ten cases")
    predictor = load_predictor(kind, path)
    out = Path(output_dir)
    report = evaluate_longrun_policy(
        predictor,
        stage=stage,
        mode=mode,
        seeds=MULTIGATE_RELIABILITY_VALIDATION_SEEDS[:cases],
        output_dir=out,
        env_kwargs={
            "adapter": adapter,
            "allow_fallback": adapter == "fallback",
            "current_profile": "none",
            "max_steps": int(max_steps),
            "observation_encoding_version": "onboard_multigate_rl_v3",
        },
        reward_config=MultiGateRewardConfig(reward_phase="reliability"),
        timesteps=int(getattr(predictor, "num_timesteps", 0)),
    )
    manifest = {
        "schema_version": "reliability_validation_v1",
        "created_utc": now_utc(),
        "git_sha": git_sha(),
        "model_kind": kind,
        "model_path": str(path),
        "model_sha256": sha256_file(path),
        "adapter": adapter,
        "current_profile": "none",
        "runtime_rule_actions": 0,
        "seeds": MULTIGATE_RELIABILITY_VALIDATION_SEEDS[:cases],
        "report": report,
    }
    atomic_write_json(out / "validation_manifest.json", manifest)
    return manifest


def compare_reports(named_paths: Sequence[str], output: str | Path) -> Dict[str, Any]:
    reports: Dict[str, Dict[str, Any]] = {}
    for item in named_paths:
        if "=" not in item:
            raise ValueError("report inputs must use NAME=PATH")
        name, raw_path = item.split("=", 1)
        report = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        if "report" in report:
            report = report["report"]
        reports[name] = report
    fields = (
        "completion_rate",
        "single_gate_completion_rate",
        "straight_completion_rate",
        "left_completion_rate",
        "right_completion_rate",
        "episodes_with_any_safety",
        "previous_gate_returns",
        "mean_penalized_time_s",
        "mean_action_jerk",
        "mean_inference_ms",
    )
    result = {
        "schema_version": "reliability_paired_comparison_v1",
        "created_utc": now_utc(),
        "git_sha": git_sha(),
        "controllers": {
            name: {field: report.get(field) for field in fields}
            for name, report in reports.items()
        },
        "claim": (
            "Descriptive paired evidence only; improvement requires the measured "
            "metrics in this file and is never inferred from training reward."
        ),
    }
    atomic_write_json(output, result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate-model")
    evaluate.add_argument("--kind", choices=("bc", "ppo"), required=True)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--stage", default="C0")
    evaluate.add_argument("--mode", choices=("light", "full"), default="full")
    evaluate.add_argument("--cases", type=int, default=20)
    evaluate.add_argument("--adapter", choices=("holoocean", "fallback"), default="holoocean")
    evaluate.add_argument("--max-steps", type=int, default=1800)
    compare = sub.add_parser("compare-reports")
    compare.add_argument("--report", action="append", required=True)
    compare.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "evaluate-model":
        result = evaluate_model(
            kind=args.kind,
            checkpoint=args.model,
            output_dir=args.out,
            stage=args.stage,
            mode=args.mode,
            cases=args.cases,
            adapter=args.adapter,
            max_steps=args.max_steps,
        )
    else:
        result = compare_reports(args.report, args.out)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
