"""Train or resume the observation-v3 multi-gate PPO policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from marine_race_arena.learning.config_v3 import OBS_ENCODING_VERSION_V3
from marine_race_arena.learning.model_contract_v3 import assert_clean_worktree
from marine_race_arena.learning.multigate_curriculum import stage
from marine_race_arena.learning.reward_v3 import MultiGateRewardConfig
from marine_race_arena.learning.seed_registry import (
    MULTIGATE_V3_DEV_EVAL_SEEDS,
    MULTIGATE_V3_PPO_TRAINING_SEEDS,
)
from marine_race_arena.learning.train_workflow import run_ppo_training

DEFAULT_BC_V1 = "results/rl_public/stage1/bc/model/best_model.pt"

KL_SAFE_PPO = {
    "learning_rate": 1e-5,
    "n_steps": 500,
    "batch_size": 100,
    "n_epochs": 1,
    "clip_range": 0.05,
    "target_kl": 0.01,
}


def _parse_seeds(spec: str):
    values = []
    for token in str(spec).split(","):
        token = token.strip()
        if "-" in token:
            lo, hi = token.split("-", 1)
            values.extend(range(int(lo), int(hi) + 1))
        elif token:
            values.append(int(token))
    return values


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="R1", choices=[f"R{i}" for i in range(6)])
    parser.add_argument("--track", default=None)
    parser.add_argument("--track-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--train-seed", type=int, default=MULTIGATE_V3_PPO_TRAINING_SEEDS[0])
    parser.add_argument(
        "--eval-seeds",
        default=f"{MULTIGATE_V3_DEV_EVAL_SEEDS[0]}-{MULTIGATE_V3_DEV_EVAL_SEEDS[4]}",
    )
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--bc-model", default=None)
    initialization.add_argument(
        "--ppo-model",
        default=None,
        help="selected observation-v3 PPO checkpoint for curriculum continuation",
    )
    parser.add_argument("--output-root", default="results/rl/multigate_v3")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--adapter", default="holoocean", choices=("holoocean", "fallback"))
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--current-profile", default="none")
    parser.add_argument("--benchmark-task", default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=2400)
    parser.add_argument("--checkpoint-freq", type=int, default=500)
    parser.add_argument("--eval-freq", type=int, default=500)
    parser.add_argument("--no-initial-eval", action="store_true")
    args = parser.parse_args(argv)

    assert_clean_worktree()
    curriculum_stage = stage(args.stage)
    if args.track:
        track = args.track
    else:
        if not 0 <= args.track_index < len(curriculum_stage.tracks):
            raise ValueError(
                f"track-index {args.track_index} outside stage {args.stage} "
                f"range 0..{len(curriculum_stage.tracks) - 1}"
            )
        track = curriculum_stage.tracks[args.track_index]
    if args.adapter == "holoocean" and args.allow_fallback:
        raise ValueError("real HoloOcean multi-gate training must not allow fallback")
    if args.train_seed not in MULTIGATE_V3_PPO_TRAINING_SEEDS:
        raise ValueError("train seed is outside the allocated v3 PPO training range")
    eval_seeds = _parse_seeds(args.eval_seeds)
    if not set(eval_seeds) <= set(MULTIGATE_V3_DEV_EVAL_SEEDS):
        raise ValueError("development evaluation seeds must use the allocated v3 dev range")
    if args.resume and not args.run_dir:
        raise ValueError("--resume requires --run-dir")
    bc_model = args.bc_model
    if bc_model is None and args.ppo_model is None:
        bc_model = DEFAULT_BC_V1
    if args.ppo_model is not None:
        from marine_race_arena.learning.model_contract_v3 import validate_v3_model

        contract = validate_v3_model(args.ppo_model)
        if contract["kind"] != "ppo":
            raise ValueError("--ppo-model must point to an observation-v3 PPO ZIP")

    benchmark_task = args.benchmark_task
    if Path(track).name == "marine_race_mixed_endurance.json" and benchmark_task is None:
        benchmark_task = "clean_gate"
    env_kwargs = {
        "adapter": args.adapter,
        "allow_fallback": bool(args.allow_fallback),
        "current_profile": args.current_profile,
        "benchmark_task": benchmark_task,
        "duration_s": args.duration,
        "max_steps": args.max_steps,
        "observation_encoding_version": OBS_ENCODING_VERSION_V3,
    }
    run_path, model = run_ppo_training(
        track,
        stage=args.stage.lower(),
        algorithm="ppo_multigate_v3",
        total_timesteps=args.steps,
        train_seed=args.train_seed,
        eval_seeds=eval_seeds,
        output_root=args.output_root,
        run_dir=args.run_dir,
        resume=args.resume,
        bc_model_path=bc_model,
        initial_ppo_model_path=args.ppo_model,
        arm=(
            "ppo_curriculum_transfer"
            if args.ppo_model is not None
            else "bc_v1_transfer_v3"
            if bc_model == DEFAULT_BC_V1
            else "bc_v3_warm_start"
        ),
        action_std_strategy=(
            "preserve_checkpoint" if args.ppo_model is not None else "fixed"
        ),
        action_std_value=(None if args.ppo_model is not None else 0.10),
        max_acceptable_kl=0.02,
        reward_config=MultiGateRewardConfig(),
        env_kwargs=env_kwargs,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        initial_eval=not args.no_initial_eval,
        ppo_kwargs=KL_SAFE_PPO,
    )
    best_metrics = run_path / "best_model" / "best_metrics.json"
    summary = (
        json.loads(best_metrics.read_text(encoding="utf-8"))
        if best_metrics.exists()
        else {}
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_path),
                "final_num_timesteps": int(model.num_timesteps),
                "best_metrics": summary,
                "controller": "rl_multigate_controller",
                "observation_encoding_version": OBS_ENCODING_VERSION_V3,
                "rule_action_weight": 0,
                "hybrid_blending": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
