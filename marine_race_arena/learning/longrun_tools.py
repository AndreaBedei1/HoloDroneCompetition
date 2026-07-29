"""Prepare, inspect, stop, select, and evaluate multi-gate long-run jobs."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from marine_race_arena.learning.longrun_checkpoint import (
    latest_valid_checkpoint,
    load_checkpoint_state,
    sha256_file,
)
from marine_race_arena.learning.longrun_config import LongRunConfig
from marine_race_arena.learning.longrun_evaluation import (
    evaluate_longrun_policy,
    official_evaluation_unlocked,
)
from marine_race_arena.learning.longrun_rollback import (
    enrich_rollbacks_with_curriculum,
    jsonl_rows_through,
    merge_curriculum_histories,
    merge_rollback_histories,
    rollback_status_fields,
)
from marine_race_arena.learning.reward_v3 import MultiGateRewardConfig
from marine_race_arena.learning.seed_registry import (
    MULTIGATE_LONGRUN_CHECKPOINT_SELECTION_SEEDS,
    MULTIGATE_RELIABILITY_CHECKPOINT_SELECTION_SEEDS,
)
from marine_race_arena.learning.train_multigate_longrun import (
    preflight,
    run_contract_hash,
)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if platform.system() == "Windows":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
        )
        return str(pid) in result.stdout and "No tasks" not in result.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _compact_curriculum_history(
    changes: Sequence[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    return [
        {
            "timesteps": int(change.get("timesteps", 0)),
            "from": change.get("from"),
            "to": change.get("to"),
            "reason": change.get(
                "reason",
                "evaluation_promotion" if change.get("metrics") else None,
            ),
        }
        for change in changes
    ]


def run_status(run_dir: str | Path) -> Dict[str, Any]:
    path = Path(run_dir)
    status_path = path / "status.json"
    status: Dict[str, Any] = {}
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception as exc:
            status = {"status_read_error": f"{type(exc).__name__}: {exc}"}
    pid = 0
    try:
        pid = int((path / "pid.txt").read_text(encoding="utf-8").strip())
    except Exception:
        pass
    alive = _pid_alive(pid)
    update_unix = float(status.get("last_log_update_unix", 0.0) or 0.0)
    age = time.time() - update_unix if update_unix else None
    config = None
    try:
        config = LongRunConfig.load(path / "config.json")
    except Exception:
        pass
    stall_timeout = (
        config.reliability.stall_timeout_seconds if config is not None else 900
    )
    crash_reports = sorted((path / "crash_reports").glob("*.json"))
    latest = latest_valid_checkpoint(
        path,
        expected_contract_hash=(
            run_contract_hash(config) if config is not None else None
        ),
        safe_only=False,
    )
    checkpoint_state: Dict[str, Any] = {}
    if latest is not None:
        try:
            checkpoint_state = load_checkpoint_state(latest)
        except Exception:
            checkpoint_state = {}
    current = int(
        status.get(
            "total_timesteps",
            checkpoint_state.get("total_timesteps", 0),
        )
        or 0
    )
    checkpoint_extra = dict(checkpoint_state.get("extra", {}))
    rollback_history = merge_rollback_histories(
        status.get("rollback_history", []),
        checkpoint_extra.get("rollback_history", []),
        jsonl_rows_through(
            path / "logs" / "rollback_history.jsonl", current
        ),
    )
    checkpoint_curriculum = dict(
        checkpoint_state.get("curriculum", {})
    ).get("state", {})
    curriculum_history = merge_curriculum_histories(
        status.get("curriculum_stage_history", []),
        checkpoint_curriculum.get("stage_changes", []),
        jsonl_rows_through(path / "curriculum_history.jsonl", current),
        rollback_history=rollback_history,
    )
    rollback_history = enrich_rollbacks_with_curriculum(
        rollback_history, curriculum_history
    )
    if rollback_history:
        latest_event = rollback_history[-1]
        checkpoint_retention_state = dict(
            checkpoint_state.get("model_training", {})
        ).get("retention_regularizer")
        if (
            isinstance(checkpoint_retention_state, dict)
            and latest_event.get("retention_multiplier_after") is None
            and checkpoint_retention_state.get("weight_multiplier") is not None
        ):
            multiplier_after = float(
                checkpoint_retention_state["weight_multiplier"]
            )
            latest_event["retention_multiplier_after"] = multiplier_after
            if config is not None:
                scale = float(
                    config.reliability.rollback.retention_weight_scale
                )
                if scale:
                    latest_event["retention_multiplier_before"] = (
                        multiplier_after / scale
                    )
    effective_retention_state = status.get("retention_state")
    checkpoint_retention_state = dict(
        checkpoint_state.get("model_training", {})
    ).get("retention_regularizer")
    if isinstance(checkpoint_retention_state, dict):
        effective_retention_state = {
            **checkpoint_retention_state,
            **(
                effective_retention_state
                if isinstance(effective_retention_state, dict)
                else {}
            ),
        }
        # The running process may still publish a pre-rollback multiplier while
        # newer sidecars already prove that the rollback restored correctly.
        # Preserve the fresher live counters/RNG, but report the persisted
        # post-rollback multiplier.
        effective_retention_state["weight_multiplier"] = (
            checkpoint_retention_state.get("weight_multiplier")
        )
    rollback_fields = rollback_status_fields(rollback_history)
    rollback_fields["rollback_count"] = max(
        int(status.get("rollback_count", 0) or 0),
        int(checkpoint_extra.get("rollback_count", 0) or 0),
        int(rollback_fields["rollback_count"]),
    )
    if not rollback_history:
        rollback_fields["last_rollback_reason"] = status.get(
            "last_rollback_reason"
        )
        rollback_fields["last_rollback_source"] = status.get(
            "last_rollback_source"
        )
    result = {
        "run_dir": str(path),
        "pid": pid or None,
        "process_alive": alive,
        "state": status.get("state", "UNKNOWN"),
        "current_total_timesteps": current or None,
        "target_total_timesteps": status.get("target_total_timesteps"),
        "current_curriculum_stage": status.get("curriculum_stage"),
        "curriculum_stage_history": _compact_curriculum_history(
            curriculum_history
        ),
        "curriculum_evaluation_history_count": status.get(
            "curriculum_evaluation_history_count"
        ),
        "evaluation_history_count": status.get("evaluation_history_count"),
        "last_checkpoint": (
            str(latest.model_path) if latest is not None else status.get("last_checkpoint")
        ),
        "best_checkpoint": status.get("best_checkpoint"),
        "latest_evaluation": status.get("latest_evaluation"),
        "latest_full_evaluation": status.get("latest_full_evaluation"),
        "overall_completion": status.get("overall_completion"),
        "evaluation_mode": status.get("evaluation_mode"),
        "evaluation_cases_completed": status.get("evaluation_cases_completed"),
        "evaluation_cases_total": status.get("evaluation_cases_total"),
        "left_success": status.get("left_success"),
        "right_success": status.get("right_success"),
        "straight_retention": status.get("straight_retention"),
        "single_gate_retention": status.get("single_gate_retention"),
        "three_gate_success": status.get("three_gate_success"),
        "approximate_kl": status.get("approx_kl"),
        "initial_policy_kl": status.get("initial_policy_kl"),
        "ppo_policy_loss": status.get("ppo_policy_loss"),
        "bc_retention_loss": status.get("bc_retention_loss"),
        "combined_policy_loss": status.get("combined_policy_loss"),
        "retention_weight": status.get("retention_weight"),
        "reward_phase": status.get("reward_phase"),
        "replay_mixture": status.get("replay_mixture"),
        "consecutive_reliable_full_evaluations": status.get(
            "consecutive_reliable_full_evaluations"
        ),
        "stage_entry_timesteps": status.get("stage_entry_timesteps"),
        "steps_in_current_stage": status.get("steps_in_current_stage"),
        "next_promotion_eligibility_step": status.get(
            "next_promotion_eligibility_step"
        ),
        "stage_minimum_remaining_timesteps": status.get(
            "stage_minimum_remaining_timesteps"
        ),
        **rollback_fields,
        "collision_events": status.get("collision_events"),
        "collision_frames": status.get("collision_frames"),
        "out_of_bounds_events": status.get("out_of_bounds_events"),
        "out_of_bounds_frames": status.get("out_of_bounds_frames"),
        "wrong_direction_count": status.get("wrong_direction_count"),
        "safety_warning_events": status.get("safety_warning_events"),
        "safety_warning_frames": status.get("safety_warning_frames"),
        "episodes_with_any_safety": status.get("episodes_with_any_safety"),
        "collision_episodes": status.get("collision_episodes"),
        "out_of_bounds_episodes": status.get("out_of_bounds_episodes"),
        "wrong_direction_episodes": status.get("wrong_direction_episodes"),
        "previous_gate_returns": status.get("previous_gate_returns"),
        "mean_successful_time_s": status.get("mean_successful_time_s"),
        "mean_penalized_time_s": status.get("mean_penalized_time_s"),
        "mean_action_jerk": status.get("mean_action_jerk"),
        "current_learning_rate": status.get("current_learning_rate"),
        "best_reliable_checkpoint": status.get("best_reliable_checkpoint"),
        "best_fast_reliable_checkpoint": status.get(
            "best_fast_reliable_checkpoint"
        ),
        "checkpoint_aliases": status.get("checkpoint_aliases"),
        "retention_state": effective_retention_state,
        "throughput_steps_per_s": status.get("throughput_steps_per_s"),
        "estimated_remaining_wall_s": status.get(
            "estimated_remaining_wall_s"
        ),
        "elapsed_wall_s": status.get("elapsed_wall_s"),
        "last_log_update_unix": update_unix or None,
        "last_log_age_s": round(age, 3) if age is not None else None,
        "detected_crash": (
            str(crash_reports[-1]) if crash_reports else status.get("detected_crash")
        ),
        "stall_detected": bool(alive and age is not None and age > stall_timeout),
        "simulator_restarts": status.get("simulator_restarts", 0),
        "stop_reason": status.get("stop_reason"),
    }
    return result


def request_stop(run_dir: str | Path) -> Dict[str, Any]:
    path = Path(run_dir)
    if not path.exists():
        raise FileNotFoundError(path)
    request = path / "stop.requested"
    request.write_text(
        json.dumps({"requested_unix": time.time(), "requested_by_pid": os.getpid()}),
        encoding="utf-8",
    )
    return {"stop_requested": True, "path": str(request), **run_status(path)}


def select_best(run_dir: str | Path) -> Dict[str, Any]:
    path = Path(run_dir)
    aliases = {}
    for name in (
        "best_reliable",
        "best_fast_reliable",
        "best_overall",
        "best_left",
        "best_right",
        "best_three_gate",
        "latest_safe",
        "last",
        "initial_bc_v3",
    ):
        model = path / "best_models" / f"{name}.zip"
        meta = model.with_suffix(".json")
        aliases[name] = (
            {
                "path": str(model),
                "sha256": sha256_file(model),
                "metadata": (
                    json.loads(meta.read_text(encoding="utf-8"))
                    if meta.exists()
                    else None
                ),
            }
            if model.exists()
            else None
        )
    selected = (
        aliases["best_reliable"]
        or aliases["best_overall"]
        or aliases["latest_safe"]
    )
    if selected is None:
        raise RuntimeError("no safe checkpoint has been selected")
    return {"selected": selected, "aliases": aliases}


def prepare(config: LongRunConfig, *, allow_dirty: bool = False) -> Dict[str, Any]:
    checks = preflight(
        config,
        allow_smoke=allow_dirty,
        allow_dirty_smoke=allow_dirty,
    )
    # Preparation never creates the run directory, which keeps the start command's
    # no-overwrite guard meaningful.
    return {
        "status": "LONG-RUN TRAINING PIPELINE READY",
        "run_dir": str(config.run_dir),
        "config": config.to_dict(),
        "checks": checks,
    }


def evaluate_selected(
    run_dir: str | Path,
    *,
    suite: str = "r2",
    stage: Optional[str] = None,
    alias: Optional[str] = None,
) -> Dict[str, Any]:
    from stable_baselines3 import PPO

    path = Path(run_dir)
    config = LongRunConfig.load(path / "config.json")
    selection = select_best(path)
    if alias:
        if alias not in selection["aliases"]:
            raise ValueError(f"unknown checkpoint alias {alias!r}")
        selected = selection["aliases"][alias]
        if selected is None:
            raise FileNotFoundError(path / "best_models" / f"{alias}.zip")
    else:
        selected = selection["selected"]
    model = PPO.load(selected["path"], device="cpu")
    status = run_status(path)
    stage_key = stage or str(status.get("current_curriculum_stage") or "C4")
    if suite == "official":
        latest_path = status.get("latest_evaluation")
        if not latest_path or not Path(latest_path).exists():
            raise RuntimeError("official evaluation locked: no full evaluation exists")
        latest = json.loads(Path(latest_path).read_text(encoding="utf-8"))
        if not official_evaluation_unlocked(latest):
            raise RuntimeError(
                "official evaluation locked until R2 and three-gate gates pass"
            )
        raise RuntimeError(
            "official circuit evaluation must be launched explicitly with the "
            "documented per-circuit held-out commands"
        )
    seeds = (
        MULTIGATE_RELIABILITY_CHECKPOINT_SELECTION_SEEDS
        if config.training_profile == "reliability_first"
        else MULTIGATE_LONGRUN_CHECKPOINT_SELECTION_SEEDS
    )[:20]
    report = evaluate_longrun_policy(
        model,
        stage=stage_key,
        mode="full",
        seeds=seeds,
        output_dir=path / "evaluations" / "post_training",
        env_kwargs={
            "adapter": "holoocean",
            "allow_fallback": False,
            "current_profile": "none",
            "max_steps": config.max_episode_steps,
            "observation_encoding_version": config.observation_version,
        },
        reward_config=MultiGateRewardConfig(),
        timesteps=int(model.num_timesteps),
        policy_mode=config.policy_mode,
        frame_stack=config.frame_stack,
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--config", required=True)
    prepare_parser.add_argument("--run-name", default=None)
    prepare_parser.add_argument("--total-timesteps", type=int, default=None)
    prepare_parser.add_argument("--allow-dirty", action="store_true")
    status_parser = sub.add_parser("status")
    status_parser.add_argument("run_dir")
    stop_parser = sub.add_parser("stop")
    stop_parser.add_argument("run_dir")
    select_parser = sub.add_parser("select-best")
    select_parser.add_argument("run_dir")
    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("run_dir")
    eval_parser.add_argument("--suite", choices=("r2", "official"), default="r2")
    eval_parser.add_argument("--stage", default=None)
    eval_parser.add_argument(
        "--alias",
        choices=(
            "best_reliable",
            "best_fast_reliable",
            "best_overall",
            "best_left",
            "best_right",
            "best_three_gate",
            "latest_safe",
            "last",
            "initial_bc_v3",
        ),
        default=None,
    )
    args = parser.parse_args(argv)
    if args.command == "prepare":
        config = LongRunConfig.load(args.config)
        if args.run_name:
            config.run_name = args.run_name
        if args.total_timesteps is not None:
            config.total_timesteps = args.total_timesteps
        result = prepare(config, allow_dirty=args.allow_dirty)
    elif args.command == "status":
        result = run_status(args.run_dir)
    elif args.command == "stop":
        result = request_stop(args.run_dir)
    elif args.command == "select-best":
        result = select_best(args.run_dir)
    else:
        result = evaluate_selected(
            args.run_dir, suite=args.suite, stage=args.stage, alias=args.alias
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
