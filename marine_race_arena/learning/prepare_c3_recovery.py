"""Create a history-preserving C3 continuation from a validated safe policy."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from marine_race_arena.learning.config import ACTION_CONTRACT_VERSION
from marine_race_arena.learning.config_v3 import OBS_ENCODING_VERSION_V3
from marine_race_arena.learning.longrun_checkpoint import (
    atomic_append_jsonl,
    atomic_copy_checkpoint,
    atomic_save_checkpoint,
    atomic_write_json,
    capture_rng_state,
    checkpoint_passed_safety,
    full_evaluation_passed_safety,
    load_checkpoint_state,
    restore_rng_state,
    sha256_file,
    valid_checkpoints,
)
from marine_race_arena.learning.longrun_config import LongRunConfig
from marine_race_arena.learning.longrun_rollback import (
    enrich_rollbacks_with_curriculum,
    jsonl_rows_through,
    merge_curriculum_histories,
    merge_rollback_histories,
    rollback_status_fields,
)
from marine_race_arena.learning.parametric_curriculum import (
    TransitionGeometry,
)
from marine_race_arena.learning.provenance import git_sha, now_utc
from marine_race_arena.learning.reliability_first import ReliabilityFirstPPO
from marine_race_arena.learning.train_multigate_longrun import (
    _attach_retention_regularizer,
    _make_sampler,
    _migrate_learning_rate_schedule,
    config_contract_hash,
)


def select_best_reliable_checkpoint(
    source_run_dir: str | Path,
) -> Any:
    """Resolve the selected best-reliable checkpoint, never the last snapshot."""
    source = Path(source_run_dir)
    checkpoints = list(valid_checkpoints(source))
    if not checkpoints:
        raise RuntimeError("source run has no valid checkpoints")
    latest_state = load_checkpoint_state(checkpoints[-1])
    selected = (
        dict(latest_state.get("evaluation", {}))
        .get("best", {})
        .get("best_reliable", {})
        .get("checkpoint")
    )
    if not selected:
        selected = dict(latest_state.get("extra", {})).get(
            "checkpoint_aliases", {}
        ).get("best_reliable")
    if not selected:
        raise RuntimeError("source run has no best_reliable selection")
    selected_name = Path(selected).name
    matches = [
        checkpoint
        for checkpoint in checkpoints
        if checkpoint.model_path.name == selected_name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"best_reliable checkpoint {selected_name} is unavailable"
        )
    checkpoint = matches[0]
    if not checkpoint_passed_safety(checkpoint):
        raise RuntimeError("best_reliable lacks matching safety evidence")
    return checkpoint


def resolve_recovery_checkpoint(
    source_run_dir: str | Path,
    requested: Optional[str | Path] = None,
) -> Any:
    if requested is None:
        return select_best_reliable_checkpoint(source_run_dir)
    requested_name = Path(requested).name
    matches = [
        checkpoint
        for checkpoint in valid_checkpoints(source_run_dir)
        if checkpoint.model_path.name == requested_name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"requested recovery checkpoint {requested_name} is unavailable"
        )
    checkpoint = matches[0]
    if not checkpoint_passed_safety(checkpoint):
        raise RuntimeError(
            "requested recovery checkpoint lacks matching safety evidence"
        )
    return checkpoint


def _latest_source_checkpoint(source: Path) -> Any:
    checkpoints = list(valid_checkpoints(source))
    if not checkpoints:
        raise RuntimeError("source run has no valid checkpoints")
    return checkpoints[-1]


def _copy_history(source: Path, destination: Path) -> None:
    for directory in ("evaluations", "logs", "tensorboard", "crash_reports"):
        src = source / directory
        dst = destination / directory
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            if directory == "logs":
                shutil.copytree(
                    src,
                    destination / "source_snapshot" / "logs",
                    dirs_exist_ok=True,
                )
    for name in (
        "curriculum_history.jsonl",
        "run_manifest.json",
        "status.json",
        "reward_config.json",
    ):
        src = source / name
        if src.exists():
            snapshot = destination / "source_snapshot" / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, snapshot)
    curriculum = source / "curriculum_history.jsonl"
    if curriculum.exists():
        shutil.copy2(curriculum, destination / curriculum.name)


def _recovery_curriculum(
    config: LongRunConfig,
    source_state: Dict[str, Any],
    failed_geometry: Dict[str, Any],
) -> Dict[str, Any]:
    sampler = _make_sampler(config)
    value = sampler.state_dict()
    value["state"] = copy.deepcopy(source_state["curriculum"]["state"])
    value["state"]["training_timesteps"] = int(
        source_state["total_timesteps"]
    )
    sampler.load_state_dict(value)
    geometry = TransitionGeometry(**failed_geometry)
    duplicate = any(
        row.get("geometry") == failed_geometry
        for row in sampler.state.targeted_failures
    )
    if not duplicate:
        sampler.add_failure_case(
            geometry,
            "recovery_replay_collision_seed_25308",
        )
    return sampler.state_dict()


def _scaled_recovery_model(
    config: LongRunConfig,
    policy_checkpoint: Any,
    policy_state: Dict[str, Any],
    curriculum: Dict[str, Any],
    *,
    absolute_timesteps: int,
    learning_rate_scale: float,
) -> Any:
    model = ReliabilityFirstPPO.load(
        str(policy_checkpoint.model_path), device="cpu"
    )
    _migrate_learning_rate_schedule(model, config)
    seen = set()
    for schedule in (
        getattr(model, "learning_rate", None),
        getattr(model, "lr_schedule", None),
        getattr(getattr(model, "lr_schedule", None), "value_schedule", None),
    ):
        if schedule is None or id(schedule) in seen:
            continue
        seen.add(id(schedule))
        if hasattr(schedule, "scale"):
            schedule.scale(learning_rate_scale)
    optimizer = getattr(model.policy, "optimizer", None)
    if optimizer is not None:
        for group in optimizer.param_groups:
            group["lr"] = float(group["lr"]) * float(learning_rate_scale)
    model.num_timesteps = int(absolute_timesteps)
    model._current_progress_remaining = max(
        0.0, 1.0 - int(absolute_timesteps) / int(config.total_timesteps)
    )
    model.longrun_config_contract = config_contract_hash(config)
    model.obs_encoding_version = OBS_ENCODING_VERSION_V3
    model.action_contract_version = ACTION_CONTRACT_VERSION
    model.longrun_policy_mode = config.policy_mode
    model.longrun_frame_stack = config.frame_stack
    sampler = _make_sampler(config)
    sampler.load_state_dict(curriculum)
    _attach_retention_regularizer(model, config, sampler)
    retention = dict(policy_state.get("model_training", {})).get(
        "retention_regularizer"
    )
    regularizer = getattr(model, "_retention_regularizer", None)
    if regularizer is not None and isinstance(retention, dict):
        regularizer.load_state_dict(retention)
    return model


def prepare_recovery_continuation(
    *,
    source_run_dir: str | Path,
    config_path: str | Path,
    failed_evaluation: str | Path,
    validation_report: str | Path,
    baseline_report: str | Path,
    policy_checkpoint: Optional[str | Path] = None,
    learning_rate_scale: float = 0.5,
) -> Dict[str, Any]:
    source = Path(source_run_dir)
    config = LongRunConfig.load(config_path)
    config.validate()
    destination = config.run_dir
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"recovery run already exists: {destination}")
    source_checkpoint = _latest_source_checkpoint(source)
    source_state = load_checkpoint_state(source_checkpoint)
    selected_policy = resolve_recovery_checkpoint(
        source, policy_checkpoint
    )
    configured_policy = Path(config.initialization_checkpoint)
    if configured_policy.resolve() != selected_policy.model_path.resolve():
        raise ValueError(
            "recovery config initialization_checkpoint does not match the "
            "selected policy checkpoint"
        )
    if sha256_file(selected_policy.model_path) != config.initialization_sha256:
        raise ValueError(
            "selected recovery policy does not match initialization_sha256"
        )
    policy_state = load_checkpoint_state(selected_policy)
    failed = json.loads(Path(failed_evaluation).read_text(encoding="utf-8"))
    collision_rows = [
        row
        for row in failed.get("rows", [])
        if int(row.get("collision_events", 0)) > 0
    ]
    if len(collision_rows) != 1 or not collision_rows[0].get("geometry"):
        raise ValueError("failed evaluation has no unique collision geometry")
    validation = json.loads(
        Path(validation_report).read_text(encoding="utf-8")
    )
    if not full_evaluation_passed_safety(
        validation, minimum_cases=config.evaluation.full_episodes
    ):
        raise ValueError("recovery policy did not pass the full safety suite")
    if int(validation.get("completions", 0)) != int(
        config.evaluation.full_episodes
    ):
        raise ValueError("recovery policy did not complete every validation case")
    baseline = json.loads(Path(baseline_report).read_text(encoding="utf-8"))
    for metric in ("mean_action_jerk", "mean_successful_time_s"):
        recovery_value = float(validation[metric])
        regressed_value = float(baseline[metric])
        if recovery_value > regressed_value * 1.25:
            raise ValueError(
                f"recovery policy substantially regressed {metric}: "
                f"{recovery_value} > 1.25 * {regressed_value}"
            )

    destination.mkdir(parents=True, exist_ok=False)
    _copy_history(source, destination)
    config.save(destination / "config.json")
    current = int(source_state["total_timesteps"])
    curriculum = _recovery_curriculum(
        config, source_state, collision_rows[0]["geometry"]
    )
    rollback_history = merge_rollback_histories(
        dict(source_state.get("extra", {})).get("rollback_history", []),
        jsonl_rows_through(
            source / "logs" / "rollback_history.jsonl", current
        ),
    )
    curriculum_history = merge_curriculum_histories(
        curriculum["state"].get("stage_changes", []),
        jsonl_rows_through(source / "curriculum_history.jsonl", current),
        rollback_history=rollback_history,
    )
    rollback_history = enrich_rollbacks_with_curriculum(
        rollback_history, curriculum_history
    )
    curriculum["state"]["stage_changes"] = curriculum_history
    rollback_fields = rollback_status_fields(rollback_history)
    recovery_checkpoint_path = (
        destination / "checkpoints" / f"ppo_{current}_steps.zip"
    )
    compact_validation = {
        key: value
        for key, value in validation.items()
        if key not in {"rows", "reward_component_sums"}
    }
    recovery_report = {
        **compact_validation,
        "timesteps": current,
        "recovery_validation": True,
        "validated_policy_checkpoint": str(selected_policy.model_path),
        "validated_policy_timesteps": selected_policy.timesteps,
        "source_report": str(Path(validation_report)),
    }
    persisted_validation = copy.deepcopy(validation)
    persisted_validation.update(recovery_report)
    atomic_write_json(
        destination / "evaluations" / f"full_{current:09d}.json",
        persisted_validation,
    )
    selected = {
        **recovery_report,
        "checkpoint": str(recovery_checkpoint_path),
        "policy_source_checkpoint": str(selected_policy.model_path),
        "policy_source_timesteps": selected_policy.timesteps,
    }
    evaluation = copy.deepcopy(source_state.get("evaluation", {}))
    evaluation["history"] = [
        *list(evaluation.get("history", [])),
        recovery_report,
    ]
    evaluation["last"] = recovery_report
    best = dict(evaluation.get("best", {}))
    for alias in ("best_overall", "best_reliable", "best_fast_reliable"):
        best[alias] = dict(selected)
    evaluation["best"] = best
    extra = copy.deepcopy(source_state.get("extra", {}))
    extra.update(rollback_fields)
    extra.update(
        {
            "stop_reason": None,
            "rollback_attempt_base": int(
                rollback_fields["rollback_count"]
            ),
            "reward_phase": "reliability",
            "checkpoint_aliases": {
                **dict(extra.get("checkpoint_aliases", {})),
                "last": str(recovery_checkpoint_path),
                "latest_safe": str(recovery_checkpoint_path),
                "best_overall": str(recovery_checkpoint_path),
                "best_reliable": str(recovery_checkpoint_path),
                "best_fast_reliable": str(recovery_checkpoint_path),
            },
            "safety_validation": {
                "passed": True,
                "report_path": str(Path(validation_report)),
                "report": recovery_report,
                "validated_policy_checkpoint": str(
                    selected_policy.model_path
                ),
                "validated_policy_timesteps": selected_policy.timesteps,
            },
            "recovery_provenance": {
                "schema_version": "multigate_c3_recovery_v1",
                "created_utc": now_utc(),
                "source_run": str(source),
                "source_absolute_checkpoint": str(
                    source_checkpoint.model_path
                ),
                "source_absolute_timesteps": current,
                "policy_restored_from": str(selected_policy.model_path),
                "policy_restored_from_timesteps": selected_policy.timesteps,
                "learning_rate_scale": float(learning_rate_scale),
                "failed_evaluation": str(Path(failed_evaluation)),
                "baseline_report": str(Path(baseline_report)),
                "failed_seed": int(collision_rows[0]["seed"]),
            },
        }
    )
    model = _scaled_recovery_model(
        config,
        selected_policy,
        policy_state,
        curriculum,
        absolute_timesteps=current,
        learning_rate_scale=learning_rate_scale,
    )
    previous_rng = capture_rng_state()
    try:
        restore_rng_state(source_state["rng"])
        checkpoint = atomic_save_checkpoint(
            model,
            destination,
            total_timesteps=current,
            config_contract_hash=config_contract_hash(config),
            curriculum_state=curriculum,
            evaluation_state=evaluation,
            status="safe",
            reason="recovery_from_validated_safe_policy",
            extra_state=extra,
        )
    finally:
        restore_rng_state(previous_rng)
    for alias in (
        "last",
        "latest_safe",
        "best_overall",
        "best_reliable",
        "best_fast_reliable",
    ):
        atomic_copy_checkpoint(
            checkpoint, destination / "best_models" / f"{alias}.zip"
        )
    recovery = extra["recovery_provenance"]
    atomic_write_json(destination / "recovery_manifest.json", recovery)
    atomic_write_json(
        destination / "run_manifest.json",
        {
            "schema_version": "multigate_longrun_run_v1",
            "created_utc": now_utc(),
            "status": "C3 RECOVERY CONTINUATION READY",
            "git_sha": git_sha(),
            "config_contract_sha256": config_contract_hash(config),
            "derived_from": recovery,
            "initialization_mode": "history_preserving_policy_restore",
        },
    )
    atomic_append_jsonl(
        destination / "logs" / "recovery_history.jsonl", recovery
    )
    status = {
        "run_name": config.run_name,
        "state": "READY",
        "process_alive": False,
        "total_timesteps": current,
        "target_total_timesteps": config.total_timesteps,
        "curriculum_stage": curriculum["state"]["current_stage"],
        "curriculum_stage_history": curriculum_history,
        "stage_entry_timesteps": curriculum["state"][
            "stage_entry_timesteps"
        ],
        "next_promotion_eligibility_step": (
            int(curriculum["state"]["stage_entry_timesteps"])
            + int(
                config.curriculum.promotion.minimum_stage_timesteps.get(
                    curriculum["state"]["current_stage"], 0
                )
            )
        ),
        "reward_phase": "reliability",
        "rollback_attempt_base": extra["rollback_attempt_base"],
        **rollback_fields,
        "last_checkpoint": str(checkpoint.model_path),
        "best_checkpoint": str(checkpoint.model_path),
        "best_reliable_checkpoint": str(checkpoint.model_path),
        "best_fast_reliable_checkpoint": str(checkpoint.model_path),
        "checkpoint_aliases": extra["checkpoint_aliases"],
        "recovery_provenance": recovery,
        "overall_completion": recovery_report["completion_rate"],
        "collision_events": recovery_report["collision_events"],
        "out_of_bounds_events": recovery_report["out_of_bounds_events"],
        "wrong_direction_count": recovery_report["wrong_direction_count"],
        "previous_gate_returns": recovery_report["previous_gate_returns"],
        "mean_successful_time_s": recovery_report["mean_successful_time_s"],
        "mean_penalized_time_s": recovery_report["mean_penalized_time_s"],
        "mean_action_jerk": recovery_report["mean_action_jerk"],
    }
    atomic_write_json(destination / "status.json", status)
    return status


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--failed-evaluation", required=True)
    parser.add_argument("--validation-report", required=True)
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--policy-checkpoint", default=None)
    parser.add_argument("--learning-rate-scale", type=float, default=0.5)
    args = parser.parse_args(argv)
    result = prepare_recovery_continuation(
        source_run_dir=args.source_run_dir,
        config_path=args.config,
        failed_evaluation=args.failed_evaluation,
        validation_report=args.validation_report,
        baseline_report=args.baseline_report,
        policy_checkpoint=args.policy_checkpoint,
        learning_rate_scale=args.learning_rate_scale,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
