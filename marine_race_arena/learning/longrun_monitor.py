"""Per-update metrics, status, KL safeguards, evaluation, and safe stop callback."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np

from marine_race_arena.learning.longrun_checkpoint import (
    atomic_append_jsonl,
    atomic_copy_checkpoint,
    atomic_save_checkpoint,
    atomic_write_json,
    capture_model_training_state,
    checkpoint_passed_safety,
    full_evaluation_passed_safety,
    valid_checkpoints,
)
from marine_race_arena.learning.longrun_evaluation import (
    directional_metric_key,
    fast_reliable_metric_key,
    is_better,
    plateau_detected,
)
from marine_race_arena.learning.parametric_curriculum import (
    reliability_requirements_met,
)
from marine_race_arena.learning.longrun_rollback import (
    ROLLBACK_EVENT_SCHEMA_VERSION,
    enrich_rollbacks_with_curriculum,
    jsonl_rows_through,
    merge_curriculum_histories,
    merge_rollback_histories,
    rollback_status_fields,
)


class LongRunStopReason:
    NONE = None
    GRACEFUL = "GRACEFUL_STOP"
    ABSOLUTE_KL = "UNSAFE_ABSOLUTE_KL"
    REPEATED_HIGH_KL = "PAUSED_REPEATED_HIGH_KL"
    NUMERICAL = "NUMERICAL_FAILURE"
    DISK = "DISK_SPACE_GUARD"
    PLATEAU = "PLATEAU"
    REPEATED_REGRESSION = "REPEATED_FULL_EVALUATION_REGRESSION"


class StatusStore:
    def __init__(self, run_dir: str | Path, initial: Optional[Dict[str, Any]] = None):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "status.json"
        self.started_wall = time.time()
        self.data: Dict[str, Any] = {
            "schema_version": "multigate_longrun_status_v1",
            "state": "INITIALIZING",
            "pid": os.getpid(),
            "process_alive": True,
            "total_timesteps": 0,
            "curriculum_stage": "C0",
            "last_checkpoint": None,
            "best_checkpoint": None,
            "latest_evaluation": None,
            "latest_full_evaluation": None,
            "overall_completion": None,
            "left_success": None,
            "right_success": None,
            "straight_retention": None,
            "single_gate_retention": None,
            "three_gate_success": None,
            "approx_kl": None,
            "initial_policy_kl": None,
            "ppo_policy_loss": None,
            "bc_retention_loss": None,
            "combined_policy_loss": None,
            "retention_weight": None,
            "reward_phase": "legacy",
            "consecutive_reliable_full_evaluations": 0,
            "stage_entry_timesteps": 0,
            "steps_in_current_stage": 0,
            "next_promotion_eligibility_step": 0,
            "stage_minimum_remaining_timesteps": 0,
            "rollback_count": 0,
            "rollback_attempt_base": 0,
            "last_rollback_reason": None,
            "last_rollback_source": None,
            "last_rollback_timestep": None,
            "last_rollback_attempt": None,
            "last_rollback_outcome": None,
            "last_rollback_event": None,
            "rollback_history": [],
            "curriculum_stage_history": [],
            "collision_events": None,
            "collision_frames": None,
            "out_of_bounds_events": None,
            "out_of_bounds_frames": None,
            "wrong_direction_count": None,
            "safety_warning_events": None,
            "safety_warning_frames": None,
            "episodes_with_any_safety": None,
            "collision_episodes": None,
            "out_of_bounds_episodes": None,
            "wrong_direction_episodes": None,
            "previous_gate_returns": None,
            "mean_successful_time_s": None,
            "mean_penalized_time_s": None,
            "mean_action_jerk": None,
            "current_learning_rate": None,
            "best_reliable_checkpoint": None,
            "best_fast_reliable_checkpoint": None,
            "throughput_steps_per_s": None,
            "estimated_remaining_wall_s": None,
            "elapsed_wall_s": 0.0,
            "last_log_update_unix": time.time(),
            "detected_crash": None,
            "stall_detected": False,
            "simulator_restarts": 0,
            "nan_events": 0,
            "stop_reason": None,
        }
        self.data.update(initial or {})
        self.write()

    def update(self, **values: Any) -> None:
        self.data.update(values)
        self.data["elapsed_wall_s"] = round(time.time() - self.started_wall, 3)
        self.data["last_log_update_unix"] = time.time()
        self.write()

    def write(self) -> None:
        atomic_write_json(self.path, self.data)


def _evaluation_path(
    run_dir: str | Path, report: Optional[Mapping[str, Any]]
) -> Optional[str]:
    if not report:
        return None
    mode = report.get("mode")
    timesteps = report.get("timesteps")
    if mode not in {"light", "full"} or timesteps is None:
        return None
    return str(
        Path(run_dir)
        / "evaluations"
        / f"{mode}_{int(timesteps):09d}.json"
    )


def _future_trigger(current: int, frequency: int) -> int:
    return (int(current) // int(frequency) + 1) * int(frequency)


def _compact_stage_history(changes: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
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


def resume_status_snapshot(
    *,
    run_dir: str | Path,
    config: Any,
    sampler: Any,
    state: Mapping[str, Any],
    last_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a complete status image from one validated checkpoint sidecar."""
    current = int(state.get("total_timesteps", 0))
    evaluation = dict(state.get("evaluation", {}))
    history = list(evaluation.get("history", []))
    latest = dict(evaluation.get("last") or (history[-1] if history else {}))
    latest_full = next(
        (
            dict(report)
            for report in reversed(history)
            if report.get("mode") == "full"
        ),
        {},
    )
    report = latest or latest_full
    best = dict(evaluation.get("best", {}))
    extra = dict(state.get("extra", {}))
    model_training = dict(state.get("model_training", {}))
    optimizer_lrs = list(model_training.get("optimizer_learning_rates", []))
    minimum = int(
        config.curriculum.promotion.minimum_stage_timesteps.get(
            sampler.current_stage, 0
        )
    )
    stage_entry = int(sampler.state.stage_entry_timesteps)
    checkpoint_aliases = dict(extra.get("checkpoint_aliases", {}))
    rollback_history = merge_rollback_histories(
        extra.get("rollback_history", []),
        jsonl_rows_through(
            Path(run_dir) / "logs" / "rollback_history.jsonl", current
        ),
    )
    curriculum_history = merge_curriculum_histories(
        sampler.state.stage_changes,
        jsonl_rows_through(Path(run_dir) / "curriculum_history.jsonl", current),
        rollback_history=rollback_history,
    )
    rollback_history = enrich_rollbacks_with_curriculum(
        rollback_history, curriculum_history
    )
    rollback_fields = rollback_status_fields(rollback_history)
    rollback_fields["rollback_count"] = max(
        int(extra.get("rollback_count", 0)),
        int(rollback_fields["rollback_count"]),
    )
    if not rollback_history:
        rollback_fields["last_rollback_reason"] = extra.get(
            "last_rollback_reason"
        )
        rollback_fields["last_rollback_source"] = extra.get(
            "last_rollback_source"
        )
    if not checkpoint_aliases:
        checkpoint_aliases = {
            alias: selected.get("checkpoint")
            for alias, selected in best.items()
            if selected.get("checkpoint")
        }
    if last_checkpoint:
        checkpoint_aliases["last"] = str(last_checkpoint)
    return {
        "total_timesteps": current,
        "curriculum_stage": sampler.current_stage,
        "curriculum_stage_history": _compact_stage_history(
            curriculum_history
        ),
        "curriculum_evaluation_history_count": len(
            sampler.state.evaluation_history
        ),
        "evaluation_history_count": len(history),
        "last_checkpoint": last_checkpoint,
        "best_checkpoint": best.get("best_overall", {}).get("checkpoint"),
        "latest_evaluation": _evaluation_path(run_dir, latest),
        "latest_full_evaluation": _evaluation_path(run_dir, latest_full),
        "overall_completion": report.get("completion_rate"),
        "left_success": report.get("left_completion_rate"),
        "right_success": report.get("right_completion_rate"),
        "straight_retention": report.get("straight_completion_rate"),
        "single_gate_retention": report.get("single_gate_completion_rate"),
        "three_gate_success": report.get("three_gate_completion_rate"),
        "consecutive_reliable_full_evaluations": int(
            sampler.state.reliable_full_streak
        ),
        "stage_entry_timesteps": stage_entry,
        "steps_in_current_stage": current - stage_entry,
        "next_promotion_eligibility_step": stage_entry + minimum,
        "stage_minimum_remaining_timesteps": max(
            0, minimum - (current - stage_entry)
        ),
        "reward_phase": str(extra.get("reward_phase", "legacy")),
        "replay_mixture": list(sampler.active_mixture),
        "recovery_provenance": extra.get("recovery_provenance"),
        "rollback_attempt_base": int(
            extra.get("rollback_attempt_base", 0)
        ),
        **rollback_fields,
        "checkpoint_aliases": checkpoint_aliases,
        "collision_events": report.get("collision_events"),
        "collision_frames": report.get("collision_frames"),
        "out_of_bounds_events": report.get("out_of_bounds_events"),
        "out_of_bounds_frames": report.get("out_of_bounds_frames"),
        "wrong_direction_count": report.get("wrong_direction_count"),
        "safety_warning_events": report.get("safety_warning_events"),
        "safety_warning_frames": report.get("safety_warning_frames"),
        "episodes_with_any_safety": report.get("episodes_with_any_safety"),
        "collision_episodes": report.get("episodes_with_collision"),
        "out_of_bounds_episodes": report.get("episodes_with_out_of_bounds"),
        "wrong_direction_episodes": report.get("episodes_with_wrong_direction"),
        "previous_gate_returns": report.get("previous_gate_returns"),
        "mean_successful_time_s": report.get("mean_successful_time_s"),
        "mean_penalized_time_s": report.get("mean_penalized_time_s"),
        "mean_action_jerk": report.get("mean_action_jerk"),
        "current_learning_rate": (
            float(optimizer_lrs[0]) if optimizer_lrs else None
        ),
        "best_reliable_checkpoint": best.get("best_reliable", {}).get(
            "checkpoint"
        ),
        "best_fast_reliable_checkpoint": best.get(
            "best_fast_reliable", {}
        ).get("checkpoint"),
        "retention_state": model_training.get("retention_regularizer"),
        "previous_stop_reason": extra.get("stop_reason"),
    }


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def evaluation_regression_reasons(
    report: Mapping[str, Any],
    incumbent: Optional[Mapping[str, Any]],
    rollback: Any,
) -> List[str]:
    """Pure full-suite regression gate used before automatic restoration."""
    if not rollback.enabled or incumbent is None:
        return []
    reasons: List[str] = []
    if float(report.get("completion_rate", 0.0)) < (
        float(incumbent.get("completion_rate", 0.0)) - rollback.completion_drop
    ):
        reasons.append("overall_completion_drop")
    if float(report.get("single_gate_completion_rate", 0.0)) < rollback.single_gate_floor:
        reasons.append("single_gate_retention_floor")
    if float(report.get("straight_completion_rate", 0.0)) < rollback.straight_floor:
        reasons.append("straight_retention_floor")
    if min(
        float(report.get("left_completion_rate", 0.0)),
        float(report.get("right_completion_rate", 0.0)),
    ) < rollback.directional_floor:
        reasons.append("directional_completion_floor")
    if int(report.get("episodes_with_any_safety", 0)) > rollback.maximum_safety_episodes:
        reasons.append("safety_episode_regression")
    if int(report.get("previous_gate_returns", 0)) > rollback.maximum_previous_gate_returns:
        reasons.append("previous_gate_return_regression")
    return reasons


def rollback_target_stage(current_stage: str) -> str:
    return f"C{max(0, int(current_stage[1:]) - 1)}"


def rollback_attempt_allowed(attempt: int, maximum_attempts: int) -> bool:
    return 1 <= int(attempt) <= int(maximum_attempts)


def restore_ppo_training_state(model: Any, source: Any, learning_rate_scale: float) -> None:
    """Restore policy, optimizer and compatible schedule without rewinding steps."""
    current_timesteps = int(model.num_timesteps)
    model.policy.load_state_dict(source.policy.state_dict())
    model.policy.optimizer.load_state_dict(source.policy.optimizer.state_dict())
    schedule = getattr(model, "learning_rate", None)
    source_schedule = getattr(source, "learning_rate", None)
    if hasattr(schedule, "load_state_dict") and source_schedule is not None:
        schedule.load_state_dict(
            {
                "schema_version": "absolute_learning_rate_schedule_v1",
                "start": float(source_schedule.start),
                "end": float(source_schedule.end),
                "kind": str(source_schedule.kind),
                "multiplier": float(source_schedule.multiplier),
                "last_value": float(source_schedule.last_value),
            }
        )
    else:
        model.lr_schedule = source.lr_schedule
        schedule = getattr(
            getattr(model, "lr_schedule", None), "value_schedule", None
        ) or getattr(model, "lr_schedule", None)
    if hasattr(schedule, "scale"):
        schedule.scale(learning_rate_scale)
    model.num_timesteps = current_timesteps


def learning_rate_snapshot(model: Any) -> Dict[str, Any]:
    """Capture the optimizer and absolute-schedule LR view without mutation."""
    optimizer = getattr(getattr(model, "policy", None), "optimizer", None)
    optimizer_lrs = (
        [float(group["lr"]) for group in optimizer.param_groups]
        if optimizer is not None
        else []
    )
    schedule_state: Dict[str, Any] = {}
    seen = set()
    for schedule in (
        getattr(model, "learning_rate", None),
        getattr(model, "lr_schedule", None),
        getattr(getattr(model, "lr_schedule", None), "value_schedule", None),
    ):
        if schedule is None or id(schedule) in seen:
            continue
        seen.add(id(schedule))
        if hasattr(schedule, "state_dict"):
            candidate = schedule.state_dict()
            if isinstance(candidate, dict):
                schedule_state = dict(candidate)
                break
        if hasattr(schedule, "last_value"):
            schedule_state = {
                "multiplier": float(getattr(schedule, "multiplier", 1.0)),
                "last_value": float(schedule.last_value),
            }
            break
    return {
        "optimizer_learning_rates": optimizer_lrs,
        "schedule_multiplier": schedule_state.get("multiplier"),
        "schedule_value": schedule_state.get("last_value"),
    }


class UpdateMetricsRecorder:
    FIELDS = (
        "num_timesteps",
        "n_updates",
        "approx_kl",
        "clip_fraction",
        "policy_loss",
        "ppo_policy_loss",
        "bc_retention_loss",
        "initial_policy_kl",
        "combined_policy_loss",
        "retention_weight",
        "value_loss",
        "entropy",
        "explained_variance",
        "action_std",
        "gradient_norm",
        "action_saturation",
        "completion_rate",
        "mean_gate_count",
        "episode_count",
        "learning_rate",
        "n_epochs",
    )

    def __init__(self, model: Any, run_dir: str | Path):
        self.model = model
        self.run_dir = Path(run_dir)
        self.csv_path = self.run_dir / "logs" / "update_metrics.csv"
        self.jsonl_path = self.run_dir / "update_metrics.jsonl"
        self.rows: List[Dict[str, Any]] = []
        self.last_timestep = -1
        self.episode_rows: List[Dict[str, Any]] = []
        self.reward_sums: Dict[str, float] = {}
        self.reward_samples = 0
        maximum_timestep = int(getattr(model, "num_timesteps", 0))
        self._quarantine_future_jsonl_rows(maximum_timestep)
        if self.csv_path.exists():
            try:
                with self.csv_path.open(newline="", encoding="utf-8") as handle:
                    all_rows = list(csv.DictReader(handle))
                self.rows = [
                    row
                    for row in all_rows
                    if int(float(row["num_timesteps"])) <= maximum_timestep
                ]
                if len(self.rows) != len(all_rows):
                    self._write_csv()
                if self.rows:
                    self.last_timestep = int(float(self.rows[-1]["num_timesteps"]))
            except Exception:
                self.rows = []

    def _quarantine_future_jsonl_rows(self, maximum_timestep: int) -> None:
        if not self.jsonl_path.exists():
            return
        kept_lines: List[str] = []
        future_rows: List[Dict[str, Any]] = []
        for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if int(row["num_timesteps"]) > maximum_timestep:
                    future_rows.append(row)
                else:
                    kept_lines.append(line)
            except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                kept_lines.append(line)
        if not future_rows:
            return
        recovery_path = (
            self.run_dir
            / "recovery"
            / f"update_metrics_after_{maximum_timestep}_{int(time.time())}.json"
        )
        atomic_write_json(
            recovery_path,
            {
                "resume_checkpoint_timesteps": maximum_timestep,
                "source": str(self.jsonl_path),
                "rows": future_rows,
            },
        )
        tmp = self.jsonl_path.with_suffix(self.jsonl_path.suffix + ".tmp")
        text = "\n".join(kept_lines)
        tmp.write_text(text + ("\n" if text else ""), encoding="utf-8")
        os.replace(tmp, self.jsonl_path)

    def observe_infos(self, infos: List[Mapping[str, Any]]) -> None:
        for info in infos:
            components = info.get("reward_components", {})
            if components:
                self.reward_samples += 1
                for name, value in components.items():
                    self.reward_sums[name] = self.reward_sums.get(name, 0.0) + float(
                        value
                    )
            if "episode_finished" in info:
                self.episode_rows.append(
                    {
                        "finished": bool(info.get("episode_finished")),
                        "gate_count": int(info.get("episode_gate_count", 0)),
                        "direction": info.get("transition_direction"),
                        "collisions": int(info.get("episode_collisions", 0)),
                        "collision_frames": int(
                            info.get("episode_collision_frames", 0)
                        ),
                        "out_of_bounds": int(info.get("episode_out_of_bounds", 0)),
                        "out_of_bounds_frames": int(
                            info.get("episode_out_of_bounds_frames", 0)
                        ),
                        "wrong_direction": int(info.get("episode_wrong_direction", 0)),
                        "warning_events": int(
                            info.get("episode_safety_warning_events", 0)
                        ),
                        "warning_frames": int(
                            info.get("episode_safety_warning_frames", 0)
                        ),
                    }
                )

    def _logger(self, key: str) -> Optional[float]:
        return _finite_or_none(self.model.logger.name_to_value.get(key))

    def _action_std(self) -> Optional[float]:
        try:
            return float(
                np.mean(np.exp(self.model.policy.log_std.detach().cpu().numpy()))
            )
        except Exception:
            return None

    def _gradient_norm(self) -> Optional[float]:
        try:
            squares = []
            for parameter in self.model.policy.parameters():
                if parameter.grad is not None:
                    value = parameter.grad.detach().norm(2).item()
                    squares.append(float(value) ** 2)
            return math.sqrt(sum(squares)) if squares else 0.0
        except Exception:
            return None

    def _action_saturation(self) -> Optional[float]:
        try:
            actions = np.asarray(self.model.rollout_buffer.actions)
            return float(np.mean(np.abs(actions) > 0.98))
        except Exception:
            return None

    def record(self) -> Optional[Dict[str, Any]]:
        approx_kl = self._logger("train/approx_kl")
        timestep = int(getattr(self.model, "num_timesteps", 0))
        if approx_kl is None or timestep == self.last_timestep:
            return None
        self.last_timestep = timestep
        completed = sum(row["finished"] for row in self.episode_rows)
        episode_count = len(self.episode_rows)
        row: Dict[str, Any] = {
            "num_timesteps": timestep,
            "n_updates": int(getattr(self.model, "_n_updates", len(self.rows) + 1)),
            "approx_kl": approx_kl,
            "clip_fraction": self._logger("train/clip_fraction"),
            "policy_loss": self._logger("train/policy_gradient_loss"),
            "ppo_policy_loss": self._logger("train/ppo_policy_loss"),
            "bc_retention_loss": self._logger("train/bc_retention_loss"),
            "initial_policy_kl": self._logger("train/initial_policy_kl"),
            "combined_policy_loss": self._logger("train/combined_policy_loss"),
            "retention_weight": self._logger("train/retention_weight"),
            "value_loss": self._logger("train/value_loss"),
            "entropy": (
                abs(float(self._logger("train/entropy_loss")))
                if self._logger("train/entropy_loss") is not None
                else None
            ),
            "explained_variance": self._logger("train/explained_variance"),
            "action_std": self._action_std(),
            "gradient_norm": self._gradient_norm(),
            "action_saturation": self._action_saturation(),
            "completion_rate": completed / episode_count if episode_count else None,
            "mean_gate_count": (
                float(np.mean([value["gate_count"] for value in self.episode_rows]))
                if episode_count
                else None
            ),
            "episode_count": episode_count,
            "learning_rate": float(
                self.model.policy.optimizer.param_groups[0]["lr"]
            ),
            "n_epochs": int(self.model.n_epochs),
            "reward_components": {
                key: value / max(1, self.reward_samples)
                for key, value in sorted(self.reward_sums.items())
            },
            "directional_training": {
                direction: {
                    "episodes": sum(
                        item["direction"] == direction
                        for item in self.episode_rows
                    ),
                    "successes": sum(
                        item["direction"] == direction and item["finished"]
                        for item in self.episode_rows
                    ),
                }
                for direction in ("straight", "left", "right")
            },
        }
        self.rows.append(row)
        self._write_csv()
        atomic_append_jsonl(self.jsonl_path, row)
        self.episode_rows = []
        self.reward_sums = {}
        self.reward_samples = 0
        return row

    def _write_csv(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.csv_path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({name: row.get(name) for name in self.FIELDS})
        os.replace(tmp, self.csv_path)


def make_longrun_callback(
    *,
    run_dir: str | Path,
    config: Any,
    sampler: Any,
    status_store: StatusStore,
    contract_hash: str,
    evaluate_fn: Callable[[Any, str, int], Dict[str, Any]],
    reward_config: Optional[Any] = None,
):
    from stable_baselines3.common.callbacks import BaseCallback

    class LongRunCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self.run_dir = Path(run_dir)
            self.metrics: Optional[UpdateMetricsRecorder] = None
            self.eval_history: List[Dict[str, Any]] = []
            self.best: Dict[str, Dict[str, Any]] = {}
            self.last_eval: Optional[Dict[str, Any]] = None
            self.last_checkpoint = None
            self.stop_reason = LongRunStopReason.NONE
            self.low_kl_count = 0
            self.high_kl_count = 0
            self.automatic_changes: List[Dict[str, Any]] = []
            self.rollback_count = 0
            self.rollback_attempt_base = 0
            self.last_rollback_reason: Optional[List[str]] = None
            self.last_rollback_source: Optional[str] = None
            self.rollback_history: List[Dict[str, Any]] = []
            self.recovery_provenance: Optional[Dict[str, Any]] = None
            self.safety_validation: Optional[Dict[str, Any]] = None
            self._rollback_this_update = False
            self._last_status_wall = 0.0
            self._started_wall = time.time()
            self._started_timesteps = 0
            self._next_checkpoint = config.evaluation.checkpoint_frequency
            self._next_light_eval = config.evaluation.light_frequency
            self._next_full_eval = config.evaluation.full_frequency

        def restore_pipeline_state(
            self,
            state: Mapping[str, Any],
            *,
            resume_checkpoint: Optional[Any] = None,
        ) -> None:
            evaluation = dict(state.get("evaluation", {}))
            self.eval_history = list(evaluation.get("history", []))
            self.best = dict(evaluation.get("best", {}))
            self.last_eval = evaluation.get("last")
            if self.last_eval is None and self.eval_history:
                self.last_eval = dict(self.eval_history[-1])
            extra = dict(state.get("extra", {}))
            self.automatic_changes = list(
                extra.get("automatic_changes", [])
            )
            current = int(state.get("total_timesteps", 0))
            self.rollback_history = merge_rollback_histories(
                extra.get("rollback_history", []),
                jsonl_rows_through(
                    self.run_dir / "logs" / "rollback_history.jsonl",
                    current,
                ),
            )
            restored_rollback = rollback_status_fields(
                self.rollback_history
            )
            self.rollback_count = max(
                int(extra.get("rollback_count", 0)),
                int(restored_rollback["rollback_count"]),
            )
            self.rollback_attempt_base = int(
                extra.get("rollback_attempt_base", 0)
            )
            self.last_rollback_reason = (
                restored_rollback["last_rollback_reason"]
                if self.rollback_history
                else extra.get("last_rollback_reason")
            )
            self.last_rollback_source = (
                restored_rollback["last_rollback_source"]
                if self.rollback_history
                else extra.get("last_rollback_source")
            )
            self.low_kl_count = int(extra.get("low_kl_count", 0))
            self.high_kl_count = int(extra.get("high_kl_count", 0))
            self.recovery_provenance = extra.get("recovery_provenance")
            self.safety_validation = extra.get("safety_validation")
            if reward_config is not None:
                reward_config.set_phase(
                    str(extra.get("reward_phase", reward_config.reward_phase))
                )
            self._next_checkpoint = int(
                extra.get(
                    "next_checkpoint",
                    _future_trigger(
                        current, config.evaluation.checkpoint_frequency
                    ),
                )
            )
            self._next_light_eval = int(
                extra.get(
                    "next_light_evaluation",
                    _future_trigger(current, config.evaluation.light_frequency),
                )
            )
            self._next_full_eval = int(
                extra.get(
                    "next_full_evaluation",
                    _future_trigger(current, config.evaluation.full_frequency),
                )
            )
            self.last_checkpoint = resume_checkpoint
            if resume_checkpoint is not None:
                self._restore_checkpoint_aliases(resume_checkpoint)
            status_store.update(
                **resume_status_snapshot(
                    run_dir=self.run_dir,
                    config=config,
                    sampler=sampler,
                    state=state,
                    last_checkpoint=(
                        str(resume_checkpoint.model_path)
                        if resume_checkpoint is not None
                        else None
                    ),
                )
            )

        def _restore_checkpoint_aliases(self, resume_checkpoint: Any) -> None:
            """Re-materialize every alias from the checkpoint-selection state."""
            checkpoint_rows = list(
                valid_checkpoints(
                    self.run_dir,
                    expected_contract_hash=contract_hash,
                )
            )
            available = {
                checkpoint.model_path.name: checkpoint
                for checkpoint in checkpoint_rows
            }
            atomic_copy_checkpoint(
                resume_checkpoint, self.run_dir / "best_models" / "last.zip"
            )
            safe_rows = [
                checkpoint
                for checkpoint in checkpoint_rows
                if checkpoint_passed_safety(checkpoint)
            ]
            if safe_rows:
                atomic_copy_checkpoint(
                    safe_rows[-1],
                    self.run_dir / "best_models" / "latest_safe.zip",
                )
            for alias, selected in self.best.items():
                source = selected.get("checkpoint")
                checkpoint = available.get(Path(source).name) if source else None
                if checkpoint is not None:
                    atomic_copy_checkpoint(
                        checkpoint,
                        self.run_dir / "best_models" / f"{alias}.zip",
                    )

        def _init_callback(self) -> None:
            self.metrics = UpdateMetricsRecorder(self.model, self.run_dir)
            self._started_timesteps = int(self.model.num_timesteps)
            status_store.update(
                state="RUNNING",
                total_timesteps=int(self.model.num_timesteps),
                curriculum_stage=sampler.current_stage,
                reward_phase=(
                    reward_config.reward_phase if reward_config is not None else "legacy"
                ),
            )

        def _on_step(self) -> bool:
            assert self.metrics is not None
            self.metrics.observe_infos(self.locals.get("infos", []) or [])
            sampler.set_timesteps(int(self.num_timesteps))
            now = time.time()
            if (
                self.n_calls % config.reliability.status_frequency_steps == 0
                or now - self._last_status_wall
                >= config.reliability.status_frequency_seconds
            ):
                self._last_status_wall = now
                elapsed = max(1e-6, now - self._started_wall)
                advanced = max(
                    0, int(self.num_timesteps) - self._started_timesteps
                )
                throughput = advanced / elapsed
                target = int(
                    status_store.data.get(
                        "target_total_timesteps", self.num_timesteps
                    )
                )
                remaining = max(0, target - int(self.num_timesteps))
                status_store.update(
                    total_timesteps=int(self.num_timesteps),
                    curriculum_stage=sampler.current_stage,
                    state="STOPPING" if self.stop_reason else "RUNNING",
                    replay_mixture=list(sampler.active_mixture),
                    throughput_steps_per_s=throughput,
                    estimated_remaining_wall_s=(
                        remaining / throughput if throughput > 0 else None
                    ),
                )
            if (self.run_dir / "stop.requested").exists():
                self.stop_reason = LongRunStopReason.GRACEFUL
            free_gb = shutil.disk_usage(self.run_dir).free / (1024**3)
            if free_gb < config.reliability.minimum_free_disk_gb:
                self.stop_reason = LongRunStopReason.DISK
            return self.stop_reason is None

        def _on_rollout_start(self) -> None:
            self._after_update()

        def _after_update(self) -> None:
            assert self.metrics is not None
            self._rollback_this_update = False
            row = self.metrics.record()
            if row is not None:
                self._apply_kl_policy(row)
                status_store.update(
                    total_timesteps=int(self.model.num_timesteps),
                    approx_kl=row["approx_kl"],
                    ppo_policy_loss=row.get("ppo_policy_loss"),
                    bc_retention_loss=row.get("bc_retention_loss"),
                    initial_policy_kl=row.get("initial_policy_kl"),
                    combined_policy_loss=row.get("combined_policy_loss"),
                    retention_weight=row.get("retention_weight"),
                    current_learning_rate=row.get("learning_rate"),
                    curriculum_stage=sampler.current_stage,
                )
            current = int(self.model.num_timesteps)
            if current >= self._next_full_eval:
                self.last_eval = evaluate_fn(self.model, "full", current)
                self._record_evaluation(self.last_eval)
                while self._next_full_eval <= current:
                    self._next_full_eval += config.evaluation.full_frequency
                while self._next_light_eval <= current:
                    self._next_light_eval += config.evaluation.light_frequency
            elif current >= self._next_light_eval:
                self.last_eval = evaluate_fn(self.model, "light", current)
                self._record_evaluation(self.last_eval)
                while self._next_light_eval <= current:
                    self._next_light_eval += config.evaluation.light_frequency
            if (
                current >= self._next_checkpoint
                and (
                    self.last_checkpoint is None
                    or self.last_checkpoint.timesteps != current
                )
                and not self._rollback_this_update
                and self.stop_reason
                not in {
                    LongRunStopReason.ABSOLUTE_KL,
                    LongRunStopReason.NUMERICAL,
                }
            ):
                self._save_checkpoint(reason="periodic", safe=False)
                while self._next_checkpoint <= current:
                    self._next_checkpoint += config.evaluation.checkpoint_frequency

        def _apply_kl_policy(self, row: Mapping[str, Any]) -> None:
            approx_kl = float(row["approx_kl"])
            if not math.isfinite(approx_kl):
                self.stop_reason = LongRunStopReason.NUMERICAL
                return
            if approx_kl > config.ppo.absolute_kl_stop:
                self.stop_reason = LongRunStopReason.ABSOLUTE_KL
                self._save_checkpoint(reason="absolute_kl", safe=False)
                return
            if approx_kl > config.ppo.high_kl:
                self.high_kl_count += 1
                self.low_kl_count = 0
                if self.high_kl_count == 1:
                    schedule = getattr(self.model, "lr_schedule", None)
                    if hasattr(schedule, "scale"):
                        schedule.scale(0.5)
                    self.model.n_epochs = max(1, int(self.model.n_epochs) - 1)
                    self._log_adaptation(
                        "high_kl_reduction",
                        row,
                        {"learning_rate_scale": 0.5, "n_epochs": self.model.n_epochs},
                    )
                else:
                    self.stop_reason = LongRunStopReason.REPEATED_HIGH_KL
                return
            self.high_kl_count = 0
            if approx_kl < config.ppo.low_kl_threshold:
                self.low_kl_count += 1
            else:
                self.low_kl_count = 0
            if (
                self.low_kl_count >= config.ppo.low_kl_updates
                and len(self.automatic_changes) < config.ppo.max_automatic_changes
                and self._evaluation_flat()
            ):
                schedule = getattr(self.model, "lr_schedule", None)
                if hasattr(schedule, "scale"):
                    schedule.scale(1.25)
                    self._log_adaptation(
                        "low_kl_flat_eval_increase",
                        row,
                        {"learning_rate_scale": 1.25},
                    )
                    self.low_kl_count = 0

        def _evaluation_flat(self) -> bool:
            if len(self.eval_history) < 3:
                return False
            recent = self.eval_history[-3:]
            return (
                max(float(row.get("completion_rate", 0.0)) for row in recent)
                <= float(recent[0].get("completion_rate", 0.0))
                and max(float(row.get("mean_gates", 0.0)) for row in recent)
                <= float(recent[0].get("mean_gates", 0.0))
            )

        def _log_adaptation(
            self, kind: str, row: Mapping[str, Any], change: Mapping[str, Any]
        ) -> None:
            event = {
                "timesteps": int(self.model.num_timesteps),
                "kind": kind,
                "measured_approx_kl": row.get("approx_kl"),
                "change": dict(change),
            }
            self.automatic_changes.append(event)
            atomic_append_jsonl(
                self.run_dir / "logs" / "automatic_changes.jsonl", event
            )

        def _record_evaluation(self, report: Dict[str, Any]) -> None:
            latest_update = self.metrics.rows[-1] if self.metrics and self.metrics.rows else {}
            compact = {
                key: value
                for key, value in report.items()
                if key not in {"rows", "reward_component_sums", "category_metrics"}
            }
            compact["approx_kl"] = latest_update.get("approx_kl")
            self.eval_history.append(compact)
            if report.get("mode") == "full":
                previous_stage = sampler.current_stage
                change = sampler.record_evaluation(
                    report,
                    timesteps=int(self.model.num_timesteps),
                    auto_promote=config.curriculum.auto_promote,
                    allow_demotion=config.curriculum.allow_demotion,
                )
                if change:
                    atomic_append_jsonl(
                        self.run_dir / "curriculum_history.jsonl", change
                    )
                phase_changed = self._update_reward_phase()
                regression_reasons = self._regression_reasons(report)
                if regression_reasons:
                    self._save_checkpoint(
                        reason="regressed_full_evaluation", safe=False
                    )
                    self._perform_rollback(regression_reasons)
                else:
                    self._save_checkpoint(
                        reason="full_evaluation",
                        safe=full_evaluation_passed_safety(
                            report,
                            minimum_cases=config.evaluation.full_episodes,
                        ),
                    )
                    if change or phase_changed:
                        self._reset_training_episode()
                plateau = plateau_detected(
                    self.eval_history,
                    current_timesteps=int(self.model.num_timesteps),
                    plateau_steps=config.evaluation.plateau_steps,
                    min_evaluations=config.evaluation.plateau_min_evaluations,
                )
                if plateau and not self._rollback_this_update:
                    atomic_write_json(
                        self.run_dir / "logs" / "plateau_recommendation.json",
                        plateau,
                    )
                    self.stop_reason = LongRunStopReason.PLATEAU
            rollback_fields = rollback_status_fields(self.rollback_history)
            model_training = capture_model_training_state(self.model)
            lr_state = learning_rate_snapshot(self.model)
            current_learning_rate = lr_state.get("schedule_value")
            if current_learning_rate is None:
                optimizer_lrs = lr_state.get("optimizer_learning_rates", [])
                current_learning_rate = (
                    optimizer_lrs[0] if optimizer_lrs else None
                )
            status_store.update(
                latest_evaluation=str(
                    self.run_dir
                    / "evaluations"
                    / f"{report['mode']}_{int(self.model.num_timesteps):09d}.json"
                ),
                latest_full_evaluation=(
                    str(
                        self.run_dir
                        / "evaluations"
                        / f"full_{int(self.model.num_timesteps):09d}.json"
                    )
                    if report.get("mode") == "full"
                    else status_store.data.get("latest_full_evaluation")
                ),
                overall_completion=report.get("completion_rate"),
                left_success=report.get("left_completion_rate"),
                right_success=report.get("right_completion_rate"),
                straight_retention=report.get("straight_completion_rate"),
                single_gate_retention=report.get("single_gate_completion_rate"),
                three_gate_success=report.get("three_gate_completion_rate"),
                curriculum_stage=sampler.current_stage,
                curriculum_stage_history=_compact_stage_history(
                    list(sampler.state.stage_changes)
                ),
                curriculum_evaluation_history_count=len(
                    sampler.state.evaluation_history
                ),
                consecutive_reliable_full_evaluations=(
                    sampler.state.reliable_full_streak
                ),
                stage_entry_timesteps=sampler.state.stage_entry_timesteps,
                steps_in_current_stage=(
                    int(self.model.num_timesteps)
                    - sampler.state.stage_entry_timesteps
                ),
                next_promotion_eligibility_step=(
                    sampler.state.stage_entry_timesteps
                    + int(
                        config.curriculum.promotion.minimum_stage_timesteps.get(
                            sampler.current_stage, 0
                        )
                    )
                ),
                stage_minimum_remaining_timesteps=max(
                    0,
                    int(
                        config.curriculum.promotion.minimum_stage_timesteps.get(
                            sampler.current_stage, 0
                        )
                    )
                    - (
                        int(self.model.num_timesteps)
                        - sampler.state.stage_entry_timesteps
                    ),
                ),
                reward_phase=(
                    reward_config.reward_phase if reward_config is not None else "legacy"
                ),
                **rollback_fields,
                current_learning_rate=current_learning_rate,
                retention_state=model_training.get("retention_regularizer"),
                collision_events=report.get("collision_events"),
                collision_frames=report.get("collision_frames"),
                out_of_bounds_events=report.get("out_of_bounds_events"),
                out_of_bounds_frames=report.get("out_of_bounds_frames"),
                wrong_direction_count=report.get("wrong_direction_count"),
                safety_warning_events=report.get("safety_warning_events"),
                safety_warning_frames=report.get("safety_warning_frames"),
                episodes_with_any_safety=report.get(
                    "episodes_with_any_safety"
                ),
                collision_episodes=report.get("episodes_with_collision"),
                out_of_bounds_episodes=report.get(
                    "episodes_with_out_of_bounds"
                ),
                wrong_direction_episodes=report.get(
                    "episodes_with_wrong_direction"
                ),
                previous_gate_returns=report.get("previous_gate_returns"),
                mean_successful_time_s=report.get("mean_successful_time_s"),
                mean_penalized_time_s=report.get("mean_penalized_time_s"),
                mean_action_jerk=report.get("mean_action_jerk"),
                best_reliable_checkpoint=self.best.get(
                    "best_reliable", {}
                ).get("checkpoint"),
                best_fast_reliable_checkpoint=self.best.get(
                    "best_fast_reliable", {}
                ).get("checkpoint"),
            )

        def _update_reward_phase(self) -> bool:
            if reward_config is None or not config.reliability.reward_phase.enabled:
                return False
            required = config.reliability.reward_phase.reliable_full_evaluations
            desired = (
                "efficiency"
                if sampler.state.reliable_full_streak >= required
                else "reliability"
            )
            if reward_config.reward_phase == desired:
                return False
            reward_config.set_phase(desired)
            atomic_append_jsonl(
                self.run_dir / "logs" / "reward_phase_history.jsonl",
                {
                    "timesteps": int(self.model.num_timesteps),
                    "phase": desired,
                    "reliable_full_streak": sampler.state.reliable_full_streak,
                },
            )
            return True

        def _regression_reasons(self, report: Mapping[str, Any]) -> List[str]:
            return evaluation_regression_reasons(
                report,
                self.best.get("best_reliable"),
                config.reliability.rollback,
            )

        def _perform_rollback(self, reasons: List[str]) -> None:
            rollback = config.reliability.rollback
            self.rollback_count += 1
            recovery_attempt = (
                self.rollback_count - self.rollback_attempt_base
            )
            self.last_rollback_reason = list(reasons)
            incumbent = self.best.get("best_reliable", {})
            source_path = incumbent.get("checkpoint")
            self.last_rollback_source = str(source_path) if source_path else None
            current_timesteps = int(self.model.num_timesteps)
            stage_before = sampler.current_stage
            regularizer = getattr(self.model, "_retention_regularizer", None)
            retention_multiplier_before = (
                float(regularizer.weight_multiplier)
                if regularizer is not None
                and hasattr(regularizer, "weight_multiplier")
                else None
            )
            learning_rate_before = learning_rate_snapshot(self.model)
            event = {
                "schema_version": ROLLBACK_EVENT_SCHEMA_VERSION,
                "timesteps": current_timesteps,
                "attempt": self.rollback_count,
                "recovery_attempt": recovery_attempt,
                "reason": (
                    reasons[0] if len(reasons) == 1 else "+".join(reasons)
                ),
                "reasons": list(reasons),
                "source": self.last_rollback_source,
                "source_checkpoint": self.last_rollback_source,
                "stage_before": stage_before,
                "retention_multiplier_before": retention_multiplier_before,
                "learning_rate_before": learning_rate_before,
            }
            if (
                not rollback_attempt_allowed(
                    recovery_attempt, rollback.maximum_attempts
                )
                or not source_path
            ):
                event.update(
                    {
                        "outcome": "stopped",
                        "stage_after": stage_before,
                        "stage": stage_before,
                        "retention_multiplier_after": (
                            retention_multiplier_before
                        ),
                        "learning_rate_after": learning_rate_before,
                        "curriculum_transition": None,
                    }
                )
                self.stop_reason = LongRunStopReason.REPEATED_REGRESSION
                self.rollback_history.append(dict(event))
                atomic_append_jsonl(
                    self.run_dir / "logs" / "rollback_history.jsonl", event
                )
                self._publish_rollback_status()
                return
            source = type(self.model).load(str(source_path), device="cpu")
            restore_ppo_training_state(
                self.model, source, rollback.learning_rate_scale
            )
            if regularizer is not None:
                event["retention_weight_after"] = regularizer.increase_weight(
                    rollback.retention_weight_scale
                )
            retention_multiplier_after = (
                float(regularizer.weight_multiplier)
                if regularizer is not None
                and hasattr(regularizer, "weight_multiplier")
                else retention_multiplier_before
            )
            target_stage = rollback_target_stage(sampler.current_stage)
            change = sampler.force_stage(
                target_stage,
                timesteps=current_timesteps,
                reason="automatic_full_evaluation_rollback",
            )
            if reward_config is not None:
                reward_config.set_phase("reliability")
            self._reset_training_episode()
            self._rollback_this_update = True
            self.last_eval = None
            event.update(
                {
                    "outcome": "restored",
                    "stage_after": target_stage,
                    "stage": target_stage,
                    "retention_multiplier_after": (
                        retention_multiplier_after
                    ),
                    "learning_rate_after": learning_rate_snapshot(self.model),
                    "curriculum_transition": dict(change),
                }
            )
            self.rollback_history.append(dict(event))
            atomic_append_jsonl(
                self.run_dir / "logs" / "rollback_history.jsonl", event
            )
            atomic_append_jsonl(
                self.run_dir / "curriculum_history.jsonl", change
            )
            self._publish_rollback_status()

        def _publish_rollback_status(self) -> None:
            rollback_fields = rollback_status_fields(self.rollback_history)
            model_training = capture_model_training_state(self.model)
            lr_state = learning_rate_snapshot(self.model)
            current_learning_rate = lr_state.get("schedule_value")
            if current_learning_rate is None:
                optimizer_lrs = lr_state.get("optimizer_learning_rates", [])
                current_learning_rate = (
                    optimizer_lrs[0] if optimizer_lrs else None
                )
            minimum = int(
                config.curriculum.promotion.minimum_stage_timesteps.get(
                    sampler.current_stage, 0
                )
            )
            current = int(self.model.num_timesteps)
            stage_entry = int(sampler.state.stage_entry_timesteps)
            status_store.update(
                **rollback_fields,
                curriculum_stage=sampler.current_stage,
                curriculum_stage_history=_compact_stage_history(
                    list(sampler.state.stage_changes)
                ),
                stage_entry_timesteps=stage_entry,
                steps_in_current_stage=current - stage_entry,
                next_promotion_eligibility_step=stage_entry + minimum,
                stage_minimum_remaining_timesteps=max(
                    0, minimum - (current - stage_entry)
                ),
                reward_phase=(
                    reward_config.reward_phase
                    if reward_config is not None
                    else "legacy"
                ),
                current_learning_rate=current_learning_rate,
                retention_state=model_training.get("retention_regularizer"),
            )

        def _reset_training_episode(self) -> None:
            try:
                self.model._last_obs = self.model.env.reset()
                self.model._last_episode_starts = np.ones(
                    (self.model.n_envs,), dtype=bool
                )
            except Exception as exc:
                self.stop_reason = LongRunStopReason.NUMERICAL
                atomic_append_jsonl(
                    self.run_dir / "logs" / "automatic_changes.jsonl",
                    {
                        "timesteps": int(self.model.num_timesteps),
                        "kind": "training_episode_reset_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )

        def _selection_state(
            self, checkpoint_path: Path, *, safe: bool
        ) -> tuple[Dict[str, Dict[str, Any]], List[str]]:
            proposed = dict(self.best)
            changed: List[str] = []
            if not (
                safe and self.last_eval and self.last_eval.get("mode") == "full"
            ):
                return proposed, changed
            compact = {
                key: value
                for key, value in self.last_eval.items()
                if key not in {"rows", "reward_component_sums"}
            }
            aliases = (
                ("best_overall", None),
                ("best_left", "left"),
                ("best_right", "right"),
                ("best_three_gate", "three_gate"),
            )
            for alias, direction in aliases:
                previous = proposed.get(alias)
                if direction == "three_gate":
                    three_gate_n = int(
                        self.last_eval.get("category_metrics", {})
                        .get("three_gate", {})
                        .get("n", 0)
                    )
                    if three_gate_n <= 0:
                        continue
                    better = previous is None or float(
                        self.last_eval.get("three_gate_completion_rate", 0.0)
                    ) > float(previous.get("three_gate_completion_rate", 0.0))
                else:
                    better = is_better(
                        self.last_eval, previous, direction=direction
                    )
                if better:
                    proposed[alias] = {
                        **compact,
                        "checkpoint": str(checkpoint_path),
                    }
                    changed.append(alias)
            reliable = reliability_requirements_met(
                self.last_eval, config.curriculum.promotion
            )
            if reliable:
                reliable_previous = proposed.get("best_reliable")
                if is_better(self.last_eval, reliable_previous):
                    proposed["best_reliable"] = {
                        **compact,
                        "checkpoint": str(checkpoint_path),
                    }
                    changed.append("best_reliable")
                fast_previous = proposed.get("best_fast_reliable")
                fast_key = fast_reliable_metric_key(
                    self.last_eval, config.curriculum.promotion
                )
                previous_key = (
                    fast_reliable_metric_key(
                        fast_previous, config.curriculum.promotion
                    )
                    if fast_previous
                    else None
                )
                if previous_key is None or fast_key > previous_key:
                    proposed["best_fast_reliable"] = {
                        **compact,
                        "checkpoint": str(checkpoint_path),
                    }
                    changed.append("best_fast_reliable")
            return proposed, changed

        def _evaluation_state(
            self, best: Optional[Dict[str, Dict[str, Any]]] = None
        ) -> Dict[str, Any]:
            return {
                "history": self.eval_history,
                "best": self.best if best is None else best,
                "last": self.last_eval,
            }

        def _save_checkpoint(self, *, reason: str, safe: bool = False):
            current = int(self.model.num_timesteps)
            safe = bool(
                safe
                and isinstance(self.last_eval, dict)
                and int(self.last_eval.get("timesteps", -1)) == current
                and full_evaluation_passed_safety(
                    self.last_eval,
                    minimum_cases=config.evaluation.full_episodes,
                )
            )
            checkpoint_path = (
                self.run_dir / "checkpoints" / f"ppo_{current}_steps.zip"
            )
            proposed_best, changed_aliases = self._selection_state(
                checkpoint_path, safe=safe
            )
            checkpoint_aliases = dict(
                status_store.data.get("checkpoint_aliases", {})
            )
            checkpoint_aliases.update(
                {
                    alias: selected.get("checkpoint")
                    for alias, selected in proposed_best.items()
                    if selected.get("checkpoint")
                }
            )
            checkpoint_aliases["last"] = str(checkpoint_path)
            if safe:
                checkpoint_aliases["latest_safe"] = str(checkpoint_path)
            rollback_fields = rollback_status_fields(self.rollback_history)
            rollback_fields["rollback_count"] = max(
                self.rollback_count, int(rollback_fields["rollback_count"])
            )
            if not self.rollback_history:
                rollback_fields["last_rollback_reason"] = (
                    self.last_rollback_reason
                )
                rollback_fields["last_rollback_source"] = (
                    self.last_rollback_source
                )
            checkpoint = atomic_save_checkpoint(
                self.model,
                self.run_dir,
                total_timesteps=current,
                config_contract_hash=contract_hash,
                curriculum_state=sampler.state_dict(),
                evaluation_state=self._evaluation_state(proposed_best),
                status="safe" if safe else "unsafe",
                reason=reason,
                extra_state={
                    "automatic_changes": self.automatic_changes,
                    "stop_reason": self.stop_reason,
                    "recovery_provenance": self.recovery_provenance,
                    "safety_validation": self.safety_validation,
                    "rollback_attempt_base": self.rollback_attempt_base,
                    **rollback_fields,
                    "reward_phase": (
                        reward_config.reward_phase
                        if reward_config is not None
                        else "legacy"
                    ),
                    "low_kl_count": self.low_kl_count,
                    "high_kl_count": self.high_kl_count,
                    "next_checkpoint": _future_trigger(
                        current, config.evaluation.checkpoint_frequency
                    ),
                    "next_light_evaluation": _future_trigger(
                        current, config.evaluation.light_frequency
                    ),
                    "next_full_evaluation": _future_trigger(
                        current, config.evaluation.full_frequency
                    ),
                    "checkpoint_aliases": checkpoint_aliases,
                },
            )
            self.best = proposed_best
            self.last_checkpoint = checkpoint
            atomic_copy_checkpoint(
                checkpoint, self.run_dir / "best_models" / "last.zip"
            )
            if safe:
                atomic_copy_checkpoint(
                    checkpoint,
                    self.run_dir / "best_models" / "latest_safe.zip",
                )
            for alias in changed_aliases:
                atomic_copy_checkpoint(
                    checkpoint,
                    self.run_dir / "best_models" / f"{alias}.zip",
                )
            status_store.update(
                last_checkpoint=str(checkpoint.model_path),
                best_checkpoint=self.best.get("best_overall", {}).get(
                    "checkpoint"
                ),
                best_reliable_checkpoint=self.best.get(
                    "best_reliable", {}
                ).get("checkpoint"),
                best_fast_reliable_checkpoint=self.best.get(
                    "best_fast_reliable", {}
                ).get("checkpoint"),
                checkpoint_aliases=checkpoint_aliases,
                total_timesteps=current,
            )
            return checkpoint

        def finalize(self) -> None:
            self._after_update()
            if (
                self.last_checkpoint is None
                or self.last_checkpoint.timesteps != int(self.model.num_timesteps)
            ):
                self._save_checkpoint(
                    reason="final_or_stop",
                    safe=(
                        self.stop_reason is None
                        and isinstance(self.last_eval, dict)
                        and int(self.last_eval.get("timesteps", -1))
                        == int(self.model.num_timesteps)
                        and full_evaluation_passed_safety(
                            self.last_eval,
                            minimum_cases=config.evaluation.full_episodes,
                        )
                    ),
                )
            state = "COMPLETED" if self.stop_reason is None else "STOPPED"
            status_store.update(
                state=state,
                stop_reason=self.stop_reason,
                total_timesteps=int(self.model.num_timesteps),
                curriculum_stage=sampler.current_stage,
                process_alive=True,
            )

        def _on_training_end(self) -> None:
            pass

    return LongRunCallback()
