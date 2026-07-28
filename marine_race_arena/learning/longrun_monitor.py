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
            "last_rollback_reason": None,
            "last_rollback_source": None,
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
    model.lr_schedule = source.lr_schedule
    schedule = getattr(model, "lr_schedule", None)
    if hasattr(schedule, "scale"):
        schedule.scale(learning_rate_scale)
    model.num_timesteps = current_timesteps


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
        if self.csv_path.exists():
            try:
                with self.csv_path.open(newline="", encoding="utf-8") as handle:
                    self.rows = list(csv.DictReader(handle))
                if self.rows:
                    self.last_timestep = int(float(self.rows[-1]["num_timesteps"]))
            except Exception:
                self.rows = []

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
            self.last_rollback_reason: Optional[List[str]] = None
            self.last_rollback_source: Optional[str] = None
            self._rollback_this_update = False
            self._last_status_wall = 0.0
            self._started_wall = time.time()
            self._started_timesteps = 0
            self._next_checkpoint = config.evaluation.checkpoint_frequency
            self._next_light_eval = config.evaluation.light_frequency
            self._next_full_eval = config.evaluation.full_frequency

        def restore_pipeline_state(self, state: Mapping[str, Any]) -> None:
            evaluation = dict(state.get("evaluation", {}))
            self.eval_history = list(evaluation.get("history", []))
            self.best = dict(evaluation.get("best", {}))
            self.automatic_changes = list(
                state.get("extra", {}).get("automatic_changes", [])
            )
            self.rollback_count = int(
                state.get("extra", {}).get("rollback_count", 0)
            )
            if reward_config is not None:
                reward_config.set_phase(
                    str(state.get("extra", {}).get("reward_phase", reward_config.reward_phase))
                )
            current = int(state.get("total_timesteps", 0))
            self._next_checkpoint = (
                current // config.evaluation.checkpoint_frequency + 1
            ) * config.evaluation.checkpoint_frequency
            self._next_light_eval = (
                current // config.evaluation.light_frequency + 1
            ) * config.evaluation.light_frequency
            self._next_full_eval = (
                current // config.evaluation.full_frequency + 1
            ) * config.evaluation.full_frequency

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
                and not self._rollback_this_update
                and self.stop_reason
                not in {
                    LongRunStopReason.ABSOLUTE_KL,
                    LongRunStopReason.NUMERICAL,
                }
            ):
                self._save_checkpoint(reason="periodic")
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
                        reason="full_evaluation", safe=True
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
                rollback_count=self.rollback_count,
                last_rollback_reason=self.last_rollback_reason,
                last_rollback_source=self.last_rollback_source,
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
            self.last_rollback_reason = list(reasons)
            incumbent = self.best.get("best_reliable", {})
            source_path = incumbent.get("checkpoint")
            self.last_rollback_source = str(source_path) if source_path else None
            event = {
                "timesteps": int(self.model.num_timesteps),
                "attempt": self.rollback_count,
                "reasons": list(reasons),
                "source": self.last_rollback_source,
            }
            if (
                not rollback_attempt_allowed(
                    self.rollback_count, rollback.maximum_attempts
                )
                or not source_path
            ):
                event["outcome"] = "stopped"
                self.stop_reason = LongRunStopReason.REPEATED_REGRESSION
                atomic_append_jsonl(
                    self.run_dir / "logs" / "rollback_history.jsonl", event
                )
                return
            source = type(self.model).load(str(source_path), device="cpu")
            current_timesteps = int(self.model.num_timesteps)
            restore_ppo_training_state(
                self.model, source, rollback.learning_rate_scale
            )
            regularizer = getattr(self.model, "_retention_regularizer", None)
            if regularizer is not None:
                event["retention_weight_after"] = regularizer.increase_weight(
                    rollback.retention_weight_scale
                )
            target_stage = rollback_target_stage(sampler.current_stage)
            change = sampler.force_stage(
                target_stage,
                timesteps=current_timesteps,
                reason="automatic_full_evaluation_rollback",
            )
            atomic_append_jsonl(
                self.run_dir / "curriculum_history.jsonl", change
            )
            if reward_config is not None:
                reward_config.set_phase("reliability")
            self._reset_training_episode()
            self._rollback_this_update = True
            self.last_eval = None
            event.update({"outcome": "restored", "stage": target_stage})
            atomic_append_jsonl(
                self.run_dir / "logs" / "rollback_history.jsonl", event
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

        def _evaluation_state(self) -> Dict[str, Any]:
            return {"history": self.eval_history, "best": self.best}

        def _save_checkpoint(
            self, *, reason: str, safe: bool = True
        ):
            current = int(self.model.num_timesteps)
            checkpoint = atomic_save_checkpoint(
                self.model,
                self.run_dir,
                total_timesteps=current,
                config_contract_hash=contract_hash,
                curriculum_state=sampler.state_dict(),
                evaluation_state=self._evaluation_state(),
                status="safe" if safe else "unsafe",
                reason=reason,
                extra_state={
                    "automatic_changes": self.automatic_changes,
                    "stop_reason": self.stop_reason,
                    "rollback_count": self.rollback_count,
                    "reward_phase": (
                        reward_config.reward_phase
                        if reward_config is not None
                        else "legacy"
                    ),
                },
            )
            self.last_checkpoint = checkpoint
            atomic_copy_checkpoint(
                checkpoint, self.run_dir / "best_models" / "last.zip"
            )
            if safe:
                atomic_copy_checkpoint(
                    checkpoint,
                    self.run_dir / "best_models" / "latest_safe.zip",
                )
            if safe and self.last_eval and self.last_eval.get("mode") == "full":
                aliases = (
                    ("best_overall", None),
                    ("best_left", "left"),
                    ("best_right", "right"),
                    ("best_three_gate", "three_gate"),
                )
                for alias, direction in aliases:
                    previous = self.best.get(alias)
                    if direction == "three_gate":
                        better = (
                            previous is None
                            or float(
                                self.last_eval.get(
                                    "three_gate_completion_rate", 0.0
                                )
                            )
                            > float(
                                previous.get("three_gate_completion_rate", 0.0)
                            )
                        )
                    else:
                        better = is_better(
                            self.last_eval, previous, direction=direction
                        )
                    if better:
                        self.best[alias] = {
                            **{
                                key: value
                                for key, value in self.last_eval.items()
                                if key not in {"rows", "reward_component_sums"}
                            },
                            "checkpoint": str(checkpoint.model_path),
                        }
                        atomic_copy_checkpoint(
                            checkpoint,
                            self.run_dir / "best_models" / f"{alias}.zip",
                        )
                reliable = reliability_requirements_met(
                    self.last_eval, config.curriculum.promotion
                )
                if reliable:
                    reliable_previous = self.best.get("best_reliable")
                    if is_better(self.last_eval, reliable_previous):
                        self._update_alias(
                            "best_reliable", checkpoint, self.last_eval
                        )
                    fast_previous = self.best.get("best_fast_reliable")
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
                        self._update_alias(
                            "best_fast_reliable", checkpoint, self.last_eval
                        )
            status_store.update(
                last_checkpoint=str(checkpoint.model_path),
                best_checkpoint=self.best.get("best_overall", {}).get(
                    "checkpoint"
                ),
                total_timesteps=current,
            )
            return checkpoint

        def _update_alias(
            self,
            alias: str,
            checkpoint: Any,
            report: Mapping[str, Any],
        ) -> None:
            self.best[alias] = {
                **{
                    key: value
                    for key, value in report.items()
                    if key not in {"rows", "reward_component_sums"}
                },
                "checkpoint": str(checkpoint.model_path),
            }
            atomic_copy_checkpoint(
                checkpoint,
                self.run_dir / "best_models" / f"{alias}.zip",
            )

        def finalize(self) -> None:
            self._after_update()
            if (
                self.last_checkpoint is None
                or self.last_checkpoint.timesteps != int(self.model.num_timesteps)
            ):
                self._save_checkpoint(
                    reason="final_or_stop",
                    safe=self.stop_reason
                    not in {
                        LongRunStopReason.ABSOLUTE_KL,
                        LongRunStopReason.NUMERICAL,
                    },
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
