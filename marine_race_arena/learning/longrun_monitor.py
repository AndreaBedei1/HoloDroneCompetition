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
    is_better,
    plateau_detected,
)


class LongRunStopReason:
    NONE = None
    GRACEFUL = "GRACEFUL_STOP"
    ABSOLUTE_KL = "UNSAFE_ABSOLUTE_KL"
    REPEATED_HIGH_KL = "PAUSED_REPEATED_HIGH_KL"
    NUMERICAL = "NUMERICAL_FAILURE"
    DISK = "DISK_SPACE_GUARD"
    PLATEAU = "PLATEAU"


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
            "left_success": None,
            "right_success": None,
            "straight_retention": None,
            "three_gate_success": None,
            "approx_kl": None,
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


class UpdateMetricsRecorder:
    FIELDS = (
        "num_timesteps",
        "n_updates",
        "approx_kl",
        "clip_fraction",
        "policy_loss",
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
                        "out_of_bounds": int(info.get("episode_out_of_bounds", 0)),
                        "wrong_direction": int(info.get("episode_wrong_direction", 0)),
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
            self._last_status_wall = 0.0
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
            status_store.update(
                state="RUNNING",
                total_timesteps=int(self.model.num_timesteps),
                curriculum_stage=sampler.current_stage,
            )

        def _on_step(self) -> bool:
            assert self.metrics is not None
            self.metrics.observe_infos(self.locals.get("infos", []) or [])
            now = time.time()
            if (
                self.n_calls % config.reliability.status_frequency_steps == 0
                or now - self._last_status_wall
                >= config.reliability.status_frequency_seconds
            ):
                self._last_status_wall = now
                status_store.update(
                    total_timesteps=int(self.num_timesteps),
                    curriculum_stage=sampler.current_stage,
                    state="STOPPING" if self.stop_reason else "RUNNING",
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
            row = self.metrics.record()
            if row is not None:
                self._apply_kl_policy(row)
                status_store.update(
                    total_timesteps=int(self.model.num_timesteps),
                    approx_kl=row["approx_kl"],
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
                plateau = plateau_detected(
                    self.eval_history,
                    current_timesteps=int(self.model.num_timesteps),
                    plateau_steps=config.evaluation.plateau_steps,
                    min_evaluations=config.evaluation.plateau_min_evaluations,
                )
                if plateau:
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
                left_success=report.get("left_completion_rate"),
                right_success=report.get("right_completion_rate"),
                straight_retention=report.get("straight_completion_rate"),
                three_gate_success=report.get("three_gate_completion_rate"),
                curriculum_stage=sampler.current_stage,
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
            status_store.update(
                last_checkpoint=str(checkpoint.model_path),
                best_checkpoint=self.best.get("best_overall", {}).get(
                    "checkpoint"
                ),
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
