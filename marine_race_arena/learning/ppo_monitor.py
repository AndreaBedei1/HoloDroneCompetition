"""Structured PPO update monitoring with a hard KL safety stop.

Stable-Baselines3's ``target_kl`` only early-stops the epochs *within* an update. This
module records, after every PPO update, the KL/clip/loss/entropy/std/saturation metrics
to a structured CSV and enforces an additional *hard* ``max_acceptable_kl``: if an update
exceeds it (or produces non-finite values), training is stopped cleanly and the run is
tagged with a documented status. A KL safety stop is NOT a simulator failure.

Implemented as an SB3 callback (not a ``model.train`` wrapper), so ``model.save`` still
pickles cleanly. The metrics SB3 records during ``train()`` are read at the start of the
next rollout (they persist in the logger until the next dump); the final update is
captured post-``learn`` via :meth:`PPOUpdateRecorder.record`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class RunStatus:
    COMPLETED = "COMPLETED"
    EARLY_STOP_TARGET_KL = "EARLY_STOP_TARGET_KL"
    ABORT_MAX_KL = "ABORT_MAX_KL"
    ABORT_ACTION_SATURATION = "ABORT_ACTION_SATURATION"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"
    SIMULATOR_FAILURE = "SIMULATOR_FAILURE"

    ALL = (COMPLETED, EARLY_STOP_TARGET_KL, ABORT_MAX_KL, ABORT_ACTION_SATURATION,
           NUMERICAL_FAILURE, SIMULATOR_FAILURE)


class KLSafetyAbort(Exception):
    """Raised to stop training cleanly when a hard KL / numerical threshold is crossed."""


class WarmStartAbort(Exception):
    """Raised before training if a BC warm-start's timestep-zero eval looks broken."""


_CSV_FIELDS = ["num_timesteps", "n_updates", "approx_kl", "target_kl", "kl_ratio", "max_acceptable_kl",
               "clip_fraction", "policy_gradient_loss", "value_loss", "entropy_loss",
               "explained_variance", "policy_std", "action_saturation", "target_kl_early_stop"]


class PPOUpdateRecorder:
    """Reads SB3 update metrics from ``model.logger`` and enforces the hard KL stop.

    Pure logic (no SB3 dependency): call :meth:`record` after each update (or once at the
    start of the next rollout) and inspect :attr:`status` / :meth:`should_abort`.
    """

    def __init__(self, model, csv_path, *, target_kl: Optional[float], max_acceptable_kl: Optional[float],
                 max_action_saturation: Optional[float] = None):
        self.model = model
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.target_kl = target_kl
        self.max_acceptable_kl = max_acceptable_kl
        self.max_action_saturation = max_action_saturation
        self.rows: List[Dict[str, Any]] = []
        self.status = RunStatus.COMPLETED
        self.max_kl_seen = 0.0
        self.n_updates = 0
        self.target_kl_early_stops = 0
        self._last_recorded_ts = -1

    def _logger_value(self, key: str) -> Optional[float]:
        val = self.model.logger.name_to_value.get(key)
        try:
            return float(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    def _policy_std(self) -> Optional[float]:
        try:
            import torch  # noqa: F401

            return float(np.mean(np.exp(self.model.policy.log_std.detach().cpu().numpy())))
        except Exception:
            return None

    def _action_saturation(self):
        try:
            acts = np.asarray(self.model.rollout_buffer.actions).reshape(-1, self.model.action_space.shape[0])
            return float(np.mean(np.abs(acts) > 0.98)), bool(np.all(np.isfinite(acts)))
        except Exception:  # pragma: no cover
            return None, True

    def record(self) -> bool:
        """Record the current update's metrics if fresh. Returns True if a row was added."""
        approx_kl = self._logger_value("train/approx_kl")
        if approx_kl is None:
            return False
        ts = int(getattr(self.model, "num_timesteps", 0))
        if ts == self._last_recorded_ts and self.rows:
            return False  # already captured this update
        self._last_recorded_ts = ts
        self.n_updates += 1
        sat, actions_finite = self._action_saturation()
        early = (self.target_kl is not None and approx_kl >= self.target_kl)
        if early:
            self.target_kl_early_stops += 1
        self.rows.append({
            "num_timesteps": ts, "n_updates": self.n_updates, "approx_kl": approx_kl,
            "target_kl": self.target_kl,
            "kl_ratio": (approx_kl / self.target_kl if self.target_kl else None),
            "max_acceptable_kl": self.max_acceptable_kl,
            "clip_fraction": self._logger_value("train/clip_fraction"),
            "policy_gradient_loss": self._logger_value("train/policy_gradient_loss"),
            "value_loss": self._logger_value("train/value_loss"),
            "entropy_loss": self._logger_value("train/entropy_loss"),
            "explained_variance": self._logger_value("train/explained_variance"),
            "policy_std": self._policy_std(), "action_saturation": sat, "target_kl_early_stop": early,
        })
        self._write_csv()
        self.max_kl_seen = max(self.max_kl_seen, approx_kl)
        if not np.isfinite(approx_kl) or not actions_finite:
            self.status = RunStatus.NUMERICAL_FAILURE
        elif self.max_acceptable_kl is not None and approx_kl > self.max_acceptable_kl:
            self.status = RunStatus.ABORT_MAX_KL
        elif (self.max_action_saturation is not None and sat is not None
              and sat > self.max_action_saturation):
            self.status = RunStatus.ABORT_ACTION_SATURATION
        return True

    def should_abort(self) -> bool:
        return self.status in (RunStatus.ABORT_MAX_KL, RunStatus.ABORT_ACTION_SATURATION,
                               RunStatus.NUMERICAL_FAILURE)

    def _write_csv(self) -> None:
        import csv

        tmp = self.csv_path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({k: row.get(k) for k in _CSV_FIELDS})
        tmp.replace(self.csv_path)

    def summary(self) -> Dict[str, Any]:
        return {
            "status": self.status, "n_updates": self.n_updates,
            "max_approx_kl": round(self.max_kl_seen, 6), "target_kl": self.target_kl,
            "max_acceptable_kl": self.max_acceptable_kl, "target_kl_early_stops": self.target_kl_early_stops,
            "final_approx_kl": (self.rows[-1]["approx_kl"] if self.rows else None),
            "final_clip_fraction": (self.rows[-1]["clip_fraction"] if self.rows else None),
            "final_policy_std": (self.rows[-1]["policy_std"] if self.rows else None),
            "final_action_saturation": (self.rows[-1]["action_saturation"] if self.rows else None),
        }


def make_kl_monitor_callback(recorder: PPOUpdateRecorder):
    """SB3 callback that records each update at the next rollout start and hard-stops on KL."""
    from stable_baselines3.common.callbacks import BaseCallback

    class KLMonitorCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self.recorder = recorder

        def _on_rollout_start(self) -> None:
            # Metrics from the previous update persist in the logger until the next dump.
            self.recorder.record()

        def _on_step(self) -> bool:
            return not self.recorder.should_abort()  # False stops learn() cleanly

    return KLMonitorCallback()
