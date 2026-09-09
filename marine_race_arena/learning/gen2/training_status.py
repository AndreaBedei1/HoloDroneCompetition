"""Training status logger and callback for Gen-2 PPO.

Tracks:
- timestep
- completed episodes
- completion rate
- mean gates completed
- collision / OOB counts
- mean reward
- approx KL
- policy loss
- value loss
- entropy loss
- learning rate
- vision availability rate
- orientation availability rate
- clipping / feature saturation rate
- best checkpoint tracking
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from marine_race_arena.learning.config_local_transition_27d import (
    FEATURE_BOUNDS_LOCAL_TRANSITION_27D,
    OBS_DIM_LOCAL_TRANSITION_27D,
)


class Gen2TrainingStatusCallback(BaseCallback):
    """Periodic status logger for Gen-2 27-D PPO."""

    def __init__(
        self,
        *,
        log_dir: str | Path,
        log_interval_steps: int = 1024,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval_steps = max(1, int(log_interval_steps))

        self.jsonl_path = self.log_dir / "training_status.jsonl"
        self.latest_json_path = self.log_dir / "latest_training_status.json"

        self.completed_episodes = 0
        self.completed_successes = 0
        self.total_gates_crossed = 0
        self.collision_episodes = 0
        self.oob_episodes = 0

        # Step-level metrics
        self._step_rewards: List[float] = []
        self._vision_available_steps = 0
        self._orientation_available_steps = 0
        self._saturated_features_count = 0
        self._total_feature_elements = 0
        self._window_steps = 0

        # Best tracking
        self.best_mean_gates = -1.0
        self.best_completion_rate = -1.0
        self.best_checkpoint_info: Optional[Dict[str, Any]] = None
        self._last_logged_step = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is not None:
            self._step_rewards.extend([float(r) for r in np.asarray(rewards).reshape(-1)])

        # Inspect observation tensor if available
        new_obs = self.locals.get("new_obs")
        if new_obs is not None:
            obs_arr = np.asarray(new_obs)
            if obs_arr.ndim == 2 and obs_arr.shape[1] == OBS_DIM_LOCAL_TRANSITION_27D:
                batch_size = obs_arr.shape[0]
                self._window_steps += batch_size
                # Feature 5 is vision_center_x, 7 is area
                vision_on = (np.abs(obs_arr[:, 5]) > 1e-4) | (obs_arr[:, 7] > 1e-4)
                self._vision_available_steps += int(np.sum(vision_on))
                # Feature 24 is gate_orientation_present
                orient_on = obs_arr[:, 24] > 0.5
                self._orientation_available_steps += int(np.sum(orient_on))

                # Clipping / saturation check against bounds
                lows = np.array([b[0] for b in FEATURE_BOUNDS_LOCAL_TRANSITION_27D], dtype=np.float32)
                highs = np.array([b[1] for b in FEATURE_BOUNDS_LOCAL_TRANSITION_27D], dtype=np.float32)
                at_low = np.abs(obs_arr - lows) < 1e-4
                at_high = np.abs(obs_arr - highs) < 1e-4
                self._saturated_features_count += int(np.sum(at_low | at_high))
                self._total_feature_elements += int(obs_arr.size)

        # Inspect completed episodes
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is not None and infos is not None:
            for done, info in zip(dones, infos):
                if done:
                    self.completed_episodes += 1
                    gate_count = int(info.get("gate_crossings", 0))
                    self.total_gates_crossed += gate_count
                    if bool(info.get("collision_contact_frame", False)):
                        self.collision_episodes += 1
                    if bool(info.get("out_of_bounds_frame", False)):
                        self.oob_episodes += 1
                    # A completed run has completed status or all gates
                    if bool(info.get("completed", False)):
                        self.completed_successes += 1

        if self.num_timesteps - self._last_logged_step >= self.log_interval_steps:
            self._log_status()
            self._last_logged_step = self.num_timesteps

        return True

    def _on_rollout_end(self) -> None:
        self._log_status()

    def _log_status(self) -> None:
        # Retrieve PPO training metrics from logger
        logger_values = getattr(self.logger, "name_to_value", {}) if self.logger else {}

        approx_kl = float(logger_values.get("train/approx_kl", 0.0))
        policy_loss = float(logger_values.get("train/policy_gradient_loss", 0.0))
        value_loss = float(logger_values.get("train/value_loss", 0.0))
        entropy_loss = float(logger_values.get("train/entropy_loss", 0.0))
        learning_rate = float(logger_values.get("train/learning_rate", 0.0))

        completion_rate = (
            float(self.completed_successes) / max(1, self.completed_episodes)
            if self.completed_episodes > 0 else 0.0
        )
        mean_gates = (
            float(self.total_gates_crossed) / max(1, self.completed_episodes)
            if self.completed_episodes > 0 else 0.0
        )
        mean_reward = (
            float(np.mean(self._step_rewards[-500:]))
            if self._step_rewards else 0.0
        )

        n_steps = max(1, self._window_steps)
        vis_avail_rate = float(self._vision_available_steps) / n_steps
        orient_avail_rate = float(self._orientation_available_steps) / n_steps
        sat_rate = (
            float(self._saturated_features_count) / max(1, self._total_feature_elements)
        )

        # Update best checkpoint
        is_best = False
        if completion_rate > self.best_completion_rate or (
            completion_rate == self.best_completion_rate and mean_gates > self.best_mean_gates
        ):
            self.best_completion_rate = completion_rate
            self.best_mean_gates = mean_gates
            is_best = True
            self.best_checkpoint_info = {
                "timestep": self.num_timesteps,
                "completion_rate": round(completion_rate, 4),
                "mean_gates": round(mean_gates, 2),
            }

        status_record = {
            "timestep": self.num_timesteps,
            "completed_episodes": self.completed_episodes,
            "completion_rate": round(completion_rate, 4),
            "mean_gates": round(mean_gates, 2),
            "collision_episodes": self.collision_episodes,
            "out_of_bounds_episodes": self.oob_episodes,
            "reward": round(mean_reward, 4),
            "approx_kl": round(approx_kl, 6),
            "policy_loss": round(policy_loss, 6),
            "value_loss": round(value_loss, 6),
            "entropy": round(entropy_loss, 6),
            "learning_rate": round(learning_rate, 8),
            "vision_availability": round(vis_avail_rate, 4),
            "orientation_availability": round(orient_avail_rate, 4),
            "clipping_saturation_fraction": round(sat_rate, 4),
            "best_checkpoint": self.best_checkpoint_info,
            "is_best": is_best,
        }

        # Append to jsonl
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(status_record) + "\n")

        # Update latest json atomically
        tmp_path = self.latest_json_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(status_record, indent=2), encoding="utf-8")
        tmp_path.replace(self.latest_json_path)

        if self.verbose:
            print(
                f"[Gen2 Status] step={self.num_timesteps:06d} | "
                f"episodes={self.completed_episodes} | "
                f"mean_gates={mean_gates:.2f} | "
                f"reward={mean_reward:+.3f} | "
                f"vis={vis_avail_rate*100:.1f}% | "
                f"orient={orient_avail_rate*100:.1f}% | "
                f"sat={sat_rate*100:.1f}% | "
                f"best={self.best_checkpoint_info}",
                flush=True,
            )
