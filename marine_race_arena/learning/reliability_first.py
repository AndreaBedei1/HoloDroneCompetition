"""Offline retention and initial-policy anchoring for reliability-first PPO.

The auxiliary update consumes only recorded BC observations/actions.  It never
constructs or queries an expert controller inside the live rollout environment.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np


class OfflineRetentionRegularizer:
    """One bounded auxiliary policy update after each configured PPO update."""

    def __init__(
        self,
        *,
        dataset_paths: Sequence[str],
        initial_bc_checkpoint: str,
        seed: int,
        batch_size: int,
        weight: float,
        maximum_weight: float,
        final_weight: float,
        initial_policy_kl_weight: float,
        decay_until_timesteps: int,
        active_through_stage: str,
        every_updates: int,
        initial_action_std: float,
        maximum_gradient_norm: float,
    ) -> None:
        import torch

        from marine_race_arena.learning.bc_longrun_v3 import prepare_datasets
        from marine_race_arena.learning.bc_v3_transfer import load_v3_policy

        missing = [path for path in dataset_paths if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"retention dataset(s) missing: {missing}")
        data = prepare_datasets({"offline_retention": list(dataset_paths)})
        train_mask = data.splits == "train"
        if not np.any(train_mask):
            raise ValueError("offline retention datasets contain no training rows")
        self.observations = np.asarray(
            data.observations[train_mask], dtype=np.float32
        )
        self.actions = np.asarray(data.actions[train_mask], dtype=np.float32)
        self.reference_policy = load_v3_policy(initial_bc_checkpoint)
        self.reference_policy.eval()
        for parameter in self.reference_policy.parameters():
            parameter.requires_grad_(False)
        self.rng = np.random.default_rng(int(seed) + 9173)
        self.batch_size = int(batch_size)
        self.base_weight = float(weight)
        self.maximum_weight = float(maximum_weight)
        self.final_weight = float(final_weight)
        self.kl_weight = float(initial_policy_kl_weight)
        self.decay_until_timesteps = int(decay_until_timesteps)
        self.active_through_stage_index = int(active_through_stage[1:])
        self.every_updates = int(every_updates)
        self.initial_action_std = float(initial_action_std)
        self.maximum_gradient_norm = float(maximum_gradient_norm)
        self.weight_multiplier = 1.0
        self.calls = 0
        self.last_metrics: Dict[str, Optional[float]] = {
            "bc_retention_loss": None,
            "initial_policy_kl": None,
            "retention_weight": self.base_weight,
        }
        self._torch = torch

    def effective_weight(self, timesteps: int, stage: str) -> float:
        if int(stage[1:]) > self.active_through_stage_index:
            return 0.0
        progress = min(1.0, max(0.0, int(timesteps) / self.decay_until_timesteps))
        scheduled = self.base_weight + (
            self.final_weight - self.base_weight
        ) * progress
        return float(
            np.clip(
                scheduled * self.weight_multiplier,
                self.final_weight,
                self.maximum_weight,
            )
        )

    def increase_weight(self, factor: float) -> float:
        self.weight_multiplier *= max(1.0, float(factor))
        return min(self.maximum_weight, self.base_weight * self.weight_multiplier)

    def apply(self, model: Any, *, timesteps: int, stage: str) -> Dict[str, float]:
        torch = self._torch
        self.calls += 1
        weight = self.effective_weight(timesteps, stage)
        if weight <= 0.0 or self.calls % self.every_updates:
            metrics = {
                "bc_retention_loss": 0.0,
                "initial_policy_kl": 0.0,
                "retention_weight": weight,
            }
            self.last_metrics = metrics
            return metrics
        indices = self.rng.integers(
            0, len(self.observations), size=min(self.batch_size, len(self.observations))
        )
        observations = torch.as_tensor(
            self.observations[indices], dtype=torch.float32, device=model.device
        )
        expert_actions = torch.as_tensor(
            self.actions[indices], dtype=torch.float32, device=model.device
        )
        with torch.no_grad():
            reference_mean = self.reference_policy(
                observations.detach().cpu()
            ).to(model.device)

        model.policy.set_training_mode(True)
        distribution = model.policy.get_distribution(observations).distribution
        current_mean = distribution.mean
        current_std = distribution.stddev
        bc_loss = torch.mean(torch.square(current_mean - expert_actions))
        reference_std = torch.full_like(current_std, self.initial_action_std)
        initial_kl = torch.mean(
            torch.log(reference_std / current_std)
            + (
                torch.square(current_std)
                + torch.square(current_mean - reference_mean)
            )
            / (2.0 * torch.square(reference_std))
            - 0.5
        )
        auxiliary_loss = weight * bc_loss + self.kl_weight * initial_kl
        model.policy.optimizer.zero_grad()
        auxiliary_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.policy.parameters(), self.maximum_gradient_norm
        )
        model.policy.optimizer.step()
        metrics = {
            "bc_retention_loss": float(bc_loss.detach().cpu()),
            "initial_policy_kl": float(initial_kl.detach().cpu()),
            "retention_weight": float(weight),
            "retention_gradient_norm": float(gradient_norm.detach().cpu()),
        }
        if not all(math.isfinite(float(value)) for value in metrics.values()):
            raise FloatingPointError("non-finite offline retention metric")
        self.last_metrics = metrics
        return metrics


def reliability_first_ppo_class():
    """Create the subclass lazily so non-RL environments can import this module."""
    from stable_baselines3 import PPO

    class ReliabilityFirstPPO(PPO):
        def _excluded_save_params(self):
            return super()._excluded_save_params() + [
                "_retention_regularizer",
                "_retention_stage_getter",
            ]

        def attach_retention_regularizer(
            self,
            regularizer: Optional[OfflineRetentionRegularizer],
            stage_getter: Optional[Callable[[], str]],
        ) -> None:
            self._retention_regularizer = regularizer
            self._retention_stage_getter = stage_getter

        def train(self) -> None:
            super().train()
            ppo_policy_loss = self.logger.name_to_value.get(
                "train/policy_gradient_loss"
            )
            metrics = {
                "bc_retention_loss": 0.0,
                "initial_policy_kl": 0.0,
                "retention_weight": 0.0,
            }
            regularizer = getattr(self, "_retention_regularizer", None)
            if regularizer is not None:
                getter = getattr(self, "_retention_stage_getter", None)
                stage = getter() if getter is not None else "C0"
                metrics = regularizer.apply(
                    self, timesteps=int(self.num_timesteps), stage=stage
                )
            ppo_value = (
                float(ppo_policy_loss) if ppo_policy_loss is not None else 0.0
            )
            combined = (
                ppo_value
                + metrics["retention_weight"] * metrics["bc_retention_loss"]
                + (
                    regularizer.kl_weight * metrics["initial_policy_kl"]
                    if regularizer is not None
                    else 0.0
                )
            )
            self.logger.record("train/ppo_policy_loss", ppo_value)
            self.logger.record(
                "train/bc_retention_loss", metrics["bc_retention_loss"]
            )
            self.logger.record(
                "train/initial_policy_kl", metrics["initial_policy_kl"]
            )
            self.logger.record(
                "train/retention_weight", metrics["retention_weight"]
            )
            self.logger.record("train/combined_policy_loss", combined)

    ReliabilityFirstPPO.__name__ = "ReliabilityFirstPPO"
    ReliabilityFirstPPO.__qualname__ = "ReliabilityFirstPPO"
    return ReliabilityFirstPPO


ReliabilityFirstPPO = reliability_first_ppo_class()
