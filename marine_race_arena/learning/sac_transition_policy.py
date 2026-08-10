"""Independent multi-step SAC networks for universal local transitions.

The actor consumes exactly the existing 35-feature onboard observation.  Its
warm mean path is byte-for-byte compatible with the PPO policy MLP; the SAC
Gaussian location is the inverse-tanh of that bounded action, which preserves
the deterministic PPO action while still producing a conventional
tanh-squashed stochastic policy.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from marine_race_arena.learning.config import ACTION_DIM
from marine_race_arena.learning.config_local_transition import (
    FEATURE_BOUNDS_LOCAL_TRANSITION,
    OBS_DIM_LOCAL_TRANSITION,
    OBS_ENCODING_VERSION_LOCAL_TRANSITION,
)


LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0
ACTION_EPSILON = 1e-6

# The Gaussian location is the inverse tanh of the bounded mean head, which
# preserves the deterministic PPO action exactly.  Unbounded, that inverse has
# derivative 1/(1-m^2): the collapsed v1 actor reached a mean Jacobian of 30,121
# and a p95 of 493,448, so a single mean-head component approaching +-1 raised
# the effective actor step by five orders of magnitude and drove the policy into
# permanent tanh saturation.  Bounding the pre-tanh location caps that
# amplification.  The warm anchor's largest mean-head magnitude is 0.605 and its
# largest Jacobian is 1.6, so the default bound is inert at initialization while
# capping any runaway at 14.2.
DEFAULT_PRE_TANH_LIMIT = 2.0


def compute_sac_bootstrap_target(
    rewards: Any,
    discounts: Any,
    minimum_target_q: Any,
    entropy_coefficient: Any,
    next_log_probability: Any,
) -> Any:
    """Pure SAC target used by tests and the twin-critic update."""

    return rewards + discounts * (
        minimum_target_q - entropy_coefficient * next_log_probability
    )


def _torch():
    import torch
    import torch.nn as nn

    return torch, nn


class SquashedGaussianActor:  # wrapped below to avoid importing torch at module import
    """Factory-compatible namespace retained for lightweight CLI imports."""

    def __new__(
        cls,
        observation_dim: int = OBS_DIM_LOCAL_TRANSITION,
        action_dim: int = ACTION_DIM,
        hidden_sizes: Sequence[int] = (256, 256),
        initial_std: float = 0.075,
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
        pre_tanh_limit: float = DEFAULT_PRE_TANH_LIMIT,
    ):
        torch, nn = _torch()

        class _Actor(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                sizes = [int(observation_dim), *map(int, hidden_sizes)]
                layers = []
                for source, target in zip(sizes, sizes[1:]):
                    layers.extend((nn.Linear(source, target), nn.Tanh()))
                self.policy_net = nn.Sequential(*layers)
                self.mean_head = nn.Linear(sizes[-1], int(action_dim))
                self.log_std_head = nn.Linear(sizes[-1], int(action_dim))
                nn.init.zeros_(self.log_std_head.weight)
                nn.init.constant_(
                    self.log_std_head.bias,
                    math.log(float(initial_std)),
                )
                self.observation_dim = int(observation_dim)
                self.action_dim = int(action_dim)
                self.hidden_sizes = tuple(map(int, hidden_sizes))
                self.initial_std = float(initial_std)
                self.log_std_min = float(log_std_min)
                self.log_std_max = float(log_std_max)
                self.pre_tanh_limit = float(pre_tanh_limit)
                if self.pre_tanh_limit <= 0.0:
                    raise ValueError("pre_tanh_limit must be positive")
                self.mean_clip = float(math.tanh(self.pre_tanh_limit))
                if not self.log_std_min < self.log_std_max:
                    raise ValueError("log_std_min must be smaller than log_std_max")
                initial_log_std = math.log(self.initial_std)
                if not self.log_std_min <= initial_log_std <= self.log_std_max:
                    raise ValueError("initial_std must fall within the log_std clip")

            def _distribution_parameters(self, observations):
                features = self.policy_net(observations)
                # PPO emits an unsquashed Gaussian mean and clips it at the
                # action boundary.  Inverting tanh after that clip makes the SAC
                # deterministic action identical without copying PPO variance.
                # The clip is at tanh(pre_tanh_limit), not at 1, so the inverse
                # tanh Jacobian stays bounded by 1/(1-mean_clip^2) instead of
                # diverging as a mean-head component approaches the action
                # boundary.
                bounded_mean = torch.clamp(
                    self.mean_head(features), -self.mean_clip, self.mean_clip
                )
                pre_tanh_mean = torch.atanh(bounded_mean)
                log_std = torch.clamp(
                    self.log_std_head(features), self.log_std_min, self.log_std_max
                )
                return pre_tanh_mean, log_std

            def deterministic(self, observations):
                mean, _ = self._distribution_parameters(observations)
                return torch.tanh(mean)

            def sample(self, observations, deterministic: bool = False):
                mean, log_std = self._distribution_parameters(observations)
                if deterministic:
                    pre_tanh = mean
                else:
                    noise = torch.randn_like(mean)
                    pre_tanh = mean + log_std.exp() * noise
                action = torch.tanh(pre_tanh)
                if deterministic:
                    log_probability = torch.zeros(
                        (action.shape[0], 1),
                        dtype=action.dtype,
                        device=action.device,
                    )
                else:
                    normal_log_probability = -0.5 * (
                        ((pre_tanh - mean) / log_std.exp()).pow(2)
                        + 2.0 * log_std
                        + math.log(2.0 * math.pi)
                    )
                    log_probability = normal_log_probability.sum(
                        dim=-1, keepdim=True
                    ) - torch.log(
                        1.0 - action.pow(2) + ACTION_EPSILON
                    ).sum(dim=-1, keepdim=True)
                return action, log_probability, torch.tanh(mean)

            def forward(self, observations):
                return self.deterministic(observations)

            def health(self, observations) -> Dict[str, float]:
                """Training-only saturation/conditioning probe (never an input)."""

                with torch.no_grad():
                    features = self.policy_net(observations)
                    raw = self.mean_head(features)
                    bounded = torch.clamp(raw, -self.mean_clip, self.mean_clip)
                    action = torch.tanh(torch.atanh(bounded))
                    jacobian = 1.0 / (1.0 - bounded.pow(2))
                    log_std = torch.clamp(
                        self.log_std_head(features),
                        self.log_std_min, self.log_std_max,
                    )
                absolute = action.abs()
                axes = ("surge", "sway", "heave", "yaw")
                return {
                    "mean_absolute_action": float(absolute.mean()),
                    **{
                        f"mean_absolute_action_{name}": float(absolute[:, index].mean())
                        for index, name in enumerate(axes[: absolute.shape[1]])
                    },
                    "max_absolute_action_axis": float(absolute.mean(dim=0).max()),
                    "action_saturation_fraction": float((absolute > 0.95).float().mean()),
                    "mean_head_clipped_fraction": float(
                        (raw.abs() >= self.mean_clip).float().mean()
                    ),
                    "pre_tanh_jacobian_mean": float(jacobian.mean()),
                    "pre_tanh_jacobian_p95": float(jacobian.flatten().quantile(0.95)),
                    "pre_tanh_jacobian_max": float(jacobian.max()),
                    "policy_std_mean": float(log_std.exp().mean()),
                }

            def config(self) -> Dict[str, Any]:
                return {
                    "observation_dim": self.observation_dim,
                    "action_dim": self.action_dim,
                    "hidden_sizes": list(self.hidden_sizes),
                    "initial_std": self.initial_std,
                    "log_std_clip": [self.log_std_min, self.log_std_max],
                    "pre_tanh_limit": self.pre_tanh_limit,
                    "mean_clip": self.mean_clip,
                    "maximum_pre_tanh_jacobian": 1.0 / (1.0 - self.mean_clip ** 2),
                }

        return _Actor()


def action_drift(current_actions: Any, anchor_actions: Any) -> Dict[str, float]:
    """Per-sample deterministic action drift between the actor and its anchor."""

    import torch

    delta = (current_actions - anchor_actions).abs()
    per_sample = delta.mean(dim=-1)
    return {
        "anchor_mean_drift": float(per_sample.mean()),
        "anchor_p95_drift": float(per_sample.flatten().quantile(0.95)),
        "anchor_max_drift": float(per_sample.max()),
        "anchor_mean_squared_drift": float(delta.pow(2).mean()),
    }


class TwinQCritic:
    def __new__(
        cls,
        observation_dim: int = OBS_DIM_LOCAL_TRANSITION,
        action_dim: int = ACTION_DIM,
        hidden_sizes: Sequence[int] = (256, 256),
    ):
        torch, nn = _torch()

        def network():
            sizes = [int(observation_dim) + int(action_dim), *map(int, hidden_sizes), 1]
            layers = []
            for index, (source, target) in enumerate(zip(sizes, sizes[1:])):
                layers.append(nn.Linear(source, target))
                if index < len(sizes) - 2:
                    layers.append(nn.Tanh())
            return nn.Sequential(*layers)

        class _TwinQ(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q1 = network()
                self.q2 = network()

            def forward(self, observations, actions):
                value = torch.cat((observations, actions), dim=-1)
                return self.q1(value), self.q2(value)

        return _TwinQ()


def transfer_ppo_mean_actor(
    actor: Any,
    source_checkpoint: str | Path,
    *,
    validation_samples: int = 2048,
    validation_seed: int = 91337,
    tolerance: float = 2e-6,
) -> Dict[str, Any]:
    """Copy only PPO policy-mean weights and prove deterministic parity."""

    import torch
    from stable_baselines3 import PPO

    source = PPO.load(str(source_checkpoint), device="cpu")
    if tuple(source.observation_space.shape) != (OBS_DIM_LOCAL_TRANSITION,):
        raise ValueError("PPO warm source does not use the 35-feature contract")
    state = source.policy.state_dict()
    target = actor.state_dict()
    copied = {}
    ppo_layers = [
        ("policy_net.0.weight", "mlp_extractor.policy_net.0.weight"),
        ("policy_net.0.bias", "mlp_extractor.policy_net.0.bias"),
        ("policy_net.2.weight", "mlp_extractor.policy_net.2.weight"),
        ("policy_net.2.bias", "mlp_extractor.policy_net.2.bias"),
        ("mean_head.weight", "action_net.weight"),
        ("mean_head.bias", "action_net.bias"),
    ]
    for destination, origin in ppo_layers:
        if destination not in target or origin not in state:
            raise ValueError(f"missing warm actor tensor {destination} <- {origin}")
        if tuple(target[destination].shape) != tuple(state[origin].shape):
            raise ValueError(f"warm actor tensor shape changed for {destination}")
        copied[destination] = state[origin].detach().clone()
    target.update(copied)
    actor.load_state_dict(target, strict=True)

    rng = np.random.default_rng(int(validation_seed))
    low = np.asarray([value[0] for value in FEATURE_BOUNDS_LOCAL_TRANSITION])
    high = np.asarray([value[1] for value in FEATURE_BOUNDS_LOCAL_TRANSITION])
    observations = rng.uniform(
        low, high, size=(int(validation_samples), OBS_DIM_LOCAL_TRANSITION)
    ).astype(np.float32)
    ppo_actions, _ = source.predict(observations, deterministic=True)
    with torch.no_grad():
        sac_actions = actor.deterministic(torch.from_numpy(observations)).cpu().numpy()
    error = np.abs(np.asarray(ppo_actions) - sac_actions)
    maximum_error = float(error.max(initial=0.0))
    mean_error = float(error.mean())
    if maximum_error > float(tolerance):
        raise ValueError(
            f"SAC warm actor parity error {maximum_error:.9g} exceeds {tolerance}"
        )
    return {
        "schema_version": "sac_ppo_actor_transfer_v1",
        "source_checkpoint": str(source_checkpoint),
        "source_observation_version": OBS_ENCODING_VERSION_LOCAL_TRANSITION,
        "source_observation_dim": OBS_DIM_LOCAL_TRANSITION,
        "target_observation_version": OBS_ENCODING_VERSION_LOCAL_TRANSITION,
        "target_observation_dim": OBS_DIM_LOCAL_TRANSITION,
        "copied_tensors": sorted(copied),
        "ppo_critic_weights_copied": False,
        "ppo_optimizer_state_copied": False,
        "ppo_rollout_data_imported": False,
        "validation_samples": int(validation_samples),
        "validation_seed": int(validation_seed),
        "maximum_absolute_action_error": maximum_error,
        "mean_absolute_action_error": mean_error,
        "tolerance": float(tolerance),
        "passed": True,
    }


class SACTransitionAgent:
    """Small CPU-first SAC learner with independent actor/critic optimizers."""

    def __init__(
        self,
        *,
        observation_dim: int = OBS_DIM_LOCAL_TRANSITION,
        action_dim: int = ACTION_DIM,
        hidden_sizes: Sequence[int] = (256, 256),
        actor_learning_rate: float = 3e-5,
        critic_learning_rate: float = 1e-4,
        entropy_learning_rate: float = 3e-5,
        tau: float = 0.005,
        target_entropy: float = -4.0,
        initial_alpha: float = 0.02,
        initial_std: float = 0.075,
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
        pre_tanh_limit: float = DEFAULT_PRE_TANH_LIMIT,
        gradient_clip_actor: float = 1.0,
        gradient_clip_critic: float = 5.0,
        critic_loss: str = "huber",
        huber_delta: float = 10.0,
        alpha_min: float = 0.001,
        alpha_max: float = 0.02,
        anchor_coefficient: float = 0.0,
        anchor_target_drift: float = 0.05,
        anchor_coefficient_min: float = 0.0,
        anchor_coefficient_max: float = 1000.0,
        anchor_increase_factor: float = 1.5,
        anchor_decrease_factor: float = 0.8,
        device: str = "cpu",
    ) -> None:
        import torch

        if critic_loss not in {"huber", "mse"}:
            raise ValueError(f"unknown critic loss {critic_loss!r}")
        if not 0.0 < float(alpha_min) <= float(alpha_max):
            raise ValueError("alpha bounds must satisfy 0 < alpha_min <= alpha_max")
        self.device = torch.device(device)
        self.actor = SquashedGaussianActor(
            observation_dim, action_dim, hidden_sizes, initial_std,
            log_std_min, log_std_max, pre_tanh_limit,
        ).to(self.device)
        # Frozen copy of the competent warm initialization.  Training-only: it
        # never produces an action for evaluation or inference.
        self.anchor_actor = copy.deepcopy(self.actor).to(self.device)
        for parameter in self.anchor_actor.parameters():
            parameter.requires_grad_(False)
        self.anchor_actor.eval()
        self.anchor_coefficient = float(anchor_coefficient)
        self.anchor_target_drift = float(anchor_target_drift)
        self.anchor_coefficient_min = float(anchor_coefficient_min)
        self.anchor_coefficient_max = float(anchor_coefficient_max)
        self.anchor_increase_factor = float(anchor_increase_factor)
        self.anchor_decrease_factor = float(anchor_decrease_factor)
        self.gradient_clip_actor = float(gradient_clip_actor)
        self.gradient_clip_critic = float(gradient_clip_critic)
        self.critic_loss_kind = str(critic_loss)
        self.huber_delta = float(huber_delta)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.critic = TwinQCritic(
            observation_dim, action_dim, hidden_sizes
        ).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        for parameter in self.target_critic.parameters():
            parameter.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=float(actor_learning_rate)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=float(critic_learning_rate)
        )
        self.log_alpha = torch.tensor(
            math.log(float(initial_alpha)),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.entropy_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=float(entropy_learning_rate)
        )
        self.tau = float(tau)
        self.target_entropy = float(target_entropy)
        self.gradient_updates = 0
        self.actor_updates = 0
        self.entropy_updates = 0
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(map(int, hidden_sizes))
        self.initial_std = float(initial_std)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.learning_rates = {
            "actor": float(actor_learning_rate),
            "critic": float(critic_learning_rate),
            "entropy": float(entropy_learning_rate),
        }

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.detach().exp().cpu())

    def act(self, observations: np.ndarray, *, deterministic: bool = False) -> np.ndarray:
        import torch

        array = np.asarray(observations, dtype=np.float32)
        single = array.ndim == 1
        if single:
            array = array[None, :]
        with torch.no_grad():
            actions, _, _ = self.actor.sample(
                torch.as_tensor(array, device=self.device),
                deterministic=deterministic,
            )
        result = actions.cpu().numpy().astype(np.float32)
        return result[0] if single else result

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """SB3-compatible evaluator surface."""

        return self.act(observation, deterministic=deterministic), None

    def update(
        self,
        batch: Mapping[str, np.ndarray],
        *,
        update_actor: bool = True,
        update_entropy: bool | None = None,
    ) -> Dict[str, float]:
        import torch
        import torch.nn.functional as functional

        tensor = {
            key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for key, value in batch.items()
            if key in {"observations", "actions", "rewards", "next_observations", "discounts"}
        }
        observations = tensor["observations"]
        actions = tensor["actions"]
        rewards = tensor["rewards"].reshape(-1, 1)
        next_observations = tensor["next_observations"]
        discounts = tensor["discounts"].reshape(-1, 1)

        with torch.no_grad():
            next_actions, next_log_probability, _ = self.actor.sample(next_observations)
            target_q1, target_q2 = self.target_critic(next_observations, next_actions)
            target = compute_sac_bootstrap_target(
                rewards,
                discounts,
                torch.minimum(target_q1, target_q2),
                self.log_alpha.exp(),
                next_log_probability,
            )

        q1, q2 = self.critic(observations, actions)
        if self.critic_loss_kind == "huber":
            # Multi-step targets built from strong one-time safety penalties
            # produce heavy-tailed TD errors; squaring them let a handful of
            # samples dominate the critic step (v1 reached a critic loss of 543).
            critic_loss = functional.huber_loss(
                q1, target, delta=self.huber_delta
            ) + functional.huber_loss(q2, target, delta=self.huber_delta)
        else:
            critic_loss = functional.mse_loss(q1, target) + functional.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = float(torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.gradient_clip_critic
        ))
        self.critic_optimizer.step()
        td_error = torch.minimum(
            (q1 - target).abs(), (q2 - target).abs()
        ).detach().flatten()
        actor_grad_norm = 0.0
        anchor_metrics = {
            "anchor_mean_drift": 0.0, "anchor_p95_drift": 0.0,
            "anchor_max_drift": 0.0, "anchor_mean_squared_drift": 0.0,
        }
        anchor_penalty = 0.0

        if update_entropy is None:
            update_entropy = update_actor
        if update_entropy and not update_actor:
            raise ValueError("entropy updates require an actor update")

        if update_actor:
            sampled_actions, log_probability, deterministic_action = self.actor.sample(
                observations
            )
            actor_q1, actor_q2 = self.critic(observations, sampled_actions)
            sac_actor_loss = (
                self.log_alpha.detach().exp() * log_probability
                - torch.minimum(actor_q1, actor_q2)
            ).mean()
            # Protected phase: penalize deterministic drift from the frozen warm
            # actor so the competent behaviour cannot be destroyed faster than
            # the critics can justify replacing it.
            with torch.no_grad():
                anchor_action = self.anchor_actor.deterministic(observations)
            drift = deterministic_action - anchor_action
            anchor_term = drift.pow(2).mean()
            anchor_penalty = float(anchor_term.detach().cpu())
            actor_loss = sac_actor_loss + self.anchor_coefficient * anchor_term
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad_norm = float(torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.gradient_clip_actor
            ))
            self.actor_optimizer.step()
            self.actor_updates += 1
            with torch.no_grad():
                anchor_metrics = action_drift(
                    self.actor.deterministic(observations), anchor_action
                )
            self._adapt_anchor_coefficient(anchor_metrics["anchor_mean_drift"])

            alpha_loss = -(
                self.log_alpha * (log_probability + self.target_entropy).detach()
            ).mean()
            if update_entropy:
                self.entropy_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.entropy_optimizer.step()
                self.entropy_updates += 1
                self._clamp_alpha()
        else:
            # Preserve finite learner diagnostics during critic-only warm-up
            # without constructing gradients through either policy optimizer.
            with torch.no_grad():
                sampled_actions, log_probability, _ = self.actor.sample(observations)
                actor_q1, actor_q2 = self.critic(observations, sampled_actions)
                actor_loss = (
                    self.log_alpha.exp() * log_probability
                    - torch.minimum(actor_q1, actor_q2)
                ).mean()
                alpha_loss = -(
                    self.log_alpha * (log_probability + self.target_entropy)
                ).mean()

        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_parameter.mul_(1.0 - self.tau).add_(
                    parameter, alpha=self.tau
                )
        self.gradient_updates += 1
        values = {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "entropy_coefficient": self.alpha,
            "mean_target_q": float(target.mean().detach().cpu()),
            "target_q_p05": float(target.flatten().quantile(0.05).detach().cpu()),
            "target_q_p95": float(target.flatten().quantile(0.95).detach().cpu()),
            "mean_q": float(torch.minimum(q1, q2).mean().detach().cpu()),
            "q_p05": float(torch.minimum(q1, q2).flatten().quantile(0.05).detach().cpu()),
            "q_p95": float(torch.minimum(q1, q2).flatten().quantile(0.95).detach().cpu()),
            "td_error_p50": float(td_error.quantile(0.50).cpu()),
            "td_error_p95": float(td_error.quantile(0.95).cpu()),
            "td_error_max": float(td_error.max().cpu()),
            "actor_gradient_norm": actor_grad_norm,
            "critic_gradient_norm": critic_grad_norm,
            "anchor_coefficient": self.anchor_coefficient,
            "anchor_penalty": anchor_penalty,
            **anchor_metrics,
            "mean_log_probability": float(log_probability.mean().detach().cpu()),
            "actor_updated": float(bool(update_actor)),
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError(f"non-finite SAC update: {values}")
        return values

    def _clamp_alpha(self) -> None:
        """Keep the entropy coefficient inside the verified safe interval."""

        import torch

        with torch.no_grad():
            self.log_alpha.clamp_(
                math.log(self.alpha_min), math.log(self.alpha_max)
            )

    def _adapt_anchor_coefficient(self, mean_drift: float) -> None:
        """Raise the anchor immediately on drift; lower it only on demand.

        The coefficient is never reduced here.  Relaxation is an explicit,
        evaluation-driven decision (see ``relax_anchor``) so a transition-count
        threshold alone can never remove the protection.
        """

        if self.anchor_coefficient_max <= 0.0:
            return
        if mean_drift > self.anchor_target_drift:
            base = max(self.anchor_coefficient, 1e-3)
            self.anchor_coefficient = min(
                self.anchor_coefficient_max, base * self.anchor_increase_factor
            )

    def relax_anchor(self) -> float:
        """Reduce the anchor after a competent official evaluation."""

        self.anchor_coefficient = max(
            self.anchor_coefficient_min,
            self.anchor_coefficient * self.anchor_decrease_factor,
        )
        return self.anchor_coefficient

    def set_anchor_from_actor(self) -> None:
        """Freeze the current actor as the anchor (used only at initialization)."""

        self.anchor_actor.load_state_dict(self.actor.state_dict())
        for parameter in self.anchor_actor.parameters():
            parameter.requires_grad_(False)
        self.anchor_actor.eval()

    def anchor_health(self, observations: np.ndarray) -> Dict[str, float]:
        """Drift and saturation probe on real replay observations."""

        import torch

        array = torch.as_tensor(
            np.asarray(observations, dtype=np.float32), device=self.device
        )
        with torch.no_grad():
            current = self.actor.deterministic(array)
            anchor = self.anchor_actor.deterministic(array)
        return {
            **self.actor.health(array),
            **action_drift(current, anchor),
            "anchor_coefficient": self.anchor_coefficient,
            "entropy_coefficient": self.alpha,
        }

    def checkpoint_state(self) -> Dict[str, Any]:
        return {
            "schema_version": "sac_transition_agent_v1",
            "config": {
                "observation_dim": self.observation_dim,
                "action_dim": self.action_dim,
                "hidden_sizes": list(self.hidden_sizes),
                "initial_std": self.initial_std,
                "log_std_min": self.log_std_min,
                "log_std_max": self.log_std_max,
                "pre_tanh_limit": self.actor.pre_tanh_limit,
                "gradient_clip_actor": self.gradient_clip_actor,
                "gradient_clip_critic": self.gradient_clip_critic,
                "critic_loss": self.critic_loss_kind,
                "huber_delta": self.huber_delta,
                "alpha_min": self.alpha_min,
                "alpha_max": self.alpha_max,
                "anchor_target_drift": self.anchor_target_drift,
                "anchor_coefficient_min": self.anchor_coefficient_min,
                "anchor_coefficient_max": self.anchor_coefficient_max,
                "anchor_increase_factor": self.anchor_increase_factor,
                "anchor_decrease_factor": self.anchor_decrease_factor,
                "learning_rates": dict(self.learning_rates),
                "tau": self.tau,
                "target_entropy": self.target_entropy,
            },
            "anchor_coefficient": self.anchor_coefficient,
            "anchor_actor": self.anchor_actor.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "entropy_optimizer": self.entropy_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "gradient_updates": self.gradient_updates,
            "actor_updates": self.actor_updates,
            "entropy_updates": self.entropy_updates,
        }

    @classmethod
    def from_checkpoint_state(cls, state: Mapping[str, Any], *, device: str = "cpu"):
        import torch

        if state.get("schema_version") != "sac_transition_agent_v1":
            raise ValueError("unsupported SAC agent checkpoint")
        config = dict(state["config"])
        learning_rates = dict(config.pop("learning_rates"))
        # Version-1 checkpoints predate the bounded pre-tanh location and the
        # anchored actor phase; they are loaded with the legacy unbounded limit
        # so historical evidence stays reproducible.
        config.setdefault("pre_tanh_limit", math.atanh(1.0 - ACTION_EPSILON))
        alpha = float(torch.as_tensor(state["log_alpha"]).exp())
        config.setdefault("alpha_min", min(alpha, 0.001))
        config.setdefault("alpha_max", max(alpha, 0.02))
        agent = cls(
            **config,
            actor_learning_rate=learning_rates["actor"],
            critic_learning_rate=learning_rates["critic"],
            entropy_learning_rate=learning_rates["entropy"],
            initial_alpha=alpha,
            anchor_coefficient=float(state.get("anchor_coefficient", 0.0) or 0.0),
            device=device,
        )
        agent.actor.load_state_dict(state["actor"], strict=True)
        anchor_state = state.get("anchor_actor")
        agent.anchor_actor.load_state_dict(
            anchor_state if anchor_state is not None else state["actor"], strict=True
        )
        for parameter in agent.anchor_actor.parameters():
            parameter.requires_grad_(False)
        agent.critic.load_state_dict(state["critic"], strict=True)
        agent.target_critic.load_state_dict(state["target_critic"], strict=True)
        agent.actor_optimizer.load_state_dict(state["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(state["critic_optimizer"])
        agent.entropy_optimizer.load_state_dict(state["entropy_optimizer"])
        with torch.no_grad():
            agent.log_alpha.copy_(torch.as_tensor(state["log_alpha"], device=agent.device))
        agent.gradient_updates = int(state.get("gradient_updates", 0))
        # Version-1 checkpoints written before delayed actor updates performed
        # one actor and entropy update for every critic update.
        agent.actor_updates = int(state.get("actor_updates", agent.gradient_updates))
        agent.entropy_updates = int(state.get("entropy_updates", agent.actor_updates))
        return agent


def rebuild_critics_from_actor(
    source_checkpoint: str | Path,
    *,
    device: str = "cpu",
    anchor_coefficient: float = 10.0,
    **overrides: Any,
) -> tuple:
    """Keep a validated actor, discard degraded critics, start clean.

    SAC v2 lost competence because its critics drifted (mean target Q reached
    -63 with a p05 of -216 and a TD-error tail of 192) while the actor itself
    stayed well behaved.  Actor competence and critic health are therefore
    checkpointed and restored independently: the actor transfers verbatim and
    becomes its own anchor, while both critics, both target critics, all critic
    optimizer state and the entropy state are rebuilt from scratch.
    """

    import torch

    payload = torch.load(str(source_checkpoint), map_location=device, weights_only=False)
    state = payload.get("agent", payload)
    if state.get("schema_version") != "sac_transition_agent_v1":
        raise ValueError("unsupported SAC agent checkpoint for critic rebuild")
    config = dict(state["config"])
    learning_rates = dict(config.pop("learning_rates"))
    config.setdefault("pre_tanh_limit", DEFAULT_PRE_TANH_LIMIT)
    config.pop("alpha_min", None)
    config.pop("alpha_max", None)
    settings = {
        "actor_learning_rate": learning_rates["actor"],
        "critic_learning_rate": learning_rates["critic"],
        "entropy_learning_rate": learning_rates["entropy"],
    }
    settings.update(overrides)
    # Explicit overrides win over whatever the degraded run was configured with.
    for key in settings:
        config.pop(key, None)
    agent = cls_factory(config, settings, anchor_coefficient, device)
    # Actor only: nothing else crosses over from the degraded run.
    agent.actor.load_state_dict(state["actor"], strict=True)
    agent.set_anchor_from_actor()
    report = {
        "schema_version": "sac_actor_only_critic_rebuild_v1",
        "source_checkpoint": str(source_checkpoint),
        "actor_transferred": True,
        "anchor_actor_frozen_from_transferred_actor": True,
        "critic_1_reinitialized": True,
        "critic_2_reinitialized": True,
        "target_critic_1_reinitialized": True,
        "target_critic_2_reinitialized": True,
        "critic_optimizer_reinitialized": True,
        "actor_optimizer_reinitialized": True,
        "entropy_state_reinitialized": True,
        "replay_imported": False,
        "source_gradient_updates": int(state.get("gradient_updates", 0)),
        "pre_tanh_limit": agent.actor.pre_tanh_limit,
        "maximum_pre_tanh_jacobian": agent.actor.config()[
            "maximum_pre_tanh_jacobian"
        ],
        "anchor_coefficient": agent.anchor_coefficient,
    }
    return agent, report


def cls_factory(config, settings, anchor_coefficient, device):
    return SACTransitionAgent(
        **config, **settings, anchor_coefficient=float(anchor_coefficient),
        device=device,
    )


def critics_are_independent(agent: Any, reference: Any) -> bool:
    """True when no critic tensor was inherited from ``reference``."""

    import torch

    left = agent.critic.state_dict()
    right = reference.critic.state_dict()
    return not any(
        torch.allclose(left[name], right[name]) for name in left if left[name].numel()
    )


def actor_parameter_sha256(agent: Any) -> str:
    """Stable content hash of the actor's parameters only.

    Used to assert that a critic rebuild left the actor bit-identical.  Critics,
    target critics, optimizers and entropy state are deliberately excluded --
    those are meant to change.
    """

    import hashlib

    import torch

    digest = hashlib.sha256()
    state = agent.actor.state_dict()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(
            torch.as_tensor(state[name]).detach().cpu().contiguous().numpy().tobytes()
        )
    return digest.hexdigest()


def load_sac_transition_policy(checkpoint: str | Path, *, device: str = "cpu") -> SACTransitionAgent:
    import torch

    payload = torch.load(str(checkpoint), map_location=device, weights_only=False)
    state = payload.get("agent", payload)
    return SACTransitionAgent.from_checkpoint_state(state, device=device)
