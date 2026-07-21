"""PPO training scaffold and BC->PPO weight transfer (Stable-Baselines3).

Provides an environment factory, a PPO builder whose policy network mirrors the
:class:`BCPolicy` architecture (so BC weights transfer exactly), a verified
BC->PPO warm-start, and a short ``train_ppo`` entry point that checkpoints and
logs. Stable-Baselines3, Gymnasium and PyTorch are RL-only dependencies.

Reinforcement learning against HoloOcean is sample-expensive; use the fallback
backend only for fast plumbing smoke tests, and real HoloOcean for actual
training runs. This module never claims a policy works — that requires closed-loop
evaluation under the unchanged referee (see ``evaluate_policy``/``docs/rl_progress.md``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from marine_race_arena.learning.gym_env import MarineRaceGymEnv


def make_env(track: str, *, seed: int = 0, **env_kwargs) -> MarineRaceGymEnv:
    """Construct one MarineRaceGymEnv (SB3 wraps it in a DummyVecEnv itself)."""
    return MarineRaceGymEnv(track, seed=seed, **env_kwargs)


def build_ppo(env, *, hidden_sizes: Sequence[int] = (256, 256), seed: int = 0, **ppo_kwargs):
    """Build a PPO whose policy/value nets match the BC architecture (Tanh MLP)."""
    from stable_baselines3 import PPO
    import torch.nn as nn

    policy_kwargs = dict(net_arch=dict(pi=list(hidden_sizes), vf=list(hidden_sizes)), activation_fn=nn.Tanh)
    defaults: Dict[str, Any] = dict(
        n_steps=256, batch_size=64, n_epochs=4, gamma=0.99, gae_lambda=0.95,
        learning_rate=3e-4, verbose=0, seed=seed, device="cpu",
    )
    defaults.update(ppo_kwargs)
    return PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, **defaults)


def transfer_bc_to_ppo(bc_policy, ppo_model) -> None:
    """Copy the BC MLP weights into the PPO policy (extractor + action head).

    The BC extractor mirrors SB3's ``mlp_extractor.policy_net`` and the BC head
    mirrors ``action_net``; the value network is left for PPO to learn. After this
    transfer, PPO's deterministic action equals the BC network output (before the
    BC clip) for the same observation, when observations are not renormalized.
    """
    import torch
    import torch.nn as nn

    ppo_policy_net = ppo_model.policy.mlp_extractor.policy_net
    with torch.no_grad():
        bc_linears = [m for m in bc_policy.extractor if isinstance(m, nn.Linear)]
        ppo_linears = [m for m in ppo_policy_net if isinstance(m, nn.Linear)]
        if len(bc_linears) != len(ppo_linears):
            raise ValueError(
                f"architecture mismatch: BC has {len(bc_linears)} hidden linears, "
                f"PPO has {len(ppo_linears)}"
            )
        for ppo_layer, bc_layer in zip(ppo_linears, bc_linears):
            if ppo_layer.weight.shape != bc_layer.weight.shape:
                raise ValueError(f"layer shape mismatch {ppo_layer.weight.shape} vs {bc_layer.weight.shape}")
            ppo_layer.weight.copy_(bc_layer.weight)
            ppo_layer.bias.copy_(bc_layer.bias)
        ppo_model.policy.action_net.weight.copy_(bc_policy.head.weight)
        ppo_model.policy.action_net.bias.copy_(bc_policy.head.bias)


def train_ppo(
    track: str,
    *,
    total_timesteps: int = 5000,
    seed: int = 0,
    output_dir: Optional[str] = None,
    bc_policy=None,
    checkpoint_every: int = 0,
    env_kwargs: Optional[Dict[str, Any]] = None,
    hidden_sizes: Sequence[int] = (256, 256),
    **ppo_kwargs,
):
    """Train PPO on the given track. Returns the trained model.

    If ``bc_policy`` is provided, PPO is warm-started from it. Checkpoints, a CSV
    log and the final model are written under ``output_dir`` when given. Uses the
    fallback backend unless ``env_kwargs`` selects the HoloOcean adapter.
    """
    from stable_baselines3.common.logger import configure

    env = make_env(track, seed=seed, **(env_kwargs or {}))
    model = build_ppo(env, hidden_sizes=hidden_sizes, seed=seed, **ppo_kwargs)
    if bc_policy is not None:
        transfer_bc_to_ppo(bc_policy, model)

    callbacks = []
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        model.set_logger(configure(str(out), ["csv"]))
        if checkpoint_every and checkpoint_every > 0:
            from stable_baselines3.common.callbacks import CheckpointCallback

            callbacks.append(
                CheckpointCallback(save_freq=checkpoint_every, save_path=str(out / "checkpoints"), name_prefix="ppo")
            )

    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks or None, progress_bar=False)
        if output_dir is not None:
            model.save(str(Path(output_dir) / "ppo_model"))
    finally:
        env.close()
    return model
