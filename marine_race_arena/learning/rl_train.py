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
    """Copy the BC MLP weights into the PPO policy, absorbing BC's obs normalization.

    The BC policy normalizes observations internally,
    ``y = W1 @ ((x - mean) / std) + b1``, while SB3's PPO ``MlpPolicy`` consumes raw
    observations. To make the transfer exact for *any* normalization, the BC
    normalization is folded into PPO's first policy layer:

        ``W_new = W1 / std`` (column-wise),   ``b_new = b1 - W_new @ mean``

    The remaining hidden layers and the action head are copied directly; the value
    network is left for PPO to learn. After this transfer, PPO's deterministic action
    equals the BC network output (before the BC clip) for the same raw observation.
    """
    import torch
    import torch.nn as nn

    ppo_policy_net = ppo_model.policy.mlp_extractor.policy_net
    bc_linears = [m for m in bc_policy.extractor if isinstance(m, nn.Linear)]
    ppo_linears = [m for m in ppo_policy_net if isinstance(m, nn.Linear)]
    if len(bc_linears) != len(ppo_linears):
        raise ValueError(
            f"architecture mismatch: BC has {len(bc_linears)} hidden linears, "
            f"PPO has {len(ppo_linears)}"
        )
    if not bc_linears:
        raise ValueError("BC policy has no hidden linear layers to transfer")

    mean = bc_policy.obs_mean.detach().to(torch.float32)
    std = bc_policy.obs_std.detach().to(torch.float32).clone()
    # Guard against invalid/near-zero std (features with no variance are left as-is).
    std = torch.where(std.abs() < 1e-6, torch.ones_like(std), std)

    with torch.no_grad():
        for idx, (ppo_layer, bc_layer) in enumerate(zip(ppo_linears, bc_linears)):
            if ppo_layer.weight.shape != bc_layer.weight.shape:
                raise ValueError(f"layer shape mismatch {ppo_layer.weight.shape} vs {bc_layer.weight.shape}")
            if idx == 0:
                if bc_layer.weight.shape[1] != std.shape[0]:
                    raise ValueError(
                        f"first-layer input {bc_layer.weight.shape[1]} != normalization dim {std.shape[0]}"
                    )
                w_new = bc_layer.weight / std.reshape(1, -1)   # divide each input column j by std[j]
                b_new = bc_layer.bias - w_new @ mean
                ppo_layer.weight.copy_(w_new)
                ppo_layer.bias.copy_(b_new)
            else:
                ppo_layer.weight.copy_(bc_layer.weight)
                ppo_layer.bias.copy_(bc_layer.bias)
        if ppo_model.policy.action_net.weight.shape != bc_policy.head.weight.shape:
            raise ValueError("action head shape mismatch between BC and PPO")
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
