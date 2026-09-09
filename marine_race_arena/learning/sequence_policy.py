"""Feed-forward and recurrent PPO construction for the sequence experiment."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from marine_race_arena.learning.config_sequence import (
    OBS_DIM_SEQUENCE,
    OBS_ENCODING_VERSION_SEQUENCE,
)
from marine_race_arena.learning.config_v3 import OBS_DIM_V3


def build_sequence_ppo(
    env: Any,
    *,
    architecture: str,
    seed: int,
    learning_rate: Any,
    hidden_sizes: Sequence[int] = (256, 256),
    n_steps: int = 2048,
    batch_size: int = 256,
    n_epochs: int = 3,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    clip_range: float = 0.05,
    target_kl: float = 0.004,
    ent_coef: float = 0.0002,
) -> Any:
    import torch.nn as nn

    common: Dict[str, Any] = dict(
        env=env,
        learning_rate=learning_rate,
        n_steps=int(n_steps),
        batch_size=int(batch_size),
        n_epochs=int(n_epochs),
        gamma=float(gamma),
        gae_lambda=float(gae_lambda),
        clip_range=float(clip_range),
        target_kl=float(target_kl),
        ent_coef=float(ent_coef),
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=int(seed),
        device="cpu",
        verbose=1,
    )
    net_arch = dict(pi=list(hidden_sizes), vf=list(hidden_sizes))
    if architecture == "feedforward_ppo":
        from stable_baselines3 import PPO

        model = PPO(
            "MlpPolicy",
            policy_kwargs=dict(net_arch=net_arch, activation_fn=nn.Tanh),
            **common,
        )
    elif architecture == "recurrent_ppo_lstm":
        from sb3_contrib import RecurrentPPO

        model = RecurrentPPO(
            "MlpLstmPolicy",
            policy_kwargs=dict(
                net_arch=net_arch,
                activation_fn=nn.Tanh,
                lstm_hidden_size=OBS_DIM_SEQUENCE,
                n_lstm_layers=1,
                shared_lstm=False,
                enable_critic_lstm=True,
            ),
            **common,
        )
    else:
        raise ValueError(f"unknown PPO architecture {architecture!r}")
    model.obs_encoding_version = OBS_ENCODING_VERSION_SEQUENCE
    model.longrun_observation_version = OBS_ENCODING_VERSION_SEQUENCE
    model.longrun_policy_mode = "recurrent" if architecture.startswith("recurrent") else "feedforward"
    model.longrun_architecture = architecture
    model.longrun_frame_stack = 1
    return model


def _expanded_tensor(origin: Any, destination: Any) -> Any:
    import torch

    if origin.shape == destination.shape:
        return origin
    if (origin.ndim == destination.ndim == 2
            and origin.shape[0] == destination.shape[0]
            and origin.shape[1] == OBS_DIM_V3
            and destination.shape[1] == OBS_DIM_SEQUENCE):
        value = torch.zeros_like(destination)
        value[:, :OBS_DIM_V3] = origin
        return value
    raise ValueError(f"cannot transfer tensor {tuple(origin.shape)} -> {tuple(destination.shape)}")


def initialize_from_ppo900462(source: Any, target: Any, architecture: str) -> Dict[str, Any]:
    """Initialize a v4 policy solely from the selected PPO checkpoint.

    Feed-forward transfer is exact for the first 59 features (new feature
    columns start at zero).  Recurrent transfer copies both PPO MLPs and heads;
    its LSTMs start as memory-free near-identity transforms and are then learned.
    """
    import torch

    source_state = source.policy.state_dict()
    target_state = target.policy.state_dict()
    transferred = []
    if architecture == "feedforward_ppo":
        updated = {}
        for name, destination in target_state.items():
            if name not in source_state:
                raise ValueError(f"source PPO is missing {name}")
            updated[name] = _expanded_tensor(source_state[name], destination)
            transferred.append(name)
        target.policy.load_state_dict(updated, strict=True)
    elif architecture == "recurrent_ppo_lstm":
        updated = dict(target_state)
        for name, destination in target_state.items():
            if name.startswith("lstm_"):
                continue
            origin = source_state.get(name)
            if origin is not None:
                updated[name] = _expanded_tensor(origin, destination)
                transferred.append(name)

        hidden = OBS_DIM_SEQUENCE
        for prefix in ("lstm_actor", "lstm_critic"):
            weight_ih = torch.zeros_like(updated[f"{prefix}.weight_ih_l0"])
            weight_hh = torch.zeros_like(updated[f"{prefix}.weight_hh_l0"])
            bias_ih = torch.zeros_like(updated[f"{prefix}.bias_ih_l0"])
            bias_hh = torch.zeros_like(updated[f"{prefix}.bias_hh_l0"])
            weight_ih[2 * hidden:3 * hidden, :hidden] = torch.eye(hidden)
            bias_ih[:hidden] = 5.0       # input gate open
            bias_ih[hidden:2 * hidden] = -5.0  # forget gate initially closed
            bias_ih[3 * hidden:] = 5.0   # output gate open
            updated[f"{prefix}.weight_ih_l0"] = weight_ih
            updated[f"{prefix}.weight_hh_l0"] = weight_hh
            updated[f"{prefix}.bias_ih_l0"] = bias_ih
            updated[f"{prefix}.bias_hh_l0"] = bias_hh
        target.policy.load_state_dict(updated, strict=True)
    else:
        raise ValueError(architecture)
    with torch.no_grad():
        target.policy.log_std.copy_(source.policy.log_std)
    return {
        "source_observation_dim": int(source.observation_space.shape[0]),
        "target_observation_dim": int(target.observation_space.shape[0]),
        "architecture": architecture,
        "transferred_tensors": transferred,
        "new_observation_columns_initialized_to_zero": architecture == "feedforward_ppo",
        "recurrent_memory_initialization": (
            "near_identity_memory_free" if architecture == "recurrent_ppo_lstm" else None
        ),
    }


def load_sequence_model(path: str, *, env: Any = None, architecture: str) -> Any:
    if architecture == "recurrent_ppo_lstm":
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO.load(path, env=env, device="cpu")
    from stable_baselines3 import PPO

    return PPO.load(path, env=env, device="cpu")


def predict_sequence_action(
    model: Any,
    observation: np.ndarray,
    *,
    architecture: str,
    recurrent_state: Any,
    episode_start: np.ndarray,
) -> tuple[np.ndarray, Any]:
    if architecture == "recurrent_ppo_lstm":
        action, next_state = model.predict(
            observation,
            state=recurrent_state,
            episode_start=episode_start,
            deterministic=True,
        )
        return np.asarray(action, dtype=np.float32), next_state
    action, _ = model.predict(observation, deterministic=True)
    return np.asarray(action, dtype=np.float32), None
