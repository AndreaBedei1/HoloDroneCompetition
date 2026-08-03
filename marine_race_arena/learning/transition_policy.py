"""Feed-forward PPO construction and selective sequence-policy warm start."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

from marine_race_arena.learning.config_local_transition import (
    FEATURE_NAMES_LOCAL_TRANSITION,
    LOCAL_TO_SEQUENCE_FEATURE_MAP,
    OBS_DIM_LOCAL_TRANSITION,
    OBS_ENCODING_VERSION_LOCAL_TRANSITION,
)
from marine_race_arena.learning.config_sequence import (
    FEATURE_NAMES_SEQUENCE,
    OBS_DIM_SEQUENCE,
    OBS_ENCODING_VERSION_SEQUENCE,
)


def build_local_transition_ppo(
    env: Any,
    *,
    seed: int,
    learning_rate: Any,
    hidden_sizes: Sequence[int] = (256, 256),
    n_steps: int = 1024,
    batch_size: int = 256,
    n_epochs: int = 3,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    clip_range: float = 0.05,
    target_kl: float = 0.004,
    ent_coef: float = 0.0002,
) -> Any:
    import torch.nn as nn
    from stable_baselines3 import PPO

    model = PPO(
        "MlpPolicy",
        env,
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
        policy_kwargs={
            "net_arch": dict(pi=list(hidden_sizes), vf=list(hidden_sizes)),
            "activation_fn": nn.Tanh,
        },
        seed=int(seed),
        device="cpu",
        verbose=1,
    )
    model.obs_encoding_version = OBS_ENCODING_VERSION_LOCAL_TRANSITION
    model.longrun_observation_version = OBS_ENCODING_VERSION_LOCAL_TRANSITION
    model.longrun_architecture = "feedforward_ppo"
    model.longrun_policy_mode = "feedforward"
    model.longrun_frame_stack = 1
    return model


def selective_warm_start_from_sequence(
    source: Any,
    target: Any,
) -> Dict[str, Any]:
    """Copy reusable policy columns/tensors; leave removed shortcuts unreachable."""

    import torch

    source_shape = tuple(getattr(source.observation_space, "shape", ()) or ())
    target_shape = tuple(getattr(target.observation_space, "shape", ()) or ())
    if source_shape != (OBS_DIM_SEQUENCE,):
        raise ValueError(f"source policy observation shape is {source_shape}, expected {(OBS_DIM_SEQUENCE,)}")
    if target_shape != (OBS_DIM_LOCAL_TRANSITION,):
        raise ValueError(f"target policy observation shape is {target_shape}, expected {(OBS_DIM_LOCAL_TRANSITION,)}")
    source_version = getattr(source, "obs_encoding_version", None)
    if source_version not in {None, OBS_ENCODING_VERSION_SEQUENCE}:
        raise ValueError(f"source observation version is {source_version!r}")

    source_state = source.policy.state_dict()
    target_state = target.policy.state_dict()
    source_index = {name: index for index, name in enumerate(FEATURE_NAMES_SEQUENCE)}
    target_index = {name: index for index, name in enumerate(FEATURE_NAMES_LOCAL_TRANSITION)}
    mapped = {
        local: sequence
        for local, sequence in LOCAL_TO_SEQUENCE_FEATURE_MAP.items()
        if local in target_index and sequence in source_index
    }
    new_temporal = [
        name for name in FEATURE_NAMES_LOCAL_TRANSITION if name not in mapped
    ]
    removed = [
        name for name in FEATURE_NAMES_SEQUENCE if name not in set(mapped.values())
    ]

    updated = {}
    first_layers = {
        "mlp_extractor.policy_net.0.weight",
        "mlp_extractor.value_net.0.weight",
    }
    copied_tensors = []
    for name, destination in target_state.items():
        origin = source_state.get(name)
        if name in first_layers:
            if origin is None or origin.ndim != 2 or destination.ndim != 2:
                raise ValueError(f"missing compatible first layer {name}")
            value = torch.zeros_like(destination)
            for local_name, sequence_name in mapped.items():
                value[:, target_index[local_name]] = origin[:, source_index[sequence_name]]
            updated[name] = value
            copied_tensors.append(name)
        elif origin is not None and tuple(origin.shape) == tuple(destination.shape):
            updated[name] = origin.detach().clone()
            copied_tensors.append(name)
        else:
            updated[name] = destination
    target.policy.load_state_dict(updated, strict=True)
    if any(not parameter.requires_grad for parameter in target.policy.parameters()):
        raise ValueError("selective warm start must not freeze parameters")
    target.num_timesteps = 0
    target.local_transition_initialization_mode = "selective_warm_start"
    return {
        "source_observation_version": source_version,
        "source_observation_dim": OBS_DIM_SEQUENCE,
        "target_observation_version": OBS_ENCODING_VERSION_LOCAL_TRANSITION,
        "target_observation_dim": OBS_DIM_LOCAL_TRANSITION,
        "mapped_features": mapped,
        "new_zero_initialized_features": new_temporal,
        "removed_source_features": removed,
        "copied_tensors": copied_tensors,
        "optimizer_state_copied": False,
        "all_parameters_trainable": True,
    }


def initialize_local_transition_policy(
    target: Any,
    *,
    mode: str,
    source_checkpoint: str | None = None,
) -> Dict[str, Any]:
    if mode == "scratch":
        target.num_timesteps = 0
        target.local_transition_initialization_mode = "scratch"
        return {
            "mode": "scratch",
            "target_observation_version": OBS_ENCODING_VERSION_LOCAL_TRANSITION,
            "target_observation_dim": OBS_DIM_LOCAL_TRANSITION,
            "all_parameters_trainable": True,
        }
    if mode != "selective_warm_start" or not source_checkpoint:
        raise ValueError("warm start requires source_checkpoint")
    from stable_baselines3 import PPO

    source = PPO.load(source_checkpoint, device="cpu")
    report = selective_warm_start_from_sequence(source, target)
    report.update({"mode": mode, "source_checkpoint": source_checkpoint})
    return report


def predict_local_transition_action(model: Any, observation: np.ndarray) -> np.ndarray:
    action, _ = model.predict(
        np.asarray(observation, dtype=np.float32), deterministic=True
    )
    return np.asarray(action, dtype=np.float32).reshape(4)

