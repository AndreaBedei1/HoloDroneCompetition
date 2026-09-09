"""Safe v1-to-v3 policy expansion for the multi-gate warm start."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from marine_race_arena.learning.bc_train import BCPolicy, load_policy, save_policy
from marine_race_arena.learning.config import ACTION_DIM, OBS_DIM, OBS_ENCODING_VERSION
from marine_race_arena.learning.config_v3 import (
    FEATURE_NAMES_V3,
    OBS_DIM_V3,
    OBS_ENCODING_VERSION_V3,
)


def expand_bc_v1_to_v3(
    bc_v1: BCPolicy,
    *,
    new_column_scale: float = 0.0,
    seed: int = 0,
) -> BCPolicy:
    """Expand a frozen v1 BC policy while preserving its neutral-feature output.

    The original 36 input columns, normalization buffers, hidden layers, and
    action head are copied exactly.  New columns default to zero; a small
    deterministic initialization can be requested for experiments, but parity
    is guaranteed only with the default zero initialization.
    """
    import torch
    import torch.nn as nn

    if int(bc_v1.obs_dim) != OBS_DIM:
        raise ValueError(f"expected v1 obs_dim={OBS_DIM}, got {bc_v1.obs_dim}")
    if int(bc_v1.act_dim) != ACTION_DIM:
        raise ValueError(f"expected action_dim={ACTION_DIM}, got {bc_v1.act_dim}")
    if new_column_scale < 0.0:
        raise ValueError("new_column_scale must be non-negative")

    mean = np.zeros(OBS_DIM_V3, dtype=np.float32)
    std = np.ones(OBS_DIM_V3, dtype=np.float32)
    mean[:OBS_DIM] = bc_v1.obs_mean.detach().cpu().numpy()
    std[:OBS_DIM] = bc_v1.obs_std.detach().cpu().numpy()
    expanded = BCPolicy(
        obs_dim=OBS_DIM_V3,
        act_dim=ACTION_DIM,
        hidden_sizes=bc_v1.hidden_sizes,
        obs_mean=mean,
        obs_std=std,
    )

    source_linears = [m for m in bc_v1.extractor if isinstance(m, nn.Linear)]
    target_linears = [m for m in expanded.extractor if isinstance(m, nn.Linear)]
    if len(source_linears) != len(target_linears):
        raise ValueError("hidden architecture mismatch during v1-to-v3 transfer")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    with torch.no_grad():
        for index, (source, target) in enumerate(zip(source_linears, target_linears)):
            if index == 0:
                if target.weight.shape[1] != OBS_DIM_V3:
                    raise ValueError("expanded first layer has the wrong input dimension")
                target.weight.zero_()
                target.weight[:, :OBS_DIM].copy_(source.weight)
                if new_column_scale > 0.0:
                    noise = torch.randn(
                        target.weight[:, OBS_DIM:].shape,
                        generator=generator,
                        dtype=target.weight.dtype,
                    )
                    target.weight[:, OBS_DIM:].copy_(noise * float(new_column_scale))
                target.bias.copy_(source.bias)
            else:
                target.weight.copy_(source.weight)
                target.bias.copy_(source.bias)
        target_head = expanded.head
        target_head.weight.copy_(bc_v1.head.weight)
        target_head.bias.copy_(bc_v1.head.bias)
    expanded.eval()
    return expanded


def save_v3_policy(policy: BCPolicy, path: Any) -> None:
    if int(policy.obs_dim) != OBS_DIM_V3:
        raise ValueError(f"v3 policy must have obs_dim={OBS_DIM_V3}, got {policy.obs_dim}")
    save_policy(
        policy,
        path,
        obs_encoding_version=OBS_ENCODING_VERSION_V3,
        feature_names=FEATURE_NAMES_V3,
    )


def load_v3_policy(path: Any) -> BCPolicy:
    """Load only an explicitly versioned v3 BC policy."""
    import torch

    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    version = checkpoint.get("obs_encoding_version")
    if version != OBS_ENCODING_VERSION_V3:
        raise ValueError(
            f"incompatible observation encoding: model={version!r}, "
            f"controller={OBS_ENCODING_VERSION_V3!r}"
        )
    if int(checkpoint.get("obs_dim", -1)) != OBS_DIM_V3:
        raise ValueError(
            f"incompatible observation dimension: model={checkpoint.get('obs_dim')}, "
            f"controller={OBS_DIM_V3}"
        )
    names = checkpoint.get("feature_names")
    if names is not None and tuple(names) != tuple(FEATURE_NAMES_V3):
        raise ValueError("v3 model feature order does not match the controller")
    return load_policy(path)


def transfer_bc_v1_to_v3_ppo(bc_v1: BCPolicy, ppo_model: Any) -> BCPolicy:
    """Expand v1, copy it into a v3 PPO policy, and stamp compatibility."""
    from marine_race_arena.learning.rl_train import transfer_bc_to_ppo

    expanded = expand_bc_v1_to_v3(bc_v1)
    transfer_bc_to_ppo(expanded, ppo_model)
    ppo_model.obs_encoding_version = OBS_ENCODING_VERSION_V3
    ppo_model.action_contract_version = "surge_sway_heave_yaw_pm1_v1"
    return expanded


def transfer_file(source: Any, destination: Any) -> BCPolicy:
    """Transfer a frozen v1 checkpoint to a versioned v3 BC checkpoint."""
    import torch

    raw = torch.load(Path(source), map_location="cpu", weights_only=False)
    source_version = raw.get("obs_encoding_version")
    if source_version not in (None, OBS_ENCODING_VERSION):
        raise ValueError(f"source model is not observation v1: {source_version!r}")
    policy = expand_bc_v1_to_v3(load_policy(source))
    save_v3_policy(policy, destination)
    return policy
