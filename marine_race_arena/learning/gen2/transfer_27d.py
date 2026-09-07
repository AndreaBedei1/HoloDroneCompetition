"""Semantic warm-start transfer from 35-D parent policy to 27-D Gen-2 model.

Maps the 24 retained features from their legacy columns in the 35-D parent,
copies normalization statistics, biases, encoder layers, LSTM trunks, and
actor/critic heads, and initializes the 3 new orientation features to zero
(neutral weights) with unit std.

The resulting network is fully trainable across all layers.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch as th

from marine_race_arena.learning.config_local_transition import (
    FEATURE_NAMES_LOCAL_TRANSITION,
    OBS_DIM_LOCAL_TRANSITION,
)
from marine_race_arena.learning.config_local_transition_27d import (
    FEATURE_NAMES_LOCAL_TRANSITION_27D,
    NEW_ORIENTATION_FEATURES_27D,
    OBS_DIM_LOCAL_TRANSITION_27D,
    REMOVED_FEATURES_FROM_35D,
)
from marine_race_arena.learning.gen2.recurrent_policy import (
    GEN2_27D_ARCH,
    Gen2RecurrentController,
    build_gen2_policy_for_training,
    build_gen2_recurrent_ppo,
)


FIRST_WEIGHT_SUFFIX = "encoder.0.weight"
FIRST_BIAS_SUFFIX = "encoder.0.bias"
MEAN_SUFFIX = "obs_mean"
STD_SUFFIX = "obs_std"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_feature_index_mapping() -> Tuple[Dict[int, int], List[int], List[int]]:
    """Return (retained_map: new_idx -> old_idx, new_indices, removed_indices)."""
    old_names = list(FEATURE_NAMES_LOCAL_TRANSITION)
    new_names = list(FEATURE_NAMES_LOCAL_TRANSITION_27D)

    retained_map: Dict[int, int] = {}
    new_indices: List[int] = []

    for new_idx, name in enumerate(new_names):
        if name in old_names:
            old_idx = old_names.index(name)
            retained_map[new_idx] = old_idx
        else:
            new_indices.append(new_idx)

    removed_indices = [
        old_idx for old_idx, name in enumerate(old_names)
        if name not in new_names
    ]

    assert len(retained_map) == 24
    assert len(new_indices) == 3
    assert len(removed_indices) == 11
    return retained_map, new_indices, removed_indices


def _transfer_tensor(
    key: str,
    parent_tensor: th.Tensor,
    child_tensor: th.Tensor,
    retained_map: Dict[int, int],
    new_indices: List[int],
) -> th.Tensor:
    """Transform or copy one parameter tensor from parent to child."""
    if tuple(parent_tensor.shape) == tuple(child_tensor.shape):
        return parent_tensor.detach().clone()

    if key.endswith(FIRST_WEIGHT_SUFFIX):
        if (
            parent_tensor.shape[0] != child_tensor.shape[0]
            or parent_tensor.shape[1] != OBS_DIM_LOCAL_TRANSITION
            or child_tensor.shape[1] != OBS_DIM_LOCAL_TRANSITION_27D
        ):
            raise ValueError(
                f"unexpected weight shapes for {key}: "
                f"{parent_tensor.shape} -> {child_tensor.shape}"
            )
        out = th.zeros_like(child_tensor)
        for new_idx, old_idx in retained_map.items():
            out[:, new_idx] = parent_tensor[:, old_idx]
        for new_idx in new_indices:
            out[:, new_idx] = 0.0
        return out

    if key.endswith(MEAN_SUFFIX):
        if (
            parent_tensor.shape != (OBS_DIM_LOCAL_TRANSITION,)
            or child_tensor.shape != (OBS_DIM_LOCAL_TRANSITION_27D,)
        ):
            raise ValueError(f"unexpected mean shapes for {key}")
        out = th.zeros_like(child_tensor)
        for new_idx, old_idx in retained_map.items():
            out[new_idx] = parent_tensor[old_idx]
        for new_idx in new_indices:
            out[new_idx] = 0.0
        return out

    if key.endswith(STD_SUFFIX):
        if (
            parent_tensor.shape != (OBS_DIM_LOCAL_TRANSITION,)
            or child_tensor.shape != (OBS_DIM_LOCAL_TRANSITION_27D,)
        ):
            raise ValueError(f"unexpected std shapes for {key}")
        out = th.ones_like(child_tensor)
        for new_idx, old_idx in retained_map.items():
            out[new_idx] = parent_tensor[old_idx]
        for new_idx in new_indices:
            out[new_idx] = 1.0
        return out

    raise ValueError(
        f"tensor {key!r} shape incompatible: "
        f"{tuple(parent_tensor.shape)} -> {tuple(child_tensor.shape)}"
    )


def transfer_parent_35d_to_27d(
    parent_path: str | Path,
    *,
    env=None,
    learning_rate: float = 3e-5,
    n_steps: int = 256,
    batch_size: int = 128,
    n_epochs: int = 2,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    clip_range: float = 0.08,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    target_kl: float = 0.01,
    seed: int = 0,
    device: str = "cpu",
):
    """Build a 27-D policy and warm-start from the 35-D parent checkpoint."""
    from sb3_contrib import RecurrentPPO

    parent_path = Path(parent_path)
    parent_sha256 = sha256_file(parent_path)
    parent = RecurrentPPO.load(str(parent_path), device="cpu")
    if tuple(parent.observation_space.shape) != (OBS_DIM_LOCAL_TRANSITION,):
        raise ValueError(
            f"parent must use the 35-D contract, got {parent.observation_space}"
        )

    retained_map, new_indices, _removed = get_feature_index_mapping()

    build_kwargs = dict(
        architecture=GEN2_27D_ARCH,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        target_kl=target_kl,
        seed=seed,
        device=device,
    )

    if env is None:
        child = build_gen2_policy_for_training(**build_kwargs)
    else:
        child = build_gen2_recurrent_ppo(env, **build_kwargs)

    old_state = parent.policy.state_dict()
    new_state = child.policy.state_dict()

    transferred = {}
    for key, new_tensor in new_state.items():
        if key not in old_state:
            raise KeyError(f"child key {key!r} not found in parent policy state")
        transferred[key] = _transfer_tensor(
            key, old_state[key], new_tensor, retained_map, new_indices
        )

    child.policy.load_state_dict(transferred, strict=True)

    # Ensure all parameters remain trainable for subsequent training
    for param in child.policy.parameters():
        param.requires_grad = True

    child.gen2_parent_checkpoint = str(parent_path)
    child.gen2_parent_sha256 = parent_sha256
    child.gen2_transfer = "semantic_35d_to_27d_warm_start_v1"
    return child


def verify_transfer_27d_tensors(
    parent_path: str | Path, child_model
) -> Dict[str, Any]:
    """Audit the transferred tensors and verify weight mapping and zeroed features."""
    from sb3_contrib import RecurrentPPO

    parent = RecurrentPPO.load(str(parent_path), device="cpu")
    old_state = parent.policy.state_dict()
    new_state = child_model.policy.state_dict()

    retained_map, new_indices, removed_indices = get_feature_index_mapping()
    mismatches = []
    same_shape_count = 0

    for key, old_val in old_state.items():
        new_val = new_state[key]
        if old_val.shape == new_val.shape:
            if not th.equal(old_val.cpu(), new_val.cpu()):
                mismatches.append(f"identical_shape_mismatch:{key}")
            else:
                same_shape_count += 1
            continue

        if key.endswith(FIRST_WEIGHT_SUFFIX):
            for new_i, old_i in retained_map.items():
                if not th.equal(old_val[:, old_i].cpu(), new_val[:, new_i].cpu()):
                    mismatches.append(f"col_mismatch:{key}:new{new_i}!=old{old_i}")
            for new_i in new_indices:
                if th.count_nonzero(new_val[:, new_i]).item() != 0:
                    mismatches.append(f"nonzero_new_col:{key}:new{new_i}")

        elif key.endswith(MEAN_SUFFIX):
            for new_i, old_i in retained_map.items():
                if not th.equal(old_val[old_i].cpu(), new_val[new_i].cpu()):
                    mismatches.append(f"mean_mismatch:{key}:new{new_i}!=old{old_i}")
            for new_i in new_indices:
                if float(new_val[new_i].item()) != 0.0:
                    mismatches.append(f"nonzero_new_mean:{key}:new{new_i}")

        elif key.endswith(STD_SUFFIX):
            for new_i, old_i in retained_map.items():
                if not th.equal(old_val[old_i].cpu(), new_val[new_i].cpu()):
                    mismatches.append(f"std_mismatch:{key}:new{new_i}!=old{old_i}")
            for new_i in new_indices:
                if float(new_val[new_i].item()) != 1.0:
                    mismatches.append(f"nonunit_new_std:{key}:new{new_i}")
        else:
            mismatches.append(f"unhandled_shape_mismatch:{key}")

    all_trainable = all(
        p.requires_grad for p in child_model.policy.parameters()
    )

    return {
        "verified": len(mismatches) == 0,
        "mismatches": mismatches,
        "same_shape_tensors_copied": same_shape_count,
        "retained_features_mapped": len(retained_map),
        "new_features_neutral": len(new_indices),
        "removed_features_count": len(removed_indices),
        "all_parameters_trainable": all_trainable,
        "parent_sha256": sha256_file(parent_path),
    }
