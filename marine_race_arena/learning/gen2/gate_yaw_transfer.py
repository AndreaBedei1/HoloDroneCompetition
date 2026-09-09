"""Exact 35 -> 38 recurrent-policy expansion for gate-plane yaw."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch as th

from marine_race_arena.learning.config_local_transition import (
    OBS_DIM_LOCAL_TRANSITION,
)
from marine_race_arena.learning.config_local_transition_gate_yaw import (
    OBS_DIM_LOCAL_TRANSITION_GATE_YAW,
)
from marine_race_arena.learning.gen2.recurrent_policy import (
    GEN2_GATE_YAW_ARCH,
    Gen2RecurrentController,
    build_gen2_policy_for_training,
)


FIRST_WEIGHT_SUFFIX = "encoder.0.weight"
MEAN_SUFFIX = "obs_mean"
STD_SUFFIX = "obs_std"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expanded_tensor(key: str, parent: th.Tensor, child: th.Tensor) -> th.Tensor:
    if tuple(parent.shape) == tuple(child.shape):
        return parent.detach().clone()
    if key.endswith(FIRST_WEIGHT_SUFFIX):
        if parent.shape[0] != child.shape[0] or parent.shape[1] != OBS_DIM_LOCAL_TRANSITION:
            raise ValueError(f"unexpected input weight shapes for {key}: {parent.shape} -> {child.shape}")
        out = th.zeros_like(child)
        out[:, :OBS_DIM_LOCAL_TRANSITION].copy_(parent)
        return out
    if key.endswith(MEAN_SUFFIX):
        if parent.shape != (OBS_DIM_LOCAL_TRANSITION,) or child.shape != (
            OBS_DIM_LOCAL_TRANSITION_GATE_YAW,
        ):
            raise ValueError(f"unexpected normalization mean shapes for {key}")
        out = th.zeros_like(child)
        out[:OBS_DIM_LOCAL_TRANSITION].copy_(parent)
        return out
    if key.endswith(STD_SUFFIX):
        if parent.shape != (OBS_DIM_LOCAL_TRANSITION,) or child.shape != (
            OBS_DIM_LOCAL_TRANSITION_GATE_YAW,
        ):
            raise ValueError(f"unexpected normalization std shapes for {key}")
        out = th.ones_like(child)
        out[:OBS_DIM_LOCAL_TRANSITION].copy_(parent)
        return out
    raise ValueError(
        f"policy tensor {key!r} changed shape unexpectedly: "
        f"{tuple(parent.shape)} -> {tuple(child.shape)}"
    )


def transfer_parent_to_gate_yaw(
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
    """Build a 38-D policy and copy every possible parent tensor exactly."""

    from sb3_contrib import RecurrentPPO

    parent_path = Path(parent_path)
    parent = RecurrentPPO.load(str(parent_path), device="cpu")
    if tuple(parent.observation_space.shape) != (OBS_DIM_LOCAL_TRANSITION,):
        raise ValueError(
            f"parent must use the immutable 35-D contract, got {parent.observation_space}"
        )
    build_kwargs = dict(
        architecture=GEN2_GATE_YAW_ARCH,
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
        from marine_race_arena.learning.gen2.recurrent_policy import (
            build_gen2_recurrent_ppo,
        )

        child = build_gen2_recurrent_ppo(env, **build_kwargs)
    old = parent.policy.state_dict()
    new = child.policy.state_dict()
    if set(old) != set(new):
        raise ValueError(
            "parent/child policy state keys differ: "
            f"missing={sorted(set(old) - set(new))}, extra={sorted(set(new) - set(old))}"
        )
    expanded = {
        key: _expanded_tensor(key, old[key], new[key])
        for key in new
    }
    child.policy.load_state_dict(expanded, strict=True)
    child.gen2_parent_checkpoint = str(parent_path)
    child.gen2_parent_sha256 = sha256_file(parent_path)
    child.gen2_transfer = "exact_prefix_35_zero_new_columns_v1"
    return child


def verify_transfer_tensors(parent_path: str | Path, child) -> Dict[str, Any]:
    """Assert copied tensors, legacy columns, and neutral columns separately."""

    from sb3_contrib import RecurrentPPO

    parent = RecurrentPPO.load(str(parent_path), device="cpu")
    old = parent.policy.state_dict()
    new = child.policy.state_dict()
    mismatches = []
    expanded_keys = []
    for key, old_value in old.items():
        new_value = new[key]
        if old_value.shape == new_value.shape:
            if not th.equal(old_value.cpu(), new_value.cpu()):
                mismatches.append(key)
            continue
        expanded_keys.append(key)
        if key.endswith(FIRST_WEIGHT_SUFFIX):
            ok = th.equal(old_value.cpu(), new_value[:, :35].cpu()) and bool(
                th.count_nonzero(new_value[:, 35:]).item() == 0
            )
        elif key.endswith(MEAN_SUFFIX):
            ok = th.equal(old_value.cpu(), new_value[:35].cpu()) and bool(
                th.count_nonzero(new_value[35:]).item() == 0
            )
        elif key.endswith(STD_SUFFIX):
            ok = th.equal(old_value.cpu(), new_value[:35].cpu()) and bool(
                th.equal(new_value[35:].cpu(), th.ones_like(new_value[35:].cpu()))
            )
        else:
            ok = False
        if not ok:
            mismatches.append(key)
    return {
        "exact": not mismatches,
        "mismatches": mismatches,
        "expanded_keys": expanded_keys,
        "same_shape_tensors_copied": sum(
            int(old[key].shape == new[key].shape) for key in old
        ),
        "parent_sha256": sha256_file(parent_path),
        "new_input_columns_zero": all(
            bool(th.count_nonzero(value[:, 35:]).item() == 0)
            for key, value in new.items()
            if key.endswith(FIRST_WEIGHT_SUFFIX)
        ),
    }


def verify_expanded_recurrent_parity(
    parent_path: str | Path,
    child,
    *,
    samples: int = 32,
    tolerance: float = 0.0,
    stream: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compare recurrent actions for ``old + [0, 0, 0]`` sequences."""

    from sb3_contrib import RecurrentPPO

    if stream is None:
        stream = np.random.default_rng(8128).uniform(
            -0.75, 0.75, size=(int(samples), OBS_DIM_LOCAL_TRANSITION)
        ).astype(np.float32)
    else:
        stream = np.asarray(stream, dtype=np.float32)
    if stream.shape != (len(stream), OBS_DIM_LOCAL_TRANSITION):
        raise ValueError("parity stream must contain 35-D observations")
    extended = np.concatenate(
        (stream, np.zeros((len(stream), 3), dtype=np.float32)), axis=1
    )
    parent = RecurrentPPO.load(str(parent_path), device="cpu")

    def rollout(model, rows):
        controller = Gen2RecurrentController(model, deterministic=True)
        return np.asarray(
            [
                controller.act(row, first_step=(index == 0))
                for index, row in enumerate(rows)
            ],
            dtype=np.float32,
        )

    expected = rollout(parent, stream)
    observed = rollout(child, extended)
    deviation = float(np.max(np.abs(expected - observed)))
    return {
        "parity": bool(deviation <= float(tolerance)),
        "bit_exact": bool(np.array_equal(expected, observed)),
        "max_abs_deviation": deviation,
        "tolerance": float(tolerance),
        "samples": len(stream),
    }
