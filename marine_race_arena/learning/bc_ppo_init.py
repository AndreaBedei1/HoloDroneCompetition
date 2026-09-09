"""Safe stochastic warm-start for BC-initialized PPO.

The BC->PPO transfer (:func:`rl_train.transfer_bc_to_ppo`) reproduces the BC
deterministic policy *mean* exactly. But a PPO rollout samples actions from a
diagonal Gaussian whose per-axis standard deviation Stable-Baselines3 initializes
near 1.0 -- enormous for actions clipped to ``[-1, 1]``. Beginning exploration that
wide would saturate the actions and destroy the imitation warm-start on the very
first update.

This module derives a small, documented per-axis exploration standard deviation
from the BC validation residuals (``std = sqrt(per-axis val MSE)``, clamped to a
safe range) and installs it as the PPO policy's ``log_std``. If no usable BC report
is available it uses a documented fixed fallback (``exp(log_std_fallback)``). The
scratch arm keeps SB3's own initialization and is never reduced automatically.

``compute_bc_action_std`` is pure NumPy (no torch) so it is unit-testable anywhere;
``apply_bc_action_std`` / ``initialize_bc_action_std`` touch the torch policy.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from marine_race_arena.learning.config import ACTION_AXES, ACTION_DIM


def compute_bc_action_std(
    bc_report: Optional[Dict[str, Any]],
    *,
    action_dim: int = ACTION_DIM,
    mode: str = "from_validation",
    std_min: float = 0.05,
    std_max: float = 0.15,
    log_std_fallback: float = -2.5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Per-axis exploration std for a BC-initialized PPO, plus a provenance dict.

    ``mode="from_validation"`` converts each per-axis BC validation MSE to a residual
    std and clamps it to ``[std_min, std_max]``; a missing/invalid axis, or an absent
    report, uses ``exp(log_std_fallback)``. ``mode="fixed"`` uses that fallback for
    every axis. Returns ``(std_per_axis float32 (action_dim,), info)``.
    """
    fallback_std = float(np.exp(log_std_fallback))
    axes = list(ACTION_AXES[:action_dim])
    per_axis_source = {a: "fixed_fallback" for a in axes}
    stds = np.full(action_dim, fallback_std, dtype=np.float64)
    source = "fixed_fallback"

    per_axis_mse = bc_report.get("best_val_mse_per_axis") if isinstance(bc_report, dict) else None

    if mode == "from_validation" and isinstance(per_axis_mse, dict) and len(per_axis_mse) >= action_dim:
        source = "bc_validation_residual"
        for i, axis in enumerate(axes):
            raw = per_axis_mse.get(axis)
            try:
                mse = float(raw)
            except (TypeError, ValueError):
                mse = float("nan")
            if not np.isfinite(mse) or mse < 0:
                stds[i] = fallback_std
                per_axis_source[axis] = "fixed_fallback (invalid mse)"
            else:
                stds[i] = float(np.sqrt(mse))
                per_axis_source[axis] = "validation_residual"
        stds = np.clip(stds, std_min, std_max)
    elif mode == "from_validation":
        # Requested residual-based init but no usable per-axis report -> fixed fallback.
        source = "fixed_fallback (no bc report)"
    # mode == "fixed": keep the documented fallback std for every axis.

    stds = stds.astype(np.float32)
    log_stds = np.log(stds)
    info = {
        "mode": mode,
        "source": source,
        "std_min": std_min,
        "std_max": std_max,
        "log_std_fallback": log_std_fallback,
        "std_per_axis": {a: float(stds[i]) for i, a in enumerate(axes)},
        "log_std_per_axis": {a: float(log_stds[i]) for i, a in enumerate(axes)},
        "per_axis_source": per_axis_source,
    }
    return stds, info


def resolve_action_std(
    strategy: str,
    *,
    bc_report: Optional[Dict[str, Any]] = None,
    value=None,
    action_dim: int = ACTION_DIM,
    std_min: float = 0.05,
    std_max: float = 0.15,
    log_std_fallback: float = -2.5,
):
    """Resolve a PPO exploration std for any arm; returns ``(std_array | None, info)``.

    Strategies:
      * ``"residual"``  — per-axis std from BC validation residuals (BC-init only).
      * ``"fixed"``     — a documented scalar or per-axis override (``value``).
      * ``"sb3_default"`` — leave SB3's own initialization untouched (returns ``None``).
    """
    axes = list(ACTION_AXES[:action_dim])
    if strategy == "sb3_default":
        return None, {"strategy": "sb3_default", "source": "sb3_default",
                      "note": "PPO keeps Stable-Baselines3's default log_std (~1.0 std)."}
    if strategy == "fixed":
        if value is None:
            raise ValueError("action-std strategy 'fixed' requires a value (scalar or per-axis list)")
        if isinstance(value, (int, float)):
            std = np.full(action_dim, float(value), dtype=np.float32)
            per_axis_source = {a: "fixed_scalar" for a in axes}
        else:
            std = np.asarray(value, dtype=np.float32)
            if std.shape != (action_dim,):
                raise ValueError(f"per-axis action-std must have exactly {action_dim} values, got {std.shape}")
            per_axis_source = {a: "fixed_per_axis" for a in axes}
        if np.any(std <= 0) or not np.all(np.isfinite(std)):
            raise ValueError(f"action-std must be positive and finite, got {std}")
        log_std = np.log(std)
        info = {
            "strategy": "fixed", "source": "fixed_override",
            "std_per_axis": {a: float(std[i]) for i, a in enumerate(axes)},
            "log_std_per_axis": {a: float(log_std[i]) for i, a in enumerate(axes)},
            "per_axis_source": per_axis_source,
        }
        return std, info
    if strategy == "residual":
        std, info = compute_bc_action_std(bc_report, action_dim=action_dim, mode="from_validation",
                                          std_min=std_min, std_max=std_max, log_std_fallback=log_std_fallback)
        info["strategy"] = "residual"
        return std, info
    raise ValueError(f"unknown action-std strategy {strategy!r}")


def initialize_action_std(ppo_model, strategy: str, **kwargs) -> Dict[str, Any]:
    """Resolve and (if applicable) install an exploration std on ``ppo_model``.

    For ``sb3_default`` nothing is changed; the returned info records that choice.
    """
    std, info = resolve_action_std(strategy, **kwargs)
    if std is not None:
        apply_bc_action_std(ppo_model, std)
    return info


def apply_bc_action_std(ppo_model, std_per_axis) -> None:
    """Copy ``log(std_per_axis)`` into the PPO policy's ``log_std`` parameter."""
    import torch

    log_std_param = ppo_model.policy.log_std
    target = torch.log(
        torch.as_tensor(np.asarray(std_per_axis, dtype=np.float32),
                        dtype=log_std_param.dtype, device=log_std_param.device)
    )
    if tuple(target.shape) != tuple(log_std_param.shape):
        raise ValueError(
            f"action-std shape {tuple(target.shape)} does not match PPO log_std "
            f"{tuple(log_std_param.shape)}"
        )
    with torch.no_grad():
        log_std_param.data.copy_(target)


def initialize_bc_action_std(
    ppo_model,
    bc_report: Optional[Dict[str, Any]] = None,
    *,
    action_dim: int = ACTION_DIM,
    mode: str = "from_validation",
    std_min: float = 0.05,
    std_max: float = 0.15,
    log_std_fallback: float = -2.5,
) -> Dict[str, Any]:
    """Compute the safe per-axis exploration std and install it on ``ppo_model``.

    Returns the provenance ``info`` dict (also suitable for run metadata).
    """
    std, info = compute_bc_action_std(
        bc_report, action_dim=action_dim, mode=mode,
        std_min=std_min, std_max=std_max, log_std_fallback=log_std_fallback,
    )
    apply_bc_action_std(ppo_model, std)
    return info
