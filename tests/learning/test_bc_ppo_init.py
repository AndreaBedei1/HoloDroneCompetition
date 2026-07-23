"""Tests for the BC->PPO exploration-std computation (numpy-only, no torch/SB3)."""

import numpy as np
import pytest

from marine_race_arena.learning.bc_ppo_init import compute_bc_action_std
from marine_race_arena.learning.config import ACTION_AXES, ACTION_DIM


def _report(**per_axis):
    return {"best_val_mse_per_axis": dict(per_axis)}


def test_per_axis_residual_std_is_sqrt_mse_clipped():
    # raw std = sqrt(mse) = [0.3, 0.2, 0.01, 0.5] -> clip to [0.05, 0.15]
    report = _report(surge=0.09, sway=0.04, heave=0.0001, yaw=0.25)
    std, info = compute_bc_action_std(report, action_dim=4, std_min=0.05, std_max=0.15)
    assert info["source"] == "bc_validation_residual"
    np.testing.assert_allclose(std, [0.15, 0.15, 0.05, 0.15], atol=1e-6)
    # log_std is exactly log(std)
    for i, axis in enumerate(ACTION_AXES):
        assert info["log_std_per_axis"][axis] == pytest.approx(float(np.log(std[i])))
        assert info["std_per_axis"][axis] == pytest.approx(float(std[i]))


def test_real_stage1_report_floors_every_axis_at_std_min():
    # The committed BC model's residuals are tiny (near-perfect open-loop fit),
    # so every axis clamps to std_min = 0.05 (documented Stage-1 warm-start).
    report = _report(surge=0.00021, sway=1.8e-05, heave=2.3e-05, yaw=1.1e-05)
    std, info = compute_bc_action_std(report, action_dim=4, std_min=0.05, std_max=0.15)
    np.testing.assert_allclose(std, [0.05, 0.05, 0.05, 0.05], atol=1e-6)
    assert all(s == "validation_residual" for s in info["per_axis_source"].values())


def test_absent_report_uses_fixed_fallback():
    std, info = compute_bc_action_std(None, action_dim=4, log_std_fallback=-2.5)
    np.testing.assert_allclose(std, np.exp(-2.5), atol=1e-6)
    assert "fixed_fallback" in info["source"]


def test_report_without_per_axis_uses_fallback():
    std, info = compute_bc_action_std({"best_val_mse": 1e-4}, action_dim=4)
    np.testing.assert_allclose(std, np.exp(-2.5), atol=1e-6)
    assert "fixed_fallback" in info["source"]


def test_zero_mse_clamps_to_std_min():
    report = _report(surge=0.0, sway=0.0, heave=0.0, yaw=0.0)
    std, _ = compute_bc_action_std(report, action_dim=4, std_min=0.05, std_max=0.15)
    np.testing.assert_allclose(std, 0.05, atol=1e-6)


def test_invalid_mse_falls_back_per_axis():
    report = _report(surge=-1.0, sway="bad", heave=float("nan"), yaw=0.09)
    std, info = compute_bc_action_std(report, action_dim=4, std_min=0.05, std_max=0.15)
    # surge/sway/heave invalid -> fallback std; yaw valid -> sqrt(0.09)=0.3 clipped to 0.15
    assert info["per_axis_source"]["surge"].startswith("fixed_fallback")
    assert info["per_axis_source"]["sway"].startswith("fixed_fallback")
    assert info["per_axis_source"]["heave"].startswith("fixed_fallback")
    assert info["per_axis_source"]["yaw"] == "validation_residual"
    assert std[3] == pytest.approx(0.15)
    assert std[0] == pytest.approx(np.exp(-2.5), abs=1e-6)


def test_dimension_mismatch_uses_fallback():
    report = _report(surge=0.09, sway=0.04)  # only 2 of 4 axes
    std, info = compute_bc_action_std(report, action_dim=4)
    np.testing.assert_allclose(std, np.exp(-2.5), atol=1e-6)
    assert "fixed_fallback" in info["source"]


def test_fixed_mode_ignores_report():
    report = _report(surge=0.09, sway=0.04, heave=0.0001, yaw=0.25)
    std, info = compute_bc_action_std(report, action_dim=4, mode="fixed", log_std_fallback=-2.0)
    np.testing.assert_allclose(std, np.exp(-2.0), atol=1e-6)
    assert info["mode"] == "fixed"


def test_output_shape_and_dtype():
    std, _ = compute_bc_action_std(None, action_dim=ACTION_DIM)
    assert std.shape == (ACTION_DIM,) and std.dtype == np.float32
