"""BC->PPO safe stochastic warm-start validation (skipped without torch/SB3)."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

import gymnasium as gym
import torch
from gymnasium import spaces

from marine_race_arena.learning.bc_ppo_init import compute_bc_action_std, initialize_bc_action_std
from marine_race_arena.learning.bc_train import BCPolicy
from marine_race_arena.learning.config import ACTION_AXES, ACTION_DIM, OBS_DIM
from marine_race_arena.learning.rl_train import build_ppo, transfer_bc_to_ppo


class _DummyEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(-1.0, 1.0, (OBS_DIM,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, True, {}


def _report(**per_axis):
    return {"best_val_mse_per_axis": dict(per_axis)}


def _nontrivial_bc(hidden=(64, 64), seed=1, zero_output=False):
    rng = np.random.default_rng(seed)
    mean = rng.uniform(-0.5, 0.5, OBS_DIM).astype(np.float32)
    std = rng.uniform(0.3, 2.0, OBS_DIM).astype(np.float32)
    bc = BCPolicy(hidden_sizes=hidden, obs_mean=mean, obs_std=std)
    if zero_output:  # output identically 0 -> samples center on 0 for a clean saturation test
        with torch.no_grad():
            bc.head.weight.zero_()
            bc.head.bias.zero_()
    return bc


def _init(report=None, mode="from_validation", zero_output=False):
    bc = _nontrivial_bc(zero_output=zero_output)
    ppo = build_ppo(_DummyEnv(), hidden_sizes=(64, 64), seed=0)
    transfer_bc_to_ppo(bc, ppo)
    info = initialize_bc_action_std(ppo, report, mode=mode)
    return bc, ppo, info


def test_log_std_init_preserves_deterministic_parity():
    """Setting log_std must not change the deterministic (mean) action."""
    bc, ppo, _ = _init(_report(surge=0.09, sway=0.04, heave=0.0025, yaw=0.01))
    rng = np.random.default_rng(5)
    for _ in range(15):
        obs = rng.uniform(-1.5, 1.5, OBS_DIM).astype(np.float32)
        ppo_action, _ = ppo.predict(obs, deterministic=True)
        with torch.no_grad():
            bc_out = bc.forward(torch.as_tensor(obs).reshape(1, -1)).numpy().reshape(-1)
        assert np.max(np.abs(ppo_action - bc_out)) < 1e-4


def test_apply_sets_expected_per_axis_log_std():
    # sqrt(mse) = [0.3, 0.2, 0.05, 0.1] -> clip [0.15, 0.15, 0.05, 0.10]
    _, ppo, info = _init(_report(surge=0.09, sway=0.04, heave=0.0025, yaw=0.01))
    expected_std = np.array([0.15, 0.15, 0.05, 0.10])
    np.testing.assert_allclose(ppo.policy.log_std.detach().numpy(), np.log(expected_std), atol=1e-5)
    assert [info["std_per_axis"][a] for a in ACTION_AXES] == pytest.approx(list(expected_std))


def test_stochastic_rollout_is_gentle_and_near_bc():
    """Sampled actions cluster tightly (small std) around the BC mean, not saturated."""
    report = _report(surge=0.01, sway=0.01, heave=0.01, yaw=0.01)  # std ~ 0.1
    bc, ppo, info = _init(report, zero_output=True)
    obs = np.zeros(OBS_DIM, np.float32)
    obs_batch = torch.as_tensor(np.tile(obs, (5000, 1)))
    dist = ppo.policy.get_distribution(obs_batch)
    samples = dist.get_actions().detach().numpy()
    assert np.all(np.isfinite(samples))  # no NaN/inf
    bc_mean = bc.forward(torch.as_tensor(obs).reshape(1, -1)).detach().numpy().reshape(-1)
    configured = np.array([info["std_per_axis"][a] for a in ACTION_AXES])
    np.testing.assert_allclose(samples.mean(0), bc_mean, atol=0.02)   # mean ~ BC output
    np.testing.assert_allclose(samples.std(0), configured, rtol=0.15)  # std ~ configured
    assert np.mean(np.abs(samples) > 0.98) < 0.02  # negligible saturation


def test_uninitialized_default_std_would_saturate_much_more():
    """Contrast: SB3's default log_std (~1.0 std) saturates far more than the warm-start."""
    bc = _nontrivial_bc(zero_output=True)
    ppo = build_ppo(_DummyEnv(), hidden_sizes=(64, 64), seed=0)
    transfer_bc_to_ppo(bc, ppo)  # NO action-std init: SB3 default log_std=0 -> std=1.0
    obs_batch = torch.as_tensor(np.tile(np.zeros(OBS_DIM, np.float32), (5000, 1)))
    default_sat = np.mean(np.abs(ppo.policy.get_distribution(obs_batch).get_actions().detach().numpy()) > 0.98)
    initialize_bc_action_std(ppo, _report(surge=0.01, sway=0.01, heave=0.01, yaw=0.01))
    warm_sat = np.mean(np.abs(ppo.policy.get_distribution(obs_batch).get_actions().detach().numpy()) > 0.98)
    assert default_sat > 0.25 and warm_sat < 0.02 and warm_sat < default_sat


def test_no_report_applies_fixed_fallback_std():
    _, ppo, info = _init(None)
    np.testing.assert_allclose(np.exp(ppo.policy.log_std.detach().numpy()), np.exp(-2.5), atol=1e-5)
    assert "fixed_fallback" in info["source"]


def test_save_load_preserves_log_std_and_actions(tmp_path):
    from stable_baselines3 import PPO

    _, ppo, _ = _init(_report(surge=0.09, sway=0.04, heave=0.0025, yaw=0.01))
    path = tmp_path / "m.zip"
    ppo.save(str(path))
    loaded = PPO.load(str(path), device="cpu")
    np.testing.assert_allclose(loaded.policy.log_std.detach().numpy(),
                               ppo.policy.log_std.detach().numpy(), atol=1e-6)
    rng = np.random.default_rng(0)
    for _ in range(8):
        obs = rng.uniform(-1, 1, OBS_DIM).astype(np.float32)
        a1, _ = ppo.predict(obs, deterministic=True)
        a2, _ = loaded.predict(obs, deterministic=True)
        np.testing.assert_allclose(a1, a2, atol=1e-5)


def test_shape_mismatch_raises():
    from marine_race_arena.learning.bc_ppo_init import apply_bc_action_std

    _, ppo, _ = _init(None)
    with pytest.raises(ValueError):
        apply_bc_action_std(ppo, np.array([0.1, 0.1, 0.1], dtype=np.float32))  # 3 != 4 axes
