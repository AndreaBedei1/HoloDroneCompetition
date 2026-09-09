"""Tests for the PPO scaffold and BC->PPO transfer (skipped without SB3)."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

import gymnasium as gym
import torch
from gymnasium import spaces

from marine_race_arena.learning.bc_train import BCPolicy
from marine_race_arena.learning.config import ACTION_DIM, OBS_DIM
from marine_race_arena.learning.rl_train import build_ppo, make_env, train_ppo, transfer_bc_to_ppo

TRACK = "marine_race_arena/tracks/tests/single_gate_yaw_0.json"


class _DummyEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(-1.0, 1.0, (OBS_DIM,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, True, {}


def _nontrivial_bc(hidden=(64, 64), seed=1):
    """A BC policy with NON-identity normalization (non-zero mean, non-unit std)."""
    rng = np.random.default_rng(seed)
    mean = rng.uniform(-0.5, 0.5, OBS_DIM).astype(np.float32)
    std = rng.uniform(0.3, 2.0, OBS_DIM).astype(np.float32)  # positive, non-unit
    return BCPolicy(hidden_sizes=hidden, obs_mean=mean, obs_std=std)


def test_bc_to_ppo_transfer_is_exact_with_nonidentity_normalization():
    """PPO's deterministic action equals the normalizing BC output after transfer."""
    bc = _nontrivial_bc()
    # sanity: the BC really does normalize (mean != 0, std != 1)
    assert not np.allclose(bc.obs_mean.numpy(), 0.0)
    assert not np.allclose(bc.obs_std.numpy(), 1.0)

    ppo = build_ppo(_DummyEnv(), hidden_sizes=(64, 64), seed=0)
    transfer_bc_to_ppo(bc, ppo)
    rng = np.random.default_rng(7)
    max_err = 0.0
    for _ in range(20):
        obs = rng.uniform(-1.5, 1.5, size=OBS_DIM).astype(np.float32)
        ppo_action, _ = ppo.predict(obs, deterministic=True)
        with torch.no_grad():
            bc_out = bc.forward(torch.as_tensor(obs).reshape(1, -1)).numpy().reshape(-1)
        max_err = max(max_err, float(np.max(np.abs(ppo_action - bc_out))))
    assert max_err < 1e-4, f"normalization-aware BC->PPO transfer not exact (max err {max_err})"


def test_transfer_is_exact_after_bc_save_load(tmp_path):
    from marine_race_arena.learning.bc_train import load_policy, save_policy

    bc = _nontrivial_bc(seed=3)
    path = tmp_path / "bc.pt"
    save_policy(bc, path)
    loaded = load_policy(path)
    ppo = build_ppo(_DummyEnv(), hidden_sizes=(64, 64), seed=0)
    transfer_bc_to_ppo(loaded, ppo)
    rng = np.random.default_rng(11)
    for _ in range(10):
        obs = rng.uniform(-1, 1, size=OBS_DIM).astype(np.float32)
        ppo_action, _ = ppo.predict(obs, deterministic=True)
        with torch.no_grad():
            bc_out = loaded.forward(torch.as_tensor(obs).reshape(1, -1)).numpy().reshape(-1)
        assert np.allclose(ppo_action, bc_out, atol=1e-4)


def test_transfer_architecture_mismatch_raises():
    bc = _nontrivial_bc()
    ppo = build_ppo(_DummyEnv(), hidden_sizes=(32,), seed=0)  # different hidden layout
    with pytest.raises(ValueError):
        transfer_bc_to_ppo(bc, ppo)


def test_ppo_smoke_trains_on_fallback_env(tmp_path):
    """Plumbing only: PPO runs a few steps against the fallback backend."""
    model = train_ppo(
        TRACK,
        total_timesteps=64,
        seed=0,
        output_dir=str(tmp_path / "ppo_run"),
        env_kwargs=dict(adapter="fallback", allow_fallback=True, max_steps=20, dt=0.1),
        n_steps=32,
        batch_size=16,
    )
    # A deterministic prediction is produced and bounded.
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (ACTION_DIM,)
    assert (tmp_path / "ppo_run" / "ppo_model.zip").exists()


def test_bc_initialized_ppo_smoke(tmp_path):
    """A BC-initialized PPO trains without error on the fallback backend."""
    bc = BCPolicy(hidden_sizes=(64, 64))
    model = train_ppo(
        TRACK,
        total_timesteps=64,
        seed=0,
        bc_policy=bc,
        env_kwargs=dict(adapter="fallback", allow_fallback=True, max_steps=20, dt=0.1),
        hidden_sizes=(64, 64),
        n_steps=32,
        batch_size=16,
    )
    action, _ = model.predict(np.zeros(OBS_DIM, dtype=np.float32), deterministic=True)
    assert np.all(np.isfinite(action))
