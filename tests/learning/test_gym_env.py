"""Gymnasium API compliance for MarineRaceGymEnv (skipped without gymnasium)."""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from marine_race_arena.learning.config import ACTION_DIM, OBS_DIM
from marine_race_arena.learning.gym_env import MarineRaceGymEnv

TRACK = "marine_race_arena/tracks/tests/single_gate_yaw_0.json"


def _make(**kwargs):
    params = dict(seed=0, dt=0.1, adapter="fallback", allow_fallback=True, max_steps=25)
    params.update(kwargs)
    return MarineRaceGymEnv(TRACK, **params)


def test_spaces_shapes_and_dtypes():
    env = _make()
    try:
        assert env.observation_space.shape == (OBS_DIM,)
        assert env.action_space.shape == (ACTION_DIM,)
        assert env.observation_space.dtype == np.float32
        assert env.action_space.dtype == np.float32
        assert np.all(env.action_space.low == -1.0)
        assert np.all(env.action_space.high == 1.0)
    finally:
        env.close()


def test_reset_returns_obs_and_info():
    env = _make()
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs)
        assert isinstance(info, dict)
    finally:
        env.close()


def test_step_signature_and_bounds():
    env = _make()
    try:
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(np.array([0.5, 0.0, 0.0, 0.1], dtype=np.float32))
        assert obs.shape == (OBS_DIM,)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "reward_components" in info
    finally:
        env.close()


def test_action_out_of_bounds_is_clipped_not_crashed():
    env = _make()
    try:
        env.reset(seed=0)
        obs, reward, term, trunc, info = env.step(np.array([5.0, -5.0, np.nan, 2.0], dtype=np.float32))
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
    finally:
        env.close()


def test_episode_runs_to_truncation():
    env = _make(max_steps=15)
    try:
        env.reset(seed=0)
        steps = 0
        done = False
        while not done and steps < 100:
            _, _, term, trunc, _ = env.step(env.action_space.sample())
            done = term or trunc
            steps += 1
        assert done
        assert steps <= 16
    finally:
        env.close()


def test_determinism_with_seed():
    env_a = _make()
    env_b = _make()
    try:
        obs_a, _ = env_a.reset(seed=42)
        obs_b, _ = env_b.reset(seed=42)
        assert np.allclose(obs_a, obs_b)
        for _ in range(10):
            a = np.array([0.4, 0.1, 0.0, 0.2], dtype=np.float32)
            oa, ra, ta, ua, _ = env_a.step(a)
            ob, rb, tb, ub, _ = env_b.step(a)
            assert np.allclose(oa, ob)
            assert ra == pytest.approx(rb)
            assert ta == tb and ua == ub
    finally:
        env_a.close()
        env_b.close()


def test_gymnasium_env_checker():
    env = _make(max_steps=20)
    try:
        from gymnasium.utils.env_checker import check_env

        check_env(env.unwrapped, skip_render_check=True)
    finally:
        env.close()


def test_reward_components_reported():
    env = _make()
    try:
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([0.6, 0.0, 0.0, 0.0], dtype=np.float32))
        comps = info["reward_components"]
        assert "time_cost" in comps and "gate_bonus" in comps
        assert comps["time_cost"] < 0.0  # per-step time cost is negative
    finally:
        env.close()


def test_close_is_idempotent_and_safe_after_error():
    env = _make()
    env.reset(seed=0)
    env.close()
    env.close()  # safe twice
