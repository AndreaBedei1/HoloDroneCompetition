"""Observation-v3 Gym wiring and bounded multi-gate reward tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from marine_race_arena.learning.config_v3 import OBS_DIM_V3, OBS_ENCODING_VERSION_V3
from marine_race_arena.learning.gym_env import MarineRaceGymEnv
from marine_race_arena.learning.reward_v3 import (
    MultiGateRewardConfig,
    MultiGateTrainingReward,
)

TRACK = "marine_race_arena/tracks/tests/two_gate_straight.json"


def test_v3_gym_space_context_and_reward_components():
    config = MultiGateRewardConfig()
    env = MarineRaceGymEnv(
        TRACK,
        seed=42,
        adapter="fallback",
        allow_fallback=True,
        max_steps=3,
        observation_encoding_version=OBS_ENCODING_VERSION_V3,
        reward_fn=MultiGateTrainingReward(config),
    )
    try:
        observation, info = env.reset(seed=42)
        assert observation.shape == (OBS_DIM_V3,)
        assert env.observation_space.shape == (OBS_DIM_V3,)
        assert env.tracker.expected_beacon_id == "B01"
        observation, reward, _, _, info = env.step(
            np.array([0.4, 0.1, -0.1, 0.2], dtype=np.float32)
        )
        assert observation.shape == (OBS_DIM_V3,)
        assert math.isfinite(reward)
        assert abs(reward) <= config.total_abs_bound
        assert info["reward_components"]
        assert "next_beacon_alignment" in info["reward_components"]
        assert all(
            math.isfinite(value) and abs(value) <= config.component_abs_bound
            for value in info["reward_components"].values()
        )
    finally:
        env.close()


def test_turn_reward_defaults_match_measured_r2_failure():
    config = MultiGateRewardConfig()
    assert config.offcenter_surge_threshold < 0.2
    assert config.post_gate_window_steps >= 50
    assert config.next_beacon_alignment_scale > 0


def test_non_finished_truncation_has_large_terminal_penalty():
    config = MultiGateRewardConfig()
    env = MarineRaceGymEnv(
        TRACK,
        seed=7,
        adapter="fallback",
        allow_fallback=True,
        max_steps=1,
        observation_encoding_version=OBS_ENCODING_VERSION_V3,
        reward_fn=MultiGateTrainingReward(config),
    )
    try:
        env.reset(seed=7)
        _, _, terminated, truncated, info = env.step(
            np.zeros(4, dtype=np.float32)
        )
        assert truncated and not terminated
        assert info["reward_components"]["timeout_penalty"] == -30.0
    finally:
        env.close()


def test_unknown_observation_version_is_rejected():
    with pytest.raises(ValueError, match="unsupported observation encoding"):
        MarineRaceGymEnv(
            TRACK,
            adapter="fallback",
            observation_encoding_version="unknown_v99",
        )
