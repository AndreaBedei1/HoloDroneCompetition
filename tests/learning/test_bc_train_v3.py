"""Multi-gate BC warm-start tests."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from marine_race_arena.learning.bc_train import BCPolicy
from marine_race_arena.learning.bc_train_v3 import (
    BCV3Config,
    fine_tune_bc_v3,
    rebase_observation_normalization,
)
from marine_race_arena.learning.bc_v3_transfer import expand_bc_v1_to_v3
from marine_race_arena.learning.config import OBS_DIM
from marine_race_arena.learning.config_v3 import OBS_DIM_V3
from marine_race_arena.learning.config_v3 import (
    FEATURE_NAMES_V3,
    OBS_ENCODING_VERSION_V3,
)
from marine_race_arena.learning.dataset import BCDataset, EpisodeMeta


def test_normalization_rebase_preserves_policy_output():
    rng = np.random.default_rng(41)
    source = BCPolicy(
        hidden_sizes=(16, 16),
        obs_mean=rng.normal(size=OBS_DIM).astype(np.float32),
        obs_std=rng.uniform(0.3, 1.5, OBS_DIM).astype(np.float32),
    )
    policy = expand_bc_v1_to_v3(source)
    observations = rng.normal(size=(20, OBS_DIM_V3)).astype(np.float32)
    before = np.stack([policy.act(row) for row in observations])
    rebase_observation_normalization(
        policy,
        rng.normal(size=OBS_DIM_V3).astype(np.float32),
        rng.uniform(0.2, 2.0, OBS_DIM_V3).astype(np.float32),
    )
    after = np.stack([policy.act(row) for row in observations])
    np.testing.assert_allclose(after, before, atol=2e-6)


def test_transition_only_training_preserves_prechange_function():
    rng = np.random.default_rng(8)
    source = BCPolicy(hidden_sizes=(16, 16))
    observations = rng.normal(size=(8, OBS_DIM_V3)).astype(np.float32)
    for name in (
        "expected_beacon_changed",
        "steps_since_beacon_change_norm",
        "previous_gate_in_rear_sector",
        "previous_gate_bearing_present",
    ):
        observations[:, FEATURE_NAMES_V3.index(name)] = 0.0
    actions = rng.uniform(-0.5, 0.5, size=(8, 4)).astype(np.float32)
    # Later rows contain real transition values, so their dataset mean is not
    # zero; prechange rows must nevertheless remain exactly neutral.
    for name in (
        "expected_beacon_changed",
        "steps_since_beacon_change_norm",
        "previous_gate_in_rear_sector",
        "previous_gate_bearing_present",
    ):
        observations[4:, FEATURE_NAMES_V3.index(name)] = rng.uniform(
            0.2, 1.0, size=4
        )
    groups = np.repeat(np.arange(4), 2)
    dataset = BCDataset(
        observations,
        actions,
        groups,
        np.repeat(np.arange(4), 2),
        groups,
        np.tile([0, 1], 4),
        np.tile([False, True], 4),
        np.zeros(8, dtype=bool),
        [
            EpisodeMeta(i, i, i, "track", "expert", 2, "FINISHED", 2)
            for i in range(4)
        ],
        observation_encoding_version=OBS_ENCODING_VERSION_V3,
    )
    baseline = expand_bc_v1_to_v3(source)
    expected = np.stack([baseline.act(row) for row in observations[:4]])
    trained, _ = fine_tune_bc_v3(
        dataset,
        source,
        BCV3Config(max_epochs=2, patience=2, batch_size=4, seed=2),
    )
    actual = np.stack([trained.act(row) for row in observations[:4]])
    np.testing.assert_allclose(actual, expected, atol=3e-6)
