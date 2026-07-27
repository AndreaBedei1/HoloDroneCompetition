"""Multi-gate BC warm-start tests."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from marine_race_arena.learning.bc_train import BCPolicy
from marine_race_arena.learning.bc_train_v3 import (
    rebase_observation_normalization,
)
from marine_race_arena.learning.bc_v3_transfer import expand_bc_v1_to_v3
from marine_race_arena.learning.config import OBS_DIM
from marine_race_arena.learning.config_v3 import OBS_DIM_V3


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
