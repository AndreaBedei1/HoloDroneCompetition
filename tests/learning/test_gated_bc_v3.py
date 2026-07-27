"""Gated residual BC-v3 policy tests."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from marine_race_arena.learning.bc_train import BCPolicy
from marine_race_arena.learning.bc_v3_transfer import expand_bc_v1_to_v3
from marine_race_arena.learning.config_v3 import FEATURE_NAMES_V3, OBS_DIM_V3
from marine_race_arena.learning.gated_bc_v3 import (
    build_gated_policy,
    load_gated_policy,
    save_gated_policy,
)


def test_gate_zero_is_exact_frozen_baseline():
    source = BCPolicy(hidden_sizes=(16, 16))
    baseline = expand_bc_v1_to_v3(source)
    gated = build_gated_policy(source, residual_hidden_sizes=(8, 8))
    rng = np.random.default_rng(12)
    gate_index = FEATURE_NAMES_V3.index("previous_gate_bearing_present")
    for _ in range(20):
        observation = rng.normal(size=OBS_DIM_V3).astype(np.float32)
        observation[gate_index] = 0.0
        np.testing.assert_allclose(
            gated.act(observation), baseline.act(observation), atol=1e-7
        )


def test_gated_checkpoint_round_trip(tmp_path):
    policy = build_gated_policy(
        BCPolicy(hidden_sizes=(16, 16)), residual_hidden_sizes=(8, 8)
    )
    path = tmp_path / "gated.pt"
    save_gated_policy(policy, path)
    loaded = load_gated_policy(path)
    observation = np.zeros(OBS_DIM_V3, dtype=np.float32)
    np.testing.assert_allclose(
        loaded.act(observation), policy.act(observation), atol=1e-7
    )
