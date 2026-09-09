"""Tests for the behavioral-cloning policy and trainer (skipped without torch)."""

import numpy as np
import pytest

pytest.importorskip("torch")

from marine_race_arena.learning.bc_train import BCConfig, BCPolicy, load_policy, save_policy, train_bc
from marine_race_arena.learning.config import ACTION_DIM, OBS_DIM
from marine_race_arena.learning.dataset import BCDataset
from marine_race_arena.learning.trajectory_recorder import collect_dataset

TRACK = "marine_race_arena/tracks/tests/single_gate_yaw_0.json"


def _dataset(n_seeds=4, max_steps=30):
    recs = collect_dataset(TRACK, controller="rule_gate_center_then_commit", seeds=list(range(n_seeds)), max_steps=max_steps)
    ds = BCDataset.from_records(recs)
    ds.check_integrity()
    return ds


def test_policy_act_shape_and_bounds():
    policy = BCPolicy(hidden_sizes=(32, 32))
    action = policy.act(np.zeros(OBS_DIM, dtype=np.float32))
    assert action.shape == (ACTION_DIM,)
    assert action.dtype == np.float32
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_inference_determinism():
    policy = BCPolicy(hidden_sizes=(32, 32))
    obs = np.linspace(-1, 1, OBS_DIM, dtype=np.float32)
    assert np.array_equal(policy.act(obs), policy.act(obs))


def test_save_load_roundtrip(tmp_path):
    ds = _dataset()
    policy, _ = train_bc(ds, BCConfig(hidden_sizes=(32, 32), max_epochs=5, patience=5))
    path = tmp_path / "bc.pt"
    save_policy(policy, path)
    loaded = load_policy(path)
    for i in range(5):
        obs = ds.observations[i]
        assert np.allclose(policy.act(obs), loaded.act(obs), atol=1e-6)


def test_training_reduces_validation_loss():
    ds = _dataset(n_seeds=5, max_steps=30)
    _, history = train_bc(ds, BCConfig(hidden_sizes=(64, 64), max_epochs=40, patience=20, seed=0))
    assert len(history) >= 2
    best = min(h["val_mse"] for h in history)
    assert best < history[0]["val_mse"], "validation loss did not improve"
    # Per-axis validation losses are reported.
    for axis in ("surge", "sway", "heave", "yaw"):
        assert f"val_mse_{axis}" in history[-1]


def test_csv_logging(tmp_path):
    ds = _dataset()
    log = tmp_path / "bc_log.csv"
    train_bc(ds, BCConfig(hidden_sizes=(16, 16), max_epochs=3, patience=3), log_csv=str(log))
    assert log.exists()
    assert "val_mse" in log.read_text(encoding="utf-8").splitlines()[0]
