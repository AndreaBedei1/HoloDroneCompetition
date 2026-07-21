"""Tests for the trajectory recorder and the BC dataset pipeline."""

import numpy as np
import pytest

from marine_race_arena.learning.config import ACTION_DIM, OBS_DIM
from marine_race_arena.learning.dataset import BCDataset, DatasetIntegrityError, EpisodeMeta
from marine_race_arena.learning.trajectory_recorder import collect_dataset, record_episode

TRACK = "marine_race_arena/tracks/tests/single_gate_yaw_0.json"


def _records(n_seeds=3, max_steps=12):
    return collect_dataset(
        TRACK,
        controller="rule_gate_center_then_commit",
        seeds=list(range(n_seeds)),
        dt=0.1,
        adapter="fallback",
        allow_fallback=True,
        max_steps=max_steps,
    )


def test_record_episode_shapes_and_bounds():
    rec = record_episode(TRACK, seed=0, max_steps=10, adapter="fallback", allow_fallback=True)
    assert rec.observations.shape[1] == OBS_DIM
    assert rec.actions.shape[1] == ACTION_DIM
    assert rec.observations.shape[0] == rec.actions.shape[0] == rec.length
    assert np.all(np.isfinite(rec.observations))
    assert np.all(rec.actions >= -1.0) and np.all(rec.actions <= 1.0)
    # Ends terminally, and privileged diagnostics are separate.
    assert bool(rec.dones[-1] or rec.truncated[-1])
    assert "positions" in rec.diagnostics and "gate_crossings" in rec.diagnostics
    assert rec.diagnostics["positions"].shape == (rec.length, 3)


def test_dataset_from_records_integrity():
    ds = BCDataset.from_records(_records())
    ds.check_integrity()
    assert ds.observations.shape[1] == OBS_DIM
    assert ds.num_episodes == 3
    assert len(ds) == sum(m.length for m in ds.episodes)
    # The dataset exposes no privileged position/gate arrays.
    assert not hasattr(ds, "positions")


def test_dataset_save_load_roundtrip(tmp_path):
    ds = BCDataset.from_records(_records())
    path = tmp_path / "demo.npz"
    ds.save(path)
    loaded = BCDataset.load(path)
    loaded.check_integrity()
    assert np.array_equal(ds.observations, loaded.observations)
    assert np.array_equal(ds.actions, loaded.actions)
    assert loaded.num_episodes == ds.num_episodes
    assert [m.seed for m in loaded.episodes] == [m.seed for m in ds.episodes]


def test_train_val_split_by_episode_no_leakage():
    ds = BCDataset.from_records(_records(n_seeds=5, max_steps=10))
    train, val = ds.train_val_split(val_fraction=0.4, seed=1)
    train.check_integrity()
    val.check_integrity()
    train_groups = set(train.group_ids.tolist())
    val_groups = set(val.group_ids.tolist())
    assert not (train_groups & val_groups), "an episode leaked across the split"
    assert len(train) + len(val) == len(ds)
    assert train.num_episodes + val.num_episodes == ds.num_episodes


def test_normalization_stats_shapes():
    ds = BCDataset.from_records(_records())
    mean, std = ds.normalization_stats()
    assert mean.shape == (OBS_DIM,) and std.shape == (OBS_DIM,)
    assert np.all(std > 0.0)  # floored, never zero


def test_integrity_rejects_nan():
    ds = BCDataset.from_records(_records())
    ds.observations[0, 0] = np.nan
    with pytest.raises(DatasetIntegrityError):
        ds.check_integrity()


def test_integrity_rejects_out_of_bounds_action():
    ds = BCDataset.from_records(_records())
    ds.actions[0, 0] = 5.0
    with pytest.raises(DatasetIntegrityError):
        ds.check_integrity()


def test_integrity_rejects_duplicate_episode_identity():
    recs = _records(n_seeds=2, max_steps=8)
    # Force both episodes to share the same (seed, episode_id, track, controller).
    recs[1].seed = recs[0].seed
    recs[1].episode_id = recs[0].episode_id
    ds = BCDataset.from_records(recs)
    with pytest.raises(DatasetIntegrityError):
        ds.check_integrity()


def test_recorded_expert_action_matches_applied_when_in_bounds():
    rec = record_episode(TRACK, seed=0, max_steps=8, adapter="fallback", allow_fallback=True)
    # The official controllers stay well inside [-1, 1], so applied == raw here.
    assert np.allclose(rec.actions, np.clip(rec.expert_actions_raw, -1.0, 1.0))
