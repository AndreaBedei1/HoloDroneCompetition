"""Pure best-model selection + resume-compatibility logic (no SB3 needed)."""

import json

import pytest

from dataclasses import asdict

from marine_race_arena.learning.config import ACTION_DIM, OBS_ENCODING_VERSION
from marine_race_arena.learning.randomization import StartRandomization
from marine_race_arena.learning.reward import RewardConfig
from marine_race_arena.learning.train_workflow import (
    _serializable_env_kwargs,
    _validate_resume_compatibility,
    best_metric_key,
    strictly_better,
)


def _row(rate, gates, coll, time=None):
    return {"completion_rate": rate, "mean_gates": gates, "mean_collisions": coll, "mean_time_finished": time}


def test_best_metric_key_lexicographic():
    # completion rate dominates
    assert best_metric_key(_row(0.9, 1, 5)) > best_metric_key(_row(0.8, 3, 0))
    # then gates
    assert best_metric_key(_row(0.9, 3, 5)) > best_metric_key(_row(0.9, 2, 0))
    # then fewer collisions
    assert best_metric_key(_row(0.9, 3, 1)) > best_metric_key(_row(0.9, 3, 4))
    # then lower finished time
    assert best_metric_key(_row(0.9, 3, 1, time=10.0)) > best_metric_key(_row(0.9, 3, 1, time=20.0))


def test_strictly_better():
    assert strictly_better(_row(0.5, 1, 0), None) is True  # first is always better
    assert strictly_better(_row(0.9, 1, 0), _row(0.8, 1, 0)) is True
    assert strictly_better(_row(0.8, 1, 0), _row(0.9, 1, 0)) is False
    assert strictly_better(_row(0.9, 1, 0), _row(0.9, 1, 0)) is False  # equal -> not better (keep earlier)


def test_serializable_env_kwargs_handles_randomization():
    env_kwargs = {"adapter": "holoocean", "start_randomization": StartRandomization(lateral_offset_m=1.0)}
    out = _serializable_env_kwargs(env_kwargs)
    json.dumps(out)  # must be JSON-serializable
    assert out["adapter"] == "holoocean"
    assert out["start_randomization"] == asdict(StartRandomization(lateral_offset_m=1.0))


def _make_run(tmp_path, *, track_hash="abc", hidden=(64, 64), bc=False, adapter="holoocean",
              current=None, randomized=False, reward=None, ppo=None):
    run = tmp_path / "run"
    run.mkdir()
    (run / "track_sha256.txt").write_text(track_hash, encoding="utf-8")
    env_kwargs = {"adapter": adapter, "current_profile": current}
    if randomized:
        env_kwargs["start_randomization"] = asdict(StartRandomization(lateral_offset_m=1.0))
    (run / "run_config.json").write_text(json.dumps({
        "obs_encoding_version": OBS_ENCODING_VERSION, "action_dim": ACTION_DIM,
        "hidden_sizes": list(hidden), "bc_initialized": bc, "env_kwargs": env_kwargs,
        "ppo_kwargs": ppo or {"n_steps": 256, "batch_size": 64, "n_epochs": 4},
    }), encoding="utf-8")
    (run / "reward_config.json").write_text(json.dumps(asdict(reward or RewardConfig())), encoding="utf-8")
    return run


def _current(**over):
    base = dict(
        track_sha256="abc", hidden_sizes=[64, 64], bc_initialized=False, adapter_requested="holoocean",
        current_profile=None, randomized=False, obs_encoding_version=OBS_ENCODING_VERSION,
        action_dim=ACTION_DIM, reward_config=asdict(RewardConfig()), ppo_kwargs={"n_steps": 256, "batch_size": 64, "n_epochs": 4},
    )
    base.update(over)
    return base


def test_compatible_resume_passes(tmp_path):
    run = _make_run(tmp_path)
    _validate_resume_compatibility(run, current=_current())  # no raise


@pytest.mark.parametrize("bad", [
    {"track_sha256": "different"},
    {"hidden_sizes": [128, 128]},
    {"adapter_requested": "fallback"},
    {"randomized": True},
    {"obs_encoding_version": "other_v2"},
    {"reward_config": asdict(RewardConfig(gate_bonus=999.0))},
    {"ppo_kwargs": {"n_steps": 128, "batch_size": 64, "n_epochs": 4}},
])
def test_incompatible_resume_rejected(tmp_path, bad):
    run = _make_run(tmp_path)
    with pytest.raises(ValueError, match="incompatible"):
        _validate_resume_compatibility(run, current=_current(**bad))
