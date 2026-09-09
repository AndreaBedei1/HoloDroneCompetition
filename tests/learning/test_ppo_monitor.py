"""Tests for structured KL monitoring and the hard KL safety stop (no SB3 needed)."""

import csv

import numpy as np
import pytest

from marine_race_arena.learning.ppo_monitor import PPOUpdateRecorder, RunStatus


class _FakeTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeModel:
    """Minimal stand-in for an SB3 PPO model exposing what the recorder reads."""

    def __init__(self, actions):
        self.logger = type("L", (), {"name_to_value": {}})()
        self.policy = type("P", (), {"log_std": _FakeTensor(np.log(np.full(4, 0.1, np.float32)))})()
        self.action_space = type("S", (), {"shape": (4,)})()
        self.rollout_buffer = type("B", (), {"actions": np.asarray(actions, dtype=np.float32)})()
        self.num_timesteps = 0

    def do_update(self, kl, ts, clip=0.3):
        """Simulate SB3 train(): record metrics and advance the step counter."""
        self.num_timesteps = ts
        self.logger.name_to_value.update({
            "train/approx_kl": kl, "train/clip_fraction": clip,
            "train/policy_gradient_loss": 0.01, "train/value_loss": 1.0,
            "train/entropy_loss": -0.5, "train/explained_variance": 0.2,
        })


def _recorder(tmp_path, model, max_kl=0.02):
    return PPOUpdateRecorder(model, tmp_path / "m.csv", target_kl=0.01, max_acceptable_kl=max_kl)


def test_run_status_enum():
    assert set(RunStatus.ALL) == {"COMPLETED", "EARLY_STOP_TARGET_KL", "ABORT_MAX_KL",
                                  "ABORT_ACTION_SATURATION", "NUMERICAL_FAILURE", "SIMULATOR_FAILURE"}


def test_safe_update_records_metrics(tmp_path):
    model = _FakeModel(np.zeros((100, 4), np.float32))
    rec = _recorder(tmp_path, model)
    model.do_update(kl=0.008, ts=500)
    assert rec.record() is True
    assert rec.status == RunStatus.COMPLETED and rec.n_updates == 1 and not rec.should_abort()
    rows = list(csv.DictReader(open(tmp_path / "m.csv", encoding="utf-8")))
    assert rows[0]["approx_kl"] == "0.008" and rows[0]["target_kl_early_stop"] == "False"


def test_no_metrics_yet_records_nothing(tmp_path):
    rec = _recorder(tmp_path, _FakeModel(np.zeros((4, 4), np.float32)))
    assert rec.record() is False and rec.n_updates == 0  # first rollout, no train() yet


def test_duplicate_timestep_not_double_counted(tmp_path):
    model = _FakeModel(np.zeros((10, 4), np.float32))
    rec = _recorder(tmp_path, model)
    model.do_update(kl=0.005, ts=500)
    assert rec.record() is True
    assert rec.record() is False and rec.n_updates == 1  # same ts -> skipped


def test_hard_kl_abort(tmp_path):
    model = _FakeModel(np.zeros((100, 4), np.float32))
    rec = _recorder(tmp_path, model)
    model.do_update(kl=0.125, ts=500)
    rec.record()
    assert rec.status == RunStatus.ABORT_MAX_KL and rec.should_abort()


def test_numerical_failure_on_nonfinite(tmp_path):
    model = _FakeModel(np.zeros((10, 4), np.float32))
    rec = _recorder(tmp_path, model)
    model.do_update(kl=float("nan"), ts=500)
    rec.record()
    assert rec.status == RunStatus.NUMERICAL_FAILURE and rec.should_abort()


def test_saturated_actions_reported(tmp_path):
    model = _FakeModel(np.ones((100, 4), np.float32))
    rec = _recorder(tmp_path, model)
    model.do_update(kl=0.005, ts=500)
    rec.record()
    assert rec.rows[0]["action_saturation"] == pytest.approx(1.0)


def test_target_kl_early_stop_flagged(tmp_path):
    model = _FakeModel(np.zeros((10, 4), np.float32))
    rec = _recorder(tmp_path, model)
    model.do_update(kl=0.015, ts=500)  # >= target_kl but < max
    rec.record()
    assert rec.status == RunStatus.COMPLETED and rec.target_kl_early_stops == 1


def test_summary_tracks_max_kl(tmp_path):
    model = _FakeModel(np.zeros((10, 4), np.float32))
    rec = _recorder(tmp_path, model)
    model.do_update(kl=0.006, ts=500)
    rec.record()
    model.do_update(kl=0.009, ts=1000)
    rec.record()
    assert rec.summary()["max_approx_kl"] == pytest.approx(0.009) and rec.n_updates == 2
