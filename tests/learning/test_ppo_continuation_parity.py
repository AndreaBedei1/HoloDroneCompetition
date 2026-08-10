"""The continuation must actually continue, and inactivity must be caught.

Root cause of the lost run: ``_setup_model()`` was called after ``PPO.load`` to
rebuild the rollout buffer for a new worker sharding, and it replaced the loaded
policy with a freshly initialised one.  The run then trained a random network for
~51k transitions while every report described it as an 84%-success continuation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from marine_race_arena.learning.ppo_activity_gate import (
    STATUS_HEALTHY,
    STATUS_INACTIVE,
    STATUS_WARNING,
    ActivityMonitor,
    ActivityThresholds,
    activity_thresholds_from_mapping,
    evaluate_activity,
)
from marine_race_arena.learning.train_ppo_transition import (
    assert_policy_matches_parent,
    policy_parameter_sha256,
)


# ------------------------------------------------------------ policy parity


def _tiny_ppo(tmp_path, seed=0):
    """A saved 'parent' whose weights are NOT its initialisation.

    A seeded PPO re-initialises to identical weights, so an unperturbed model
    could not distinguish "restored" from "freshly built" -- the very thing
    under test.  Nudging the parameters first makes re-initialisation visible.
    """

    import gymnasium as gym
    import torch
    from stable_baselines3 import PPO

    env = gym.make("Pendulum-v1")
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, seed=seed, device="cpu")
    with torch.no_grad():
        for parameter in model.policy.parameters():
            parameter.add_(0.05)
    path = tmp_path / f"parent_{seed}.zip"
    model.save(str(path))
    return model, path


def test_hash_is_deterministic_and_parameter_sensitive(tmp_path):
    import torch

    model, _ = _tiny_ppo(tmp_path)
    first = policy_parameter_sha256(model)
    assert first == policy_parameter_sha256(model)
    with torch.no_grad():
        model.policy.log_std.add_(0.5)
    assert policy_parameter_sha256(model) != first


def test_parity_passes_for_an_untouched_load(tmp_path):
    from stable_baselines3 import PPO

    _, path = _tiny_ppo(tmp_path)
    loaded = PPO.load(str(path), device="cpu")
    report = assert_policy_matches_parent(loaded, path)
    assert report["policy_parity_verified"] is True


def test_parity_catches_a_reinitialised_policy(tmp_path):
    """Exactly the failure: _setup_model() silently replaces the policy."""

    from stable_baselines3 import PPO

    _, path = _tiny_ppo(tmp_path, seed=1)
    loaded = PPO.load(str(path), device="cpu")
    loaded._setup_model()  # discards the loaded weights, as SB3 does
    with pytest.raises(ValueError, match="did not preserve the parent policy"):
        assert_policy_matches_parent(loaded, path)


def test_snapshot_and_restore_survives_setup_model(tmp_path):
    """The fix: snapshot around _setup_model, then restore."""

    import copy

    from stable_baselines3 import PPO

    _, path = _tiny_ppo(tmp_path, seed=2)
    loaded = PPO.load(str(path), device="cpu")
    policy_state = copy.deepcopy(loaded.policy.state_dict())
    optimizer_state = copy.deepcopy(loaded.policy.optimizer.state_dict())
    loaded.n_steps = 128
    loaded._setup_model()
    loaded.policy.load_state_dict(policy_state)
    loaded.policy.optimizer.load_state_dict(optimizer_state)
    assert assert_policy_matches_parent(loaded, path)["policy_parity_verified"]


def test_the_continuation_path_snapshots_before_setup_model():
    from marine_race_arena.learning import train_ppo_transition as trainer

    body = Path(trainer.__file__).read_text(encoding="utf-8").split(
        "def _continue_from_parent", 1
    )[1]
    snapshot = body.index("policy_state = copy.deepcopy")
    setup = body.index("model._setup_model()")
    restore = body.index("model.policy.load_state_dict(policy_state)")
    assert snapshot < setup < restore, "order must be snapshot -> setup -> restore"
    assert "assert_policy_matches_parent(model, path)" in body
    assert "optimizer.load_state_dict(optimizer_state)" in body


# --------------------------------------------------------- inactivity gate


ACTIVE = {
    "episodes": 40, "mean_absolute_action": 0.098,
    "nontrivial_action_fraction": 0.62, "mean_distance_travelled_m": 7.2,
    "first_gate_reach_fraction": 1.0, "completed_gate_rate": 1.27,
}
# The measured collapse window: |a| 0.0089, nontrivial 0, 3.05 m, 9.8% first gate.
COLLAPSED = {
    "episodes": 40, "mean_absolute_action": 0.0089,
    "nontrivial_action_fraction": 0.0, "mean_distance_travelled_m": 3.05,
    "first_gate_reach_fraction": 0.098, "completed_gate_rate": 0.098,
}


def test_a_healthy_policy_never_triggers():
    monitor = ActivityMonitor()
    for step in range(20):
        record = monitor.observe(ACTIVE, timesteps=step * 2048)
        assert record["status"] == STATUS_HEALTHY
        assert record["should_rollback"] is False


def test_the_measured_collapse_is_detected():
    monitor = ActivityMonitor()
    record = None
    for step in range(6):
        record = monitor.observe(COLLAPSED, timesteps=step * 2048)
    assert record["status"] == STATUS_INACTIVE
    assert record["should_rollback"] is True
    assert record["action"] == "checkpoint_and_restore_latest_competent"
    assert "mean_absolute_action" in record["reasons"]
    assert "nontrivial_action_fraction" in record["reasons"]
    assert "first_gate_reach_fraction" in record["task_reasons"]


def test_one_quiet_rollout_is_not_a_collapse():
    monitor = ActivityMonitor()
    record = monitor.observe(COLLAPSED, timesteps=0)
    assert record["status"] == STATUS_HEALTHY
    assert record["should_rollback"] is False


def test_sustained_quiet_without_task_failure_warns_but_never_rolls_back():
    """A gentle-but-effective policy must not be thrown away."""

    gentle = dict(COLLAPSED)
    gentle.update(
        mean_distance_travelled_m=8.0,
        first_gate_reach_fraction=0.95,
        completed_gate_rate=1.1,
    )
    monitor = ActivityMonitor()
    record = None
    for step in range(12):
        record = monitor.observe(gentle, timesteps=step * 2048)
    assert record["status"] == STATUS_WARNING
    assert record["should_rollback"] is False


def test_a_small_window_is_never_judged():
    monitor = ActivityMonitor()
    tiny = dict(COLLAPSED, episodes=3)
    for step in range(10):
        record = monitor.observe(tiny, timesteps=step)
    assert record["should_rollback"] is False
    assert record["note"] == "insufficient_episodes"


def test_recovery_resets_the_counters():
    monitor = ActivityMonitor()
    for step in range(6):
        monitor.observe(COLLAPSED, timesteps=step)
    assert monitor.status == STATUS_INACTIVE
    monitor.reset_after_rollback()
    assert monitor.status == STATUS_HEALTHY
    assert monitor.consecutive_failing == 0


def test_thresholds_mirror_the_competence_gate():
    from marine_race_arena.learning.transition_selection import (
        DEFAULT_COMPETENCE_THRESHOLDS,
    )

    activity = ActivityThresholds()
    assert activity.min_mean_absolute_action == pytest.approx(
        DEFAULT_COMPETENCE_THRESHOLDS.min_mean_absolute_action
    )
    assert activity.min_nontrivial_action_fraction == pytest.approx(
        DEFAULT_COMPETENCE_THRESHOLDS.min_nontrivial_action_fraction
    )


def test_unknown_threshold_keys_are_rejected():
    with pytest.raises(ValueError, match="unknown activity threshold keys"):
        activity_thresholds_from_mapping({"whatever": 1})


def test_state_round_trips():
    import json

    monitor = ActivityMonitor()
    for step in range(6):
        monitor.observe(COLLAPSED, timesteps=step)
    payload = json.loads(json.dumps(monitor.state_dict()))
    restored = ActivityMonitor()
    restored.load_state_dict(payload)
    assert restored.status == STATUS_INACTIVE
    assert restored.consecutive_failing == monitor.consecutive_failing
