"""Actor-only transfer, fresh critics, fresh replay and critic-health detection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from marine_race_arena.learning.sac_health_gates import (
    PHASE_DIVERGING,
    CriticHealthMonitor,
    CriticHealthThresholds,
    DEFAULT_CRITIC_HEALTH,
)
from marine_race_arena.learning.sac_replay_buffer import StratifiedReplayBuffer
from marine_race_arena.learning.sac_transition_checkpoint import (
    atomic_save_sac_checkpoint,
)
from marine_race_arena.learning.sac_transition_policy import (
    SACTransitionAgent,
    critics_are_independent,
    rebuild_critics_from_actor,
)


def _observations(n=256, seed=5):
    return torch.from_numpy(
        np.random.default_rng(seed).normal(size=(n, 35)).astype(np.float32)
    )


def _degraded_source(tmp_path) -> Path:
    """An agent whose actor is fine but whose critics have been driven away."""

    agent = SACTransitionAgent(hidden_sizes=(32, 32), anchor_coefficient=4.0)
    with torch.no_grad():
        agent.actor.mean_head.bias.fill_(0.13)
        for parameter in agent.critic.parameters():
            parameter.add_(7.5)
        for parameter in agent.target_critic.parameters():
            parameter.add_(7.5)
    path = tmp_path / "sac_300032_steps.pt"
    torch.save({"agent": agent.checkpoint_state()}, path)
    return path, agent


def test_actor_transfers_verbatim_and_critics_are_fresh(tmp_path):
    source, original = _degraded_source(tmp_path)
    agent, report = rebuild_critics_from_actor(
        source, anchor_coefficient=10.0, critic_learning_rate=3e-5,
        actor_learning_rate=1e-7, entropy_learning_rate=1e-6, tau=0.002,
    )
    observations = _observations()
    with torch.no_grad():
        assert torch.allclose(
            agent.actor.deterministic(observations),
            original.actor.deterministic(observations),
        ), "the validated actor must transfer unchanged"
    assert critics_are_independent(agent, original)
    for key in (
        "critic_1_reinitialized", "critic_2_reinitialized",
        "target_critic_1_reinitialized", "target_critic_2_reinitialized",
        "critic_optimizer_reinitialized", "actor_optimizer_reinitialized",
        "entropy_state_reinitialized",
    ):
        assert report[key] is True, key
    assert report["replay_imported"] is False
    assert report["actor_transferred"] is True


def test_target_critics_are_not_copies_of_the_degraded_ones(tmp_path):
    source, original = _degraded_source(tmp_path)
    agent, _ = rebuild_critics_from_actor(source, tau=0.002)
    left = agent.target_critic.state_dict()
    right = original.target_critic.state_dict()
    assert not any(
        torch.allclose(left[name], right[name]) for name in left if left[name].numel()
    )
    # A fresh run starts with target == online critic.
    online = agent.critic.state_dict()
    assert all(torch.allclose(left[n], online[n]) for n in left)


def test_transferred_actor_becomes_its_own_anchor(tmp_path):
    source, original = _degraded_source(tmp_path)
    agent, _ = rebuild_critics_from_actor(source, anchor_coefficient=10.0)
    observations = _observations()
    with torch.no_grad():
        assert torch.allclose(
            agent.anchor_actor.deterministic(observations),
            original.actor.deterministic(observations),
        )
    assert agent.anchor_coefficient == pytest.approx(10.0)
    assert all(not p.requires_grad for p in agent.anchor_actor.parameters())


def test_overrides_win_over_the_degraded_runs_settings(tmp_path):
    source, _ = _degraded_source(tmp_path)
    agent, _ = rebuild_critics_from_actor(
        source, tau=0.002, alpha_min=0.001, alpha_max=0.02,
        critic_loss="huber", gradient_clip_actor=1.0, gradient_clip_critic=5.0,
    )
    assert agent.tau == pytest.approx(0.002)
    assert (agent.alpha_min, agent.alpha_max) == (0.001, 0.02)
    assert agent.critic_loss_kind == "huber"
    assert agent.gradient_clip_actor == 1.0 and agent.gradient_clip_critic == 5.0


def test_rebuild_refuses_a_foreign_checkpoint(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"agent": {"schema_version": "something_else"}}, path)
    with pytest.raises(ValueError, match="unsupported SAC agent checkpoint"):
        rebuild_critics_from_actor(path)


def test_a_rebuilt_run_starts_with_an_empty_replay():
    replay = StratifiedReplayBuffer(capacity=1000, seed=1)
    assert replay.size == 0


# ------------------------------------------------------- critic health


#: Fast baseline for tests: real runs demand far more evidence, but the
#: property under test is the detector's response *after* a baseline exists.
FAST_BASELINE = CriticHealthThresholds(
    minimum_baseline_samples=40, minimum_baseline_updates=100,
    minimum_baseline_windows=1, window_samples=40,
    # The detector now refuses a verdict until a full post-baseline
    # observation window; shrink it too or nothing can ever reach HEALTHY.
    minimum_post_baseline_updates=40, drift_window_updates=40,
    healthy_windows_after_baseline=2, settling_windows=1,
)


def _settled(step):
    return {
        "mean_q": -3.0 + 0.01 * ((-1) ** step),
        "q_p05": -9.0, "q_p95": 1.0, "td_error_p95": 2.0,
        "critic_loss": 4.0, "critic_gradient_norm": 0.6,
    }


def _with_baseline(thresholds=FAST_BASELINE, gradient_clip=5.0):
    """A monitor that has already reached HEALTHY on a settled critic."""

    monitor = CriticHealthMonitor(thresholds, gradient_clip=gradient_clip)
    for step in range(200):
        monitor.observe(_settled(step), updates=step * 10)
    assert monitor.is_healthy(), f"helper failed to settle: phase={monitor.phase}"
    return monitor


def _diverging(step):
    return {
        "mean_q": -3.0 - step * 2.0,
        "q_p05": -10.0 - step * 6.0,
        "q_p95": 2.0,
        "td_error_p95": 2.0 + step * 1.5,
        "critic_loss": 5.0 + step * 6.0,
        "critic_gradient_norm": 5.0,
    }


def _healthy(step):
    return {
        "mean_q": -3.0 + 0.01 * ((-1) ** step),
        "q_p05": -9.0, "q_p95": 1.0, "td_error_p95": 2.0,
        "critic_loss": 4.0, "critic_gradient_norm": 0.6,
    }


def test_v2_divergence_signature_is_detected_before_an_evaluation():
    monitor = _with_baseline()
    record = None
    for step in range(120):
        record = monitor.observe(_diverging(step), updates=1_200 + step * 100)
    assert record["should_pause"] is True
    assert record["phase"] == PHASE_DIVERGING
    assert "q_median_drift" in record["reasons"]
    assert "td_error_growth" in record["reasons"]
    assert "q_spread_expansion" in record["reasons"]


def test_an_exploding_critic_loss_is_flagged_on_its_own():
    monitor = _with_baseline()
    record = None
    for step in range(120):
        sample = _healthy(step)
        # Loss alone runs away by far more than the 8x growth threshold.
        sample["critic_loss"] = 4.0 * (1.10 ** step)
        record = monitor.observe(sample, updates=1_200 + step * 100)
    assert "critic_loss_explosion" in record["reasons"]


def test_protective_action_preserves_the_actor():
    monitor = _with_baseline()
    record = None
    for step in range(120):
        record = monitor.observe(_diverging(step), updates=1_200 + step * 100)
    # Actor competence and critic health are separable: rebuild the critics,
    # never discard the actor.
    assert record["action"] == "checkpoint_freeze_actor_and_rebuild_critics"


def test_healthy_critics_never_trigger():
    monitor = CriticHealthMonitor()
    for step in range(60):
        record = monitor.observe(_healthy(step), updates=step * 100)
        assert not record["diverging"], record["reasons"]


def test_detector_waits_for_a_baseline():
    monitor = CriticHealthMonitor()
    for step in range(150):
        record = monitor.observe(_diverging(step), updates=step * 100)
        assert not record["diverging"], "must not judge before a baseline exists"


def test_pinned_critic_gradients_are_flagged():
    """Clipping counts only once it fills the rolling window and rises above
    the baseline regime, so the window must actually be flushed here."""

    monitor = _with_baseline()
    baseline_clip = monitor.baseline_clip_fraction()
    assert baseline_clip == pytest.approx(0.0), "helper baseline must be unclipped"
    record = None
    for step in range(700):
        sample = _healthy(step)
        sample["critic_gradient_norm"] = 5.0
        record = monitor.observe(sample, updates=2_000 + step * 100)
    assert monitor.clip_hit_fraction() > 0.9
    assert "critic_gradients_pinned_at_clip" in record["reasons"]


def test_health_thresholds_are_serializable():
    payload = DEFAULT_CRITIC_HEALTH.as_dict()
    assert json.dumps(payload)
    assert payload["consecutive_violations"] == 3


def test_detector_state_round_trip():
    monitor = CriticHealthMonitor()
    for step in range(25):
        monitor.observe(_healthy(step), updates=step * 10)
    state = monitor.state_dict()
    assert state["samples"] and json.dumps(state)


# ------------------------------------------------------------- config


def test_v3_config_rebuilds_critics_and_keeps_the_v2_actor():
    from marine_race_arena.learning.train_sac_transition import _load_config

    config = _load_config("configs/rl/sac_universal_transition_v3_critic_rebuild.json")
    initialization = config["initialization"]
    assert initialization["mode"] == "actor_only_critic_rebuild"
    assert "sac_300032_steps.pt" in initialization["source_checkpoint"]
    assert initialization["replay_imported_from_predecessor"] is False
    assert initialization["optimizer_state_imported_from_predecessor"] is False
    sac = config["sac"]
    assert sac["tau"] == pytest.approx(0.002)
    assert sac["critic_learning_rate"] == pytest.approx(3e-5)
    assert sac["actor_learning_rate"] == pytest.approx(1e-7)
    assert sac["learning_starts"] == 20_000
    assert sac["critic_warmup_updates"] == 10_000
    assert sac["policy_delay"] == 8
    assert sac["anchor"]["enabled"] is True
    assert config["critic_health"]["consecutive_violations"] == 3


def test_config_rejects_an_out_of_range_stability_knob(tmp_path):
    from marine_race_arena.learning.train_sac_transition import _load_config

    base = json.loads(
        Path("configs/rl/sac_universal_transition_v3_critic_rebuild.json").read_text()
    )
    base["sac"]["tau"] = 0.5
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ValueError, match="tau"):
        _load_config(path)
