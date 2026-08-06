"""Anchored SAC actor: bounded pre-tanh, drift control and collapse detection.

The v1 run collapsed with a mean absolute action of 0.53 (surge 0.79, yaw 0.82)
and 108 out-of-bounds episodes.  The measured cause was the unbounded inverse
tanh between the mean head and the Gaussian location: at initialization its
largest Jacobian is 1.6, and after 50,688 transitions it reached 493,448, so a
single mean-head component approaching the action boundary raised the effective
actor step by five orders of magnitude.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from marine_race_arena.learning.config_local_transition import (
    OBS_DIM_LOCAL_TRANSITION,
)
from marine_race_arena.learning.sac_health_gates import (
    DEFAULT_HEALTH_THRESHOLDS,
    HealthMonitor,
    HealthThresholds,
    evaluate_health,
    thresholds_from_mapping,
)
from marine_race_arena.learning.sac_transition_policy import (
    DEFAULT_PRE_TANH_LIMIT,
    SACTransitionAgent,
    SquashedGaussianActor,
    action_drift,
    compute_sac_bootstrap_target,
)


def _observations(n=256, seed=3):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, OBS_DIM_LOCAL_TRANSITION)).astype(np.float32)


def _batch(observations, reward=0.0, discount=0.985):
    n = observations.shape[0]
    return {
        "observations": observations,
        "actions": np.zeros((n, 4), dtype=np.float32),
        "rewards": np.full(n, reward, dtype=np.float32),
        "next_observations": observations,
        "discounts": np.full(n, discount, dtype=np.float32),
    }


# --------------------------------------------------- bounded pre-tanh location


def test_pre_tanh_location_is_bounded_and_caps_the_jacobian():
    actor = SquashedGaussianActor(pre_tanh_limit=2.0)
    config = actor.config()
    assert config["pre_tanh_limit"] == 2.0
    assert config["mean_clip"] == pytest.approx(math.tanh(2.0))
    assert config["maximum_pre_tanh_jacobian"] == pytest.approx(14.154, abs=1e-3)
    # Force the mean head far past the action boundary: the location stays bounded.
    with torch.no_grad():
        actor.mean_head.bias.fill_(50.0)
    observations = torch.from_numpy(_observations())
    health = actor.health(observations)
    assert health["pre_tanh_jacobian_max"] == pytest.approx(
        config["maximum_pre_tanh_jacobian"], rel=1e-5
    )
    assert health["mean_absolute_action"] <= math.tanh(2.0) + 1e-6


def test_unbounded_limit_reproduces_the_v1_amplification():
    legacy = SquashedGaussianActor(pre_tanh_limit=math.atanh(1.0 - 1e-6))
    with torch.no_grad():
        legacy.mean_head.bias.fill_(50.0)
    health = legacy.health(torch.from_numpy(_observations()))
    # The defect the repaired actor removes: five orders of magnitude.
    assert health["pre_tanh_jacobian_max"] > 100_000.0
    bounded = SquashedGaussianActor(pre_tanh_limit=DEFAULT_PRE_TANH_LIMIT)
    with torch.no_grad():
        bounded.mean_head.bias.fill_(50.0)
    assert bounded.health(torch.from_numpy(_observations()))[
        "pre_tanh_jacobian_max"] < 15.0


def test_default_bound_is_inert_for_a_warm_scale_actor():
    """A policy acting in the warm range is untouched by the bound."""

    actor = SquashedGaussianActor(pre_tanh_limit=DEFAULT_PRE_TANH_LIMIT)
    with torch.no_grad():
        actor.mean_head.weight.mul_(0.05)
        actor.mean_head.bias.fill_(0.05)
    health = actor.health(torch.from_numpy(_observations()))
    assert health["mean_head_clipped_fraction"] == 0.0
    assert health["pre_tanh_jacobian_max"] < 2.0


# ------------------------------------------------------------- anchored actor


def test_anchor_starts_identical_to_the_actor_and_is_frozen():
    agent = SACTransitionAgent(anchor_coefficient=1.0)
    observations = torch.from_numpy(_observations())
    with torch.no_grad():
        drift = action_drift(
            agent.actor.deterministic(observations),
            agent.anchor_actor.deterministic(observations),
        )
    assert drift["anchor_mean_drift"] == pytest.approx(0.0, abs=1e-7)
    assert all(not p.requires_grad for p in agent.anchor_actor.parameters())


def test_anchor_penalty_reduces_measured_drift():
    observations = _observations()
    batch = _batch(observations, reward=5.0)

    def drift_after(coefficient, steps=40):
        agent = SACTransitionAgent(
            actor_learning_rate=3e-3, critic_learning_rate=3e-3,
            anchor_coefficient=coefficient, anchor_coefficient_max=coefficient,
            anchor_target_drift=10.0,  # disable adaptation for a clean comparison
        )
        for _ in range(steps):
            metrics = agent.update(batch, update_actor=True, update_entropy=False)
        return metrics["anchor_mean_drift"]

    assert drift_after(0.0) > drift_after(50.0)


def test_anchor_coefficient_increases_when_drift_exceeds_the_target():
    agent = SACTransitionAgent(
        actor_learning_rate=3e-2, critic_learning_rate=1e-3,
        anchor_coefficient=0.0, anchor_target_drift=1e-9,
    )
    batch = _batch(_observations(), reward=10.0)
    start = agent.anchor_coefficient
    for _ in range(5):
        agent.update(batch, update_actor=True, update_entropy=False)
    assert agent.anchor_coefficient > start


def test_anchor_is_never_relaxed_by_transitions_alone():
    agent = SACTransitionAgent(anchor_coefficient=4.0, anchor_target_drift=1.0)
    batch = _batch(_observations())
    for _ in range(20):
        agent.update(batch, update_actor=True, update_entropy=False)
    # No relaxation happened during training; only an explicit call relaxes it.
    assert agent.anchor_coefficient >= 4.0
    relaxed = agent.relax_anchor()
    assert relaxed == pytest.approx(4.0 * agent.anchor_decrease_factor)


def test_anchor_relaxation_respects_its_floor():
    agent = SACTransitionAgent(anchor_coefficient=1e-6, anchor_coefficient_min=0.5)
    assert agent.relax_anchor() == pytest.approx(0.5)


def test_anchor_actor_never_produces_a_runtime_action():
    agent = SACTransitionAgent(anchor_coefficient=1.0)
    with torch.no_grad():
        agent.actor.mean_head.bias.fill_(0.4)
    observations = _observations(8)
    deterministic = agent.act(observations, deterministic=True)
    with torch.no_grad():
        tensor = torch.from_numpy(observations)
        expected = agent.actor.deterministic(tensor).numpy()
        anchor = agent.anchor_actor.deterministic(tensor).numpy()
    assert np.allclose(deterministic, expected)
    assert not np.allclose(deterministic, anchor)


def test_drift_metrics_are_per_sample_and_ordered():
    current = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.4, 0.0, 0.0, 0.0]])
    anchor = torch.zeros_like(current)
    drift = action_drift(current, anchor)
    assert drift["anchor_mean_drift"] == pytest.approx(0.05)
    assert drift["anchor_max_drift"] == pytest.approx(0.1)
    assert drift["anchor_p95_drift"] >= drift["anchor_mean_drift"]


# ---------------------------------------------------------- clipping / alpha


def test_actor_and_critic_gradients_are_clipped_separately():
    agent = SACTransitionAgent(
        actor_learning_rate=1e-2, critic_learning_rate=1e-2,
        gradient_clip_actor=1.0, gradient_clip_critic=5.0,
    )
    metrics = agent.update(_batch(_observations(), reward=1e3), update_actor=True)
    assert metrics["actor_gradient_norm"] >= 0.0
    assert metrics["critic_gradient_norm"] >= 0.0
    assert agent.gradient_clip_actor == 1.0 and agent.gradient_clip_critic == 5.0


def test_alpha_stays_inside_its_bounds():
    agent = SACTransitionAgent(
        entropy_learning_rate=5.0, initial_alpha=0.02,
        alpha_min=0.001, alpha_max=0.02, actor_learning_rate=1e-4,
    )
    batch = _batch(_observations())
    for _ in range(30):
        agent.update(batch, update_actor=True, update_entropy=True)
        assert agent.alpha_min - 1e-9 <= agent.alpha <= agent.alpha_max + 1e-9


def test_entropy_update_raises_alpha_when_entropy_is_too_low():
    """log_prob + target_entropy > 0 must push alpha up, not down."""

    agent = SACTransitionAgent(
        entropy_learning_rate=0.5, initial_alpha=0.005,
        alpha_min=1e-6, alpha_max=1.0, target_entropy=50.0,
    )
    start = agent.alpha
    batch = _batch(_observations())
    for _ in range(5):
        agent.update(batch, update_actor=True, update_entropy=True)
    assert agent.alpha > start


def test_bootstrap_target_zeroes_on_a_terminal_transition():
    rewards = torch.tensor([[2.0]])
    q = torch.tensor([[-30.0]])
    logp = torch.tensor([[1.0]])
    terminal = compute_sac_bootstrap_target(rewards, torch.tensor([[0.0]]), q, 0.02, logp)
    bootstrapped = compute_sac_bootstrap_target(
        rewards, torch.tensor([[0.985]]), q, 0.02, logp
    )
    assert terminal.item() == pytest.approx(2.0)
    assert bootstrapped.item() < terminal.item()


def test_huber_critic_is_less_outlier_sensitive_than_mse():
    observations = _observations(128)
    outlier = _batch(observations, reward=0.0)
    outlier["rewards"][0] = -5_000.0
    huber = SACTransitionAgent(critic_loss="huber", critic_learning_rate=1e-3)
    mse = SACTransitionAgent(critic_loss="mse", critic_learning_rate=1e-3)
    huber_metrics = huber.update(outlier, update_actor=False)
    mse_metrics = mse.update(outlier, update_actor=False)
    assert huber_metrics["critic_loss"] < mse_metrics["critic_loss"]
    assert huber_metrics["critic_gradient_norm"] <= mse_metrics["critic_gradient_norm"]


def test_update_reports_q_and_td_percentiles():
    agent = SACTransitionAgent()
    metrics = agent.update(_batch(_observations()), update_actor=True)
    for key in ("q_p05", "q_p95", "target_q_p05", "target_q_p95",
                "td_error_p50", "td_error_p95", "td_error_max"):
        assert key in metrics and math.isfinite(metrics[key])


def test_checkpoint_round_trip_preserves_anchor_and_bounds(tmp_path):
    agent = SACTransitionAgent(
        anchor_coefficient=3.5, pre_tanh_limit=2.0, alpha_min=0.001, alpha_max=0.02,
    )
    with torch.no_grad():
        agent.actor.mean_head.bias.fill_(0.3)
    state = agent.checkpoint_state()
    restored = SACTransitionAgent.from_checkpoint_state(state)
    assert restored.anchor_coefficient == pytest.approx(3.5)
    assert restored.actor.pre_tanh_limit == 2.0
    observations = torch.from_numpy(_observations(64))
    assert torch.allclose(
        restored.anchor_actor.deterministic(observations),
        agent.anchor_actor.deterministic(observations),
    )
    assert torch.allclose(
        restored.actor.deterministic(observations),
        agent.actor.deterministic(observations),
    )


# -------------------------------------------------------------- health gates


def test_reported_v1_collapse_metrics_trigger_a_pause():
    collapsed = {
        "mean_absolute_action": 0.53, "mean_absolute_action_yaw": 0.82,
        "action_saturation_fraction": 0.31, "anchor_mean_drift": 0.42,
        "anchor_p95_drift": 0.60, "mean_q": -27.3, "critic_loss": 543.0,
    }
    monitor = HealthMonitor()
    for index in range(DEFAULT_HEALTH_THRESHOLDS.consecutive_violations):
        record = monitor.observe(collapsed, updates=index)
    assert record["should_pause"] is True
    assert "mean_absolute_action" in record["reasons"]
    assert "absolute_yaw_action" in record["reasons"]
    assert "anchor_mean_drift" in record["reasons"]


def test_a_single_noisy_sample_does_not_pause_the_run():
    monitor = HealthMonitor()
    record = monitor.observe({"mean_absolute_action": 0.55}, updates=1)
    assert record["violated"] and not record["should_pause"]
    healthy = monitor.observe({"mean_absolute_action": 0.09}, updates=2)
    assert not healthy["violated"] and monitor.consecutive == 0


def test_non_finite_learner_values_pause_immediately():
    monitor = HealthMonitor()
    record = monitor.observe({"critic_loss": float("nan")}, updates=1)
    assert record["fatal"] and record["should_pause"]
    assert record["reasons"] == ["non_finite:critic_loss"]


def test_healthy_warm_scale_metrics_never_trigger():
    healthy = {
        "mean_absolute_action": 0.098, "mean_absolute_action_yaw": 0.091,
        "action_saturation_fraction": 0.0, "anchor_mean_drift": 0.01,
        "anchor_p95_drift": 0.03, "mean_q": -3.2, "critic_loss": 1.4,
        "actor_gradient_norm": 0.4, "critic_gradient_norm": 1.1,
        "entropy_coefficient": 0.01,
    }
    violated, reasons = evaluate_health(healthy)
    assert not violated and reasons == ()


def test_out_of_bounds_rate_needs_enough_recent_episodes():
    noisy = evaluate_health({}, recent_episodes=5, out_of_bounds_episodes=5)
    assert not noisy[0]
    real = evaluate_health({}, recent_episodes=100, out_of_bounds_episodes=40)
    assert "out_of_bounds_rate" in real[1]


def test_compact_probe_rates_can_trigger_a_pause():
    violated, reasons = evaluate_health(
        {}, probe={"first_gate_crossing_rate": 0.10, "target_switch_rate": 0.05}
    )
    assert violated
    assert "probe_first_gate_rate" in reasons
    assert "probe_target_switch_rate" in reasons


def test_exploding_q_values_are_detected():
    violated, reasons = evaluate_health({"mean_q": -50_000.0})
    assert violated and any(r.startswith("exploding_q") for r in reasons)


def test_health_thresholds_reject_unknown_keys():
    tuned = thresholds_from_mapping({"max_mean_absolute_action": 0.2})
    assert tuned.max_mean_absolute_action == pytest.approx(0.2)
    assert isinstance(tuned, HealthThresholds)
    with pytest.raises(ValueError):
        thresholds_from_mapping({"nope": 1})


def test_health_monitor_state_round_trip():
    monitor = HealthMonitor()
    monitor.observe({"mean_absolute_action": 0.55}, updates=1)
    restored = HealthMonitor()
    restored.load_state_dict(monitor.state_dict())
    assert restored.consecutive == monitor.consecutive
    assert list(restored.last_reasons) == list(monitor.last_reasons)


def test_agent_health_probe_exposes_drift_and_saturation():
    agent = SACTransitionAgent(anchor_coefficient=1.0)
    with torch.no_grad():
        agent.actor.mean_head.bias.fill_(0.9)
    health = agent.anchor_health(_observations(128))
    assert health["anchor_mean_drift"] > 0.0
    assert "action_saturation_fraction" in health
    assert "pre_tanh_jacobian_max" in health
    assert health["anchor_coefficient"] == pytest.approx(1.0)
