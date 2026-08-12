"""Regressions for the two defects that cost SAC v3 an entire 362k run.

1. The actor never unfroze, because the unfreeze threshold was computed from
   ``agent.gradient_updates`` *before* a rebuild that resets it to zero.
2. The divergence detector fired on a freshly initialised critic at 22 updates,
   so every rebuild immediately seeded the next one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from marine_race_arena.learning.sac_actor_gate import (
    ActorFreezeGate,
    ActorGatePolicy,
    STATE_ACTIVE,
    STATE_FROZEN,
    actor_gate_policy_from_mapping,
)
from marine_race_arena.learning.sac_health_gates import (
    PHASE_BASELINE,
    PHASE_DIVERGING,
    PHASE_HEALTHY,
    PHASE_WARNING,
    CriticHealthMonitor,
    CriticHealthThresholds,
)


# --------------------------------------------------------------- actor gate


def _gate(min_updates=100, healthy=True):
    return ActorFreezeGate(
        ActorGatePolicy(
            actor_unfreeze_min_updates=min_updates,
            require_critic_healthy=healthy,
        )
    )


def _train(gate, n, *, critic_healthy=True):
    for _ in range(n):
        gate.record_critic_update()
        gate.consider_unfreeze(critic_healthy=critic_healthy)


def test_actor_starts_frozen():
    gate = _gate()
    assert gate.actor_frozen is True
    assert gate.state == STATE_FROZEN
    assert gate.should_update_actor() is False


def test_actor_unfreezes_after_enough_healthy_critic_updates():
    gate = _gate(min_updates=100)
    _train(gate, 99)
    assert gate.actor_frozen is True, "must not wake one update early"
    _train(gate, 1)
    assert gate.actor_frozen is False
    assert gate.state == STATE_ACTIVE
    assert gate.should_update_actor() is True
    assert gate.unfroze_at_updates_since_rebuild == 100


def test_a_stale_absolute_counter_can_never_strand_the_actor():
    """The exact SAC v3 scenario: 13,548 pre-rebuild updates, then a rebuild.

    The old code stored 13,548 + 10,000 = 23,548 and compared it against a
    counter that had just restarted at zero.  Nothing here may consult that
    number: after the rebuild only the new generation's own updates count.
    """

    gate = _gate(min_updates=10_000)
    _train(gate, 13_548)
    assert gate.actor_frozen is False, "the first generation should have woken"

    gate.on_critic_rebuild()
    assert gate.actor_frozen is True
    assert gate.critic_updates_since_rebuild == 0
    assert gate.critic_rebuild_generation == 1

    _train(gate, 9_999)
    assert gate.actor_frozen is True
    _train(gate, 1)
    assert gate.actor_frozen is False, (
        "the actor must wake after 10,000 updates of the NEW generation, "
        "not after an unreachable absolute threshold"
    )
    assert gate.unfroze_at_generation == 1
    assert gate.unfroze_at_updates_since_rebuild == 10_000


def test_many_rebuild_generations_each_unfreeze_independently():
    gate = _gate(min_updates=500)
    for generation in range(5):
        _train(gate, 500)
        assert gate.actor_frozen is False, f"generation {generation} never woke"
        assert gate.critic_rebuild_generation == generation
        gate.on_critic_rebuild()
        assert gate.actor_frozen is True
    assert gate.critic_rebuild_generation == 5


def test_unhealthy_critics_hold_the_actor_frozen_even_when_trained_enough():
    gate = _gate(min_updates=50)
    _train(gate, 500, critic_healthy=False)
    assert gate.actor_frozen is True
    assert "waiting_for_critic_health" in gate.last_reason
    gate.consider_unfreeze(critic_healthy=True)
    assert gate.actor_frozen is False


def test_gate_state_round_trips_through_a_checkpoint():
    gate = _gate(min_updates=100)
    _train(gate, 150)
    gate.on_critic_rebuild()
    _train(gate, 40)
    payload = json.loads(json.dumps(gate.state_dict()))

    restored = ActorFreezeGate(ActorGatePolicy())
    restored.load_state_dict(payload)
    assert restored.critic_rebuild_generation == gate.critic_rebuild_generation
    assert restored.critic_updates_since_rebuild == 40
    assert restored.actor_frozen is True
    assert restored.policy.actor_unfreeze_min_updates == 100
    # And it still wakes on schedule after the resume.
    _train(restored, 60)
    assert restored.actor_frozen is False


def test_actor_update_counter_is_tracked():
    gate = _gate(min_updates=1)
    _train(gate, 2)
    gate.record_actor_update()
    gate.record_actor_update()
    assert gate.total_actor_updates == 2
    assert gate.describe()["total_actor_updates"] == 2


def test_policy_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown actor gate keys"):
        actor_gate_policy_from_mapping({"nope": 1})


# ------------------------------------------------------- critic baseline


def _healthy_sample(step, base=-3.0):
    return {
        "mean_q": base + 0.02 * ((-1) ** step),
        "q_p05": base - 6.0, "q_p95": base + 4.0,
        "td_error_p95": 2.0, "td_error_max": 5.0,
        "critic_loss": 4.0, "critic_gradient_norm": 0.6,
    }


def _fresh_critic_learning(step):
    """A brand-new critic legitimately walking away from random init."""

    return {
        "mean_q": -0.1 - 0.5 * step,
        "q_p05": -0.2 - 2.0 * step,
        "q_p95": 0.1 + 0.4 * step,
        "td_error_p95": 2.0 + 0.5 * step,
        "td_error_max": 10.0 + step,
        "critic_loss": 5.0 + 2.0 * step,
        "critic_gradient_norm": 1.0,
    }


def test_a_new_critic_cannot_trigger_at_22_updates():
    """The literal SAC v3 first trigger must now be impossible."""

    monitor = CriticHealthMonitor()
    record = None
    for step in range(22):
        record = monitor.observe(_fresh_critic_learning(step), updates=step)
    assert record["phase"] == PHASE_BASELINE
    assert record["diverging"] is False
    assert record["should_pause"] is False
    assert record["reasons"] == []


def test_baseline_building_ignores_even_violent_early_movement():
    monitor = CriticHealthMonitor()
    record = None
    for step in range(150):
        record = monitor.observe(_fresh_critic_learning(step), updates=step * 10)
    assert record["phase"] == PHASE_BASELINE
    assert record["should_pause"] is False


def test_baseline_requires_samples_updates_and_windows():
    t = CriticHealthThresholds(
        minimum_baseline_samples=50, minimum_baseline_updates=5_000,
        minimum_baseline_windows=1, window_samples=50,
        minimum_post_baseline_updates=50, drift_window_updates=50,
        settling_windows=1,
    )
    monitor = CriticHealthMonitor(t)
    # Plenty of samples, but nowhere near enough updates.
    for step in range(60):
        record = monitor.observe(_healthy_sample(step), updates=step)
    assert record["phase"] == PHASE_BASELINE, "few updates must not establish a baseline"
    for step in range(60):
        record = monitor.observe(_healthy_sample(step), updates=5_000 + step * 100)
    assert record["baseline_established"] is True


def test_after_baseline_a_healthy_critic_stays_healthy():
    monitor = CriticHealthMonitor(
        CriticHealthThresholds(
            minimum_baseline_samples=50, minimum_baseline_updates=100,
            minimum_baseline_windows=1, window_samples=50,
            minimum_post_baseline_updates=50, drift_window_updates=50,
            healthy_windows_after_baseline=2, settling_windows=1,
        )
    )
    for step in range(400):
        record = monitor.observe(_healthy_sample(step), updates=step * 10)
    assert record["phase"] == PHASE_HEALTHY
    assert monitor.is_healthy() is True


def test_real_divergence_is_still_caught_after_the_baseline():
    t = CriticHealthThresholds(
        minimum_baseline_samples=50, minimum_baseline_updates=100,
        minimum_baseline_windows=1, window_samples=50, consecutive_violations=3,
        minimum_post_baseline_updates=50, drift_window_updates=50,
        healthy_windows_after_baseline=2, settling_windows=1,
    )
    monitor = CriticHealthMonitor(t)
    for step in range(200):
        monitor.observe(_healthy_sample(step), updates=step * 10)
    assert monitor.is_healthy()

    # Now reproduce the v2 signature at a *different* scale to prove the
    # detector is relative, not a table of the old literal numbers.
    record = None
    for step in range(200):
        record = monitor.observe(
            {
                "mean_q": -3.0 - 4.0 * step,
                "q_p05": -9.0 - 20.0 * step,
                "q_p95": 4.0,
                "td_error_p95": 2.0 + 3.0 * step,
                "td_error_max": 5.0 + 10.0 * step,
                "critic_loss": 4.0 + 12.0 * step,
                "critic_gradient_norm": 0.6,
            },
            updates=2_000 + step * 10,
        )
    assert record["phase"] == PHASE_DIVERGING
    assert record["should_pause"] is True
    assert record["action"] == "checkpoint_freeze_actor_and_rebuild_critics"
    assert set(record["reasons"]) & {
        "q_median_drift", "q_spread_expansion", "td_error_growth",
        "critic_loss_explosion",
    }


def test_one_transient_window_is_only_a_warning():
    t = CriticHealthThresholds(
        minimum_baseline_samples=50, minimum_baseline_updates=100,
        minimum_baseline_windows=1, window_samples=50, consecutive_violations=3,
        minimum_post_baseline_updates=50, drift_window_updates=50,
        healthy_windows_after_baseline=2, settling_windows=1,
    )
    monitor = CriticHealthMonitor(t)
    for step in range(200):
        monitor.observe(_healthy_sample(step), updates=step * 10)
    record = monitor.observe(
        {"mean_q": -900.0, "q_p05": -3000.0, "q_p95": 400.0,
         "td_error_p95": 900.0, "td_error_max": 2000.0,
         "critic_loss": 5000.0, "critic_gradient_norm": 0.6},
        updates=3_000,
    )
    assert record["phase"] == PHASE_WARNING
    assert record["should_pause"] is False, "one bad window must never rebuild"


def test_a_rebuild_restarts_the_lifecycle():
    monitor = CriticHealthMonitor(
        CriticHealthThresholds(
            minimum_baseline_samples=50, minimum_baseline_updates=100,
            minimum_baseline_windows=1, window_samples=50,
            minimum_post_baseline_updates=50, drift_window_updates=50,
            healthy_windows_after_baseline=2, settling_windows=1,
        )
    )
    for step in range(200):
        monitor.observe(_healthy_sample(step), updates=step * 10)
    assert monitor.is_healthy()
    monitor.begin_generation()
    assert monitor.generation == 1
    assert monitor.phase == PHASE_BASELINE
    assert monitor.baseline_established() is False
    assert monitor.is_healthy() is False
    record = monitor.observe(_fresh_critic_learning(0), updates=99_999)
    assert record["should_pause"] is False


def test_critic_health_state_round_trips():
    monitor = CriticHealthMonitor(
        CriticHealthThresholds(
            minimum_baseline_samples=50, minimum_baseline_updates=100,
            minimum_baseline_windows=1, window_samples=50,
            minimum_post_baseline_updates=50, drift_window_updates=50,
            healthy_windows_after_baseline=2, settling_windows=1,
        )
    )
    for step in range(200):
        monitor.observe(_healthy_sample(step), updates=step * 10)
    payload = json.loads(json.dumps(monitor.state_dict()))
    restored = CriticHealthMonitor()
    restored.load_state_dict(payload)
    assert restored.phase == monitor.phase
    assert restored.generation == monitor.generation
    assert restored.baseline_established() is True
    assert restored.is_healthy() is True


# ------------------------------------------- integration with the trainer


def test_trainer_uses_the_gate_and_never_the_absolute_counter():
    from marine_race_arena.learning import train_sac_transition as trainer

    body = Path(trainer.__file__).read_text(encoding="utf-8").split(
        "def run_training", 1
    )[1]
    assert "actor_gate.consider_unfreeze(" in body
    assert "actor_gate.on_critic_rebuild()" in body
    assert "critic_health.begin_generation()" in body
    assert "actor_gate.should_update_actor()" in body
    # The defective arithmetic must be gone for good.
    assert "actor_unfreeze_after" not in body
    assert "int(agent.gradient_updates)\n" not in body.split("actor_gate")[0][-400:]


def test_recovery_asserts_the_actor_hash_is_unchanged():
    from marine_race_arena.learning import train_sac_transition as trainer

    body = Path(trainer.__file__).read_text(encoding="utf-8").split(
        "def _recover_from_critic_divergence", 1
    )[1]
    assert "actor_parameter_sha256(agent)" in body
    assert "actor_parameter_sha256(recovered)" in body
    assert "critic rebuild altered the actor" in body


def test_actor_hash_helper_is_sensitive_and_stable():
    from marine_race_arena.learning.sac_transition_policy import (
        SACTransitionAgent,
        actor_parameter_sha256,
    )

    agent = SACTransitionAgent(hidden_sizes=(16, 16))
    first = actor_parameter_sha256(agent)
    assert first == actor_parameter_sha256(agent), "hash must be deterministic"
    with torch.no_grad():
        for parameter in agent.critic.parameters():
            parameter.add_(1.0)
    assert actor_parameter_sha256(agent) == first, "critics must not affect it"
    with torch.no_grad():
        agent.actor.mean_head.bias.add_(1e-3)
    assert actor_parameter_sha256(agent) != first, "actor changes must show"


def test_checkpoint_persists_gate_and_health_state():
    from marine_race_arena.learning import sac_transition_checkpoint as ck

    source = Path(ck.__file__).read_text(encoding="utf-8")
    assert "actor_gate_state" in source
    assert "critic_health_state" in source
    assert '"actor_gate": dict(actor_gate_state or {})' in source
