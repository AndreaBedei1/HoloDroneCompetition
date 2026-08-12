"""Regression for the deterministic critic-warmup livelock.

SAC v5 rebuilt its critics seven times, every single time at exactly
``gradient_updates = 4004`` and exactly 64,050 environment transitions apart,
and never once reached the 10,000 updates the actor needed.  471,108
transitions produced zero actor updates.

Cause: the baseline froze the instant ``updates == minimum_baseline_updates``,
and the drift check then divided by ``current_updates - baseline_updates``,
which is 1, 2, 3 on the very next windows.  Extrapolating that to a per-1000
update rate turned ordinary early critic movement into an enormous apparent
drift, so three consecutive windows tripped the threshold immediately.
"""

from __future__ import annotations

import json

import pytest

from marine_race_arena.learning.sac_actor_gate import ActorFreezeGate, ActorGatePolicy
from marine_race_arena.learning.sac_health_gates import (
    PHASE_BASELINE,
    PHASE_BASELINE_NOT_SETTLING,
    PHASE_DIVERGING,
    PHASE_HEALTHY,
    PHASE_POST_BASELINE,
    PHASE_WARNING,
    CriticHealthMonitor,
    CriticHealthThresholds,
)

#: The live SAC v5 configuration, verbatim.
V5_THRESHOLDS = dict(
    max_q_drift_rate_per_1k_updates=5.0,
    max_q_spread_growth_ratio=4.0,
    max_td_p95_growth_ratio=4.0,
    max_critic_loss_growth_ratio=8.0,
    max_clipped_gradient_fraction=0.8,
    minimum_baseline_samples=2000,
    minimum_baseline_updates=4000,
    minimum_baseline_windows=2,
    window_samples=1000,
    consecutive_violations=3,
    recovery_windows=2,
)


def realistic_critic(step: int) -> dict:
    """A perfectly ordinary warming critic, matching the observed v5 curves.

    mean_q walks -0.7 -> about -5 and q_p05 -> about -28 over ~4000 updates.
    Nothing here is divergence; this is a critic learning a negative-reward
    task from a random initialisation.
    """

    frac = step / 4000.0
    return {
        "mean_q": -0.7 - 4.2 * frac,
        "q_p05": -1.9 - 26.0 * frac,
        "q_p95": 0.4 + 0.6 * frac,
        "mean_target_q": -1.0 - 7.0 * frac,
        "td_error_p50": 1.4 + 0.3 * frac,
        "td_error_p95": 52.0 + 30.0 * frac,
        "td_error_max": 100.0 + 48.0 * frac,
        "critic_loss": 190.0 + 15.0 * frac,
        "critic_gradient_norm": 4.0 + 6.0 * ((-1) ** step) * 0.1,
    }


def run_generation(monitor, start=0, count=6000):
    """Feed one critic generation; return the update at which it was rebuilt."""

    for step in range(count):
        record = monitor.observe(realistic_critic(step), updates=start + step)
        if record["should_pause"]:
            return start + step, record
    return None, record


# ------------------------------------------------------- the old failure


def test_the_v5_configuration_no_longer_rebuilds_at_4004():
    """The exact live config must survive far past the old boundary."""

    monitor = CriticHealthMonitor(
        CriticHealthThresholds(**V5_THRESHOLDS), gradient_clip=5.0
    )
    rebuilt_at, record = run_generation(monitor, count=6000)
    assert rebuilt_at is None, (
        f"critic rebuilt at update {rebuilt_at} with reasons "
        f"{record['reasons']}; the livelock is still present"
    )
    assert monitor.updates_in_generation >= 5000


def test_seven_consecutive_generations_all_survive():
    """The observed pattern was seven identical rebuilds; reproduce the loop."""

    rebuild_updates = []
    gate = ActorFreezeGate(ActorGatePolicy(actor_unfreeze_min_updates=10_000))
    monitor = CriticHealthMonitor(
        CriticHealthThresholds(**V5_THRESHOLDS), gradient_clip=5.0
    )
    for _ in range(7):
        rebuilt_at, _ = run_generation(monitor, count=6000)
        if rebuilt_at is not None:
            rebuild_updates.append(rebuilt_at)
            monitor.begin_generation()
            gate.on_critic_rebuild()
        else:
            break
    assert rebuild_updates == [], (
        f"generations still rebuilding at {rebuild_updates} "
        "(v5 produced exactly [4004] x 7)"
    )


def test_no_drift_verdict_in_the_window_right_after_the_baseline_freezes():
    """The precise arithmetic bug: span of 1-3 updates must never be used."""

    thresholds = CriticHealthThresholds(**V5_THRESHOLDS)
    monitor = CriticHealthMonitor(thresholds, gradient_clip=5.0)
    for step in range(4200):
        record = monitor.observe(realistic_critic(step), updates=step)
        if monitor.baseline_established():
            span = monitor.post_baseline_updates
            if span < thresholds.minimum_post_baseline_updates:
                assert record["reasons"] == [], (
                    f"verdict issued only {span} updates after the baseline: "
                    f"{record['reasons']}"
                )
                assert record["phase"] == PHASE_POST_BASELINE


def test_the_actor_can_actually_reach_its_unfreeze_threshold():
    """End to end: the whole point is that the actor eventually wakes."""

    gate = ActorFreezeGate(ActorGatePolicy(actor_unfreeze_min_updates=5_000))
    monitor = CriticHealthMonitor(
        CriticHealthThresholds(**V5_THRESHOLDS), gradient_clip=5.0
    )
    for step in range(9000):
        record = monitor.observe(realistic_critic(step), updates=step)
        if record["should_pause"]:
            monitor.begin_generation()
            gate.on_critic_rebuild()
            continue
        gate.record_critic_update()
        gate.consider_unfreeze(critic_healthy=monitor.is_healthy())
    assert gate.actor_frozen is False, (
        f"actor still FROZEN after 9000 updates: {gate.last_reason}"
    )
    assert gate.critic_rebuild_generation == 0, "no rebuild should have happened"


# --------------------------------------------------- post-baseline phase


def test_phases_progress_baseline_then_observation_then_healthy():
    monitor = CriticHealthMonitor(
        CriticHealthThresholds(**V5_THRESHOLDS), gradient_clip=5.0
    )
    seen = []
    for step in range(6000):
        record = monitor.observe(realistic_critic(step), updates=step)
        if not seen or seen[-1] != record["phase"]:
            seen.append(record["phase"])
    assert seen[0] == PHASE_BASELINE
    assert PHASE_POST_BASELINE in seen
    assert seen[-1] in (PHASE_HEALTHY, PHASE_POST_BASELINE)
    assert PHASE_DIVERGING not in seen


def test_drift_uses_a_full_window_not_a_three_sample_extrapolation():
    """effective_span must never fall below the configured drift window."""

    thresholds = CriticHealthThresholds(**V5_THRESHOLDS)
    monitor = CriticHealthMonitor(thresholds, gradient_clip=5.0)
    for step in range(6000):
        monitor.observe(realistic_critic(step), updates=step)
    span = monitor.effective_drift_span()
    assert span >= thresholds.drift_window_updates


# ------------------------------------------------ real divergence still caught


def diverging_critic(step: int) -> dict:
    """A genuinely runaway critic at a different scale from SAC v2."""

    return {
        "mean_q": -5.0 - 3.0 * step,
        "q_p05": -20.0 - 18.0 * step,
        "q_p95": 4.0,
        "mean_target_q": -8.0 - 5.0 * step,
        "td_error_p50": 2.0 + 2.0 * step,
        "td_error_p95": 50.0 + 40.0 * step,
        "td_error_max": 120.0 + 90.0 * step,
        "critic_loss": 190.0 + 60.0 * step,
        "critic_gradient_norm": 5.0,
    }


def test_genuine_sustained_divergence_is_still_detected():
    monitor = CriticHealthMonitor(
        CriticHealthThresholds(**V5_THRESHOLDS), gradient_clip=5.0
    )
    for step in range(6000):
        monitor.observe(realistic_critic(step), updates=step)
    assert monitor.is_healthy()
    record = None
    for step in range(4000):
        record = monitor.observe(diverging_critic(step), updates=6000 + step)
        if record["should_pause"]:
            break
    assert record["should_pause"] is True
    assert record["phase"] == PHASE_DIVERGING
    assert record["action"] == "checkpoint_freeze_actor_and_rebuild_critics"


def test_one_bad_window_is_only_a_warning():
    monitor = CriticHealthMonitor(
        CriticHealthThresholds(**V5_THRESHOLDS), gradient_clip=5.0
    )
    for step in range(6000):
        monitor.observe(realistic_critic(step), updates=step)
    record = monitor.observe(
        {"mean_q": -900.0, "q_p05": -3000.0, "q_p95": 100.0,
         "mean_target_q": -900.0, "td_error_p50": 500.0, "td_error_p95": 900.0,
         "td_error_max": 2000.0, "critic_loss": 9000.0,
         "critic_gradient_norm": 5.0},
        updates=6001,
    )
    assert record["should_pause"] is False
    assert record["phase"] in (PHASE_WARNING, PHASE_HEALTHY, PHASE_POST_BASELINE)


# --------------------------------------------------------- clip fraction


def test_clip_fraction_is_measured_over_a_window_not_one_update():
    monitor = CriticHealthMonitor(
        CriticHealthThresholds(**V5_THRESHOLDS), gradient_clip=5.0
    )
    for step in range(6000):
        monitor.observe(realistic_critic(step), updates=step)
    before = monitor.clip_hit_fraction()
    assert before < 0.5
    # A handful of clipped updates must not move the verdict.
    for step in range(5):
        sample = realistic_critic(step)
        sample["critic_gradient_norm"] = 5.0
        record = monitor.observe(sample, updates=6000 + step)
    assert "critic_gradients_pinned_at_clip" not in record["reasons"]
    assert monitor.clip_hit_fraction() < 0.5


def test_sustained_clipping_is_reported_with_its_fraction():
    monitor = CriticHealthMonitor(
        CriticHealthThresholds(**V5_THRESHOLDS), gradient_clip=5.0
    )
    for step in range(6000):
        monitor.observe(realistic_critic(step), updates=step)
    record = None
    for step in range(4000):
        sample = realistic_critic(step)
        sample["critic_gradient_norm"] = 5.0
        record = monitor.observe(sample, updates=6000 + step)
    assert monitor.clip_hit_fraction() > 0.8
    assert record["clip_hit_fraction"] > 0.8


# ------------------------------------------------------ baseline settling


def never_settling_critic(step: int) -> dict:
    """Mean Q marches forever while the spread stays bounded.

    The spread is the natural scale of the Q distribution, so a bounded spread
    with an unbounded mean is genuinely non-settling at every magnitude.
    """

    mean_q = -1.0 - 0.5 * step
    return {
        "mean_q": mean_q,
        "q_p05": mean_q - 3.0,
        "q_p95": mean_q + 3.0,
        "mean_target_q": mean_q - 1.0,
        "td_error_p50": 2.0, "td_error_p95": 60.0, "td_error_max": 120.0,
        "critic_loss": 200.0,
        "critic_gradient_norm": 1.0,
    }


def test_a_critic_that_never_settles_is_labelled_not_rebuilt_forever():
    thresholds = CriticHealthThresholds(
        **{**V5_THRESHOLDS, "maximum_baseline_updates": 3000}
    )
    monitor = CriticHealthMonitor(thresholds, gradient_clip=5.0)
    record = None
    for step in range(6000):
        record = monitor.observe(never_settling_critic(step), updates=step)
    assert record["phase"] == PHASE_BASELINE_NOT_SETTLING
    assert record["should_pause"] is False, (
        "a non-settling critic must be surfaced as a diagnostic, "
        "not silently rebuilt in a loop"
    )


def test_baseline_waits_for_local_settling_not_just_a_timestamp():
    """Reaching minimum_baseline_updates alone must not freeze a baseline."""

    monitor = CriticHealthMonitor(
        CriticHealthThresholds(**V5_THRESHOLDS), gradient_clip=5.0
    )
    for step in range(4001):
        monitor.observe(never_settling_critic(step), updates=step)
    assert not monitor.baseline_established(), (
        "baseline froze on a still-moving critic purely because the update "
        "counter crossed the minimum"
    )


# -------------------------------------------------------- livelock guard


def test_repeated_rebuilds_before_activation_raise_a_livelock_fault():
    from marine_race_arena.learning.sac_actor_gate import CriticWarmupLivelock

    gate = ActorFreezeGate(
        ActorGatePolicy(
            actor_unfreeze_min_updates=10_000,
            max_critic_rebuilds_before_actor_activation=3,
        )
    )
    for _ in range(3):
        gate.on_critic_rebuild()
    assert gate.livelock_detected() is True
    with pytest.raises(CriticWarmupLivelock, match="CRITIC_WARMUP_LIVELOCK"):
        gate.assert_not_livelocked()


def test_rebuilds_after_a_successful_activation_do_not_count_as_livelock():
    gate = ActorFreezeGate(
        ActorGatePolicy(
            actor_unfreeze_min_updates=10,
            max_critic_rebuilds_before_actor_activation=2,
        )
    )
    for _ in range(20):
        gate.record_critic_update()
        gate.consider_unfreeze(critic_healthy=True)
    assert gate.actor_frozen is False
    for _ in range(5):
        gate.on_critic_rebuild()
    assert gate.livelock_detected() is False, (
        "the guard exists to catch never-activating warmup, not normal "
        "recovery after the actor has already trained"
    )


def test_livelock_state_round_trips():
    gate = ActorFreezeGate(
        ActorGatePolicy(max_critic_rebuilds_before_actor_activation=3)
    )
    gate.on_critic_rebuild()
    gate.on_critic_rebuild()
    payload = json.loads(json.dumps(gate.state_dict()))
    restored = ActorFreezeGate(ActorGatePolicy())
    restored.load_state_dict(payload)
    assert restored.rebuilds_before_activation == 2
    assert restored.livelock_detected() is False
    restored.on_critic_rebuild()
    assert restored.livelock_detected() is True


def test_critic_health_state_round_trips_with_the_new_fields():
    monitor = CriticHealthMonitor(
        CriticHealthThresholds(**V5_THRESHOLDS), gradient_clip=5.0
    )
    for step in range(6000):
        monitor.observe(realistic_critic(step), updates=step)
    payload = json.loads(json.dumps(monitor.state_dict()))
    restored = CriticHealthMonitor()
    restored.load_state_dict(payload)
    assert restored.phase == monitor.phase
    assert restored.post_baseline_updates == monitor.post_baseline_updates
    assert restored.baseline_established() is True
