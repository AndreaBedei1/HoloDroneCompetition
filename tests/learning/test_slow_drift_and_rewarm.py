"""Slow cumulative critic drift, and KL-aware PPO learning-rate re-warm.

Two independent problems, two independent mechanisms.

SAC: the repaired detector catches *fast* divergence, but a drift of 0.6 per
1000 updates passes every rate threshold while still moving mean Q by more than
10 across 17,000 updates.  Cumulative displacement is therefore measured
separately, normalised against the baseline's own Q spread.

PPO: the live v3 policy moved 1.3e-5 (relative) per rollout and 3.98e-3 across
645,000 transitions with log_std frozen -- competent, but not learning.
"""

from __future__ import annotations

import json

import pytest

from marine_race_arena.learning.ppo_lr_rewarm import (
    STATE_HEALTHY,
    STATE_STALLED,
    STATE_TOO_HOT,
    LearningRateRewarm,
    RewarmPolicy,
    classify_update,
    rewarm_policy_from_mapping,
)
from marine_race_arena.learning.sac_health_gates import (
    REASON_CUMULATIVE_DIVERGING,
    CriticHealthMonitor,
    CriticHealthThresholds,
)

FAST = dict(
    minimum_baseline_samples=200, minimum_baseline_updates=400,
    minimum_baseline_windows=1, window_samples=200,
    minimum_post_baseline_updates=200, drift_window_updates=200,
    healthy_windows_after_baseline=2, settling_windows=1,
    cumulative_warning_windows=3, cumulative_diverging_windows=6,
)


def settled(step, q=-3.0):
    return {
        "mean_q": q + 0.02 * ((-1) ** step), "q_p05": q - 6.0, "q_p95": q + 4.0,
        "mean_target_q": q - 1.0, "td_error_p50": 1.5, "td_error_p95": 20.0,
        "td_error_max": 40.0, "critic_loss": 100.0, "critic_gradient_norm": 3.0,
    }


def _settle(monitor, n=1200):
    for step in range(n):
        monitor.observe(settled(step), updates=step)
    assert monitor.is_healthy(), monitor.phase
    return monitor


# ------------------------------------------------------- cumulative drift


def test_slow_drift_alone_warns_but_does_not_rebuild():
    """A critic legitimately walking to its fixed point must not be rebuilt."""

    monitor = _settle(CriticHealthMonitor(CriticHealthThresholds(**FAST)))
    record = None
    for step in range(7000):
        # Slow enough to clear the fast-rate check (4.0/1k vs a 5.0 limit) but
        # cumulatively large; TD/loss/gradients stay healthy. This is the
        # measured SAC v6 regime.
        q = -3.0 - 0.004 * step
        record = monitor.observe(settled(step, q=q), updates=1200 + step)
    d = monitor.cumulative_displacement()
    assert d["q_shift_ratio"] > 2.0, d
    assert "q_median_drift" not in record["reasons"], (
        "the FAST detector must not be what fires here")
    assert record["should_pause"] is False, "displacement alone must never rebuild"
    assert record["cumulative_state"] is not None


def test_slow_drift_with_td_and_loss_deterioration_escalates():
    monitor = _settle(CriticHealthMonitor(CriticHealthThresholds(**FAST)))
    record = None
    for step in range(7000):
        q = -3.0 - 0.004 * step
        s = settled(step, q=q)
        s["td_error_p95"] = 20.0 + 0.02 * step      # deteriorating
        s["critic_loss"] = 100.0 + 0.1 * step       # deteriorating
        record = monitor.observe(s, updates=1200 + step)
    assert REASON_CUMULATIVE_DIVERGING in record["reasons"]
    assert record["should_pause"] is True
    assert "td_error_deterioration" in record["reasons"]
    assert "critic_loss_deterioration" in record["reasons"]


def test_slow_drift_with_gradient_growth_escalates():
    monitor = _settle(CriticHealthMonitor(CriticHealthThresholds(**FAST)))
    record = None
    for step in range(7000):
        s = settled(step, q=-3.0 - 0.004 * step)
        s["critic_gradient_norm"] = 3.0 + 0.01 * step
        record = monitor.observe(s, updates=1200 + step)
    assert "critic_gradient_growth" in record["reasons"]
    assert record["should_pause"] is True


def test_a_stable_critic_never_reports_cumulative_drift():
    monitor = _settle(CriticHealthMonitor(CriticHealthThresholds(**FAST)))
    record = None
    for step in range(4000):
        record = monitor.observe(settled(step), updates=1200 + step)
    assert record["cumulative_reasons"] == []
    assert record["cumulative_state"] is None
    assert monitor.healthy_for_actor_activation() is True


def test_displacement_is_normalised_not_absolute():
    """A critic on a large Q scale must not trip merely for being large."""

    wide = CriticHealthMonitor(CriticHealthThresholds(**FAST))
    for step in range(1200):
        wide.observe({
            "mean_q": -500.0 + 2.0 * ((-1) ** step), "q_p05": -900.0, "q_p95": -100.0,
            "mean_target_q": -510.0, "td_error_p50": 5.0, "td_error_p95": 60.0,
            "td_error_max": 120.0, "critic_loss": 400.0, "critic_gradient_norm": 3.0,
        }, updates=step)
    assert wide.is_healthy()
    record = None
    for step in range(2000):
        record = wide.observe({
            "mean_q": -500.0 - 0.002 * step, "q_p05": -900.0, "q_p95": -100.0,
            "mean_target_q": -510.0, "td_error_p50": 5.0, "td_error_p95": 60.0,
            "td_error_max": 120.0, "critic_loss": 400.0, "critic_gradient_norm": 3.0,
        }, updates=1200 + step)
    # A few absolute units of shift against an 800-wide spread is small.
    assert record["cumulative"]["q_shift_ratio"] < 2.0
    assert record["should_pause"] is False


def test_actor_may_not_activate_under_a_cumulative_drift_warning():
    monitor = _settle(CriticHealthMonitor(CriticHealthThresholds(**FAST)))
    for step in range(7000):
        monitor.observe(settled(step, q=-3.0 - 0.004 * step), updates=1200 + step)
    assert monitor.is_healthy() or monitor.cumulative_warning
    assert monitor.healthy_for_actor_activation() is False


def test_the_trainer_gates_activation_on_the_drift_free_signal():
    from pathlib import Path

    from marine_race_arena.learning import train_sac_transition as trainer

    body = Path(trainer.__file__).read_text(encoding="utf-8")
    assert "critic_healthy=critic_health.healthy_for_actor_activation()" in body


def test_cumulative_state_round_trips():
    monitor = _settle(CriticHealthMonitor(CriticHealthThresholds(**FAST)))
    for step in range(4000):
        monitor.observe(settled(step, q=-3.0 - 0.004 * step), updates=1200 + step)
    payload = json.loads(json.dumps(monitor.state_dict()))
    restored = CriticHealthMonitor(CriticHealthThresholds(**FAST))
    restored.load_state_dict(payload)
    assert restored.cumulative_windows == monitor.cumulative_windows
    assert restored.cumulative_warning == monitor.cumulative_warning


# ------------------------------------------------------------ PPO rewarm


STALLED = {"approx_kl": 1e-5, "clip_fraction": 0.001, "relative_policy_update": 1.3e-5}
HEALTHY = {"approx_kl": 0.0025, "clip_fraction": 0.08, "relative_policy_update": 9e-4}
HOT = {"approx_kl": 0.02, "clip_fraction": 0.45, "relative_policy_update": 6e-3}


def test_the_measured_v3_activity_is_classified_stalled():
    """The live numbers: KL ~0, clipping inactive, 1.3e-5 movement per rollout."""

    assert classify_update(STALLED) == STATE_STALLED
    assert classify_update(HEALTHY) == STATE_HEALTHY
    assert classify_update(HOT) == STATE_TOO_HOT


def test_one_stalled_rollout_does_not_move_the_rate():
    r = LearningRateRewarm(learning_rate=3.5e-6)
    before = r.learning_rate
    rec = r.observe(STALLED)
    assert rec["changed"] is False
    assert r.learning_rate == before


def test_sustained_stall_raises_the_rate_within_bounds():
    policy = RewarmPolicy(sustained_rollouts=5)
    r = LearningRateRewarm(policy, learning_rate=3.5e-6)
    for _ in range(5):
        rec = r.observe(STALLED)
    assert rec["changed"] is True
    assert r.learning_rate > 3.5e-6
    for _ in range(400):
        r.observe(STALLED)
    assert r.learning_rate <= policy.maximum_learning_rate
    assert r.learning_rate == pytest.approx(policy.maximum_learning_rate)


def test_a_healthy_optimizer_is_left_alone():
    r = LearningRateRewarm(learning_rate=7e-6)
    for _ in range(200):
        r.observe(HEALTHY)
    assert r.learning_rate == pytest.approx(7e-6)
    assert r.adjustments == []


def test_excessive_kl_lowers_the_rate_immediately():
    """Protecting a competent policy outranks a smooth schedule."""

    r = LearningRateRewarm(learning_rate=1.2e-5)
    rec = r.observe(HOT)
    assert rec["changed"] is True
    assert r.learning_rate < 1.2e-5


def test_the_rate_never_leaves_its_bounds():
    policy = RewarmPolicy(minimum_learning_rate=3e-6, maximum_learning_rate=1.5e-5,
                          sustained_rollouts=1)
    r = LearningRateRewarm(policy, learning_rate=1.4e-5)
    for _ in range(100):
        r.observe(STALLED)
    assert r.learning_rate <= 1.5e-5
    for _ in range(100):
        r.observe(HOT)
    assert r.learning_rate >= 3e-6


def test_a_large_clip_fraction_also_counts_as_too_hot():
    sample = {"approx_kl": 0.0005, "clip_fraction": 0.5, "relative_policy_update": 1e-4}
    assert classify_update(sample) == STATE_TOO_HOT


def test_stall_needs_all_three_signals():
    """Movement alone being small is not enough if KL says work is happening."""

    partial = dict(STALLED, approx_kl=0.003)
    assert classify_update(partial) == STATE_HEALTHY


def test_rewarm_state_round_trips():
    r = LearningRateRewarm(RewarmPolicy(sustained_rollouts=2), learning_rate=5e-6)
    for _ in range(6):
        r.observe(STALLED)
    payload = json.loads(json.dumps(r.state_dict()))
    restored = LearningRateRewarm()
    restored.load_state_dict(payload)
    assert restored.learning_rate == pytest.approx(r.learning_rate)
    assert restored.policy.sustained_rollouts == 2


def test_unknown_rewarm_keys_are_rejected():
    with pytest.raises(ValueError, match="unknown rewarm keys"):
        rewarm_policy_from_mapping({"nope": 1})


def test_the_ppo_loop_measures_activity_and_applies_the_rate():
    """A controller nobody consults is worthless; pin the call site."""

    from pathlib import Path

    from marine_race_arena.learning import train_ppo_transition as trainer

    body = Path(trainer.__file__).read_text(encoding="utf-8").split(
        "def run_training", 1)[1]
    assert "_policy_vector(model)" in body
    assert "_optimizer_activity(model, policy_before)" in body
    assert "rewarm.observe(" in body
    assert "_apply_learning_rate(model, rewarm.learning_rate)" in body


def test_activity_reports_relative_parameter_movement():
    from marine_race_arena.learning.train_ppo_transition import _optimizer_activity

    class _P:
        def __init__(self, v): self._v = v
        def parameters(self): return iter([self._v])

    class _M:
        def __init__(self, v): self.policy = _P(v); self.logger = None

    import torch
    before = torch.ones(10)
    model = _M(torch.ones(10) * 1.01)
    a = _optimizer_activity(model, before)
    assert a["relative_policy_update"] == pytest.approx(0.01, rel=1e-3)
    assert a["approx_kl"] == 0.0


# --------------------------------------- continuation learning-rate integrity


def test_restore_overwrites_the_learning_rate_this_is_the_v4_bug():
    """Root cause: the parent's rate is reimposed on every param group."""

    from marine_race_arena.learning.longrun_checkpoint import (
        restore_model_training_state,
    )

    class _Sched:
        def __init__(self): self.multiplier = 1.0; self.last_value = 7e-6
        def load_state_dict(self, s):
            self.multiplier = float(s.get("multiplier", self.multiplier))
            self.last_value = float(s.get("last_value", self.last_value))

    class _M:
        def __init__(self, lr):
            self.policy = type("P", (), {})()
            self.policy.optimizer = type("O", (), {})()
            self.policy.optimizer.param_groups = [{"lr": lr}, {"lr": lr}]
            self.learning_rate = _Sched()
            self.num_timesteps = 528384
            self._n_updates = 0
            self._current_progress_remaining = 1.0

    m = _M(7e-6)
    restore_model_training_state(m, {
        "num_timesteps": 528384, "n_updates": 100,
        "current_progress_remaining": 0.0,
        "learning_rate_schedule": {"last_value": 3.489754098360656e-06, "multiplier": 1.0},
        "optimizer_learning_rates": [3.489754098360656e-06] * 2,
    })
    assert all(g["lr"] == pytest.approx(3.489754098360656e-06)
               for g in m.policy.optimizer.param_groups)


def test_an_exhausted_inherited_schedule_can_never_reach_the_new_rate():
    """The ratchet is deliberate; a continuation needs its OWN schedule."""

    from marine_race_arena.learning.train_multigate_longrun import (
        AbsoluteLearningRateSchedule,
    )

    inherited = AbsoluteLearningRateSchedule(7e-6, 3e-6, "linear")
    inherited.last_value = 3.489754098360656e-06
    assert inherited(1.0) == pytest.approx(3.489754098360656e-06)

    fresh = AbsoluteLearningRateSchedule(7e-6, 3e-6, "linear")
    assert fresh(1.0) == pytest.approx(7e-6)


def test_the_continuation_installs_a_fresh_schedule_after_the_restore():
    from pathlib import Path

    from marine_race_arena.learning import train_ppo_transition as trainer

    body = Path(trainer.__file__).read_text(encoding="utf-8").split(
        "def _continue_from_parent", 1)[1]
    restore = body.index("restore_model_training_state(model")
    install = body.index("_install_continuation_schedule(model, config)")
    apply_ = body.index("_apply_learning_rate(model, continuation_learning_rate)")
    verify = body.index("assert_learning_rate_applied(")
    assert restore < install < apply_ < verify, (
        "order must be restore -> new schedule -> apply rate -> verify")


def test_the_assertion_rejects_a_mismatched_optimizer_rate():
    from marine_race_arena.learning.train_ppo_transition import (
        assert_learning_rate_applied,
    )

    class _M:
        def __init__(self):
            self.policy = type("P", (), {})()
            self.policy.optimizer = type("O", (), {})()
            self.policy.optimizer.param_groups = [{"lr": 7e-6}, {"lr": 3.49e-6}]
            self.learning_rate = None
            self._current_progress_remaining = 1.0

    with pytest.raises(ValueError, match="learning rate not applied"):
        assert_learning_rate_applied(_M(), expected=7e-6, restored=3.49e-6,
                                     configured=7e-6)


def test_the_assertion_reports_every_param_group_and_marks_verified():
    from marine_race_arena.learning.train_ppo_transition import (
        assert_learning_rate_applied,
    )

    class _M:
        def __init__(self):
            self.policy = type("P", (), {})()
            self.policy.optimizer = type("O", (), {})()
            self.policy.optimizer.param_groups = [{"lr": 7e-6}, {"lr": 7e-6}]
            self.learning_rate = None
            self._current_progress_remaining = 1.0

    report = assert_learning_rate_applied(_M(), expected=7e-6, restored=3.49e-6,
                                          configured=7e-6)
    assert report["lr_rewarm_verified"] is True
    assert report["effective_optimizer_lr"] == pytest.approx(7e-6)
    assert report["restored_parent_lr"] == pytest.approx(3.49e-6)
    assert len(report["optimizer_param_group_lrs"]) == 2


def test_every_rollout_is_logged_not_only_rate_changes():
    from pathlib import Path

    from marine_race_arena.learning import train_ppo_transition as trainer

    body = Path(trainer.__file__).read_text(encoding="utf-8").split(
        "def run_training", 1)[1]
    log_at = body.index('atomic_append_jsonl(run_dir / "logs" / "lr_rewarm.jsonl"')
    change_at = body.index('if rewarm_record["changed"]:')
    assert log_at < change_at, (
        "the activity record must be written before/independently of the "
        "changed branch, so a controller that never fires is still observable")
