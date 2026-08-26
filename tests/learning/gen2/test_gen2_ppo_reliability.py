"""Recurrent PPO phase 1 must not be able to trade gates for speed."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

from marine_race_arena.learning.gen2 import ppo_reliability as ppo


def test_the_reward_hierarchy_holds():
    """Checked arithmetically, because inspection missed it once.

    The first weight set had a -0.002 per-step time penalty, which looks
    negligible and accumulates to -12 over a full circuit -- outweighing a
    10-point gate and letting speed buy missed gates in the reliability phase.
    """
    report = ppo.assert_reward_hierarchy()
    assert report["all_hold"], report["checks"]


def test_a_slow_completion_beats_a_fast_dnf():
    """The property the whole phase depends on."""
    horizon = ppo.REWARD_HORIZON_STEPS
    slow_full_run = (
        22 * ppo.GATE_REWARD + ppo.COMPLETION_BONUS + ppo.TIME_PENALTY * horizon
    )
    fast_dnf = 5 * ppo.GATE_REWARD + ppo.MISSED_GATE_PENALTY
    assert slow_full_run > fast_dnf
    # And a completion always beats the same run without the finish.
    assert ppo.COMPLETION_BONUS > 0


def test_safety_cannot_be_bought_with_gates():
    assert abs(ppo.OUT_OF_BOUNDS_PENALTY) > 2 * ppo.GATE_REWARD
    assert abs(ppo.WRONG_DIRECTION_PENALTY) >= ppo.GATE_REWARD
    assert abs(ppo.MISSED_GATE_PENALTY) > ppo.GATE_REWARD


def test_time_shaping_cannot_outweigh_one_gate():
    accumulated = abs(ppo.TIME_PENALTY) * ppo.REWARD_HORIZON_STEPS
    assert accumulated < ppo.GATE_REWARD


def test_ppo_starts_from_the_parent_with_exact_actor_parity(tmp_path):
    """No random initialization, no policy reset -- the parent is valuable."""
    from sb3_contrib import RecurrentPPO

    from marine_race_arena.learning.gen2.recurrent_policy import (
        build_gen2_policy_for_training,
    )

    parent = build_gen2_policy_for_training(seed=0)
    path = tmp_path / "parent.zip"
    parent.save(path)

    loaded = RecurrentPPO.load(path, device="cpu")
    report = ppo.verify_actor_parity(path, loaded)
    assert report["parity"], report
    assert report["max_abs_deviation"] == pytest.approx(0.0, abs=1e-9)


def test_parity_check_detects_a_changed_actor(tmp_path):
    """The check must fail when the actor really differs, or it proves nothing."""
    import torch as th
    from sb3_contrib import RecurrentPPO

    from marine_race_arena.learning.gen2.recurrent_policy import (
        build_gen2_policy_for_training,
    )

    parent = build_gen2_policy_for_training(seed=0)
    path = tmp_path / "parent.zip"
    parent.save(path)

    changed = RecurrentPPO.load(path, device="cpu")
    with th.no_grad():
        changed.policy.action_net.bias.add_(0.05)
    assert not ppo.verify_actor_parity(path, changed)["parity"]


def test_the_config_is_conservative():
    from marine_race_arena.learning.gen2.bc_recurrent import BCConfig

    config = ppo.PPOReliabilityConfig()
    assert config.learning_rate < BCConfig().learning_rate
    assert config.clip_range <= 0.10        # SB3 default is 0.2
    assert config.target_kl is not None
    assert config.total_timesteps <= 100_000  # short experiment, not a campaign


def test_training_distribution_includes_whole_circuits():
    """A whole-run failure cannot be expressed by a fragment alone."""
    plan = ppo.training_tracks(ppo.PPOReliabilityConfig())
    assert set(plan["full_circuits"]) == {
        "horseshoe_bay", "vertical_serpent", "mixed_endurance"
    }
    assert 0.0 < plan["full_circuit_fraction"] < 1.0
