"""The recurrence ablation must isolate memory, not capacity."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
pytest.importorskip("sb3_contrib")

from marine_race_arena.learning.gen2 import ablation as ab
from marine_race_arena.learning.gen2 import seeds as gen2_seeds


def _actor_parameters(policy) -> int:
    total = sum(p.numel() for p in policy.features_extractor.parameters())
    total += sum(p.numel() for p in policy.mlp_extractor.policy_net.parameters())
    total += sum(p.numel() for p in policy.action_net.parameters())
    return int(total)


def test_arm_b_matches_arm_c_actor_capacity():
    """Otherwise a C-over-B win could be capacity rather than memory."""
    target = ab.recurrent_actor_parameter_count()
    model = ab.build_feedforward_policy(seed=0)
    observed = _actor_parameters(model.policy)
    assert abs(observed - target) / target < 0.05, (
        f"arm B has {observed} actor parameters against arm C's {target}"
    )


def test_arm_b_really_has_no_recurrent_state():
    model = ab.build_feedforward_policy(seed=0)
    assert not hasattr(model.policy, "lstm_actor")
    controller = ab.FeedforwardController(model)
    probe = np.random.default_rng(0).random(35).astype(np.float32)
    fresh = controller.act(probe, first_step=True)
    for _ in range(5):
        controller.act(np.random.default_rng(1).random(35).astype(np.float32))
    # A memoryless policy must give the same action for the same observation
    # regardless of what it saw in between.
    assert np.allclose(fresh, controller.act(probe))


def test_both_arms_share_the_same_encoder_shape():
    from marine_race_arena.learning.gen2.recurrent_policy import (
        GEN2_ARCH,
        build_gen2_policy_for_training,
    )

    recurrent = build_gen2_policy_for_training(seed=0).policy.features_extractor
    feedforward = ab.build_feedforward_policy(seed=0).policy.features_extractor
    assert type(recurrent) is type(feedforward)
    assert recurrent.features_dim == feedforward.features_dim == GEN2_ARCH.encoder_hidden[-1]


def test_ablation_runs_on_matched_validation_seeds():
    seeds = ab.ablation_seeds(80)
    assert len(seeds) == 80
    assert all(gen2_seeds.band_of(s) == "VALIDATION" for s in seeds)
    # Matched means every arm sees the identical list.
    assert ab.ablation_seeds(80) == seeds
    # And the ablation band must not touch the progression bands.
    progression = set(gen2_seeds.GEN2_BC_STAGE_SEEDS) | set(
        gen2_seeds.GEN2_DAGGER_VALIDATION_SEEDS
    )
    assert progression.isdisjoint(seeds)


def test_comparison_attributes_each_effect():
    arms = [
        ab.ArmResult("A", "gen1", None, "feedforward",
                     {"gate1_to_gate2_transition_rate": 0.73, "overall_completion_rate": 0.20}),
        ab.ArmResult("B", "ff_bc", None, "feedforward",
                     {"gate1_to_gate2_transition_rate": 0.80, "overall_completion_rate": 0.40}),
        ab.ArmResult("C", "rec_bc", None, "recurrent_lstm",
                     {"gate1_to_gate2_transition_rate": 0.88, "overall_completion_rate": 0.55}),
        ab.ArmResult("D", "rec_dagger", None, "recurrent_lstm",
                     {"gate1_to_gate2_transition_rate": 0.95, "overall_completion_rate": 0.75}),
    ]
    report = ab.compare_arms(arms)
    assert report["expert_bootstrapping_effect_B_minus_A"] == pytest.approx(0.07)
    assert report["recurrence_effect_C_minus_B"] == pytest.approx(0.08)
    assert report["dagger_effect_D_minus_C"] == pytest.approx(0.07)


def test_a_missing_arm_is_reported_not_silently_dropped():
    report = ab.compare_arms([
        ab.ArmResult("C", "rec_bc", None, "recurrent_lstm",
                     {"gate1_to_gate2_transition_rate": 0.88}),
    ])
    assert report["arms"]["A"] is None
    assert report["recurrence_effect_C_minus_B"] is None
