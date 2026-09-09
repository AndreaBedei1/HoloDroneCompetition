"""Learned multi-gate curriculum and seed separation tests."""

from pathlib import Path

from marine_race_arena.learning import multigate_curriculum as curriculum
from marine_race_arena.learning import seed_registry as seeds


def test_multigate_curriculum_order_tracks_and_thresholds():
    assert [stage.key for stage in curriculum.STAGES] == [
        "R0",
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
    ]
    for stage in curriculum.STAGES:
        assert stage.tracks
        assert all(Path(track).exists() for track in stage.tracks)
        assert 0.0 < stage.min_completion_rate <= 1.0
    assert curriculum.stage("r1").tracks == (curriculum.TWO_GATE_STRAIGHT,)
    assert curriculum.stage("R2").tracks == (
        curriculum.TWO_GATE_LEFT,
        curriculum.TWO_GATE_RIGHT,
    )
    assert curriculum.stage("R5").official


def test_progression_requires_every_track_to_pass():
    assert curriculum.meets_progression_criterion("R2", [0.8, 0.9])
    assert not curriculum.meets_progression_criterion("R2", [1.0, 0.7])


def test_v3_seed_roles_are_pairwise_disjoint_and_final_held_out():
    seeds.assert_pairwise_disjoint()
    development = (
        set(seeds.MULTIGATE_V3_DEMONSTRATION_SEEDS)
        | set(seeds.MULTIGATE_V3_PPO_TRAINING_SEEDS)
        | set(seeds.MULTIGATE_V3_DEV_EVAL_SEEDS)
        | set(seeds.MULTIGATE_V3_CHECKPOINT_SELECTION_SEEDS)
    )
    final = (
        set(seeds.MULTIGATE_V3_FINAL_TWO_GATE_SEEDS)
        | set(seeds.MULTIGATE_V3_FINAL_THREE_GATE_SEEDS)
        | set(seeds.MULTIGATE_V3_FINAL_OFFICIAL_SEEDS)
    )
    forbidden = set().union(*seeds.DO_NOT_TRAIN_ON.values())
    assert development.isdisjoint(final)
    assert final <= forbidden
    assert development.isdisjoint(forbidden)
