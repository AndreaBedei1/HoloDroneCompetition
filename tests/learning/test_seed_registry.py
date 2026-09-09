"""Tests for the seed registry (separation of development and final/frozen seeds)."""

import json

from marine_race_arena.learning import seed_registry as sr


def test_new_allocations_are_unused():
    sr.assert_new_allocations_are_unused()  # raises on collision


def test_development_and_final_are_disjoint():
    assert sr.development_and_final_are_disjoint()


def test_roles_are_pairwise_disjoint():
    sr.assert_pairwise_disjoint()  # raises on any overlap between mutually-exclusive roles


def test_consumed_dev_allocations_are_in_used():
    used = sr.all_used_seeds()
    for s in (1400, 1404, 1410, 1419, 15002, 15113):  # already executed -> must be consumed
        assert s in used


def test_reserved_multigate_is_held_out():
    forbidden = set().union(*sr.DO_NOT_TRAIN_ON.values())
    for s in (1800, 1899):
        assert s in forbidden
    # v2 forward dev ranges are not held out
    assert set(sr.VISUAL_POSE_DATASET_V2_SEEDS).isdisjoint(forbidden)
    assert set(sr.BC_V2_DEV_EVAL_SEEDS).isdisjoint(forbidden)


def test_frozen_and_reserved_are_in_do_not_train_on():
    forbidden = set().union(*sr.DO_NOT_TRAIN_ON.values())
    for s in (1000, 1049, 1100, 1149):  # frozen A/B
        assert s in forbidden
    for s in (1500, 1549, 1550, 1599):  # reserved final
        assert s in forbidden


def test_calibration_and_dev_seeds_not_frozen():
    forbidden = set().union(*sr.DO_NOT_TRAIN_ON.values())
    assert set(sr.STAGE1_KL_CALIBRATION_SEEDS).isdisjoint(forbidden)
    assert set(sr.STAGE2_PPO_DEV_SEEDS).isdisjoint(forbidden)


def test_registry_written(tmp_path):
    p = tmp_path / "seed_registry.json"
    sr.write_registry(p)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["invariants"]["development_and_final_disjoint"] is True
    assert data["invariants"]["roles_pairwise_disjoint"] is True
    assert 1400 in data["used_seeds"]["stage1_kl_calibration"]  # consumed
    assert 1600 in data["new_allocations"]["visual_pose_dataset_v2"]  # v2 forward allocation
