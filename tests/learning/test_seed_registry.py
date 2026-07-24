"""Tests for the seed registry (separation of development and final/frozen seeds)."""

import json

from marine_race_arena.learning import seed_registry as sr


def test_new_allocations_are_unused():
    sr.assert_new_allocations_are_unused()  # raises on collision


def test_development_and_final_are_disjoint():
    assert sr.development_and_final_are_disjoint()


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
    assert 1400 in data["new_allocations"]["stage1_kl_calibration"]
