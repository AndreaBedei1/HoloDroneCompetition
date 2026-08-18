"""Contract, seed-separation and holdout-seal invariants for Generation 2."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from marine_race_arena.learning.config import ACTION_DIM
from marine_race_arena.learning.config_local_transition import (
    FEATURE_NAMES_LOCAL_TRANSITION,
    OBS_DIM_LOCAL_TRANSITION,
    OBS_ENCODING_VERSION_LOCAL_TRANSITION,
)
from marine_race_arena.learning.gen2 import (
    GEN2_ACTION_CONTRACT,
    GEN2_EXPERT_ID,
    GEN2_OBS_CONTRACT,
)
from marine_race_arena.learning.gen2 import course_family as cf
from marine_race_arena.learning.gen2 import seeds as gen2_seeds

REPO_ROOT = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------- contracts

def test_gen2_uses_the_unchanged_35_feature_contract():
    assert GEN2_OBS_CONTRACT == OBS_ENCODING_VERSION_LOCAL_TRANSITION
    assert OBS_DIM_LOCAL_TRANSITION == 35
    assert len(FEATURE_NAMES_LOCAL_TRANSITION) == 35


def test_gen2_uses_the_unchanged_four_axis_action_contract():
    assert GEN2_ACTION_CONTRACT == "surge_sway_heave_yaw_pm1_v1"
    assert ACTION_DIM == 4


def test_expert_identity_is_the_frozen_rule_controller():
    from marine_race_arena.participants.controller_loader import ControllerLoader

    assert GEN2_EXPERT_ID == "rule_gate_center_then_commit"
    assert GEN2_EXPERT_ID in ControllerLoader.BUILT_INS


# ------------------------------------------------------------- seed bands

def test_gen2_roles_are_pairwise_disjoint():
    gen2_seeds.assert_roles_pairwise_disjoint()


def test_gen2_seeds_never_collide_with_generation_one():
    gen2_seeds.assert_disjoint_from_generation_one()


def test_the_four_bands_do_not_overlap():
    spans = list(gen2_seeds.GEN2_BANDS.values())
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            a, b = spans[i], spans[j]
            assert a[1] < b[0] or b[1] < a[0], f"bands {a} and {b} overlap"


@pytest.mark.parametrize("seed,expected", [
    (40_000, "TRAIN"), (49_999, "TRAIN"),
    (50_000, "VALIDATION"), (50_999, "VALIDATION"),
    (51_000, "TEST"), (51_999, "TEST"),
    (52_000, "FINAL_HOLDOUT"), (52_999, "FINAL_HOLDOUT"),
])
def test_band_boundaries(seed, expected):
    assert gen2_seeds.band_of(seed) == expected


def test_expert_labels_are_refused_outside_train():
    for seed in (50_000, 50_600, 51_000, 51_500, 52_000, 52_100):
        with pytest.raises(PermissionError):
            gen2_seeds.assert_expert_labelling_seed(seed)


def test_expert_labels_are_allowed_on_train_roles():
    for seed in (40_000, 44_000, 49_500):
        assert gen2_seeds.assert_expert_labelling_seed(seed) == seed


def test_validation_seeds_may_evaluate_but_never_train():
    for seed in gen2_seeds.GEN2_VALIDATION_SEEDS[:5]:
        assert gen2_seeds.band_of(seed) == "VALIDATION"
        with pytest.raises(PermissionError):
            gen2_seeds.assert_training_seed(seed)


def test_dagger_rounds_consume_disjoint_seed_slices():
    seen: set = set()
    for index in range(1, 6):
        block = set(gen2_seeds.dagger_round_seeds(index))
        assert not (block & seen), f"DAgger round {index} reuses earlier seeds"
        seen |= block
        assert all(gen2_seeds.band_of(s) == "TRAIN" for s in block)


# --------------------------------------------------------- course family

def test_courses_are_a_pure_function_of_their_seed():
    for seed in (40_000, 44_321, 50_500):
        assert cf.sample_course(seed) == cf.sample_course(seed)


def test_courses_never_place_non_adjacent_gates_on_top_of_each_other():
    """A self-intersecting course is ambiguous, not hard.

    A monotone 45-degree turn at ~4 m spacing closes a circle of radius ~5 m,
    so without the relaxation guard a long turning course puts a later gate
    beside an earlier one and no onboard front end can disambiguate them.
    """
    specs = [cf.sample_course(seed) for seed in gen2_seeds.GEN2_EXPERT_DEMO_SEEDS[:400]]
    for spec in specs:
        if spec.gate_count < 3:
            continue
        assert spec.min_nonadjacent_separation_m >= cf.MIN_NONADJACENT_GATE_SEPARATION_M - 1e-6, (
            f"seed {spec.seed} ({spec.pattern}, {spec.gate_count} gates) self-intersects"
        )


def test_the_family_is_actually_diverse():
    specs = [cf.sample_course(seed) for seed in gen2_seeds.GEN2_EXPERT_DEMO_SEEDS[:600]]
    report = cf.family_diversity_report(specs)
    assert report["unique_geometry_sha256"] == len(specs)
    assert set(report["patterns"]) == set(cf.GEN2_PATTERNS)
    assert set(report["gate_counts"]) == {str(v) for v in cf.GEN2_GATE_COUNTS}
    # Both turn directions, both vertical directions, and reversals present.
    assert report["left_turning_courses"] > 20 and report["right_turning_courses"] > 20
    assert report["climbing_courses"] > 20 and report["descending_courses"] > 20
    assert report["courses_with_turn_reversals"] > 20
    assert report["courses_with_vertical_reversals"] > 20


def test_materialized_track_carries_no_privileged_policy_input(tmp_path):
    spec = cf.sample_course(49_500, gate_count=3)
    path = cf.materialize_course(spec, tmp_path / "course.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    # Course metadata is allowed in the track file (it is diagnostics), but it
    # must live under its own key and never be reachable as an observation.
    assert "gen2_course" in data
    assert data["currents"] == []
    assert data["gates"][0]["id"] == "G01"
    assert len(data["gates"]) == spec.gate_count


# ------------------------------------------------------------ holdout seal

def test_the_sealed_holdout_verifies():
    from marine_race_arena.learning.gen2 import holdout_seal as seal

    report = seal.verify_seal(REPO_ROOT)
    assert report["manifest_hash_matches"], report
    assert report["intact"], report
    assert len(report["circuits"]) == len(seal.HOLDOUT_PLAN)


def test_sealed_circuits_cover_the_requested_length_spread():
    from marine_race_arena.learning.gen2 import holdout_seal as seal

    manifest = seal.read_manifest(REPO_ROOT)
    lengths = sorted(entry["gate_count"] for entry in manifest["circuits"])
    assert lengths == [12, 15, 17, 20, 22]
    assert all(
        gen2_seeds.band_of(entry["seed"]) == "FINAL_HOLDOUT"
        for entry in manifest["circuits"]
    )


def test_training_cannot_load_a_sealed_circuit():
    from marine_race_arena.learning.gen2 import holdout_seal as seal

    for name in seal.HOLDOUT_PLAN:
        path = REPO_ROOT / seal.SEALED_DIR / f"{name}.json"
        assert seal.is_sealed_path(path, root=REPO_ROOT)
        with pytest.raises(seal.SealedHoldoutAccessError):
            seal.assert_course_accessible(path, context="training", root=REPO_ROOT)


def test_a_forged_unseal_record_is_rejected():
    from marine_race_arena.learning.gen2 import holdout_seal as seal

    path = REPO_ROOT / seal.SEALED_DIR / "gen2_final_holdout_01.json"
    with pytest.raises(seal.SealedHoldoutAccessError):
        seal.assert_course_accessible(
            path, context="training", root=REPO_ROOT,
            unseal_record={"record_sha256": "0" * 64, "reason": "let me in"},
        )


def test_unsealed_courses_are_untouched_by_the_guard(tmp_path):
    from marine_race_arena.learning.gen2 import holdout_seal as seal

    spec = cf.sample_course(40_000, gate_count=3)
    path = cf.materialize_course(spec, tmp_path / "train_course.json")
    assert seal.assert_course_accessible(path, context="training", root=REPO_ROOT) == path


def test_every_gen2_rollout_path_is_guarded():
    """The guard must sit in the drivers, not in an optional helper."""
    for module in ("expert_rollout", "evaluation"):
        source = (
            REPO_ROOT / "marine_race_arena" / "learning" / "gen2" / f"{module}.py"
        ).read_text(encoding="utf-8")
        assert "assert_course_accessible(" in source, module
        # Regression guard for the Gen-1 incident where the access check was
        # left inside an `if False:` block and a probe reached a sealed circuit.
        assert "if False" not in source, module
