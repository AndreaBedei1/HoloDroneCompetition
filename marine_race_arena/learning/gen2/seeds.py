"""Generation-2 seed bands: TRAIN / VALIDATION / TEST / FINAL_HOLDOUT / TRACK_SPECIFIC.

Every Generation-1 seed role lives at or below 33_999 (see
:mod:`marine_race_arena.learning.seed_registry`).  Generation 2 therefore opens
a fresh decade starting at 40_000 so no Gen-2 episode can collide with any
historical experiment, and the four Gen-2 spaces are disjoint by construction.

Rules enforced by ``tests/learning/gen2/test_gen2_seeds.py``:

* ``TRAIN`` produces the procedural experiment's gradient-bearing samples --
  expert demonstrations, DAgger learner rollouts with expert labels, and PPO
  rollouts.
* ``TRACK_SPECIFIC`` produces the track campaign's samples.  It trains on
  fragments of the three evaluation circuits by design and therefore makes no
  unseen-track claim; it is a separate protocol, not a widening of TRAIN.
* ``VALIDATION`` may evaluate learned policies and drive progression decisions.
  It must never supply expert-labelled samples to training.
* ``TEST`` stays untouched until final model selection.
* ``FINAL_HOLDOUT`` stays sealed until exactly one Gen-2 policy is frozen.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple


def _r(lo: int, hi: int) -> List[int]:
    return list(range(lo, hi + 1))


# --------------------------------------------------------------- band extents
GEN2_TRAIN_BAND: Tuple[int, int] = (40_000, 49_999)
GEN2_VALIDATION_BAND: Tuple[int, int] = (50_000, 50_999)
GEN2_TEST_BAND: Tuple[int, int] = (51_000, 51_999)
GEN2_FINAL_HOLDOUT_BAND: Tuple[int, int] = (52_000, 52_999)

#: The track-specific campaign (``track_specific_gen2_v1``) lives in its own
#: band.  It deliberately trains on fragments of the three evaluation circuits,
#: so it makes no unseen-track claim and the TRAIN/VALIDATION distinction that
#: protects the procedural experiment does not apply to it.  Keeping it in a
#: separate decade means it can never be confused with, or collide with, the
#: procedural bookkeeping.
GEN2_TRACK_BAND: Tuple[int, int] = (60_000, 69_999)

GEN2_BANDS: Dict[str, Tuple[int, int]] = {
    "TRAIN": GEN2_TRAIN_BAND,
    "VALIDATION": GEN2_VALIDATION_BAND,
    "TEST": GEN2_TEST_BAND,
    "FINAL_HOLDOUT": GEN2_FINAL_HOLDOUT_BAND,
    "TRACK_SPECIFIC": GEN2_TRACK_BAND,
}

#: Bands whose seeds may produce gradient-bearing samples.
TRAINABLE_BANDS = frozenset({"TRAIN", "TRACK_SPECIFIC"})

# ------------------------------------------------------------- TRAIN sub-roles
# Expert demonstration courses (behaviour-cloning corpus).
GEN2_EXPERT_DEMO_SEEDS: List[int] = _r(40_000, 43_999)
# DAgger learner-rollout courses.  Round r consumes a contiguous slice.
GEN2_DAGGER_SEEDS: List[int] = _r(44_000, 47_999)
# Recurrent PPO fine-tuning rollout streams.
GEN2_PPO_TRAINING_SEEDS: List[int] = _r(48_000, 49_499)
# Throwaway smoke/plumbing checks that must not pollute a real role.
GEN2_SMOKE_SEEDS: List[int] = _r(49_500, 49_999)

# -------------------------------------------------------- VALIDATION sub-roles
# Closed-loop BC/DAgger progression evaluation.  Split explicitly so a BC stage
# and a DAgger round can never be graded on the same course: both are
# validation, but comparing "before" and "after" on overlapping geometry would
# make an improvement partly a memory of the same layout.
GEN2_VALIDATION_SEEDS: List[int] = _r(50_000, 50_599)
#: BC stages A..E: 60 courses each, disjoint per stage.
GEN2_BC_STAGE_SEEDS: List[int] = _r(50_000, 50_299)
#: DAgger rounds 1..5: 60 courses each, disjoint per round.
GEN2_DAGGER_VALIDATION_SEEDS: List[int] = _r(50_300, 50_599)
# Recurrence ablation A/B/C/D on matched seeds.
GEN2_ABLATION_SEEDS: List[int] = _r(50_600, 50_899)
# PPO periodic validation.
GEN2_PPO_VALIDATION_SEEDS: List[int] = _r(50_900, 50_999)

#: Courses per BC stage / per DAgger round.  Fixed so a slice index is stable.
GEN2_VALIDATION_BLOCK = 60


def bc_stage_seeds(stage_index: int, count: int) -> List[int]:
    """Validation courses for BC stage ``stage_index`` (0-based)."""
    return _validation_block(GEN2_BC_STAGE_SEEDS, stage_index, count, "BC stage")


def dagger_validation_seeds(round_index: int, count: int) -> List[int]:
    """Validation courses for DAgger round ``round_index`` (1-based)."""
    return _validation_block(
        GEN2_DAGGER_VALIDATION_SEEDS, int(round_index) - 1, count, "DAgger round"
    )


def _validation_block(pool: List[int], index: int, count: int, label: str) -> List[int]:
    if index < 0:
        raise ValueError(f"{label} index must be non-negative")
    if int(count) > GEN2_VALIDATION_BLOCK:
        raise ValueError(
            f"{label} asked for {count} courses but each block holds "
            f"{GEN2_VALIDATION_BLOCK}; widening a block would overlap the next one"
        )
    start = index * GEN2_VALIDATION_BLOCK
    end = start + int(count)
    if end > len(pool):
        raise ValueError(f"{label} {index} exceeds its allocated validation band")
    return pool[start:end]

# -------------------------------------------------------------- TEST sub-roles
# Untouched until model selection; then used to pick the single frozen policy.
GEN2_MODEL_SELECTION_SEEDS: List[int] = _r(51_000, 51_499)
# The committed readiness-gate benchmark.
GEN2_READINESS_SEEDS: List[int] = _r(51_500, 51_999)

# ------------------------------------------------------ FINAL_HOLDOUT sub-roles
# Circuit-generation seeds for gen2_final_holdout_01..05 (sealed).
GEN2_FINAL_HOLDOUT_CIRCUIT_SEEDS: List[int] = _r(52_000, 52_099)
# Per-trial seeds for the paired frozen-policy vs frozen-rules runs (sealed).
GEN2_FINAL_HOLDOUT_TRIAL_SEEDS: List[int] = _r(52_100, 52_599)
# Retrospective diagnostic runs on the three no-longer-pristine old circuits.
GEN2_RETROSPECTIVE_OLD_CIRCUIT_SEEDS: List[int] = _r(52_600, 52_999)

# --------------------------------------------------- TRACK_SPECIFIC sub-roles
# Expert demonstrations on exact fragments of the three official circuits.
GEN2_TRACK_FRAGMENT_SEEDS: List[int] = _r(60_000, 63_999)
# DAgger rollouts on those same fragments and on failure regions.
GEN2_TRACK_DAGGER_SEEDS: List[int] = _r(64_000, 65_999)
# Closed-loop evaluation of the learner on every fragment.
GEN2_TRACK_EVAL_SEEDS: List[int] = _r(66_000, 66_999)
# Repeated trials on the three complete circuits.
GEN2_FULL_CIRCUIT_SEEDS: List[int] = _r(67_000, 67_999)
# Recurrent PPO speed fine-tuning on the circuits.
GEN2_TRACK_PPO_SEEDS: List[int] = _r(68_000, 69_499)
# Throwaway plumbing checks for the track campaign.
GEN2_TRACK_SMOKE_SEEDS: List[int] = _r(69_500, 69_999)


GEN2_ROLE_SEEDS: Dict[str, List[int]] = {
    "gen2_expert_demonstrations": GEN2_EXPERT_DEMO_SEEDS,
    "gen2_dagger_rollouts": GEN2_DAGGER_SEEDS,
    "gen2_ppo_training": GEN2_PPO_TRAINING_SEEDS,
    "gen2_smoke": GEN2_SMOKE_SEEDS,
    "gen2_validation": GEN2_VALIDATION_SEEDS,
    "gen2_ablation": GEN2_ABLATION_SEEDS,
    "gen2_ppo_validation": GEN2_PPO_VALIDATION_SEEDS,
    "gen2_model_selection": GEN2_MODEL_SELECTION_SEEDS,
    "gen2_readiness": GEN2_READINESS_SEEDS,
    "gen2_final_holdout_circuits": GEN2_FINAL_HOLDOUT_CIRCUIT_SEEDS,
    "gen2_final_holdout_trials": GEN2_FINAL_HOLDOUT_TRIAL_SEEDS,
    "gen2_retrospective_old_circuits": GEN2_RETROSPECTIVE_OLD_CIRCUIT_SEEDS,
    "gen2_track_fragments": GEN2_TRACK_FRAGMENT_SEEDS,
    "gen2_track_dagger": GEN2_TRACK_DAGGER_SEEDS,
    "gen2_track_eval": GEN2_TRACK_EVAL_SEEDS,
    "gen2_full_circuit": GEN2_FULL_CIRCUIT_SEEDS,
    "gen2_track_ppo": GEN2_TRACK_PPO_SEEDS,
    "gen2_track_smoke": GEN2_TRACK_SMOKE_SEEDS,
}

ROLE_TO_BAND: Dict[str, str] = {
    "gen2_expert_demonstrations": "TRAIN",
    "gen2_dagger_rollouts": "TRAIN",
    "gen2_ppo_training": "TRAIN",
    "gen2_smoke": "TRAIN",
    "gen2_validation": "VALIDATION",
    "gen2_ablation": "VALIDATION",
    "gen2_ppo_validation": "VALIDATION",
    "gen2_model_selection": "TEST",
    "gen2_readiness": "TEST",
    "gen2_final_holdout_circuits": "FINAL_HOLDOUT",
    "gen2_final_holdout_trials": "FINAL_HOLDOUT",
    "gen2_retrospective_old_circuits": "FINAL_HOLDOUT",
    "gen2_track_fragments": "TRACK_SPECIFIC",
    "gen2_track_dagger": "TRACK_SPECIFIC",
    "gen2_track_eval": "TRACK_SPECIFIC",
    "gen2_full_circuit": "TRACK_SPECIFIC",
    "gen2_track_ppo": "TRACK_SPECIFIC",
    "gen2_track_smoke": "TRACK_SPECIFIC",
}

# Roles that may legally be paired with a frozen-expert action label.
EXPERT_LABELLING_ALLOWED_ROLES = frozenset({
    "gen2_expert_demonstrations",
    "gen2_dagger_rollouts",
    "gen2_smoke",
    # The track-specific campaign trains on the evaluation circuits by design.
    "gen2_track_fragments",
    "gen2_track_dagger",
    "gen2_track_smoke",
})


def band_of(seed: int) -> str:
    """Return the Gen-2 band owning ``seed``; raise if the seed is out of scope."""
    value = int(seed)
    for name, (lo, hi) in GEN2_BANDS.items():
        if lo <= value <= hi:
            return name
    raise ValueError(
        f"seed {value} is outside every Generation-2 band; Gen-2 seeds live in "
        f"[{GEN2_TRAIN_BAND[0]}, {GEN2_FINAL_HOLDOUT_BAND[1]}]"
    )


def role_of(seed: int) -> str:
    """Return the Gen-2 role owning ``seed``; raise if unallocated."""
    value = int(seed)
    for name, allocated in GEN2_ROLE_SEEDS.items():
        if allocated[0] <= value <= allocated[-1]:
            return name
    raise ValueError(f"seed {value} is not allocated to any Generation-2 role")


def assert_training_seed(seed: int, *, context: str = "training") -> int:
    """Raise unless ``seed`` may be used to produce gradient-bearing samples."""
    band = band_of(seed)
    if band not in TRAINABLE_BANDS:
        raise PermissionError(
            f"{context} attempted to use seed {int(seed)} from the Gen-2 {band} band; "
            f"only {sorted(TRAINABLE_BANDS)} seeds may produce training samples"
        )
    return int(seed)


def assert_expert_labelling_seed(seed: int, *, context: str = "expert labelling") -> int:
    """Raise unless ``seed`` may be paired with a frozen-expert action label.

    VALIDATION exists to measure the learned policy.  Labelling a validation
    state with the expert would leak the validation distribution into training,
    so the ban is enforced here rather than left to convention.
    """
    role = role_of(seed)
    if role not in EXPERT_LABELLING_ALLOWED_ROLES:
        raise PermissionError(
            f"{context} attempted to query the expert on seed {int(seed)} "
            f"(role {role!r}); expert labels are only legal on "
            f"{sorted(EXPERT_LABELLING_ALLOWED_ROLES)}"
        )
    return int(seed)


def assert_not_final_holdout(seed: int, *, context: str = "operation") -> int:
    """Raise if ``seed`` belongs to the sealed final-holdout band."""
    if GEN2_FINAL_HOLDOUT_BAND[0] <= int(seed) <= GEN2_FINAL_HOLDOUT_BAND[1]:
        raise PermissionError(
            f"{context} attempted to touch sealed final-holdout seed {int(seed)}"
        )
    return int(seed)


def dagger_round_seeds(round_index: int, count: int = 500) -> List[int]:
    """Return a disjoint contiguous slice of DAgger seeds for ``round_index``."""
    if round_index < 1:
        raise ValueError("DAgger rounds are 1-indexed")
    start = GEN2_DAGGER_SEEDS[0] + (round_index - 1) * int(count)
    end = start + int(count) - 1
    if end > GEN2_DAGGER_SEEDS[-1]:
        raise ValueError(
            f"DAgger round {round_index} would exceed the allocated band "
            f"{GEN2_DAGGER_SEEDS[0]}..{GEN2_DAGGER_SEEDS[-1]}"
        )
    return _r(start, end)


def all_gen2_seeds() -> Set[int]:
    out: Set[int] = set()
    for allocated in GEN2_ROLE_SEEDS.values():
        out.update(allocated)
    return out


def assert_roles_pairwise_disjoint() -> None:
    names = sorted(GEN2_ROLE_SEEDS)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            clash = set(GEN2_ROLE_SEEDS[a]) & set(GEN2_ROLE_SEEDS[b])
            if clash:
                raise ValueError(f"Gen-2 roles {a} and {b} overlap: {sorted(clash)[:8]}")


def assert_disjoint_from_generation_one() -> None:
    """Raise if any Gen-2 seed collides with a Generation-1 allocation."""
    from marine_race_arena.learning import seed_registry

    gen1: Set[int] = set(seed_registry.all_used_seeds())
    for allocated in seed_registry.NEW_ALLOCATIONS.values():
        gen1.update(allocated)
    for allocated in seed_registry.ROLE_SEED_SETS.values():
        gen1.update(allocated)
    clash = sorted(all_gen2_seeds() & gen1)
    if clash:
        raise ValueError(f"Generation-2 seeds collide with Generation-1: {clash[:8]}")


def registry_dict() -> Dict[str, object]:
    assert_roles_pairwise_disjoint()
    assert_disjoint_from_generation_one()
    return {
        "schema_version": "gen2_seed_registry_v1",
        "bands": {name: list(bounds) for name, bounds in GEN2_BANDS.items()},
        "roles": {
            name: {
                "band": ROLE_TO_BAND[name],
                "first": allocated[0],
                "last": allocated[-1],
                "count": len(allocated),
                "expert_labelling_allowed": name in EXPERT_LABELLING_ALLOWED_ROLES,
            }
            for name, allocated in GEN2_ROLE_SEEDS.items()
        },
        "invariants": {
            "roles_pairwise_disjoint": True,
            "disjoint_from_generation_one": True,
            "expert_labels_only_on_trainable_bands": all(
                ROLE_TO_BAND[name] in TRAINABLE_BANDS
                for name in EXPERT_LABELLING_ALLOWED_ROLES
            ),
        },
    }
