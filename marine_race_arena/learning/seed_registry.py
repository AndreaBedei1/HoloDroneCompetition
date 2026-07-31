"""Canonical seed registry for the learning pipeline.

Every seed range used for demonstrations, BC development/evaluation, PPO training,
checkpoint selection and CLI smokes is recorded here so new experiments never reuse a
seed and the final held-out ranges stay untouched. Development and final ranges are
disjoint by construction; a test asserts it.

The PPO *training-env* seed is a separate namespace (it seeds the rollout env, not an
evaluation), but we still pick a value disjoint from every evaluation/test range to
avoid confusion.
"""

from __future__ import annotations

from typing import Dict, List, Set


def _r(lo: int, hi: int) -> List[int]:
    return list(range(lo, hi + 1))


# --- Seeds already consumed (never reuse) -------------------------------------
USED_SEEDS: Dict[str, List[int]] = {
    "demonstrations": _r(0, 33),
    "closed_loop_early_dev": _r(300, 319),
    "bc_development_eval": _r(400, 419),
    "frozen_eval_A_fixed": _r(1000, 1049),
    "frozen_eval_B_randomized": _r(1100, 1149),
    "ppo_1k_smoke_dev": _r(1200, 1204),
    "reset_benchmark": _r(2000, 2002),
    "cli_smokes": [3000, 3001],
    "ppo_training_env_stream": _r(9000, 9099),  # per-episode randomization stream for PPO training
    # --- Consumed development allocations (already executed; never reuse) ---
    "stage1_kl_calibration": _r(1400, 1404),
    "stage2_ppo_dev_checkpoint": _r(1410, 1419),
    "stage2_extreme_corner_eval": [15002, 15013, 15063, 15113],
}

# Backward-compatible aliases (still referenced by the launcher/tests).
STAGE1_KL_CALIBRATION_SEEDS: List[int] = _r(1400, 1404)
STAGE2_PPO_DEV_SEEDS: List[int] = _r(1410, 1419)
STAGE2_SECONDARY_DEV_SEEDS: List[int] = _r(1420, 1439)    # allocated, not yet consumed
STAGE2_EXTREME_EVAL_SEEDS: List[int] = [15002, 15013, 15063, 15113]

# --- Observation-v2 / multi-gate forward allocations (not yet consumed) -------
VISUAL_POSE_DATASET_V2_SEEDS: List[int] = _r(1600, 1699)   # v2 demonstration collection
BC_V2_DEV_EVAL_SEEDS: List[int] = _r(1700, 1729)           # single-gate v2 checkpoint selection
PPO_V2_DEV_SEEDS: List[int] = _r(1730, 1759)               # v2 PPO development eval
MULTIGATE_DEV_SEEDS: List[int] = _r(1760, 1799)            # multi-gate curriculum development
RESERVED_FINAL_FIXED_SEEDS: List[int] = _r(1500, 1549)    # DO NOT USE until the final stage
RESERVED_FINAL_RANDOMIZED_SEEDS: List[int] = _r(1550, 1599)  # DO NOT USE until the final stage
RESERVED_FINAL_MULTIGATE_SEEDS: List[int] = _r(1800, 1899)  # DO NOT USE until the final multi-gate eval

# --- Observation-v3 learned multi-gate experiment ----------------------------
# Every role is intentionally disjoint, including the three final task families.
MULTIGATE_V3_DEMONSTRATION_SEEDS: List[int] = _r(20000, 20099)
MULTIGATE_V3_PPO_TRAINING_SEEDS: List[int] = _r(20100, 20999)
MULTIGATE_V3_DEV_EVAL_SEEDS: List[int] = _r(21000, 21049)
MULTIGATE_V3_CHECKPOINT_SELECTION_SEEDS: List[int] = _r(21100, 21149)
MULTIGATE_V3_FINAL_TWO_GATE_SEEDS: List[int] = _r(21200, 21249)
MULTIGATE_V3_FINAL_THREE_GATE_SEEDS: List[int] = _r(21300, 21349)
MULTIGATE_V3_FINAL_OFFICIAL_SEEDS: List[int] = _r(21400, 21499)

# The PPO rollout-env seed (separate namespace; disjoint from all eval/test ranges).
PPO_TRAINING_ENV_SEED: int = 9000

# --- Multi-day observation-v3 long-run forward allocations -------------------
# These ranges are disjoint from the completed 5k experiment and its final seeds.
MULTIGATE_LONGRUN_PPO_TRAINING_SEEDS: List[int] = _r(22000, 22999)
MULTIGATE_LONGRUN_DEV_EVAL_SEEDS: List[int] = _r(25000, 25099)
MULTIGATE_LONGRUN_CHECKPOINT_SELECTION_SEEDS: List[int] = _r(25100, 25199)
MULTIGATE_LONGRUN_DEMONSTRATION_SEEDS: List[int] = _r(26000, 26999)
MULTIGATE_LONGRUN_BC_TRAINING_SEEDS: List[int] = _r(28000, 28199)
MULTIGATE_LONGRUN_BC_EVAL_SEEDS: List[int] = _r(28200, 28299)

# Reliability-first follow-up. These are separate from the completed long-run
# development namespaces so paired evidence cannot silently reuse tuning seeds.
MULTIGATE_RELIABILITY_PPO_TRAINING_SEEDS: List[int] = _r(23000, 23999)
MULTIGATE_RELIABILITY_DEV_EVAL_SEEDS: List[int] = _r(25300, 25399)
MULTIGATE_RELIABILITY_CHECKPOINT_SELECTION_SEEDS: List[int] = _r(25400, 25499)
MULTIGATE_RELIABILITY_VALIDATION_SEEDS: List[int] = _r(25500, 25599)

# Final common benchmark holdout. These seeds have never been used for training,
# checkpoint selection, reward design or hyper-parameter tuning, so the final
# controller comparison has a genuinely unseen half in every test group.
MULTIGATE_FINAL_BENCHMARK_HOLDOUT_SEEDS: List[int] = _r(27000, 27499)

NEW_ALLOCATIONS: Dict[str, List[int]] = {
    "stage2_secondary_dev_eval": STAGE2_SECONDARY_DEV_SEEDS,
    "visual_pose_dataset_v2": VISUAL_POSE_DATASET_V2_SEEDS,
    "bc_v2_dev_eval": BC_V2_DEV_EVAL_SEEDS,
    "ppo_v2_dev": PPO_V2_DEV_SEEDS,
    "multigate_dev": MULTIGATE_DEV_SEEDS,
    "RESERVED_final_fixed_eval": RESERVED_FINAL_FIXED_SEEDS,
    "RESERVED_final_randomized_eval": RESERVED_FINAL_RANDOMIZED_SEEDS,
    "RESERVED_final_multigate_eval": RESERVED_FINAL_MULTIGATE_SEEDS,
    "multigate_v3_demonstrations": MULTIGATE_V3_DEMONSTRATION_SEEDS,
    "multigate_v3_ppo_training": MULTIGATE_V3_PPO_TRAINING_SEEDS,
    "multigate_v3_dev_eval": MULTIGATE_V3_DEV_EVAL_SEEDS,
    "multigate_v3_checkpoint_selection": MULTIGATE_V3_CHECKPOINT_SELECTION_SEEDS,
    "multigate_v3_FINAL_two_gate": MULTIGATE_V3_FINAL_TWO_GATE_SEEDS,
    "multigate_v3_FINAL_three_gate": MULTIGATE_V3_FINAL_THREE_GATE_SEEDS,
    "multigate_v3_FINAL_official": MULTIGATE_V3_FINAL_OFFICIAL_SEEDS,
    "multigate_longrun_ppo_training": MULTIGATE_LONGRUN_PPO_TRAINING_SEEDS,
    "multigate_longrun_dev_eval": MULTIGATE_LONGRUN_DEV_EVAL_SEEDS,
    "multigate_longrun_checkpoint_selection": MULTIGATE_LONGRUN_CHECKPOINT_SELECTION_SEEDS,
    "multigate_longrun_demonstrations": MULTIGATE_LONGRUN_DEMONSTRATION_SEEDS,
    "multigate_longrun_bc_training": MULTIGATE_LONGRUN_BC_TRAINING_SEEDS,
    "multigate_longrun_bc_eval": MULTIGATE_LONGRUN_BC_EVAL_SEEDS,
    "multigate_reliability_ppo_training": MULTIGATE_RELIABILITY_PPO_TRAINING_SEEDS,
    "multigate_reliability_dev_eval": MULTIGATE_RELIABILITY_DEV_EVAL_SEEDS,
    "multigate_reliability_checkpoint_selection": MULTIGATE_RELIABILITY_CHECKPOINT_SELECTION_SEEDS,
    "multigate_reliability_validation": MULTIGATE_RELIABILITY_VALIDATION_SEEDS,
    "multigate_FINAL_benchmark_holdout": MULTIGATE_FINAL_BENCHMARK_HOLDOUT_SEEDS,
}

# Mutually-exclusive roles that must be pairwise disjoint (training/selection vs held-out).
ROLE_SEED_SETS: Dict[str, Set[int]] = {
    "demonstrations_v1": set(_r(0, 33)),
    "frozen_eval_A_fixed": set(_r(1000, 1049)),
    "frozen_eval_B_randomized": set(_r(1100, 1149)),
    "stage2_dev": set(_r(1400, 1439)) | set([15002, 15013, 15063, 15113]),
    "visual_pose_dataset_v2": set(VISUAL_POSE_DATASET_V2_SEEDS),
    "bc_v2_dev_eval": set(BC_V2_DEV_EVAL_SEEDS),
    "ppo_v2_dev": set(PPO_V2_DEV_SEEDS),
    "multigate_dev": set(MULTIGATE_DEV_SEEDS),
    "reserved_final_fixed": set(RESERVED_FINAL_FIXED_SEEDS),
    "reserved_final_randomized": set(RESERVED_FINAL_RANDOMIZED_SEEDS),
    "reserved_final_multigate": set(RESERVED_FINAL_MULTIGATE_SEEDS),
    "multigate_v3_demonstrations": set(MULTIGATE_V3_DEMONSTRATION_SEEDS),
    "multigate_v3_ppo_training": set(MULTIGATE_V3_PPO_TRAINING_SEEDS),
    "multigate_v3_dev_eval": set(MULTIGATE_V3_DEV_EVAL_SEEDS),
    "multigate_v3_checkpoint_selection": set(MULTIGATE_V3_CHECKPOINT_SELECTION_SEEDS),
    "multigate_v3_final_two_gate": set(MULTIGATE_V3_FINAL_TWO_GATE_SEEDS),
    "multigate_v3_final_three_gate": set(MULTIGATE_V3_FINAL_THREE_GATE_SEEDS),
    "multigate_v3_final_official": set(MULTIGATE_V3_FINAL_OFFICIAL_SEEDS),
    "multigate_longrun_ppo_training": set(MULTIGATE_LONGRUN_PPO_TRAINING_SEEDS),
    "multigate_longrun_dev_eval": set(MULTIGATE_LONGRUN_DEV_EVAL_SEEDS),
    "multigate_longrun_checkpoint_selection": set(MULTIGATE_LONGRUN_CHECKPOINT_SELECTION_SEEDS),
    "multigate_longrun_demonstrations": set(MULTIGATE_LONGRUN_DEMONSTRATION_SEEDS),
    "multigate_longrun_bc_training": set(MULTIGATE_LONGRUN_BC_TRAINING_SEEDS),
    "multigate_longrun_bc_eval": set(MULTIGATE_LONGRUN_BC_EVAL_SEEDS),
    "multigate_reliability_ppo_training": set(MULTIGATE_RELIABILITY_PPO_TRAINING_SEEDS),
    "multigate_reliability_dev_eval": set(MULTIGATE_RELIABILITY_DEV_EVAL_SEEDS),
    "multigate_reliability_checkpoint_selection": set(MULTIGATE_RELIABILITY_CHECKPOINT_SELECTION_SEEDS),
    "multigate_reliability_validation": set(MULTIGATE_RELIABILITY_VALIDATION_SEEDS),
    "multigate_final_benchmark_holdout": set(MULTIGATE_FINAL_BENCHMARK_HOLDOUT_SEEDS),
}

# Ranges that must never be used for training, checkpoint selection, reward or
# hyperparameter tuning (they are held out for the final scientific evaluation).
DO_NOT_TRAIN_ON: Dict[str, List[int]] = {
    "frozen_eval_A_fixed": USED_SEEDS["frozen_eval_A_fixed"],
    "frozen_eval_B_randomized": USED_SEEDS["frozen_eval_B_randomized"],
    "RESERVED_final_fixed_eval": RESERVED_FINAL_FIXED_SEEDS,
    "RESERVED_final_randomized_eval": RESERVED_FINAL_RANDOMIZED_SEEDS,
    "RESERVED_final_multigate_eval": RESERVED_FINAL_MULTIGATE_SEEDS,
    "multigate_v3_FINAL_two_gate": MULTIGATE_V3_FINAL_TWO_GATE_SEEDS,
    "multigate_v3_FINAL_three_gate": MULTIGATE_V3_FINAL_THREE_GATE_SEEDS,
    "multigate_v3_FINAL_official": MULTIGATE_V3_FINAL_OFFICIAL_SEEDS,
    "multigate_FINAL_benchmark_holdout": MULTIGATE_FINAL_BENCHMARK_HOLDOUT_SEEDS,
}


def assert_pairwise_disjoint() -> None:
    """Raise if any two mutually-exclusive seed roles overlap."""
    names = list(ROLE_SEED_SETS)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            clash = sorted(ROLE_SEED_SETS[a] & ROLE_SEED_SETS[b])
            if clash:
                raise ValueError(f"seed roles {a} and {b} overlap: {clash}")


def all_used_seeds() -> Set[int]:
    out: Set[int] = set()
    for seeds in USED_SEEDS.values():
        out.update(seeds)
    return out


def assert_new_allocations_are_unused() -> None:
    """Raise if any development allocation collides with an already-used seed."""
    used = all_used_seeds()
    for name, seeds in NEW_ALLOCATIONS.items():
        clash = sorted(set(seeds) & used)
        if clash:
            raise ValueError(f"seed allocation {name} reuses already-used seeds: {clash}")


def development_and_final_are_disjoint() -> bool:
    dev = (set(STAGE1_KL_CALIBRATION_SEEDS) | set(STAGE2_PPO_DEV_SEEDS) | set(STAGE2_SECONDARY_DEV_SEEDS)
           | set(VISUAL_POSE_DATASET_V2_SEEDS) | set(BC_V2_DEV_EVAL_SEEDS) | set(PPO_V2_DEV_SEEDS)
           | set(MULTIGATE_DEV_SEEDS))
    final = (set(RESERVED_FINAL_FIXED_SEEDS) | set(RESERVED_FINAL_RANDOMIZED_SEEDS)
             | set(RESERVED_FINAL_MULTIGATE_SEEDS)
             | set(MULTIGATE_V3_FINAL_TWO_GATE_SEEDS)
             | set(MULTIGATE_V3_FINAL_THREE_GATE_SEEDS)
             | set(MULTIGATE_V3_FINAL_OFFICIAL_SEEDS)
             | set(MULTIGATE_FINAL_BENCHMARK_HOLDOUT_SEEDS))
    dev |= (
        set(MULTIGATE_V3_DEMONSTRATION_SEEDS)
        | set(MULTIGATE_V3_PPO_TRAINING_SEEDS)
        | set(MULTIGATE_V3_DEV_EVAL_SEEDS)
        | set(MULTIGATE_V3_CHECKPOINT_SELECTION_SEEDS)
        | set(MULTIGATE_LONGRUN_PPO_TRAINING_SEEDS)
        | set(MULTIGATE_LONGRUN_DEV_EVAL_SEEDS)
        | set(MULTIGATE_LONGRUN_CHECKPOINT_SELECTION_SEEDS)
        | set(MULTIGATE_LONGRUN_DEMONSTRATION_SEEDS)
        | set(MULTIGATE_LONGRUN_BC_TRAINING_SEEDS)
        | set(MULTIGATE_LONGRUN_BC_EVAL_SEEDS)
        | set(MULTIGATE_RELIABILITY_PPO_TRAINING_SEEDS)
        | set(MULTIGATE_RELIABILITY_DEV_EVAL_SEEDS)
        | set(MULTIGATE_RELIABILITY_CHECKPOINT_SELECTION_SEEDS)
        | set(MULTIGATE_RELIABILITY_VALIDATION_SEEDS)
    )
    return dev.isdisjoint(final)


def registry_dict() -> Dict:
    return {
        "note": ("Canonical seed registry. New experiments must not reuse USED_SEEDS; the "
                 "RESERVED_final_* ranges are held out for the final scientific evaluation and "
                 "must not be used for training, checkpoint selection, reward or hyperparameter tuning."),
        "used_seeds": USED_SEEDS,
        "new_allocations": NEW_ALLOCATIONS,
        "do_not_train_on": sorted(set().union(*DO_NOT_TRAIN_ON.values())) if DO_NOT_TRAIN_ON else [],
        "ppo_training_env_seed": PPO_TRAINING_ENV_SEED,
        "roles_pairwise_disjoint_checked": sorted(ROLE_SEED_SETS),
        "invariants": {
            "new_allocations_unused": True,
            "development_and_final_disjoint": development_and_final_are_disjoint(),
            "roles_pairwise_disjoint": True,
        },
    }


def write_registry(path) -> None:
    import json
    from pathlib import Path

    assert_new_allocations_are_unused()
    assert_pairwise_disjoint()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(registry_dict(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    write_registry("results/rl_public/seed_registry.json")
    print("[seed_registry] wrote results/rl_public/seed_registry.json")
