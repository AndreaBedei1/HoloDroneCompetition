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

# The PPO rollout-env seed (separate namespace; disjoint from all eval/test ranges).
PPO_TRAINING_ENV_SEED: int = 9000

NEW_ALLOCATIONS: Dict[str, List[int]] = {
    "stage2_secondary_dev_eval": STAGE2_SECONDARY_DEV_SEEDS,
    "visual_pose_dataset_v2": VISUAL_POSE_DATASET_V2_SEEDS,
    "bc_v2_dev_eval": BC_V2_DEV_EVAL_SEEDS,
    "ppo_v2_dev": PPO_V2_DEV_SEEDS,
    "multigate_dev": MULTIGATE_DEV_SEEDS,
    "RESERVED_final_fixed_eval": RESERVED_FINAL_FIXED_SEEDS,
    "RESERVED_final_randomized_eval": RESERVED_FINAL_RANDOMIZED_SEEDS,
    "RESERVED_final_multigate_eval": RESERVED_FINAL_MULTIGATE_SEEDS,
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
}

# Ranges that must never be used for training, checkpoint selection, reward or
# hyperparameter tuning (they are held out for the final scientific evaluation).
DO_NOT_TRAIN_ON: Dict[str, List[int]] = {
    "frozen_eval_A_fixed": USED_SEEDS["frozen_eval_A_fixed"],
    "frozen_eval_B_randomized": USED_SEEDS["frozen_eval_B_randomized"],
    "RESERVED_final_fixed_eval": RESERVED_FINAL_FIXED_SEEDS,
    "RESERVED_final_randomized_eval": RESERVED_FINAL_RANDOMIZED_SEEDS,
    "RESERVED_final_multigate_eval": RESERVED_FINAL_MULTIGATE_SEEDS,
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
             | set(RESERVED_FINAL_MULTIGATE_SEEDS))
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
