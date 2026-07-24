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
}

# --- New allocations for this Stage-2 diagnostic ------------------------------
STAGE1_KL_CALIBRATION_SEEDS: List[int] = _r(1400, 1404)
STAGE2_PPO_DEV_SEEDS: List[int] = _r(1410, 1419)          # checkpoint selection
STAGE2_SECONDARY_DEV_SEEDS: List[int] = _r(1420, 1439)    # secondary development eval
RESERVED_FINAL_FIXED_SEEDS: List[int] = _r(1500, 1549)    # DO NOT USE this task
RESERVED_FINAL_RANDOMIZED_SEEDS: List[int] = _r(1550, 1599)  # DO NOT USE this task

# The PPO rollout-env seed (separate namespace; disjoint from all eval/test ranges).
PPO_TRAINING_ENV_SEED: int = 9000

NEW_ALLOCATIONS: Dict[str, List[int]] = {
    "stage1_kl_calibration": STAGE1_KL_CALIBRATION_SEEDS,
    "stage2_ppo_dev_checkpoint": STAGE2_PPO_DEV_SEEDS,
    "stage2_secondary_dev_eval": STAGE2_SECONDARY_DEV_SEEDS,
    "RESERVED_final_fixed_eval": RESERVED_FINAL_FIXED_SEEDS,
    "RESERVED_final_randomized_eval": RESERVED_FINAL_RANDOMIZED_SEEDS,
}

# Ranges that must never be used for training, checkpoint selection, reward or
# hyperparameter tuning (they are held out for the final scientific evaluation).
DO_NOT_TRAIN_ON: Dict[str, List[int]] = {
    "frozen_eval_A_fixed": USED_SEEDS["frozen_eval_A_fixed"],
    "frozen_eval_B_randomized": USED_SEEDS["frozen_eval_B_randomized"],
    "RESERVED_final_fixed_eval": RESERVED_FINAL_FIXED_SEEDS,
    "RESERVED_final_randomized_eval": RESERVED_FINAL_RANDOMIZED_SEEDS,
}


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
    dev = set(STAGE1_KL_CALIBRATION_SEEDS) | set(STAGE2_PPO_DEV_SEEDS) | set(STAGE2_SECONDARY_DEV_SEEDS)
    final = set(RESERVED_FINAL_FIXED_SEEDS) | set(RESERVED_FINAL_RANDOMIZED_SEEDS)
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
        "invariants": {
            "new_allocations_unused": True,
            "development_and_final_disjoint": development_and_final_are_disjoint(),
        },
    }


def write_registry(path) -> None:
    import json
    from pathlib import Path

    assert_new_allocations_are_unused()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(registry_dict(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    write_registry("results/rl_public/seed_registry.json")
    print("[seed_registry] wrote results/rl_public/seed_registry.json")
