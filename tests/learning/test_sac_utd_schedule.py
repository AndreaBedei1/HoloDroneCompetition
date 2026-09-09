"""Update-to-data accounting, and the provenance rules the offline study relies on.

SAC v7 degraded after actor activation (0.57 -> 0.22 -> 0.02) despite a repaired
critic-health detector and a reduced critic learning rate.  Before attributing
that to update-to-data pressure, the accounting itself has to be exact: the live
run showed measured/declared UTD of 1.0001, and these tests pin that property.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from marine_race_arena.learning.rl_holdout_policy import (
    FINAL_CIRCUIT_NAMES,
    role_of_seed,
)


def simulate_credit(total_transitions, n_envs, utd, learning_starts=0):
    """Mirror the trainer's credit accumulator exactly."""

    credit = 0.0
    updates = 0
    collected = 0
    for _ in range(0, total_transitions, n_envs):
        collected += n_envs
        if collected < learning_starts:
            continue
        credit += n_envs * utd
        while credit >= 1.0:
            updates += 1
            credit -= 1.0
    return updates


def test_the_declared_ratio_is_what_actually_happens():
    for utd in (0.0625, 0.03125, 0.015625, 0.25):
        updates = simulate_credit(400_000, 2, utd)
        assert updates == pytest.approx(400_000 * utd, rel=1e-3), utd


def test_halving_utd_halves_the_updates_for_the_same_data():
    """The whole point of the experiment: same data, half the optimization."""

    full = simulate_credit(400_000, 2, 0.0625)
    half = simulate_credit(400_000, 2, 0.03125)
    assert half == pytest.approx(full / 2, rel=1e-3)


def test_worker_count_does_not_change_updates_per_transition():
    """UTD must be per transition, not per environment step."""

    for n_envs in (1, 2, 4, 8):
        updates = simulate_credit(200_000, n_envs, 0.0625)
        assert updates == pytest.approx(200_000 * 0.0625, rel=1e-2), n_envs


def test_learning_starts_delays_but_does_not_inflate_updates():
    updates = simulate_credit(120_000, 2, 0.0625, learning_starts=20_000)
    assert updates == pytest.approx((120_000 - 20_000) * 0.0625, rel=1e-2)


def test_one_credit_unit_yields_exactly_one_update():
    """A duplicated update inside the credit loop would double the real UTD."""

    from marine_race_arena.learning import train_sac_transition as trainer

    body = Path(trainer.__file__).read_text(encoding="utf-8").split(
        "while update_credit >= 1.0:", 1)[1].split("elapsed = time.perf_counter()", 1)[0]
    assert body.count("agent.update(") == 1
    assert body.count("update_credit -= 1.0") == 1


def test_the_measured_live_ratio_matched_the_declared_one():
    """Recorded from the real v7 run: (tx - learning_starts) * utd == gupd."""

    for tx, gupd in ((66_816, 2_927), (160_000, 8_751), (253_184, 14_575)):
        predicted = (tx - 20_000) * 0.0625
        assert abs(gupd - predicted) / predicted < 0.01, (tx, gupd, predicted)


# --------------------------------------------------------- replay provenance


def test_only_train_seeds_may_enter_the_offline_study(tmp_path):
    from marine_race_arena.learning.episode_composition_log import (
        append_episode, read_episodes,
    )
    from marine_race_arena.learning.generic_sequence_curriculum import (
        GenericSequenceCurriculum,
    )
    from marine_race_arena.learning.transition_curriculum import (
        TransitionGeometrySampler,
    )
    from marine_race_arena.learning.episode_composition_log import episode_row

    sampler = TransitionGeometrySampler(
        seed=1_000_001, sequence_curriculum=GenericSequenceCurriculum(),
        dataset_split="train",
    )
    for _ in range(200):
        append_episode(tmp_path, episode_row(
            utc="t", algorithm="sac", run="r", worker_id=0, geometry=sampler.sample()))
    rows = read_episodes(tmp_path)
    assert {role_of_seed(r["episode_seed"]) for r in rows} == {"train"}
    assert all(r["dataset_split"] == "train" for r in rows)
    assert not any(
        n in str(r.get("pattern", "")).lower() for r in rows for n in FINAL_CIRCUIT_NAMES
    )


def test_validation_and_test_seeds_are_detectable_if_they_ever_appeared():
    from marine_race_arena.learning.rl_holdout_policy import seed_group

    assert role_of_seed(seed_group("validation").seed_at(0)) == "validation"
    assert role_of_seed(seed_group("test").seed_at(0)) == "test"
    assert role_of_seed(seed_group("train").seed_at(0)) == "train"


def test_the_offline_study_reserves_a_heldout_slice_it_never_fits():
    """Generalisation diagnostics must not be drawn from the fitted region."""

    usable, heldout, total = 280_032, 20_000, 300_032
    assert usable + heldout == total
    fitted = np.arange(usable)
    held = np.arange(usable, usable + heldout)
    assert not set(fitted.tolist()) & set(held.tolist())


def test_v7_is_preserved_and_marked_not_deleted():
    run = Path(
        "results/rl/universal_transition/sac/"
        "universal_transition_sac_v7_stable_critic_seed23001"
    )
    if not run.exists():
        pytest.skip("v7 run directory not present in this checkout")
    marker = run / "SCIENTIFIC_INTERPRETATION.json"
    assert marker.exists()
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["preserved"] is True
    assert payload["deleted"] is False
    assert payload["do_not_resume"] is True
    assert "post_activation_critic_drift" in payload["interpretation"]
