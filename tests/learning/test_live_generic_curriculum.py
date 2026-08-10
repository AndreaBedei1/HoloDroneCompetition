"""The generic sequence curriculum and the dataset split must be LIVE.

Both existed as importable modules for days while the live samplers ignored
them: training stayed on the binary two-gate coin flip and rollout seeds came
from arbitrary worker bases.  These tests pin the wiring itself, not the
existence of the classes.
"""

from __future__ import annotations

import inspect
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from marine_race_arena.learning.episode_composition_log import (
    append_episode,
    episode_row,
    read_episodes,
    summarize_episodes,
)
from marine_race_arena.learning.generic_sequence_curriculum import (
    BUCKET_GATE_RANGES,
    STAGES,
    GenericSequenceCurriculum,
    bucket_completion,
    empirical_mixture,
)
from marine_race_arena.learning.rl_holdout_policy import (
    SealedCircuitViolation,
    assert_not_final_circuit,
    role_of_seed,
    seed_group,
)
from marine_race_arena.learning.transition_curriculum import TransitionGeometrySampler
from marine_race_arena.learning.transition_env import UniversalTransitionEnv


def _sampler(stage=0, split="train", seed=1_234_567):
    curriculum = GenericSequenceCurriculum()
    curriculum.state.stage_index = stage
    return TransitionGeometrySampler(
        seed=seed, sequence_curriculum=curriculum, dataset_split=split
    ), curriculum


# ------------------------------------------------- curriculum is really live


def test_the_live_sampler_uses_the_curriculum_not_the_coin_flip():
    sampler, curriculum = _sampler()
    assert sampler.sequence_curriculum is curriculum
    counts = Counter(sampler.sample().gate_count for _ in range(1500))
    assert len(counts) > 5, "a coin flip would only ever produce 2 and the fixed set"
    assert max(counts) > 2


@pytest.mark.parametrize("stage_index", range(len(STAGES)))
def test_each_stage_reproduces_its_declared_mixture(stage_index):
    sampler, curriculum = _sampler(stage=stage_index, seed=1_000_003 + stage_index)
    target = curriculum.mixture
    observed = empirical_mixture(
        [sampler.sample().gate_count for _ in range(6000)]
    )
    for bucket, want in target.items():
        assert observed[bucket] == pytest.approx(want, abs=0.03), (
            f"{STAGES[stage_index]['name']} {bucket}: want {want}, got {observed[bucket]}"
        )


def test_stage_zero_is_gentle_and_has_no_long_sequences():
    """S0 must preserve existing competence while introducing chaining."""

    sampler, curriculum = _sampler(stage=0)
    assert curriculum.mixture["focus"] == pytest.approx(0.50)
    assert curriculum.mixture["long"] == pytest.approx(0.0)
    counts = [sampler.sample().gate_count for _ in range(3000)]
    assert max(counts) <= BUCKET_GATE_RANGES["medium"][1]


def test_later_stages_shift_mass_to_longer_sequences():
    shares = []
    for index in range(len(STAGES)):
        sampler, curriculum = _sampler(stage=index, seed=2_000 + index)
        observed = empirical_mixture([sampler.sample().gate_count for _ in range(3000)])
        shares.append(observed["medium"] + observed["long"])
    assert shares == sorted(shares), f"long-sequence mass must not decrease: {shares}"


def test_gate_counts_span_the_full_declared_range():
    sampler, _ = _sampler(stage=len(STAGES) - 1, seed=99)
    counts = {sampler.sample().gate_count for _ in range(8000)}
    assert min(counts) == 2
    assert max(counts) >= 20, f"long bucket under-sampled: max {max(counts)}"


# --------------------------------------------------------- promotion policy


def _metrics(success, short=None, medium=None, n_eval=112, collisions=0):
    by_length = {}
    if short is not None:
        by_length.update({"3": {"completion_rate": short}, "5": {"completion_rate": short}})
    if medium is not None:
        by_length.update({"8": {"completion_rate": medium}, "12": {"completion_rate": medium}})
    return {
        "n_eval": n_eval,
        "universal_transition_success_rate": success,
        "full_sequence_success_by_length": by_length,
        "collision_episodes": collisions,
    }


def test_promotion_requires_two_consecutive_qualifying_validations():
    curriculum = GenericSequenceCurriculum()
    good = _metrics(0.85, short=0.75)
    first = curriculum.observe_validation(good, 100_000)
    assert first["promoted"] is False, "one good result must never promote"
    second = curriculum.observe_validation(good, 150_000)
    assert second["promoted"] is True
    assert curriculum.stage["name"] == STAGES[1]["name"]


def test_a_noisy_good_result_between_bad_ones_does_not_promote():
    curriculum = GenericSequenceCurriculum()
    curriculum.observe_validation(_metrics(0.85, short=0.75), 1)
    curriculum.observe_validation(_metrics(0.50, short=0.20), 2)
    record = curriculum.observe_validation(_metrics(0.85, short=0.75), 3)
    assert record["promoted"] is False
    assert curriculum.state.stage_index == 0


def test_promotion_needs_the_sequence_bar_not_only_transition_success():
    curriculum = GenericSequenceCurriculum()
    # Transition success clears 0.80 but 3-5 gate completion is far below 0.70.
    weak = _metrics(0.95, short=0.10)
    curriculum.observe_validation(weak, 1)
    record = curriculum.observe_validation(weak, 2)
    assert record["promoted"] is False
    assert curriculum.state.stage_index == 0


def test_a_missing_sequence_measurement_never_qualifies():
    curriculum = GenericSequenceCurriculum()
    blind = _metrics(0.99, short=None)
    curriculum.observe_validation(blind, 1)
    record = curriculum.observe_validation(blind, 2)
    assert record["promoted"] is False


def test_sustained_regression_demotes():
    curriculum = GenericSequenceCurriculum()
    good = _metrics(0.85, short=0.75)
    curriculum.observe_validation(good, 1)
    curriculum.observe_validation(good, 2)
    assert curriculum.state.stage_index == 1
    bad = _metrics(0.20)
    curriculum.observe_validation(bad, 3)
    record = curriculum.observe_validation(bad, 4)
    assert record["demoted"] is True
    assert curriculum.state.stage_index == 0


def test_terminal_stage_has_no_promotion_bar():
    curriculum = GenericSequenceCurriculum()
    curriculum.state.stage_index = len(STAGES) - 1
    assert curriculum.promotion_bar() is None
    record = curriculum.observe_validation(_metrics(0.99, short=0.99, medium=0.99), 1)
    assert record["promoted"] is False


def test_bucket_completion_averages_the_right_lengths():
    metrics = {"full_sequence_success_by_length": {
        "3": {"completion_rate": 1.0}, "5": {"completion_rate": 0.5},
        "8": {"completion_rate": 0.25}, "22": {"completion_rate": 0.0},
    }}
    assert bucket_completion(metrics, "short") == pytest.approx(0.75)
    assert bucket_completion(metrics, "medium") == pytest.approx(0.25)
    assert bucket_completion(metrics, "long") == pytest.approx(0.0)
    assert bucket_completion({}, "short") is None


# ------------------------------------------------------- dataset separation


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_every_episode_seed_lands_in_its_declared_band(split):
    sampler, _ = _sampler(split=split)
    seeds = [sampler.sample().seed for _ in range(2000)]
    group = seed_group(split)
    assert all(group.contains(s) for s in seeds)
    assert {role_of_seed(s) for s in seeds} == {split}


def test_train_and_validation_seeds_can_never_collide():
    train, _ = _sampler(split="train", seed=11)
    validation, _ = _sampler(split="validation", seed=11)
    train_seeds = {train.sample().seed for _ in range(3000)}
    validation_seeds = {validation.sample().seed for _ in range(3000)}
    assert not (train_seeds & validation_seeds)


def test_a_rollout_worker_may_only_use_the_train_split(tmp_path):
    for split in ("validation", "test"):
        with pytest.raises(ValueError, match="must sample from the TRAIN split"):
            UniversalTransitionEnv(
                run_dir=tmp_path, worker_id=0, sampler_seed=1,
                difficulty="G1", adapter="fallback", allow_fallback=True,
                dataset_split=split,
            )


def test_the_env_constructor_accepts_the_curriculum():
    params = inspect.signature(UniversalTransitionEnv.__init__).parameters
    for name in ("sequence_curriculum", "dataset_split", "algorithm"):
        assert name in params


def test_both_trainers_forward_the_curriculum_to_their_workers():
    from marine_race_arena.learning import train_ppo_transition, train_sac_transition

    for module in (train_ppo_transition, train_sac_transition):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "sequence_curriculum=sequence_curriculum" in source
        assert 'config.get("sequence_curriculum")' in source
        assert 'dataset_split="train"' in source


def test_final_circuits_are_still_sealed_for_training():
    from marine_race_arena.learning.rl_holdout_policy import FINAL_CIRCUIT_TRACKS

    for track in FINAL_CIRCUIT_TRACKS:
        with pytest.raises(SealedCircuitViolation):
            assert_not_final_circuit(track, purpose="training")


def test_no_generated_track_is_a_final_circuit():
    from marine_race_arena.learning.rl_holdout_policy import FINAL_CIRCUIT_NAMES

    sampler, _ = _sampler()
    for _ in range(500):
        geometry = sampler.sample()
        assert not any(
            name in str(geometry.pattern).lower() for name in FINAL_CIRCUIT_NAMES
        )


# ------------------------------------------------------- procedural variety


def test_generated_courses_are_diverse_and_not_a_tiny_pool():
    sampler, _ = _sampler(stage=2, seed=4242)
    episodes = [sampler.sample() for _ in range(2000)]
    signatures = {
        (e.gate_count, tuple(round(v, 3) for v in e.spacings_m),
         tuple(round(v, 2) for v in e.turn_deltas_deg))
        for e in episodes
    }
    assert len(signatures) > 1900, f"only {len(signatures)} distinct courses"
    spacings = [v for e in episodes for v in e.spacings_m]
    turns = [v for e in episodes for v in e.turn_deltas_deg]
    vertical = [v for e in episodes for v in e.vertical_deltas_m]
    assert min(spacings) < max(spacings)
    assert min(turns) < 0 < max(turns), "turns must go both left and right"
    assert min(vertical) < 0 < max(vertical), "must both climb and descend"
    assert len({e.pattern for e in episodes}) >= 4


def test_generated_geometry_is_physically_sane():
    sampler, _ = _sampler(stage=3, seed=77)
    for _ in range(1000):
        e = sampler.sample()
        assert e.gate_count >= 2
        assert len(e.spacings_m) == e.gate_count - 1
        assert all(np.isfinite(v) and v > 0 for v in e.spacings_m)
        assert all(np.isfinite(v) for v in e.turn_deltas_deg)
        assert all(np.isfinite(v) for v in e.vertical_deltas_m)


# ---------------------------------------------------------- episode logging


def test_episode_rows_capture_composition_and_outcome(tmp_path):
    sampler, _ = _sampler(stage=1)
    for _ in range(50):
        geometry = sampler.sample()
        append_episode(tmp_path, episode_row(
            utc="2026-08-10T00:00:00Z", algorithm="ppo", run="r",
            worker_id=0, geometry=geometry,
            outcome={"gates_completed": 1, "collision": False, "timeout": False},
        ))
    rows = read_episodes(tmp_path)
    assert len(rows) == 50
    assert all(r["dataset_split"] == "train" for r in rows)
    assert all(r["curriculum_stage"] == STAGES[1]["name"] for r in rows)
    assert all(r["sequence_category"] in BUCKET_GATE_RANGES for r in rows)
    assert all("spacing_mean_m" in r for r in rows)
    assert all(json.dumps(r) for r in rows)


def test_the_summary_reports_what_the_agent_actually_saw(tmp_path):
    sampler, curriculum = _sampler(stage=2, seed=31337)
    for _ in range(3000):
        append_episode(tmp_path, episode_row(
            utc="2026-08-10T00:00:00Z", algorithm="sac", run="r",
            worker_id=1, geometry=sampler.sample(),
        ))
    summary = summarize_episodes(read_episodes(tmp_path))
    assert summary["episodes"] == 3000
    for bucket, want in curriculum.mixture.items():
        assert summary["category_share"][bucket] == pytest.approx(want, abs=0.04)
    assert summary["dataset_split_share"]["train"] == pytest.approx(1.0)
    assert summary["distinct_episode_seeds"] > 2900
    assert summary["spacing_m"]["min"] < summary["spacing_m"]["max"]


def test_logging_never_raises_on_an_unwritable_directory(tmp_path):
    sampler, _ = _sampler()
    append_episode(tmp_path / "nope" / "deep", episode_row(
        utc="x", algorithm="ppo", run="r", worker_id=0, geometry=sampler.sample()))


def test_the_env_logs_at_episode_end_with_matching_geometry():
    source = Path(UniversalTransitionEnv.__module__.replace(".", "/") + ".py")
    body = Path(
        __import__("marine_race_arena.learning.transition_env", fromlist=["x"]).__file__
    ).read_text(encoding="utf-8")
    tail = body.split("if terminated or truncated:", 1)[1]
    assert "append_episode(" in tail
    assert "geometry=self.current_geometry" in tail
