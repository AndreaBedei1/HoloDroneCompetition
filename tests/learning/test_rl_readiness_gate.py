"""Tests for the pre-registered readiness gate and the holdout access guard.

Every test here asks the same question in a different way: can the three sealed
circuits be reached, or their results reused, by anything other than the
declared protocol?  The interesting cases are therefore the refusals -- a
hand-edited freeze record, a swapped checkpoint, a forgiven safety criterion, a
deleted ledger entry -- not the happy path.
"""

from __future__ import annotations

import copy
import json

import pytest

from marine_race_arena.learning.config import ACTION_CONTRACT_VERSION
from marine_race_arena.learning.config_local_transition import (
    OBS_ENCODING_VERSION_LOCAL_TRANSITION,
)
from marine_race_arena.learning.generic_sequence_curriculum import CURRICULUM_VERSION
from marine_race_arena.learning.reward_audit import REWARD_CONTRACT_VERSION
from marine_race_arena.learning.rl_holdout_policy import role_of_seed
from marine_race_arena.learning.rl_readiness_gate import (
    EXPLORATORY_HOLDOUT_DECISION_VERSION,
    FINAL_CIRCUIT_TRIAL_SEEDS,
    FREEZE_RECORD_VERSION,
    GRANT_NARROW_MISS,
    GROUP_COMPETENCE,
    GROUP_EVIDENCE,
    LADDER_KEY,
    PROTOCOL_MAY_VARY,
    PROTOCOL_MUST_NOT_VARY,
    READINESS_THRESHOLDS,
    REFUSAL_BUDGET,
    REFUSAL_EVIDENCE,
    REFUSAL_NOT_MEASURED,
    REFUSAL_SAFETY,
    REFUSAL_TOO_LARGE,
    TRIALS_PER_CIRCUIT,
    FinalCircuitsLocked,
    HoldoutContaminated,
    LedgerTampered,
    PolicyAlreadyFrozen,
    ProtocolViolation,
    ReadinessNotMet,
    ReadinessOverride,
    ReadinessThresholds,
    assert_final_circuits_unlocked,
    assert_no_tuning_after_holdout,
    assert_protocol_variations_allowed,
    benchmark_inputs,
    difficulty_ladder_summary,
    evaluate_readiness,
    exploratory_holdout_decision_path,
    final_circuit_protocol,
    freeze_policy,
    freeze_record_id,
    freeze_record_path,
    gate_manifest,
    protocol_permits_variation,
    record_final_circuit_evaluation,
    record_exploratory_holdout_decision,
    resolved_dataset_split,
    verify_ledger_chain,
)

VALIDATION_SEED = 5_000_007
TEST_BAND_SEED = 8_000_003
TRAIN_BAND_SEED = 1_000_042
#: What a real report carries at the top level: the sampler master seed, which
#: belongs to no band.  The band evidence is the per-episode case seeds.
MASTER_SEED = 20_260_808

CONTRACTS = {
    "reward": REWARD_CONTRACT_VERSION,
    "observation": OBS_ENCODING_VERSION_LOCAL_TRANSITION,
    "action": ACTION_CONTRACT_VERSION,
}
GIT_SHA = "0123456789abcdef0123456789abcdef01234567"
REASON = "12-gate completion missed by one case of twenty; reviewed by two authors"
LADDER_REASON = "G3 transition success short by one case of sixty; reviewed by two authors"

#: ``run_difficulty_ladder`` defaults to 60 transition cases per rung.
LADDER_RUNG_CASES = 60


def ladder_rung(
    difficulty: str, success: float, first_gate: float, n: int = LADDER_RUNG_CASES
) -> dict:
    """One row shaped exactly like ``run_difficulty_ladder`` writes it."""

    return {
        "difficulty": difficulty,
        "n": n,
        "success": success,
        "success_ci": [max(0.0, success - 0.13), min(1.0, success + 0.13)],
        "first_gate": first_gate,
        "switch": first_gate - 0.05,
        "alignment": first_gate - 0.08,
        "collisions": 0,
        "out_of_bounds": 0,
        "missed_gate_dnf": 0,
        "long_sequence": success / 2.0,
    }


def ladder_report(*rungs: dict) -> dict:
    """A ``ppo_difficulty_ladder_v1`` report as the tool writes it to disk."""

    return {
        "schema_version": "ppo_difficulty_ladder_v1",
        "checkpoint": "results/rl/.../ppo_628736_steps.zip",
        "sha256": "a" * 64,
        "dataset_split": "validation",
        "seed": 5_000_000,
        "transition_cases": LADDER_RUNG_CASES,
        "full_cases_per_length": 2,
        "rungs": list(rungs),
    }


#: The ladder a candidate is expected to arrive with: strong on the rung it
#: trained at, exactly on the bar at the mid rung, and visibly degrading above
#: it.  The bar is read at G3, so the poor G6 row is reported without being
#: fatal.
DEFAULT_LADDER = ladder_report(
    ladder_rung("G1", 0.92, 0.99),
    ladder_rung("G3", 0.50, 0.80),
    ladder_rung("G6", 0.18, 0.55),
)


def exactly_at_threshold_metrics(**overrides) -> dict:
    """A validation report sitting exactly on every pre-registered bar."""

    metrics = {
        "seed": MASTER_SEED,
        "dataset_split": "validation",
        "episode_seeds": [VALIDATION_SEED, VALIDATION_SEED + 11, VALIDATION_SEED + 97],
        "difficulty": "G1",
        "difficulty_ladder": copy.deepcopy(DEFAULT_LADDER),
        "n_eval": 1120,
        "transition_n": 1000,
        "universal_transition_success_rate": 0.90,
        "first_gate_crossing_rate": 0.97,
        "target_switch_rate": 0.85,
        "new_target_alignment_rate": 0.85,
        "out_of_bounds_rate": 0.01,
        "collision_episode_rate": 0.20,
        "full_sequence_success_by_length": {
            "3": {"n": 20, "completion_rate": 0.95},
            "5": {"n": 20, "completion_rate": 0.90},
            "8": {"n": 20, "completion_rate": 0.75},
            "12": {"n": 20, "completion_rate": 0.50},
            "17": {"n": 20, "completion_rate": 0.25},
            "22": {"n": 20, "completion_rate": 0.25},
        },
    }
    metrics.update(overrides)
    return metrics


def with_metrics(**overrides) -> dict:
    return exactly_at_threshold_metrics(**overrides)


def with_length(length: int, **fields) -> dict:
    metrics = exactly_at_threshold_metrics()
    metrics["full_sequence_success_by_length"] = copy.deepcopy(
        metrics["full_sequence_success_by_length"]
    )
    metrics["full_sequence_success_by_length"][str(length)].update(fields)
    return metrics


def with_ladder(*rungs: dict) -> dict:
    """The standard report with its ladder replaced by ``rungs``."""

    return exactly_at_threshold_metrics(difficulty_ladder=ladder_report(*rungs))


def without_ladder() -> dict:
    """A candidate that cleared every G1 bar and was never stressed above G1."""

    metrics = exactly_at_threshold_metrics()
    metrics.pop("difficulty_ladder")
    return metrics


def frozen_policy(tmp_path, *, metrics=None, verdict=None, run_dir=None):
    """Freeze a ready policy and return (checkpoint, record, record_path)."""

    run = run_dir or tmp_path
    checkpoint = tmp_path / "ppo_8400000_steps.zip"
    if not checkpoint.exists():
        checkpoint.write_bytes(b"frozen policy weights")
    metrics = metrics if metrics is not None else exactly_at_threshold_metrics()
    verdict = verdict if verdict is not None else evaluate_readiness(metrics)
    record = freeze_policy(
        checkpoint,
        run_dir=run,
        metrics=metrics,
        verdict=verdict,
        git_sha=GIT_SHA,
        contracts=CONTRACTS,
        training_transitions=8_400_000,
    )
    return checkpoint, record, freeze_record_path(run)


def exploratory_policy(tmp_path, *, run_dir=None):
    """Record one failed-readiness exploratory decision for guard tests."""

    run = run_dir or tmp_path
    checkpoint = tmp_path / "ppo_929792_steps.zip"
    checkpoint.write_bytes(b"immutable final policy weights")
    metrics = with_metrics(
        n_eval=620,
        transition_n=500,
        universal_transition_success_rate=0.82,
        out_of_bounds_episodes=0,
        collision_episodes=0,
    )
    metrics.pop("out_of_bounds_rate")
    metrics.pop("collision_episode_rate")
    verdict = evaluate_readiness(metrics)
    assert not verdict.ready
    record = record_exploratory_holdout_decision(
        checkpoint,
        run_dir=run,
        metrics=metrics,
        verdict=verdict,
        git_sha=GIT_SHA,
        contracts=CONTRACTS,
        training_transitions=929_792,
    )
    return checkpoint, record, exploratory_holdout_decision_path(run), metrics


# ------------------------------------------------------------- thresholds

def test_thresholds_are_the_preregistered_values():
    """The bars are the contract; a silent edit must break this test."""

    thresholds = READINESS_THRESHOLDS
    assert thresholds.min_universal_transition_success == 0.90
    assert thresholds.min_first_gate_crossing == 0.97
    assert thresholds.min_target_switch == 0.85
    assert thresholds.min_new_target_alignment == 0.85
    assert dict(thresholds.completion_by_length) == {
        3: 0.95, 5: 0.90, 8: 0.75, 12: 0.50, 17: 0.25, 22: 0.25,
    }
    assert thresholds.max_out_of_bounds_rate == 0.01
    assert thresholds.max_collision_episode_rate == 0.20
    assert thresholds.required_dataset_split == "validation"
    assert thresholds.min_transition_cases >= 500
    # A 0.95 completion bar is unmeasurable with the benchmark default of five
    # cases per length, whose resolution is 0.20.
    assert thresholds.min_full_cases_per_length >= 20
    json.dumps(thresholds.as_dict())


def test_the_difficulty_bar_is_a_ladder_not_a_single_level():
    """The matched regime is pinned low; the *stress* bar is what earns the pass.

    Every candidate trained only at G1, so a matched benchmark demanded at G6
    could only ever answer "not ready", and a matched benchmark accepted at G1
    with nothing else would certify near-straight, near-level geometry.  Both
    halves have to be in the pre-registered bars for either to mean anything.
    """

    thresholds = READINESS_THRESHOLDS
    assert thresholds.primary_difficulty == "G1"
    assert thresholds.ladder_min_rung == "G3"
    assert thresholds.min_ladder_transition_success == 0.50
    assert thresholds.min_ladder_first_gate_crossing == 0.80
    # A rung claimed on a handful of cases is not a rung that was run.
    assert thresholds.min_ladder_rung_cases >= 30
    assert not hasattr(thresholds, "required_difficulty")
    payload = thresholds.as_dict()
    assert payload["primary_difficulty"] == "G1"
    assert payload["ladder_min_rung"] == "G3"
    assert "required_difficulty" not in payload


def test_threshold_hash_identifies_the_gate_that_was_applied():
    assert ReadinessThresholds().sha256() == READINESS_THRESHOLDS.sha256()
    relaxed = ReadinessThresholds(min_universal_transition_success=0.80)
    assert relaxed.sha256() != READINESS_THRESHOLDS.sha256()


@pytest.mark.parametrize("relaxation", [
    {"min_ladder_transition_success": 0.10},
    {"min_ladder_first_gate_crossing": 0.10},
    {"ladder_min_rung": "G1"},
    {"min_ladder_rung_cases": 0},
    {"primary_difficulty": "G6"},
])
def test_the_ladder_bars_are_inside_the_hash_the_freeze_record_pins(relaxation):
    """Softening the ladder must be as visible as softening any other bar.

    The freeze record pins ``readiness_thresholds_sha256`` and the access guard
    compares it to the pre-registered value, so a ladder bar left out of the hash
    would be a bar that could be quietly lowered after the fact.
    """

    assert ReadinessThresholds(**relaxation).sha256() != READINESS_THRESHOLDS.sha256()


# ------------------------------------------------------------ gate verdict

def test_metrics_exactly_at_every_threshold_pass():
    verdict = evaluate_readiness(exactly_at_threshold_metrics())
    assert verdict.ready is True
    assert verdict.overall is True
    assert verdict.ready_without_override is True
    assert verdict.failures == ()
    assert verdict.narrow_misses == ()
    assert verdict.unmeasured == ()
    json.dumps(verdict.as_dict())


def test_full_benchmark_report_is_accepted_like_a_metrics_block():
    """The fields that make the set *matched* live on the report, not in metrics.

    A real ``evaluate_checkpoint_universal_transition_benchmark`` report carries
    the difficulty, the declared split and the per-episode case seeds at the top
    level; the band evidence therefore has to survive the episode log being
    dropped.
    """

    report = {
        "schema_version": "universal_transition_benchmark_v1",
        "seed": MASTER_SEED,
        "difficulty": "G1",
        "dataset_split": "validation",
        "difficulty_ladder": copy.deepcopy(DEFAULT_LADDER),
        "episodes": [
            {"seed": VALIDATION_SEED + index, "gate_count": 2} for index in range(8)
        ],
        "metrics": {
            key: value
            for key, value in exactly_at_threshold_metrics().items()
            if key not in {
                "seed", "difficulty", "dataset_split", "episode_seeds",
                "difficulty_ladder",
            }
        },
    }
    inputs = benchmark_inputs(report)
    assert inputs["episode_seed_roles"] == ["validation"]
    assert inputs["episode_seed_n"] == 8
    assert "episodes" not in inputs
    assert evaluate_readiness(report).ready is True


JUST_BELOW = {
    "universal_transition_success": with_metrics(
        universal_transition_success_rate=0.899
    ),
    "first_gate_crossing": with_metrics(first_gate_crossing_rate=0.969),
    "target_switch": with_metrics(target_switch_rate=0.849),
    "new_target_alignment": with_metrics(new_target_alignment_rate=0.849),
    "completion_3_gate": with_length(3, completion_rate=0.90),
    "completion_5_gate": with_length(5, completion_rate=0.85),
    "completion_8_gate": with_length(8, completion_rate=0.70),
    "completion_12_gate": with_length(12, completion_rate=0.45),
    "completion_17_gate": with_length(17, completion_rate=0.20),
    "completion_22_gate": with_length(22, completion_rate=0.20),
    "out_of_bounds_rate": with_metrics(out_of_bounds_rate=0.011),
    "collision_episode_rate": with_metrics(collision_episode_rate=0.21),
    "matched_validation_split": with_metrics(episode_seeds=[TRAIN_BAND_SEED]),
    "matched_difficulty": with_metrics(difficulty="G3"),
    "transition_cases": with_metrics(transition_n=499),
    "full_sequence_cases_per_length": with_length(8, n=19),
    "ladder_transition_success": with_ladder(
        ladder_rung("G1", 0.92, 0.99), ladder_rung("G3", 0.49, 0.80)
    ),
    "ladder_first_gate_crossing": with_ladder(
        ladder_rung("G1", 0.92, 0.99), ladder_rung("G3", 0.50, 0.79)
    ),
    "ladder_mid_rung_coverage": with_ladder(
        ladder_rung("G1", 0.92, 0.99), ladder_rung("G2", 0.71, 0.93)
    ),
}


@pytest.mark.parametrize("criterion,metrics", sorted(JUST_BELOW.items()))
def test_metric_just_below_a_threshold_fails_and_is_named(criterion, metrics):
    verdict = evaluate_readiness(metrics)
    assert verdict.ready is False
    assert criterion in verdict.failures
    assert verdict.criterion(criterion).satisfied is False


def test_every_criterion_is_covered_by_the_just_below_cases():
    """A new criterion must arrive with its own failing case."""

    names = {item.name for item in evaluate_readiness(
        exactly_at_threshold_metrics()).criteria}
    assert names == set(JUST_BELOW)


def test_test_band_evidence_can_never_satisfy_the_gate():
    """The sealed TEST band is not a substitute for the validation band."""

    verdict = evaluate_readiness(
        with_metrics(episode_seeds=[TEST_BAND_SEED], dataset_split="test")
    )
    assert verdict.ready is False
    assert "matched_validation_split" in verdict.failures


def test_declared_split_cannot_contradict_the_case_seed_band():
    verdict = evaluate_readiness(
        with_metrics(episode_seeds=[TRAIN_BAND_SEED], dataset_split="validation")
    )
    assert "matched_validation_split" in verdict.failures

    # With no label to contradict, the case seeds still name the band.
    unlabelled = with_metrics(episode_seeds=[TRAIN_BAND_SEED])
    unlabelled.pop("dataset_split")
    assert resolved_dataset_split(benchmark_inputs(unlabelled)) == "train"


def test_master_sampler_seed_is_not_band_evidence():
    """The report's top-level ``seed`` seeds the sampler; it is not a case seed.

    Reading it as a band both rejects genuine validation runs (a master seed
    belongs to no band) and, worse, accepts train-band cases whenever the master
    seed happens to land in the validation range.
    """

    # A master seed inside the train band does not taint validation cases.
    ok = evaluate_readiness(with_metrics(seed=TRAIN_BAND_SEED))
    assert ok.ready is True

    # A master seed inside the validation band does not launder train cases.
    laundered = with_metrics(seed=VALIDATION_SEED, episode_seeds=[TRAIN_BAND_SEED])
    laundered.pop("dataset_split")
    verdict = evaluate_readiness(laundered)
    assert verdict.ready is False
    assert "matched_validation_split" in verdict.failures


def test_a_declared_split_without_seed_evidence_is_not_believed():
    """A self-declared label is a claim, not a measurement."""

    metrics = exactly_at_threshold_metrics()
    metrics.pop("episode_seeds")
    verdict = evaluate_readiness(metrics)
    assert verdict.ready is False
    assert "matched_validation_split" in verdict.failures
    assert resolved_dataset_split(benchmark_inputs(metrics)) == "unverified_label"


def test_cases_drawn_from_more_than_one_band_are_not_a_matched_set():
    verdict = evaluate_readiness(
        with_metrics(episode_seeds=[VALIDATION_SEED, TRAIN_BAND_SEED])
    )
    assert verdict.ready is False
    assert "matched_validation_split" in verdict.failures


def test_a_summarised_band_never_outranks_the_seeds_it_summarises():
    """A hand-written ``episode_seed_roles`` cannot relabel real case seeds."""

    verdict = evaluate_readiness(
        with_metrics(
            episode_seeds=[TEST_BAND_SEED],
            episode_seed_roles=["validation"],
        )
    )
    assert verdict.ready is False
    assert "matched_validation_split" in verdict.failures


def test_missing_metric_fails_and_is_reported_as_unmeasured():
    metrics = exactly_at_threshold_metrics()
    metrics.pop("universal_transition_success_rate")
    verdict = evaluate_readiness(metrics)
    assert verdict.ready is False
    assert "universal_transition_success" in verdict.failures
    assert "universal_transition_success" in verdict.unmeasured


def test_safety_rates_are_derived_from_episode_counts_when_absent():
    metrics = exactly_at_threshold_metrics()
    metrics.pop("out_of_bounds_rate")
    metrics.pop("collision_episode_rate")
    metrics.update({"out_of_bounds_episodes": 11, "collision_episodes": 224})
    assert evaluate_readiness(metrics).ready is True

    metrics["out_of_bounds_episodes"] = 12
    verdict = evaluate_readiness(metrics)
    assert verdict.ready is False
    assert "out_of_bounds_rate" in verdict.failures


# ---------------------------------------------------------------- override

def test_override_requires_a_written_reason_and_named_criteria():
    with pytest.raises(ValueError):
        ReadinessOverride(reason="", approved_criteria=["completion_12_gate"])
    with pytest.raises(ValueError):
        ReadinessOverride(reason="ok", approved_criteria=["completion_12_gate"])
    with pytest.raises(ValueError):
        ReadinessOverride(reason=REASON, approved_criteria=[])


def test_narrow_miss_on_a_non_safety_criterion_is_overridable_and_recorded():
    metrics = with_metrics(universal_transition_success_rate=0.885)
    assert evaluate_readiness(metrics).ready is False

    override = ReadinessOverride(
        reason=REASON,
        approved_criteria=["universal_transition_success"],
        approver="lead_author",
    )
    verdict = evaluate_readiness(metrics, override=override)
    assert verdict.narrow_misses == ("universal_transition_success",)
    assert verdict.ready is True
    # The exception is never silent: the gate still says it was not met on its
    # own terms, and the reason is carried in the serialised verdict.
    assert verdict.ready_without_override is False
    assert verdict.granted_overrides == ("universal_transition_success",)
    payload = verdict.as_dict()
    assert payload["override"]["reason"] == REASON
    assert payload["override"]["approver"] == "lead_author"
    assert payload["override"]["utc"]
    assert payload["override_decisions"] == [
        {
            "criterion": "universal_transition_success",
            "granted": True,
            "rationale": GRANT_NARROW_MISS,
        }
    ]


def test_safety_criterion_can_never_be_overridden_even_as_a_narrow_miss():
    metrics = with_metrics(collision_episode_rate=0.205)
    baseline = evaluate_readiness(metrics)
    assert "collision_episode_rate" in baseline.narrow_misses

    verdict = evaluate_readiness(
        metrics,
        override=ReadinessOverride(
            reason=REASON, approved_criteria=["collision_episode_rate"]
        ),
    )
    assert verdict.ready is False
    assert "collision_episode_rate" in verdict.failures
    assert verdict.granted_overrides == ()
    assert verdict.as_dict()["override_decisions"] == [
        {
            "criterion": "collision_episode_rate",
            "granted": False,
            "rationale": REFUSAL_SAFETY,
        }
    ]


def test_out_of_bounds_is_a_safety_criterion_too():
    verdict = evaluate_readiness(
        with_metrics(out_of_bounds_rate=0.015),
        override=ReadinessOverride(
            reason=REASON, approved_criteria=["out_of_bounds_rate"]
        ),
    )
    assert verdict.ready is False
    assert verdict.decisions[0].rationale == REFUSAL_SAFETY


def test_large_miss_cannot_be_overridden():
    verdict = evaluate_readiness(
        with_metrics(universal_transition_success_rate=0.60),
        override=ReadinessOverride(
            reason=REASON, approved_criteria=["universal_transition_success"]
        ),
    )
    assert verdict.ready is False
    assert "universal_transition_success" in verdict.failures
    assert verdict.decisions[0].rationale == REFUSAL_TOO_LARGE


def test_unmeasured_criterion_cannot_be_overridden():
    metrics = exactly_at_threshold_metrics()
    metrics.pop("target_switch_rate")
    verdict = evaluate_readiness(
        metrics,
        override=ReadinessOverride(reason=REASON, approved_criteria=["target_switch"]),
    )
    assert verdict.ready is False
    assert verdict.decisions[0].rationale == REFUSAL_NOT_MEASURED


def test_evidence_criteria_cannot_be_overridden():
    """Too little evidence is not a close call; it is no measurement at all."""

    verdict = evaluate_readiness(
        with_metrics(transition_n=499),
        override=ReadinessOverride(reason=REASON, approved_criteria=["transition_cases"]),
    )
    assert verdict.ready is False
    assert verdict.decisions[0].rationale == REFUSAL_EVIDENCE


def test_narrow_margin_follows_measurement_granularity():
    """One missed case out of twenty is narrow; the same gap over 100 is not."""

    coarse = evaluate_readiness(with_length(3, completion_rate=0.90))
    assert coarse.criterion("completion_3_gate").narrow_miss is True

    fine = evaluate_readiness(with_length(3, n=100, completion_rate=0.90))
    assert fine.criterion("completion_3_gate").narrow_miss is False
    assert fine.criterion("completion_3_gate").shortfall == pytest.approx(0.05)


@pytest.mark.parametrize(
    "length,bound", list(READINESS_THRESHOLDS.completion_by_length)
)
def test_one_missed_case_out_of_twenty_is_narrow_for_every_bucket(length, bound):
    """The margin is "one measured case" for *all* buckets, not most of them.

    ``0.75 - 0.70`` is 0.05000000000000004 in binary floating point, so an
    unpadded ``shortfall <= 1/n`` comparison silently refuses a genuine
    one-of-twenty miss on the 8- and 5-gate buckets while forgiving the
    identical miss on the 3-gate bucket.
    """

    one_case_below = (round(bound * 20) - 1) / 20
    verdict = evaluate_readiness(
        with_length(length, n=20, completion_rate=one_case_below)
    )
    criterion = verdict.criterion(f"completion_{length}_gate")
    assert criterion.satisfied is False
    assert criterion.narrow_miss is True

    two_cases_below = (round(bound * 20) - 2) / 20
    if two_cases_below >= 0.0:
        wider = evaluate_readiness(
            with_length(length, n=20, completion_rate=two_cases_below)
        )
        assert wider.criterion(f"completion_{length}_gate").narrow_miss is False


def test_override_budget_limits_how_much_judgment_can_be_applied():
    metrics = with_metrics(
        universal_transition_success_rate=0.885,
        target_switch_rate=0.835,
        new_target_alignment_rate=0.835,
    )
    override = ReadinessOverride(
        reason=REASON,
        approved_criteria=[
            "universal_transition_success", "target_switch", "new_target_alignment",
        ],
    )
    verdict = evaluate_readiness(metrics, override=override)
    assert len(verdict.granted_overrides) == READINESS_THRESHOLDS.max_overridden_criteria
    assert verdict.ready is False
    assert verdict.decisions[-1].rationale == REFUSAL_BUDGET


def test_override_of_an_unknown_criterion_raises():
    """A typo must never be mistaken for a forgiven criterion."""

    with pytest.raises(ValueError, match="unknown readiness criteria"):
        evaluate_readiness(
            exactly_at_threshold_metrics(),
            override=ReadinessOverride(
                reason=REASON, approved_criteria=["collision_rate"]
            ),
        )


# --------------------------------------------------------- difficulty ladder

def test_matched_evidence_must_come_from_the_declared_primary_difficulty():
    """"Matched" means the one regime every candidate shares, not any regime.

    G1 is where all of them trained, so it is the only difficulty at which their
    numbers are comparable; evidence from anywhere else -- harder *or* easier --
    is a different experiment and cannot be read as this one.
    """

    assert evaluate_readiness(with_metrics(difficulty="G1")).ready is True
    for elsewhere in ("G2", "G6"):
        verdict = evaluate_readiness(with_metrics(difficulty=elsewhere))
        assert verdict.ready is False
        assert "matched_difficulty" in verdict.failures
        assert verdict.criterion("matched_difficulty").overridable is False


def test_g1_competence_without_a_ladder_run_is_not_readiness():
    """The central case: a perfect G1 report, never stressed above G1.

    This is exactly the report the campaign actually produces, and it must not
    be a pass -- the sealed circuits turn and climb, and nothing here has ever
    been measured on geometry that does either.
    """

    metrics = without_ladder()
    verdict = evaluate_readiness(metrics)
    assert verdict.ready is False
    # Every G1 bar is met; the only failures are the ladder ones.
    assert set(verdict.failures) == {
        "ladder_mid_rung_coverage",
        "ladder_transition_success",
        "ladder_first_gate_crossing",
    }
    coverage = verdict.criterion("ladder_mid_rung_coverage")
    assert coverage.group == GROUP_EVIDENCE
    assert coverage.overridable is False
    # No ladder at all is unmeasured, which is not the same as a measured zero.
    assert coverage.observed is None

    forgiven = evaluate_readiness(
        metrics,
        override=ReadinessOverride(
            reason=LADDER_REASON, approved_criteria=["ladder_mid_rung_coverage"]
        ),
    )
    assert forgiven.ready is False
    assert "ladder_mid_rung_coverage" in forgiven.failures
    assert forgiven.granted_overrides == ()
    assert forgiven.decisions[0].rationale == REFUSAL_EVIDENCE


def test_a_ladder_that_stops_below_the_mid_rung_is_still_missing_evidence():
    """Testing one rung higher and stopping short of G3 is not stress evidence."""

    verdict = evaluate_readiness(
        with_ladder(ladder_rung("G1", 0.92, 0.99), ladder_rung("G2", 0.88, 0.96))
    )
    assert verdict.ready is False
    coverage = verdict.criterion("ladder_mid_rung_coverage")
    assert coverage.observed == 0.0  # a ladder ran; it covered no mid rung
    assert coverage.overridable is False
    # Strong G2 numbers do not stand in for the rung that was never run.
    assert verdict.criterion("ladder_transition_success").observed is None
    assert "ladder_transition_success" in verdict.unmeasured


def test_a_claimed_rung_with_no_cases_is_not_a_rung_that_was_run():
    """A fabricated block must fail the same way an absent one does."""

    fabricated = with_ladder(
        ladder_rung("G1", 0.92, 0.99), ladder_rung("G3", 1.00, 1.00, n=0)
    )
    verdict = evaluate_readiness(fabricated)
    assert verdict.ready is False
    assert verdict.criterion("ladder_mid_rung_coverage").observed == 0.0
    # The perfect rates attached to the unrun rung are not read off it either.
    assert verdict.criterion("ladder_transition_success").observed is None

    # A token sample is no better: the bar has to be measurable at the rung.
    thin = with_ladder(
        ladder_rung("G1", 0.92, 0.99), ladder_rung("G3", 1.00, 1.00, n=3)
    )
    thin_verdict = evaluate_readiness(thin)
    assert thin_verdict.criterion("ladder_mid_rung_coverage").observed == 0.0
    assert thin_verdict.ready is False


def test_collapse_at_the_mid_rung_is_fatal_and_cannot_be_overridden():
    metrics = with_ladder(
        ladder_rung("G1", 0.92, 0.99), ladder_rung("G3", 0.05, 0.11)
    )
    verdict = evaluate_readiness(
        metrics,
        override=ReadinessOverride(
            reason=LADDER_REASON,
            approved_criteria=[
                "ladder_transition_success", "ladder_first_gate_crossing",
            ],
        ),
    )
    assert verdict.ready is False
    assert "ladder_transition_success" in verdict.failures
    assert verdict.granted_overrides == ()
    assert [item.rationale for item in verdict.decisions] == [
        REFUSAL_TOO_LARGE, REFUSAL_TOO_LARGE,
    ]
    criterion = verdict.criterion("ladder_transition_success")
    # Competence, so forgivable in principle -- but not by this much.
    assert criterion.group == GROUP_COMPETENCE
    assert criterion.overridable is True
    assert criterion.narrow_miss is False
    assert criterion.shortfall == pytest.approx(0.45)


def test_a_narrow_miss_at_the_mid_rung_is_a_judgment_call():
    """One case of sixty short at G3 is a close call, and stays a recorded one."""

    metrics = with_ladder(
        ladder_rung("G1", 0.92, 0.99), ladder_rung("G3", 0.49, 0.80)
    )
    assert evaluate_readiness(metrics).ready is False
    assert evaluate_readiness(metrics).narrow_misses == ("ladder_transition_success",)

    verdict = evaluate_readiness(
        metrics,
        override=ReadinessOverride(
            reason=LADDER_REASON,
            approved_criteria=["ladder_transition_success"],
            approver="lead_author",
        ),
    )
    assert verdict.ready is True
    assert verdict.ready_without_override is False
    assert verdict.granted_overrides == ("ladder_transition_success",)
    assert verdict.decisions[0].rationale == GRANT_NARROW_MISS


def test_the_easiest_qualifying_rung_carries_the_bar():
    """Measuring more of the ladder must never make the gate harder to pass.

    Otherwise the honest thing -- running the whole curve, as the ladder tool
    does by default -- would be the expensive thing, and reporting would shrink
    to the single rung most likely to clear.
    """

    generous = evaluate_readiness(
        with_ladder(ladder_rung("G3", 0.55, 0.85), ladder_rung("G6", 0.05, 0.10))
    )
    assert generous.ready is True
    assert generous.criterion("ladder_transition_success").observed == 0.55

    # With G3 never run, the only mid rung on offer has to carry the bar itself.
    lonely = evaluate_readiness(with_ladder(ladder_rung("G6", 0.05, 0.10)))
    assert lonely.ready is False
    assert "ladder_transition_success" in lonely.failures


def test_a_rung_reported_twice_is_taken_at_its_worst_row():
    """A duplicate must not be able to improve on itself."""

    verdict = evaluate_readiness(
        with_ladder(ladder_rung("G3", 0.95, 0.99), ladder_rung("G3", 0.20, 0.45))
    )
    assert verdict.criterion("ladder_transition_success").observed == 0.20
    assert verdict.ready is False


def test_the_ladder_is_found_wherever_the_caller_nested_it():
    """One tool output shape, several plausible places to hang it."""

    base = without_ladder()
    ladder = copy.deepcopy(DEFAULT_LADDER)
    assert evaluate_readiness(base).ready is False

    bundled = dict(base, evidence={"stress": {"difficulty_ladder": ladder}})
    assert evaluate_readiness(bundled).ready is True

    inside_metrics = {
        "schema_version": "universal_transition_benchmark_v1",
        "ladder": ladder,
        "metrics": base,
    }
    assert evaluate_readiness(inside_metrics).ready is True


def test_difficulty_ladder_summary_reads_the_tool_output_shape():
    """``run_difficulty_ladder`` writes this; the gate must read it unaided."""

    summary = difficulty_ladder_summary(copy.deepcopy(DEFAULT_LADDER))
    assert summary["rungs_claimed"] == ["G1", "G3", "G6"]
    assert summary["mid_ladder_rungs"] == ["G3", "G6"]
    assert summary["required_min_rung"] == "G3"
    assert summary["governing_rung"] == "G3"
    assert summary["governing_rung_n"] == LADDER_RUNG_CASES
    assert summary["success"] == 0.50
    assert summary["first_gate"] == 0.80
    assert summary["rungs"][1]["success_ci"] == pytest.approx([0.37, 0.63])
    json.dumps(summary)

    # No ladder at all is None; a ladder that claims nothing is not.
    assert difficulty_ladder_summary(without_ladder()) is None
    empty = difficulty_ladder_summary({"rungs": []})
    assert empty["mid_ladder_rungs"] == []
    assert empty["governing_rung"] is None


def test_benchmark_inputs_normalises_the_ladder_into_the_evidence():
    inputs = benchmark_inputs(exactly_at_threshold_metrics())
    rungs = inputs[LADDER_KEY]["rungs"]
    assert [row["difficulty"] for row in rungs] == ["G1", "G3", "G6"]
    assert rungs[1]["n"] == LADDER_RUNG_CASES
    assert rungs[1]["success"] == 0.50
    assert rungs[1]["first_gate"] == 0.80
    # The normalised form is itself acceptable evidence, which is what lets
    # ``freeze_policy`` re-judge the summary it is about to record.
    assert benchmark_inputs(inputs)[LADDER_KEY] == inputs[LADDER_KEY]
    assert evaluate_readiness(inputs).ready is True

    # An unparseable block is dropped rather than left to look like evidence.
    junk = benchmark_inputs(exactly_at_threshold_metrics(difficulty_ladder="G3: fine"))
    assert LADDER_KEY not in junk


def test_gate_manifest_states_the_ladder_half_of_the_bar():
    """A report has to be able to quote the gate it was judged against."""

    manifest = gate_manifest()
    assert manifest["thresholds"]["primary_difficulty"] == "G1"
    assert manifest["thresholds"]["ladder_min_rung"] == "G3"
    assert manifest["thresholds"]["min_ladder_transition_success"] == 0.50
    assert manifest["thresholds_sha256"] == READINESS_THRESHOLDS.sha256()
    assert GROUP_EVIDENCE in manifest["non_overridable_groups"]
    json.dumps(manifest)


def test_a_g1_only_policy_cannot_be_frozen(tmp_path):
    """The gate is the door: no ladder evidence, no freeze, no sealed circuits."""

    with pytest.raises(ReadinessNotMet, match="ladder_mid_rung_coverage"):
        frozen_policy(tmp_path, metrics=without_ladder())
    assert not freeze_record_path(tmp_path).exists()


def test_freeze_record_pins_the_ladder_evidence(tmp_path):
    _, record, path = frozen_policy(tmp_path)
    rungs = record["validation_benchmark"][LADDER_KEY]["rungs"]
    assert [row["difficulty"] for row in rungs] == ["G1", "G3", "G6"]
    mid = rungs[1]
    assert mid["n"] == LADDER_RUNG_CASES
    assert mid["success"] == 0.50
    assert mid["success_ci"] == pytest.approx([0.37, 0.63])
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["validation_benchmark"][LADDER_KEY] == record["validation_benchmark"][LADDER_KEY]
    assert on_disk["freeze_id"] == freeze_record_id(on_disk)


# ------------------------------------------------------------ freeze record

def test_freeze_policy_writes_the_record_and_refuses_a_second_freeze(tmp_path):
    checkpoint, record, path = frozen_policy(tmp_path)

    assert path.is_file()
    assert record["schema_version"] == FREEZE_RECORD_VERSION
    assert record["policy"]["checkpoint"] == checkpoint.name
    assert record["policy"]["checkpoint_sha256"]
    assert record["policy"]["training_transitions"] == 8_400_000
    assert record["git_sha"] == GIT_SHA
    assert record["contracts"]["reward"] == REWARD_CONTRACT_VERSION
    assert record["contracts"]["observation"] == OBS_ENCODING_VERSION_LOCAL_TRANSITION
    assert record["contracts"]["action"] == ACTION_CONTRACT_VERSION
    assert record["curriculum_version"] == CURRICULUM_VERSION
    assert record["validation_benchmark"]["dataset_split"] == "validation"
    assert record["validation_benchmark"]["transition_n"] == 1000
    assert record["readiness_verdict"]["ready"] is True
    assert record["final_circuit_protocol"]["trials_per_circuit"] == TRIALS_PER_CIRCUIT
    assert record["utc"]
    assert record["freeze_id"] == freeze_record_id(record)

    with pytest.raises(PolicyAlreadyFrozen):
        frozen_policy(tmp_path)
    # The refused second freeze left the original record byte-identical.
    assert json.loads(path.read_text(encoding="utf-8"))["freeze_id"] == record["freeze_id"]


def test_freeze_policy_refuses_an_unready_verdict(tmp_path):
    metrics = with_metrics(collision_episode_rate=0.5)
    with pytest.raises(ReadinessNotMet):
        frozen_policy(tmp_path, metrics=metrics)
    assert not freeze_record_path(tmp_path).exists()


def test_freeze_policy_requires_every_contract_version(tmp_path):
    checkpoint = tmp_path / "ppo_8400000_steps.zip"
    checkpoint.write_bytes(b"frozen policy weights")
    metrics = exactly_at_threshold_metrics()
    with pytest.raises(ValueError, match="observation contract"):
        freeze_policy(
            checkpoint,
            run_dir=tmp_path,
            metrics=metrics,
            verdict=evaluate_readiness(metrics),
            git_sha=GIT_SHA,
            contracts={"reward": REWARD_CONTRACT_VERSION, "action": ACTION_CONTRACT_VERSION},
            training_transitions=8_400_000,
        )


def relaxed_thresholds() -> ReadinessThresholds:
    """Bars low enough that a policy which never crosses a gate would pass."""

    return ReadinessThresholds(
        min_universal_transition_success=0.0,
        min_first_gate_crossing=0.0,
        min_target_switch=0.0,
        min_new_target_alignment=0.0,
        completion_by_length=tuple((length, 0.0) for length in (3, 5, 8, 12, 17, 22)),
        min_ladder_transition_success=0.0,
        min_ladder_first_gate_crossing=0.0,
        max_out_of_bounds_rate=1.0,
        max_collision_episode_rate=1.0,
        min_transition_cases=0,
        min_full_cases_per_length=0,
        min_ladder_rung_cases=0,
    )


def failing_metrics() -> dict:
    metrics = with_metrics(
        universal_transition_success_rate=0.10,
        first_gate_crossing_rate=0.20,
        target_switch_rate=0.10,
        new_target_alignment_rate=0.10,
        out_of_bounds_rate=0.90,
        collision_episode_rate=0.95,
        transition_n=1,
        difficulty_ladder=ladder_report(
            ladder_rung("G1", 0.10, 0.20), ladder_rung("G3", 0.00, 0.02)
        ),
    )
    metrics["full_sequence_success_by_length"] = {
        str(length): {"n": 1, "completion_rate": 0.0}
        for length in (3, 5, 8, 12, 17, 22)
    }
    return metrics


def test_freeze_policy_refuses_home_made_thresholds(tmp_path):
    """Moving the bar must not be a way of passing the gate.

    ``evaluate_readiness`` accepts thresholds so the gate is testable; that is
    exactly why the freeze -- the moment of commitment -- has to insist on the
    pre-registered ones.  Otherwise a garbage policy unlocks the sealed circuits
    with no reason, no override budget and no safety protection.
    """

    metrics = failing_metrics()
    verdict = evaluate_readiness(metrics, relaxed_thresholds())
    assert verdict.ready is True  # ...against bars nobody pre-registered

    checkpoint = tmp_path / "ppo_8400000_steps.zip"
    checkpoint.write_bytes(b"garbage policy")
    with pytest.raises(ReadinessNotMet, match="pre-registered"):
        freeze_policy(
            checkpoint, run_dir=tmp_path, metrics=metrics, verdict=verdict,
            git_sha=GIT_SHA, contracts=CONTRACTS, training_transitions=8_400_000,
        )
    assert not freeze_record_path(tmp_path).exists()


def test_a_record_frozen_against_other_thresholds_keeps_the_circuits_sealed(tmp_path):
    """Defence in depth: relaxing the bars after freezing re-locks the door."""

    _, record, path = frozen_policy(tmp_path)
    assert assert_final_circuits_unlocked(path)["freeze_id"] == record["freeze_id"]

    moved = copy.deepcopy(record)
    moved["readiness_thresholds_sha256"] = ReadinessThresholds(
        min_universal_transition_success=0.10
    ).sha256()
    moved["freeze_id"] = freeze_record_id(moved)
    moved_path = tmp_path / "moved" / "final_policy_freeze.json"
    moved_path.parent.mkdir()
    moved_path.write_text(json.dumps(moved), encoding="utf-8")

    with pytest.raises(FinalCircuitsLocked, match="pre-registered"):
        assert_final_circuits_unlocked(moved_path, verify_checkpoint=False)


def test_freeze_policy_refuses_a_verdict_from_different_metrics(tmp_path):
    """The record pins these metrics as the evidence, so they must support it."""

    checkpoint = tmp_path / "ppo_8400000_steps.zip"
    checkpoint.write_bytes(b"frozen policy weights")
    ready_elsewhere = evaluate_readiness(exactly_at_threshold_metrics())
    assert ready_elsewhere.ready is True

    with pytest.raises(ReadinessNotMet, match="reproduce"):
        freeze_policy(
            checkpoint, run_dir=tmp_path, metrics=failing_metrics(),
            verdict=ready_elsewhere, git_sha=GIT_SHA, contracts=CONTRACTS,
            training_transitions=8_400_000,
        )
    assert not freeze_record_path(tmp_path).exists()


def test_freeze_policy_pins_the_case_seed_band_as_evidence(tmp_path):
    _, record, _ = frozen_policy(tmp_path)
    benchmark = record["validation_benchmark"]
    assert benchmark["dataset_split"] == "validation"
    assert benchmark["episode_seed_roles"] == ["validation"]
    assert benchmark["seed"] == MASTER_SEED


def test_freeze_record_stores_the_override_reason(tmp_path):
    metrics = with_metrics(universal_transition_success_rate=0.885)
    verdict = evaluate_readiness(
        metrics,
        override=ReadinessOverride(
            reason=REASON, approved_criteria=["universal_transition_success"]
        ),
    )
    _, record, path = frozen_policy(tmp_path, metrics=metrics, verdict=verdict)
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["readiness_verdict"]["override"]["reason"] == REASON
    assert on_disk["readiness_verdict"]["ready_without_override"] is False
    assert record["freeze_id"] == freeze_record_id(on_disk)


# ------------------------------------------ failed-readiness holdout amendment

def test_exploratory_decision_is_once_only_and_unlocks_failed_policy(tmp_path):
    checkpoint, record, path, metrics = exploratory_policy(tmp_path)

    assert path.is_file()
    assert record["schema_version"] == EXPLORATORY_HOLDOUT_DECISION_VERSION
    assert record["readiness_verdict"]["ready"] is False
    assert record["readiness_verdict"]["failures"] == [
        "universal_transition_success"
    ]
    assert record["policy"]["checkpoint_sha256"]
    assert record["policy"]["immutable"] is True
    decision = record["exploratory_decision"]
    assert decision["training_permanently_closed"] is True
    assert decision["final_policy_immutable"] is True
    assert decision["exploratory_holdout_authorized"] is True
    assert decision["executions_permitted"] == 1
    assert decision["post_holdout_training_permitted"] is False
    assert assert_final_circuits_unlocked(path)["freeze_id"] == record["freeze_id"]

    with pytest.raises(PolicyAlreadyFrozen):
        record_exploratory_holdout_decision(
            checkpoint,
            run_dir=tmp_path,
            metrics=metrics,
            verdict=evaluate_readiness(metrics),
            git_sha=GIT_SHA,
            contracts=CONTRACTS,
            training_transitions=929_792,
        )


def test_exploratory_decision_refuses_a_ready_policy(tmp_path):
    checkpoint = tmp_path / "ready.zip"
    checkpoint.write_bytes(b"ready")
    metrics = exactly_at_threshold_metrics()
    with pytest.raises(ProtocolViolation, match="only valid.*failed readiness"):
        record_exploratory_holdout_decision(
            checkpoint,
            run_dir=tmp_path,
            metrics=metrics,
            verdict=evaluate_readiness(metrics),
            git_sha=GIT_SHA,
            contracts=CONTRACTS,
            training_transitions=929_792,
        )


def test_rehashed_exploratory_decision_cannot_reopen_training(tmp_path):
    _, record, path, _ = exploratory_policy(tmp_path)
    edited = copy.deepcopy(record)
    edited["exploratory_decision"]["training_permanently_closed"] = False
    edited["freeze_id"] = freeze_record_id(edited)
    path.write_text(json.dumps(edited), encoding="utf-8")

    with pytest.raises(FinalCircuitsLocked, match="permanent training closure"):
        assert_final_circuits_unlocked(path)


def test_rehashed_exploratory_decision_must_reproduce_its_failure(tmp_path):
    _, record, path, _ = exploratory_policy(tmp_path)
    edited = copy.deepcopy(record)
    edited["validation_benchmark"]["universal_transition_success_rate"] = 0.90
    edited["freeze_id"] = freeze_record_id(edited)
    path.write_text(json.dumps(edited), encoding="utf-8")

    with pytest.raises(FinalCircuitsLocked, match="not reproducible"):
        assert_final_circuits_unlocked(path)


def test_exploratory_decision_locks_if_checkpoint_changes(tmp_path):
    checkpoint, _, path, _ = exploratory_policy(tmp_path)
    checkpoint.write_bytes(b"different policy")

    with pytest.raises(FinalCircuitsLocked, match="sha256"):
        assert_final_circuits_unlocked(path)


# ------------------------------------------------------------- access guard

def test_assert_final_circuits_unlocked_raises_without_a_freeze_record(tmp_path):
    """The pre-freeze peek guard: no record, no access."""

    with pytest.raises(FinalCircuitsLocked, match="no freeze record"):
        assert_final_circuits_unlocked(freeze_record_path(tmp_path))


def test_assert_final_circuits_unlocked_raises_when_the_verdict_is_not_ready(tmp_path):
    _, record, _ = frozen_policy(tmp_path)
    unready = copy.deepcopy(record)
    unready["readiness_verdict"]["ready"] = False
    unready["readiness_verdict"]["failures"] = ["completion_12_gate"]
    unready["freeze_id"] = freeze_record_id(unready)
    path = tmp_path / "unready" / "final_policy_freeze.json"
    path.parent.mkdir()
    path.write_text(json.dumps(unready), encoding="utf-8")

    with pytest.raises(FinalCircuitsLocked, match="readiness gate"):
        assert_final_circuits_unlocked(path)


def test_assert_final_circuits_unlocked_passes_with_a_valid_ready_record(tmp_path):
    _, record, path = frozen_policy(tmp_path)
    unlocked = assert_final_circuits_unlocked(path)
    assert unlocked["freeze_id"] == record["freeze_id"]


def test_a_hand_edited_ready_flag_does_not_unlock_the_circuits(tmp_path):
    """Flipping ``ready`` in the JSON breaks the record's content hash."""

    _, record, _ = frozen_policy(tmp_path)
    unready = copy.deepcopy(record)
    unready["readiness_verdict"]["ready"] = False
    unready["freeze_id"] = freeze_record_id(unready)
    path = tmp_path / "edited" / "final_policy_freeze.json"
    path.parent.mkdir()
    path.write_text(json.dumps(unready), encoding="utf-8")

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["readiness_verdict"]["ready"] = True
    path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(FinalCircuitsLocked, match="content hash"):
        assert_final_circuits_unlocked(path)


def test_replacing_the_checkpoint_after_freezing_locks_the_circuits(tmp_path):
    checkpoint, _, path = frozen_policy(tmp_path)
    checkpoint.write_bytes(b"a different policy entirely")

    with pytest.raises(FinalCircuitsLocked, match="sha256"):
        assert_final_circuits_unlocked(path)


def test_a_missing_frozen_checkpoint_locks_the_circuits(tmp_path):
    checkpoint, _, path = frozen_policy(tmp_path)
    checkpoint.unlink()

    with pytest.raises(FinalCircuitsLocked, match="missing"):
        assert_final_circuits_unlocked(path)


# ------------------------------------------------------------ holdout ledger

def test_final_circuit_results_cannot_be_recorded_before_freezing(tmp_path):
    with pytest.raises(FinalCircuitsLocked):
        record_final_circuit_evaluation(
            tmp_path / "ledger.jsonl",
            freeze_record_path=freeze_record_path(tmp_path),
            circuit="horseshoe_bay",
            trial_seeds=FINAL_CIRCUIT_TRIAL_SEEDS,
            metrics={"circuit_completed": True},
        )
    assert not (tmp_path / "ledger.jsonl").exists()


def test_recorded_evaluations_form_a_verifiable_append_only_chain(tmp_path):
    _, record, path = frozen_policy(tmp_path)
    ledger = tmp_path / "final_circuit_evaluations.jsonl"

    first = record_final_circuit_evaluation(
        ledger, freeze_record_path=path, circuit="horseshoe_bay",
        trial_seeds=FINAL_CIRCUIT_TRIAL_SEEDS, metrics={"circuit_completed": True},
    )
    second = record_final_circuit_evaluation(
        ledger, freeze_record_path=path, circuit="vertical_serpent",
        trial_seeds=FINAL_CIRCUIT_TRIAL_SEEDS, metrics={"circuit_completed": False},
    )
    assert first["entry_index"] == 0 and first["previous_sha256"] is None
    assert second["previous_sha256"] == first["entry_sha256"]
    assert second["checkpoint_sha256"] == record["policy"]["checkpoint_sha256"]
    assert len(verify_ledger_chain(ledger)) == 2


def test_editing_or_dropping_a_ledger_entry_is_detected(tmp_path):
    _, _, path = frozen_policy(tmp_path)
    ledger = tmp_path / "final_circuit_evaluations.jsonl"
    for circuit in ("horseshoe_bay", "vertical_serpent"):
        record_final_circuit_evaluation(
            ledger, freeze_record_path=path, circuit=circuit,
            trial_seeds=FINAL_CIRCUIT_TRIAL_SEEDS, metrics={"circuit_completed": False},
        )
    lines = ledger.read_text(encoding="utf-8").splitlines()

    edited = json.loads(lines[0])
    edited["metrics"]["circuit_completed"] = True
    ledger.write_text("\n".join([json.dumps(edited), lines[1]]) + "\n", encoding="utf-8")
    with pytest.raises(LedgerTampered):
        verify_ledger_chain(ledger)

    ledger.write_text(lines[1] + "\n", encoding="utf-8")
    with pytest.raises(LedgerTampered):
        verify_ledger_chain(ledger)


def test_recorded_trials_must_use_the_preregistered_seeds_and_circuits(tmp_path):
    _, _, path = frozen_policy(tmp_path)
    ledger = tmp_path / "final_circuit_evaluations.jsonl"

    with pytest.raises(ProtocolViolation):
        record_final_circuit_evaluation(
            ledger, freeze_record_path=path, circuit="horseshoe_bay",
            trial_seeds=[123], metrics={},
        )
    with pytest.raises(ValueError):
        record_final_circuit_evaluation(
            ledger, freeze_record_path=path, circuit="procedural_course_42",
            trial_seeds=FINAL_CIRCUIT_TRIAL_SEEDS, metrics={},
        )
    assert not ledger.exists()


def test_assert_no_tuning_after_holdout_raises_once_an_entry_exists(tmp_path):
    _, _, path = frozen_policy(tmp_path)
    ledger = tmp_path / "final_circuit_evaluations.jsonl"
    # Before the holdout is opened, training is unconstrained.
    assert_no_tuning_after_holdout(ledger, GIT_SHA)

    record_final_circuit_evaluation(
        ledger, freeze_record_path=path, circuit="mixed_endurance",
        trial_seeds=FINAL_CIRCUIT_TRIAL_SEEDS, metrics={"circuit_completed": False},
    )

    with pytest.raises(HoldoutContaminated):
        assert_no_tuning_after_holdout(ledger, "f" * 40)
    # Even at the frozen commit: re-training the frozen policy is the forbidden
    # act, not merely changing the code.
    with pytest.raises(HoldoutContaminated):
        assert_no_tuning_after_holdout(ledger, GIT_SHA)


def test_reproducing_the_identical_policy_from_the_identical_commit_is_allowed(tmp_path):
    _, record, path = frozen_policy(tmp_path)
    ledger = tmp_path / "final_circuit_evaluations.jsonl"
    record_final_circuit_evaluation(
        ledger, freeze_record_path=path, circuit="mixed_endurance",
        trial_seeds=FINAL_CIRCUIT_TRIAL_SEEDS, metrics={"circuit_completed": False},
    )
    sha = record["policy"]["checkpoint_sha256"]

    assert_no_tuning_after_holdout(
        ledger, GIT_SHA, policy_sha256=sha, allow_reproduction=True
    )
    with pytest.raises(HoldoutContaminated):
        assert_no_tuning_after_holdout(
            ledger, GIT_SHA, policy_sha256="0" * 64, allow_reproduction=True
        )


# ----------------------------------------------------------------- protocol

def test_protocol_never_permits_varying_circuit_geometry():
    protocol = final_circuit_protocol()
    for name in (
        "circuit_geometry", "geometry", "gate_positions", "gate_count",
        "gate_order", "course_scale", "track_layout",
    ):
        assert protocol_permits_variation(name) is False
    assert protocol_permits_variation("evaluation_seed") is True
    assert protocol_permits_variation("Initial Yaw Error Deg") is True

    forbidden = ("geometry", "gate", "course", "track", "circuit", "layout")
    assert not any(
        token in entry.lower()
        for entry in protocol["may_vary_between_trials"]
        for token in forbidden
    )
    assert set(protocol["may_vary_between_trials"]) == set(PROTOCOL_MAY_VARY)
    for pinned in ("circuit_geometry", "gate_positions", "gate_count", "gate_order"):
        assert pinned in PROTOCOL_MUST_NOT_VARY

    with pytest.raises(ProtocolViolation):
        assert_protocol_variations_allowed(["evaluation_seed", "gate_positions"])


def test_protocol_is_fully_preregistered_and_serialisable():
    protocol = final_circuit_protocol()
    assert protocol["trials_per_circuit"] == TRIALS_PER_CIRCUIT
    assert len(protocol["circuits"]) == 3
    assert protocol["total_trials"] == 3 * TRIALS_PER_CIRCUIT
    assert len(protocol["trial_seeds"]) == TRIALS_PER_CIRCUIT
    assert protocol["reporting"]["post_hoc_trial_exclusion_permitted"] is False
    assert protocol["reporting"]["executions_permitted"] == 1
    assert "circuit_completed" in protocol["metrics"]
    json.dumps(protocol)


def test_trial_seeds_belong_to_no_procedural_band():
    """Trials must not consume the train, validation or test seed bands."""

    assert all(role_of_seed(seed) is None for seed in FINAL_CIRCUIT_TRIAL_SEEDS)
    assert len(set(FINAL_CIRCUIT_TRIAL_SEEDS)) == TRIALS_PER_CIRCUIT
