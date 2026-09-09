"""Tests for the paired validation benchmark.

The module exists because unpaired n=112 evaluations could not tell two nearly
identical policies apart, so the tests assert the two properties that fix that:
every candidate really does see the same VALIDATION cases, and the paired
statistics separate a real difference from noise where the marginal intervals
cannot.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from marine_race_arena.learning import rl_matched_benchmark as mb
from marine_race_arena.learning.rl_holdout_policy import role_of_seed, seed_group


# --------------------------------------------------------------------- fixtures

def _by_length(rate: float, lengths=(3, 6, 12, 20), n: int = 5) -> dict:
    return {str(length): {"n": n, "completion_rate": rate} for length in lengths}


def _metrics(
    *,
    n_eval: int = 1012,
    success: float = 0.60,
    sequence_rate: float = 0.50,
    by_length: dict | None = None,
    collisions: int = 50,
    missed: int = 5,
    out_of_bounds: int = 0,
    wrong_direction: int = 2,
    jerk: float = 0.010,
    nontrivial_action_fraction: float = 0.95,
    mean_absolute_action: float = 0.085,
) -> dict:
    """A plausible complete metrics block, shaped like a real benchmark report."""

    return {
        "n_eval": n_eval,
        "transition_n": 1000,
        "universal_transition_success_rate": success,
        "first_gate_crossing_rate": 0.94,
        "target_switch_rate": 0.86,
        "new_target_alignment_rate": 0.71,
        "new_target_range_decrease_rate": 0.74,
        "missed_gate_dnf": missed,
        "collision_episodes": collisions,
        "out_of_bounds_episodes": out_of_bounds,
        "wrong_direction_events": wrong_direction,
        "mean_action_jerk": jerk,
        "nontrivial_action_fraction": nontrivial_action_fraction,
        "mean_absolute_action": mean_absolute_action,
        "full_sequence_success_by_length": (
            by_length if by_length is not None else _by_length(sequence_rate)
        ),
    }


def _checkpoints(tmp_path: Path, *names: str) -> dict:
    paths = {}
    for name in names:
        path = tmp_path / f"{name}.zip"
        path.write_bytes(name.encode("utf-8") * 32)
        paths[name] = str(path)
    return paths


def _fake_report(checkpoint: str, seed: int, plan: dict, *, success: bool = True) -> dict:
    """A report shaped like one the real evaluator writes.

    It carries the provenance block the resume check reads, because the point of
    that check is to notice when a file on disk describes *different* work.
    """

    episodes = [
        {
            "seed": seed + index,
            "episode_type": "transition_focus",
            "gate_count": 2,
            "geometry_group": f"{plan['difficulty']}:transition_focus:2:aligned",
            "universal_transition_success": success,
        }
        for index in range(int(plan["transition_cases"]))
    ]
    for length in (3, 6):
        for index in range(int(plan["full_cases_per_length"])):
            episodes.append({
                "seed": seed + 500 + index,
                "episode_type": "full_sequence",
                "gate_count": length,
                "geometry_group": (
                    f"{plan['difficulty']}:full_sequence:{length}:varied"
                ),
                "full_sequence_completion": success,
            })
    return {
        "schema_version": "universal_transition_benchmark_v1",
        "seed": int(seed),
        "difficulty": plan["difficulty"],
        "dataset_split": plan["dataset_split"],
        "transition_cases": int(plan["transition_cases"]),
        "full_cases_per_length": int(plan["full_cases_per_length"]),
        "checkpoint": str(checkpoint),
        "algorithm": "ppo",
        "metrics": _metrics(),
        "episodes": episodes,
    }


# ------------------------------------------------------------------- the pairing

def test_every_planned_seed_is_validation_and_shared_by_all_candidates(tmp_path):
    paths = _checkpoints(tmp_path, "alpha", "beta", "gamma")
    plan = mb.plan_matched_benchmark(
        paths, transition_cases=500, sequences_per_length=5, seeds=3
    )

    band = seed_group("validation")
    assert plan["seeds"], "a plan without seeds cannot be paired"
    for seed in plan["seeds"]:
        assert role_of_seed(seed) == "validation"
        assert band.start <= seed < band.end

    assigned = [tuple(entry["seeds"]) for entry in plan["candidates"]]
    assert len(assigned) == 3
    # The pairing property: not merely "all validation" but literally the same list.
    assert len(set(assigned)) == 1
    assert list(assigned[0]) == plan["seeds"]
    # The workload is identical too, otherwise the cases would differ.
    assert plan["transition_cases"] == 500
    assert plan["full_cases_per_length"] == 5
    assert plan["paired"] is True
    json.dumps(plan)


def test_train_or_test_seeds_can_never_enter_a_plan(tmp_path, monkeypatch):
    paths = _checkpoints(tmp_path, "alpha")
    train_seed = seed_group("train").start + 7
    test_seed = seed_group("test").start + 7

    for leaked in (train_seed, test_seed, 0, -1):
        monkeypatch.setattr(
            mb,
            "evaluation_seeds",
            lambda role, count, *, offset=0, _leaked=leaked: [_leaked] * int(count),
        )
        with pytest.raises(ValueError, match="validation"):
            mb.plan_matched_benchmark(
                paths, transition_cases=10, sequences_per_length=1
            )


def test_large_validation_offset_wraps_inside_the_validation_band(tmp_path):
    paths = _checkpoints(tmp_path, "alpha")
    band = seed_group("validation")
    plan = mb.plan_matched_benchmark(
        paths,
        transition_cases=10,
        sequences_per_length=1,
        validation_offset=band.size * 3 + 11,
        seeds=4,
    )
    assert all(role_of_seed(seed) == "validation" for seed in plan["seeds"])
    assert len(set(plan["seeds"])) == 4


def test_planned_cases_really_land_in_the_validation_band(tmp_path):
    """The benchmark seed is only a sampler seed; the band comes from the split.

    Declaring ``dataset_split`` proves nothing on its own, so this materialises
    the cases the evaluator will run and checks the seeds they actually carry.
    """

    paths = _checkpoints(tmp_path, "alpha")
    plan = mb.plan_matched_benchmark(
        paths, transition_cases=25, sequences_per_length=1
    )
    assert plan["dataset_split"] == "validation"
    checked = mb.assert_plan_uses_validation(plan)

    band = seed_group("validation")
    entry = checked["seeds"][str(plan["seeds"][0])]
    assert entry["cases"] > 25
    low, high = entry["episode_seed_range"]
    assert band.start <= low <= high < band.end
    # The episode seeds are nowhere near the sampler seed, which is the point:
    # a validation-looking benchmark seed would not by itself keep them in band.
    assert entry["distinct_episode_seeds"] > 1


def test_assert_plan_uses_validation_catches_a_plan_aimed_at_the_train_band(tmp_path):
    paths = _checkpoints(tmp_path, "alpha")
    plan = mb.plan_matched_benchmark(
        paths, transition_cases=10, sequences_per_length=1
    )
    plan["dataset_split"] = "train"
    with pytest.raises(ValueError, match="validation"):
        mb.assert_plan_uses_validation(plan)


def test_plan_rejects_an_empty_or_degenerate_workload(tmp_path):
    paths = _checkpoints(tmp_path, "alpha")
    with pytest.raises(ValueError):
        mb.plan_matched_benchmark(paths, transition_cases=0, sequences_per_length=5)
    with pytest.raises(ValueError):
        mb.plan_matched_benchmark(paths, transition_cases=10, sequences_per_length=-1)
    with pytest.raises(ValueError):
        mb.plan_matched_benchmark({}, transition_cases=10, sequences_per_length=1)


# ------------------------------------------------------------------- intervals

def test_wilson_interval_contains_the_estimate_and_narrows_with_n():
    widths = []
    for scale in (1, 10, 100):
        low, high = mb.wilson_interval(30 * scale, 100 * scale)
        assert low < 0.30 < high
        assert 0.0 <= low < high <= 1.0
        widths.append(high - low)
    assert widths[0] > widths[1] > widths[2]
    # Roughly 1/sqrt(n): a hundredfold sample should be about ten times tighter.
    assert 8.0 < widths[0] / widths[2] < 12.0


def test_wilson_interval_matches_the_published_value_and_tracks_confidence():
    # Hand-computed with z_0.975 = 1.9599639845: a wrong quantile (a 90% z, say)
    # still produces a plausible-looking interval, so pin the numbers.
    low, high = mb.wilson_interval(30, 100)
    assert low == pytest.approx(0.21894885, abs=1e-7)
    assert high == pytest.approx(0.39584855, abs=1e-7)
    # Independently evaluated from the closed form at p=0.05, n=20, where the
    # normal approximation would run below zero and Wilson must not.
    assert mb.wilson_interval(1, 20) == pytest.approx(
        (0.00888145, 0.23613119), abs=1e-7
    )

    tight = mb.wilson_interval(30, 100, confidence=0.80)
    wide = mb.wilson_interval(30, 100, confidence=0.99)
    assert (tight[1] - tight[0]) < (high - low) < (wide[1] - wide[0])
    with pytest.raises(ValueError):
        mb.wilson_interval(30, 100, confidence=1.0)


def test_wilson_interval_stays_inside_the_unit_interval_at_the_extremes():
    for n in (20, 100, 1000):
        low, high = mb.wilson_interval(0, n)
        assert low == pytest.approx(0.0, abs=1e-12) and 0.0 < high < 0.20
        low, high = mb.wilson_interval(n, n)
        assert high == pytest.approx(1.0, abs=1e-12) and 0.80 < low < 1.0
    with pytest.raises(ValueError):
        mb.wilson_interval(5, 0)
    with pytest.raises(ValueError):
        mb.wilson_interval(101, 100)


def test_sign_test_p_value_is_the_exact_binomial_on_both_code_paths():
    assert mb._binomial_two_sided_p(0, 0) == 1.0
    assert mb._binomial_two_sided_p(8, 0) == pytest.approx(2 * 0.5 ** 8)
    assert mb._binomial_two_sided_p(3, 1) == pytest.approx(0.625)
    assert mb._binomial_two_sided_p(50, 50) == pytest.approx(1.0)
    # Above the exact-integer cut-over the same binomial mass is summed in log
    # space; it must still agree with the exact rational value.
    trials, extreme = 4200, 2000
    exact = min(1.0, 2.0 * (
        sum(math.comb(trials, index) for index in range(extreme + 1)) / (1 << trials)
    ))
    assert 0.0 < exact < 1.0
    assert mb._binomial_two_sided_p(extreme, trials - extreme) == pytest.approx(
        exact, rel=1e-9
    )


def test_bootstrap_interval_contains_the_estimate_narrows_and_is_deterministic():
    widths = []
    for n in (50, 500, 5000):
        low, high = mb.bootstrap_interval(n // 2, n, resamples=4000, seed=11)
        assert low < 0.50 < high
        widths.append(high - low)
    assert widths[0] > widths[1] > widths[2]

    first = mb.bootstrap_interval(120, 400, resamples=4000, seed=7)
    again = mb.bootstrap_interval(120, 400, resamples=4000, seed=7)
    assert first == again
    # The two independent methods must agree to within their own resolution.
    wilson = mb.wilson_interval(120, 400)
    assert abs(first[0] - wilson[0]) < 0.03 and abs(first[1] - wilson[1]) < 0.03


# --------------------------------------------------------- paired discrimination

def test_paired_difference_detects_a_real_difference():
    # 30 cases only A solves, 60 both solve, 10 neither.
    a = [1] * 30 + [1] * 60 + [0] * 10
    b = [0] * 30 + [1] * 60 + [0] * 10
    result = mb.paired_difference(a, b, resamples=4000, seed=3)

    assert result["n_pairs"] == 100
    assert result["mean_difference"] == pytest.approx(0.30)
    assert result["n_a_better"] == 30 and result["n_b_better"] == 0
    assert result["difference_ci"][0] > 0.0
    assert result["sign_test_p_value"] < 1e-6
    assert result["equivalent_within_noise"] is False
    assert mb.equivalent_within_noise(result) is False


def test_paired_difference_reports_equivalence_when_only_noise_separates_them():
    # Identical policies apart from 15 coin flips in each direction.
    a = [1] * 15 + [0] * 15 + [1] * 100 + [0] * 70
    b = [0] * 15 + [1] * 15 + [1] * 100 + [0] * 70
    result = mb.paired_difference(a, b, resamples=4000, seed=3)

    assert result["mean_difference"] == pytest.approx(0.0)
    assert result["n_discordant"] == 30
    assert result["sign_test_p_value"] == pytest.approx(1.0)
    assert result["difference_ci"][0] < 0.0 < result["difference_ci"][1]
    assert result["equivalent_within_noise"] is True


def test_pairing_separates_candidates_whose_marginal_intervals_overlap():
    """The reason the benchmark is paired at all."""

    a = [1] * 8 + [1] * 200 + [0] * 192
    b = [0] * 8 + [1] * 200 + [0] * 192
    marginal_a = mb.wilson_interval(208, 400)
    marginal_b = mb.wilson_interval(200, 400)
    # Unpaired, the two candidates are indistinguishable.
    assert marginal_a[0] < marginal_b[1] and marginal_b[0] < marginal_a[1]

    paired = mb.paired_difference(a, b, resamples=4000, seed=5)
    assert paired["mean_difference"] == pytest.approx(0.02)
    assert paired["difference_ci"][0] > 0.0
    assert paired["sign_test_p_value"] < 0.05
    assert paired["equivalent_within_noise"] is False
    # The paired interval is far tighter than either marginal one.
    paired_width = paired["difference_ci"][1] - paired["difference_ci"][0]
    assert paired_width < (marginal_a[1] - marginal_a[0]) / 2.0


def test_paired_difference_refuses_unpairable_inputs():
    with pytest.raises(ValueError):
        mb.paired_difference([1, 0, 1], [1, 0])
    with pytest.raises(ValueError):
        mb.paired_difference([], [])
    with pytest.raises(ValueError):
        mb.paired_difference({"case_a": 1}, {"case_b": 0})


def test_paired_difference_aligns_mappings_on_shared_cases():
    a = {"c0": 1, "c1": 1, "c2": 0, "only_a": 1}
    b = {"c0": 0, "c1": 1, "c2": 0, "only_b": 0}
    result = mb.paired_difference(a, b, resamples=1000, seed=1)
    assert result["n_pairs"] == 3
    assert result["aligned_on_cases"] == 3
    assert result["mean_difference"] == pytest.approx(1.0 / 3.0)


def test_equivalent_within_noise_reads_the_interval():
    assert mb.equivalent_within_noise((-0.02, 0.05)) is True
    assert mb.equivalent_within_noise((0.01, 0.05)) is False
    assert mb.equivalent_within_noise((-0.05, -0.01)) is False
    assert mb.equivalent_within_noise((0.0, 0.0)) is True
    with pytest.raises(ValueError):
        mb.equivalent_within_noise((0.5, 0.1))


# --------------------------------------------------------------------- the rule

def test_selection_rule_is_serialisable_and_data_independent():
    payload = mb.SELECTION_RULE.as_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["criteria"][0] == "sequence_reliability"
    assert payload["criteria"][-1] == "smoothness"
    assert payload["step_0_reject"]["min_nontrivial_action_fraction"] == 0.25
    # Fingerprint depends only on the rule, so a plan can log it before running.
    assert mb.SELECTION_RULE.fingerprint() == mb.SELECTION_RULE.fingerprint()
    assert mb.selection_rule_from_mapping(payload) == mb.SELECTION_RULE


def test_a_plan_detects_a_selection_rule_edited_after_the_fact(tmp_path):
    paths = _checkpoints(tmp_path, "alpha")
    plan = mb.plan_matched_benchmark(
        paths, transition_cases=10, sequences_per_length=1
    )
    assert mb.rule_from_plan(plan) == mb.SELECTION_RULE

    edited = json.loads(json.dumps(plan))
    edited["selection_rule"]["criteria"] = ["smoothness"] + list(
        mb.RANKING_CRITERIA[:-1]
    )
    with pytest.raises(ValueError, match="fingerprint"):
        mb.rule_from_plan(edited)

    # Deleting the fingerprint is the same edit, so it must not be a way out.
    stripped = json.loads(json.dumps(edited))
    stripped.pop("selection_rule_fingerprint")
    with pytest.raises(ValueError, match="fingerprint"):
        mb.rule_from_plan(stripped)


def test_health_gates_refuse_to_pass_a_report_with_no_episode_count():
    """A counter with no denominator would disable three gates in silence."""

    metrics = {
        "nontrivial_action_fraction": 0.9,
        "mean_absolute_action": 0.1,
        "collision_episodes": 900,
        "out_of_bounds_episodes": 900,
        "wrong_direction_events": 900,
    }
    with pytest.raises(ValueError, match="n_eval"):
        mb.health_gate_failures(metrics)
    # With the denominator present the same numbers are rejected, not ignored.
    metrics["n_eval"] = 1000
    gates = {item["gate"] for item in mb.health_gate_failures(metrics)}
    assert gates == {"collision", "out_of_bounds", "wrong_direction"}


# ------------------------------------------------------------------- the ranking

def test_rank_candidates_rejects_an_inactive_policy_despite_high_scores():
    results = {
        "inactive": {"metrics": _metrics(
            success=0.99,
            by_length=_by_length(1.0),
            collisions=0,
            missed=0,
            wrong_direction=0,
            jerk=0.0001,
            nontrivial_action_fraction=0.0,
            mean_absolute_action=0.0034,
        )},
        "working": {"metrics": _metrics(success=0.55, sequence_rate=0.45)},
    }
    ranking = mb.rank_candidates(results, rule=mb.SELECTION_RULE)
    by_name = {row["candidate"]: row for row in ranking}

    assert by_name["inactive"]["rejected"] is True
    assert by_name["inactive"]["rank"] is None
    gates = {item["gate"] for item in by_name["inactive"]["gate_failures"]}
    assert gates == {"nontrivial_action", "mean_absolute_action"}
    assert "nontrivial_action_fraction" in by_name["inactive"]["rejection_reason"]
    # It dominates every ranking criterion and still must not win.
    assert ranking[0]["candidate"] == "working"
    assert by_name["working"]["rank"] == 1
    assert ranking[-1]["candidate"] == "inactive"


def test_rank_candidates_rejects_unsafe_candidates_on_every_hard_gate():
    results = {
        "wanderer": {"metrics": _metrics(out_of_bounds=200)},
        "backwards": {"metrics": _metrics(wrong_direction=300)},
        "crasher": {"metrics": _metrics(collisions=900)},
        "healthy": {"metrics": _metrics()},
    }
    ranking = mb.rank_candidates(results, rule=mb.SELECTION_RULE)
    by_name = {row["candidate"]: row for row in ranking}

    assert by_name["wanderer"]["gate_failures"][0]["gate"] == "out_of_bounds"
    assert by_name["backwards"]["gate_failures"][0]["gate"] == "wrong_direction"
    assert by_name["crasher"]["gate_failures"][0]["gate"] == "collision"
    assert by_name["healthy"]["rejected"] is False
    assert [row["candidate"] for row in ranking][0] == "healthy"
    assert all(row["rank"] is None for row in ranking[1:])


def test_rank_candidates_prefers_sequence_reliability_over_two_gate_success():
    results = {
        "chainer": {"metrics": _metrics(success=0.80, sequence_rate=0.70)},
        "sprinter": {"metrics": _metrics(success=0.82, sequence_rate=0.40)},
    }
    ranking = mb.rank_candidates(results, rule=mb.SELECTION_RULE)

    assert [row["candidate"] for row in ranking] == ["chainer", "sprinter"]
    assert ranking[0]["scores"]["sequence_reliability"] > (
        ranking[1]["scores"]["sequence_reliability"]
    )
    # The sprinter really is ahead on the second criterion; it just does not matter.
    assert ranking[1]["scores"]["universal_transition_success"] > (
        ranking[0]["scores"]["universal_transition_success"]
    )
    assert ranking[0]["rank"] == 1


def test_sequence_reliability_weights_longer_sequences_more_heavily():
    long_good = _by_length(0.0, lengths=(3,)) | _by_length(0.8, lengths=(20,))
    short_good = _by_length(0.8, lengths=(3,)) | _by_length(0.0, lengths=(20,))
    assert mb.sequence_reliability_score({
        "full_sequence_success_by_length": long_good
    }) > mb.sequence_reliability_score({
        "full_sequence_success_by_length": short_good
    })
    # Lengths with no measured episodes must not drag the score toward zero.
    assert mb.sequence_reliability_score({
        "full_sequence_success_by_length": {
            "3": {"n": 5, "completion_rate": 0.6},
            "20": {"n": 0, "completion_rate": None},
        }
    }) == pytest.approx(0.6)
    assert mb.sequence_reliability_score({}) is None


def test_smoothness_only_breaks_ties():
    results = {
        "smooth": {"metrics": _metrics(success=0.70, jerk=0.001)},
        "capable": {"metrics": _metrics(success=0.75, jerk=0.900)},
    }
    ranking = mb.rank_candidates(results, rule=mb.SELECTION_RULE)
    assert [row["candidate"] for row in ranking] == ["capable", "smooth"]

    tied = {
        "smooth": {"metrics": _metrics(success=0.70, jerk=0.001)},
        "jerky": {"metrics": _metrics(success=0.70, jerk=0.900)},
    }
    assert [row["candidate"] for row in mb.rank_candidates(tied, rule=mb.SELECTION_RULE)] == [
        "smooth", "jerky"
    ]


# ------------------------------------------------------------------- final pick

def _tied_outcomes() -> tuple:
    """400 matched cases where two policies differ by 10 coin flips each way."""

    keys = [f"5000000:transition_focus:2:{index}" for index in range(400)]
    a = {key: 1 for key in keys[:220]}
    a.update({key: 0 for key in keys[220:]})
    b = dict(a)
    for key in keys[:10]:      # A wins these
        b[key] = 0
    for key in keys[220:230]:  # B wins these
        b[key] = 1
    return a, b


def test_select_final_parent_breaks_a_statistical_tie_on_safety():
    a_outcomes, b_outcomes = _tied_outcomes()
    results = {
        "leader": {
            "checkpoint": "runs/leader.zip",
            "sha256": "a" * 64,
            # Marginally ahead on the primary criterion, but far more collisions.
            "metrics": _metrics(sequence_rate=0.62, collisions=200),
            "case_outcomes": a_outcomes,
        },
        "safer": {
            "checkpoint": "runs/safer.zip",
            "sha256": "b" * 64,
            "metrics": _metrics(sequence_rate=0.60, collisions=20),
            "case_outcomes": b_outcomes,
        },
    }
    # Without the tie test the pre-registered order puts "leader" first.
    assert [row["candidate"] for row in mb.rank_candidates(results, rule=mb.SELECTION_RULE)] == [
        "leader", "safer"
    ]

    selection = mb.select_final_parent(results, mb.SELECTION_RULE)
    assert selection["paired_comparison"]["equivalent_within_noise"] is True
    assert selection["selected"] == "safer"
    assert selection["checkpoint"] == "runs/safer.zip"
    assert selection["sha256"] == "b" * 64
    assert "safety" in selection["reason"]
    assert "statistically equivalent" in selection["reason"]
    assert selection["ranking"][0]["candidate"] == "leader"
    json.dumps(selection)


def test_select_final_parent_keeps_the_leader_when_the_difference_is_real():
    keys = [f"5000000:transition_focus:2:{index}" for index in range(400)]
    a = {key: int(index < 300) for index, key in enumerate(keys)}
    b = {key: int(index < 150) for index, key in enumerate(keys)}
    results = {
        "leader": {
            "checkpoint": "runs/leader.zip",
            "sha256": "a" * 64,
            "metrics": _metrics(sequence_rate=0.80, collisions=200),
            "case_outcomes": a,
        },
        "safer": {
            "checkpoint": "runs/safer.zip",
            "sha256": "b" * 64,
            "metrics": _metrics(sequence_rate=0.40, collisions=20),
            "case_outcomes": b,
        },
    }
    selection = mb.select_final_parent(results, mb.SELECTION_RULE)
    assert selection["paired_comparison"]["equivalent_within_noise"] is False
    assert selection["selected"] == "leader"
    assert "ranked first" in selection["reason"]


def test_select_final_parent_refuses_to_promote_when_all_candidates_are_rejected():
    results = {
        "dead": {"metrics": _metrics(nontrivial_action_fraction=0.0)},
        "frozen": {"metrics": _metrics(mean_absolute_action=0.001)},
    }
    selection = mb.select_final_parent(results, mb.SELECTION_RULE)
    assert selection["selected"] is None
    assert selection["checkpoint"] is None
    assert "health gate" in selection["reason"]


# ------------------------------------------------------------------ running it

def test_run_matched_benchmark_reuses_existing_results(tmp_path):
    paths = _checkpoints(tmp_path, "alpha", "beta")
    plan = mb.plan_matched_benchmark(
        paths, transition_cases=4, sequences_per_length=1
    )
    output_root = tmp_path / "matched"
    entries = {entry["name"]: entry for entry in plan["candidates"]}
    seed = plan["seeds"][0]

    # A completed alpha run from an interrupted sweep.
    existing = output_root / "alpha" / f"seed_{seed}" / "evaluation.json"
    existing.parent.mkdir(parents=True)
    existing.write_text(
        json.dumps(_fake_report(entries["alpha"]["checkpoint"], seed, plan)),
        encoding="utf-8",
    )

    calls = []

    def runner(checkpoint, **kwargs):
        calls.append((checkpoint, kwargs["seed"]))
        return _fake_report(checkpoint, kwargs["seed"], plan)

    report = mb.run_matched_benchmark(
        plan, output_root=output_root, parallel_workers=2, runner=runner
    )

    assert calls == [(entries["beta"]["checkpoint"], seed)]
    assert report["results"]["alpha"]["reused"] is True
    assert report["results"]["alpha"]["reused_seeds"] == [seed]
    assert report["results"]["beta"]["evaluated_seeds"] == [seed]

    # A second sweep must not repeat a single episode.
    again = mb.run_matched_benchmark(
        plan, output_root=output_root, parallel_workers=2, runner=runner
    )
    assert len(calls) == 1
    assert again["results"]["beta"]["reused"] is True


def test_run_matched_benchmark_does_not_reuse_a_report_for_another_policy(tmp_path):
    paths = _checkpoints(tmp_path, "alpha")
    plan = mb.plan_matched_benchmark(
        paths, transition_cases=4, sequences_per_length=1
    )
    output_root = tmp_path / "matched"
    seed = plan["seeds"][0]
    stale = output_root / "alpha" / f"seed_{seed}" / "evaluation.json"
    stale.parent.mkdir(parents=True)
    stale.write_text(
        json.dumps(_fake_report("runs/some_other_checkpoint.zip", seed, plan)),
        encoding="utf-8",
    )

    calls = []

    def runner(checkpoint, **kwargs):
        calls.append(checkpoint)
        return _fake_report(checkpoint, kwargs["seed"], plan)

    mb.run_matched_benchmark(
        plan, output_root=output_root, parallel_workers=1, runner=runner
    )
    assert calls == [plan["candidates"][0]["checkpoint"]]


def test_run_matched_benchmark_evaluates_every_candidate_on_the_planned_work(tmp_path):
    """The arguments are the pairing: same seed, same split, same workload.

    ``dataset_split`` is the one that decides which seed band the courses come
    from, so dropping it would silently benchmark on TRAIN geometry while every
    other assertion in this file still passed.
    """

    paths = _checkpoints(tmp_path, "alpha", "beta")
    plan = mb.plan_matched_benchmark(
        paths, transition_cases=4, sequences_per_length=1
    )
    seen = []

    def runner(checkpoint, **kwargs):
        seen.append((checkpoint, dict(kwargs)))
        return _fake_report(checkpoint, kwargs["seed"], plan)

    mb.run_matched_benchmark(
        plan, output_root=tmp_path / "matched", parallel_workers=3, runner=runner
    )

    assert len(seen) == 2
    workloads = {
        (kwargs["seed"], kwargs["dataset_split"], kwargs["difficulty"],
         kwargs["transition_cases"], kwargs["full_cases_per_length"])
        for _, kwargs in seen
    }
    assert workloads == {(plan["seeds"][0], "validation", "G6", 4, 1)}
    # Each policy needs its own output directory or the evaluator's own resume
    # logic would hand one candidate's episodes to the other.
    directories = {str(kwargs["output_dir"]) for _, kwargs in seen}
    assert len(directories) == 2


def test_run_matched_benchmark_resumes_a_runner_that_omits_its_provenance(tmp_path):
    """Resume must work for any runner, not only the production evaluator."""

    paths = _checkpoints(tmp_path, "alpha")
    plan = mb.plan_matched_benchmark(
        paths, transition_cases=3, sequences_per_length=1
    )
    output_root = tmp_path / "matched"
    calls = []

    def bare_runner(checkpoint, **kwargs):
        calls.append(checkpoint)
        report = _fake_report(checkpoint, kwargs["seed"], plan)
        for key in ("seed", "difficulty", "dataset_split", "transition_cases",
                    "full_cases_per_length", "checkpoint", "algorithm"):
            report.pop(key)
        return report

    mb.run_matched_benchmark(
        plan, output_root=output_root, parallel_workers=1, runner=bare_runner
    )
    again = mb.run_matched_benchmark(
        plan, output_root=output_root, parallel_workers=1, runner=bare_runner
    )
    assert len(calls) == 1
    assert again["results"]["alpha"]["reused"] is True


def test_run_matched_benchmark_refuses_a_report_describing_different_work(tmp_path):
    paths = _checkpoints(tmp_path, "alpha")
    plan = mb.plan_matched_benchmark(
        paths, transition_cases=3, sequences_per_length=1
    )

    def lying_runner(checkpoint, **kwargs):
        report = _fake_report(checkpoint, kwargs["seed"], plan)
        report["dataset_split"] = "train"  # ran the training band regardless
        return report

    with pytest.raises(ValueError, match="dataset_split"):
        mb.run_matched_benchmark(
            plan, output_root=tmp_path / "matched", parallel_workers=1,
            runner=lying_runner,
        )


@pytest.mark.parametrize(
    "field, value",
    [("difficulty", "G1"), ("dataset_split", "train"), ("algorithm", "sac")],
)
def test_run_matched_benchmark_does_not_reuse_a_report_for_other_work(
    tmp_path, field, value
):
    """A stale report from a different curriculum or band is not the same run.

    Reusing a G1 report inside a G6 sweep would compare one candidate on easy
    geometry against another on hard geometry -- the unpaired comparison this
    module exists to remove -- and reusing a train-band report would select a
    checkpoint on its own training courses.
    """

    paths = _checkpoints(tmp_path, "alpha")
    plan = mb.plan_matched_benchmark(
        paths, transition_cases=4, sequences_per_length=1
    )
    output_root = tmp_path / "matched"
    seed = plan["seeds"][0]
    stale_path = output_root / "alpha" / f"seed_{seed}" / "evaluation.json"
    stale_path.parent.mkdir(parents=True)
    stale = _fake_report(plan["candidates"][0]["checkpoint"], seed, plan)
    stale[field] = value
    stale_path.write_text(json.dumps(stale), encoding="utf-8")

    calls = []

    def runner(checkpoint, **kwargs):
        calls.append(checkpoint)
        return _fake_report(checkpoint, kwargs["seed"], plan)

    report = mb.run_matched_benchmark(
        plan, output_root=output_root, parallel_workers=1, runner=runner
    )
    assert calls == [plan["candidates"][0]["checkpoint"]]
    assert report["results"]["alpha"]["evaluated_seeds"] == [seed]
    assert json.loads(stale_path.read_text(encoding="utf-8"))[field] != value


def test_run_matched_benchmark_writes_a_serialisable_report(tmp_path):
    paths = _checkpoints(tmp_path, "alpha", "beta")
    plan = mb.plan_matched_benchmark(
        paths, transition_cases=4, sequences_per_length=1
    )
    output_root = tmp_path / "matched"

    def runner(checkpoint, **kwargs):
        return _fake_report(checkpoint, kwargs["seed"], plan)

    report = mb.run_matched_benchmark(
        plan, output_root=output_root, parallel_workers=1, runner=runner
    )
    written = json.loads(
        (output_root / "matched_benchmark.json").read_text(encoding="utf-8")
    )
    assert written["plan"]["selection_rule_fingerprint"] == (
        mb.SELECTION_RULE.fingerprint()
    )
    assert set(written["results"]) == {"alpha", "beta"}
    assert written["selection"]["selected"] in {"alpha", "beta"}
    assert report["results"]["alpha"]["n_cases"] == 6  # 4 transitions + 2 sequences
    # Both candidates were scored on exactly the same cases.
    assert set(report["results"]["alpha"]["case_outcomes"]) == set(
        report["results"]["beta"]["case_outcomes"]
    )


def test_run_matched_benchmark_refuses_an_unpaired_plan(tmp_path):
    paths = _checkpoints(tmp_path, "alpha", "beta")
    plan = mb.plan_matched_benchmark(
        paths, transition_cases=4, sequences_per_length=1
    )
    plan["candidates"][1]["seeds"] = [plan["seeds"][0] + 1]

    with pytest.raises(ValueError, match="paired"):
        mb.run_matched_benchmark(
            plan,
            output_root=tmp_path / "matched",
            parallel_workers=1,
            runner=lambda checkpoint, **kwargs: _fake_report(
                checkpoint, kwargs["seed"], plan
            ),
        )


def test_case_outcomes_key_on_case_identity_not_row_order():
    report = {
        "episodes": [
            {"seed": 5000000, "episode_type": "transition_focus",
             "gate_count": 2, "universal_transition_success": True},
            {"seed": 5000001, "episode_type": "transition_focus",
             "gate_count": 2, "universal_transition_success": False},
            {"seed": 5000002, "episode_type": "full_sequence",
             "gate_count": 6, "full_sequence_completion": True},
        ]
    }
    shuffled = {"episodes": list(reversed(report["episodes"]))}
    assert mb.case_outcomes(report) == mb.case_outcomes(shuffled)
    assert sum(mb.case_outcomes(report).values()) == 2
    assert mb.case_outcomes({"episodes": []}) == {}


def test_case_outcomes_keeps_cases_that_share_an_episode_seed():
    """Episode seeds are drawn with replacement, so identities repeat.

    Around one case in a hundred collides at benchmark scale; collapsing those
    onto one key would silently throw away paired units, so each occurrence has
    to survive as its own case.
    """

    rows = [
        {"seed": 5000042, "episode_type": "transition_focus", "gate_count": 2,
         "geometry_group": "G6:transition_focus:2:aligned",
         "universal_transition_success": True},
        {"seed": 5000042, "episode_type": "transition_focus", "gate_count": 2,
         "geometry_group": "G6:transition_focus:2:aligned",
         "universal_transition_success": False},
    ]
    outcomes = mb.case_outcomes({"seed": 5000000, "episodes": rows})
    assert len(outcomes) == 2
    assert sorted(outcomes.values()) == [0, 1]


def test_case_outcomes_namespace_keeps_two_benchmark_seeds_apart():
    """Merging seeds must add evidence, not overwrite it."""

    row = {"seed": 5000042, "episode_type": "transition_focus", "gate_count": 2,
           "geometry_group": "G6:transition_focus:2:aligned",
           "universal_transition_success": True}
    first = mb.case_outcomes({"seed": 5000000, "episodes": [row]})
    second = mb.case_outcomes({"seed": 5000001, "episodes": [dict(row)]})
    merged = {**first, **second}
    assert len(merged) == 2


def test_the_default_runner_accepts_the_call_run_matched_benchmark_actually_makes():
    """The production path, which every injected-runner test bypasses.

    ``run_matched_benchmark`` passes the checkpoint positionally.  A
    ``_default_runner(**kwargs)`` signature satisfies the whole suite and then
    raises TypeError on the first real evaluation, which is exactly what happened
    when the benchmark was launched, so the binding is pinned here rather than
    discovered again after an engine start.
    """

    import inspect

    signature = inspect.signature(mb._default_runner)
    parameters = list(signature.parameters.values())
    assert parameters, "_default_runner must accept the checkpoint positionally"
    first = parameters[0]
    assert first.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ), f"first parameter {first.name!r} cannot receive a positional checkpoint"

    # And the keywords the benchmark sends must all be accepted by the evaluator.
    from marine_race_arena.learning.transition_evaluation import (
        evaluate_checkpoint_universal_transition_benchmark as evaluator,
    )

    accepted = set(inspect.signature(evaluator).parameters)
    sent = {
        "output_dir", "seed", "difficulty", "transition_cases",
        "full_cases_per_length", "adapter", "max_steps", "parallel_workers",
        "algorithm", "dataset_split",
    }
    assert sent <= accepted, f"evaluator rejects {sorted(sent - accepted)}"


def test_the_default_runner_forwards_every_argument_to_the_evaluator(monkeypatch):
    """A forwarding bug would silently evaluate the wrong thing."""

    seen = {}

    def _fake(checkpoint, **kwargs):
        seen["checkpoint"] = checkpoint
        seen.update(kwargs)
        return {"metrics": {}, "episodes": []}

    import marine_race_arena.learning.transition_evaluation as te

    monkeypatch.setattr(
        te, "evaluate_checkpoint_universal_transition_benchmark", _fake
    )
    mb._default_runner(
        "policy.zip", output_dir="out", seed=5_000_000, difficulty="G1",
        transition_cases=7, full_cases_per_length=2, adapter="holoocean",
        max_steps=3600, parallel_workers=3, algorithm="ppo",
        dataset_split="validation",
    )
    assert seen["checkpoint"] == "policy.zip"
    assert seen["dataset_split"] == "validation"
    assert seen["difficulty"] == "G1"
    assert seen["seed"] == 5_000_000
