"""Competence gate, safety ranking and anti-inactivity metric tests.

The reported scratch and warm A/B metrics are used verbatim: the ranking fix is
only meaningful if it rejects the exact inactive policy that the previous
ranking selected.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from marine_race_arena.learning.transition_evaluation import (
    NONTRIVIAL_ACTION_THRESHOLD,
    aggregate_transition_benchmark,
    transition_checkpoint_rank_key,
)
from marine_race_arena.learning.transition_selection import (
    CLASSIFICATION_COMPETENT,
    CLASSIFICATION_DEGENERATE_INACTIVE,
    CLASSIFICATION_INSUFFICIENT_COMPETENCE,
    CLASSIFICATION_INSUFFICIENT_EVIDENCE,
    DEFAULT_COMPETENCE_THRESHOLDS,
    baseline_collapse_criteria,
    evaluate_competence_gate,
    has_collapsed_below_baseline,
    safety_rank_key,
    select_policy,
    thresholds_from_mapping,
)


# Metrics as reported by the completed definitive A/B arms over the same
# 1,012-episode unseen holdout (1,000 transitions + 12 full sequences).
REPORTED_SCRATCH = {
    "n_eval": 1012,
    "transition_n": 1000,
    "universal_transition_success_rate": 0.0,
    "first_gate_crossing_rate": 0.015,
    "target_switch_rate": 0.002,
    "new_target_alignment_rate": 0.0,
    "new_target_range_decrease_rate": 0.0,
    "previous_gate_returns": 0,
    "missed_gate_dnf": 9,
    "collision_episodes": 103,
    "collision_events": 539,
    "out_of_bounds_episodes": 0,
    "wrong_direction_events": 4,
    "acquisition_timeouts": 6,
    "mean_acquisition_time_s": None,
    "mean_action_jerk": 0.005676414031620554,
    "long_sequence_completion_score": 0.0,
    "safety_clean": False,
    "completed_gate_count": 15,
    "mean_completed_gates_per_episode": 15 / 1012,
    "fraction_of_episodes_reaching_first_gate": 15 / 1012,
    "mean_distance_travelled_m": 0.4,
    "mean_absolute_action": 0.0034,
    "nontrivial_action_fraction": 0.0,
}

REPORTED_WARM = {
    "n_eval": 1012,
    "transition_n": 1000,
    "universal_transition_success_rate": 0.314,
    "first_gate_crossing_rate": 0.912,
    "target_switch_rate": 0.819,
    "new_target_alignment_rate": 0.426,
    "new_target_range_decrease_rate": 0.815,
    "previous_gate_returns": 1,
    "missed_gate_dnf": 78,
    "collision_episodes": 417,
    "collision_events": 2591,
    "out_of_bounds_episodes": 0,
    "wrong_direction_events": 51,
    "acquisition_timeouts": 436,
    "mean_acquisition_time_s": 1.3711267605633803,
    "mean_action_jerk": 0.1098715662055336,
    "long_sequence_completion_score": 0.25,
    "safety_clean": False,
    "completed_gate_count": 946,
    "mean_completed_gates_per_episode": 946 / 1012,
    "fraction_of_episodes_reaching_first_gate": 924 / 1012,
    "mean_distance_travelled_m": 9.2,
    "mean_absolute_action": 0.0848,
    "nontrivial_action_fraction": 0.9789,
}


def test_reported_scratch_metrics_are_rejected_as_degenerate_inactive():
    verdict = evaluate_competence_gate(REPORTED_SCRATCH)
    assert not verdict.passed
    assert verdict.classification == CLASSIFICATION_DEGENERATE_INACTIVE
    assert "first_gate_crossing" in verdict.failed_criteria
    assert "nontrivial_action" in verdict.failed_criteria


def test_reported_warm_metrics_pass_the_competence_gate():
    verdict = evaluate_competence_gate(REPORTED_WARM)
    assert verdict.passed
    assert verdict.classification == CLASSIFICATION_COMPETENT
    assert verdict.failed_criteria == ()


def test_stationary_policy_with_zero_success_cannot_win():
    stationary = dict(REPORTED_SCRATCH)
    # Give it a perfect safety record: still no task competence at all.
    stationary.update({
        "collision_episodes": 0, "collision_events": 0, "collision_entries": 0,
        "missed_gate_dnf": 0, "wrong_direction_events": 0,
        "previous_gate_returns": 0, "acquisition_timeouts": 0,
        "safety_clean": True,
    })
    report = select_policy({"scratch": stationary, "warm": REPORTED_WARM})
    assert report["selected"] == "warm"
    assert report["rejected_candidates"]["scratch"] == (
        CLASSIFICATION_DEGENERATE_INACTIVE
    )
    assert transition_checkpoint_rank_key(stationary) < transition_checkpoint_rank_key(
        REPORTED_WARM
    )


def test_fewer_safety_events_do_not_compensate_for_zero_task_competence():
    flawless_but_idle = dict(REPORTED_SCRATCH)
    flawless_but_idle.update({
        "collision_episodes": 0, "collision_events": 0, "missed_gate_dnf": 0,
        "wrong_direction_events": 0, "previous_gate_returns": 0,
        "acquisition_timeouts": 0, "safety_clean": True,
    })
    # Safety alone would rank the idle policy first; the gate flag must dominate.
    assert safety_rank_key(flawless_but_idle) > safety_rank_key(REPORTED_WARM)
    assert transition_checkpoint_rank_key(
        flawless_but_idle
    ) < transition_checkpoint_rank_key(REPORTED_WARM)


def test_moving_policy_without_task_success_is_insufficient_competence():
    thrashing = dict(REPORTED_SCRATCH)
    thrashing.update({
        "mean_distance_travelled_m": 25.0,
        "mean_absolute_action": 0.4,
        "nontrivial_action_fraction": 0.95,
    })
    verdict = evaluate_competence_gate(thrashing)
    assert not verdict.passed
    assert verdict.classification == CLASSIFICATION_INSUFFICIENT_COMPETENCE


def test_safety_ranking_still_orders_two_competent_policies():
    safer = dict(REPORTED_WARM)
    safer.update({"collision_episodes": 200, "collision_entries": 260})
    riskier = dict(REPORTED_WARM)
    riskier.update({"collision_episodes": 400, "collision_entries": 900})
    for metrics in (safer, riskier):
        assert evaluate_competence_gate(metrics).passed
    report = select_policy({"riskier": riskier, "safer": safer})
    assert report["selected"] == "safer"
    assert report["competent_candidates"] == ["safer", "riskier"]


def test_collision_episodes_outrank_entries_so_one_long_contact_is_not_many():
    one_long_contact = dict(REPORTED_WARM)
    one_long_contact.update({
        "collision_episodes": 10, "collision_entries": 10,
        "collision_contact_frames": 5000,
    })
    many_distinct_impacts = dict(REPORTED_WARM)
    many_distinct_impacts.update({
        "collision_episodes": 40, "collision_entries": 90,
        "collision_contact_frames": 120,
    })
    report = select_policy({
        "one_long_contact": one_long_contact,
        "many_distinct_impacts": many_distinct_impacts,
    })
    assert report["selected"] == "one_long_contact"


def test_ranking_is_deterministic_and_order_independent():
    first = select_policy({"scratch": REPORTED_SCRATCH, "warm": REPORTED_WARM})
    second = select_policy({"warm": REPORTED_WARM, "scratch": REPORTED_SCRATCH})
    assert first == second
    assert [safety_rank_key(REPORTED_WARM) for _ in range(5)].count(
        safety_rank_key(REPORTED_WARM)
    ) == 5
    tie_a = dict(REPORTED_WARM)
    tie_b = dict(REPORTED_WARM)
    assert select_policy({"b": tie_b, "a": tie_a})["selected"] == "a"


def test_no_candidate_passing_the_gate_selects_nothing():
    report = select_policy({"scratch": REPORTED_SCRATCH})
    assert report["selected"] is None
    assert report["competent_candidates"] == []


def test_insufficient_evaluation_evidence_is_classified_separately():
    tiny = dict(REPORTED_WARM)
    tiny.update({"n_eval": 12, "transition_n": 10})
    verdict = evaluate_competence_gate(tiny)
    assert not verdict.passed
    assert verdict.classification == CLASSIFICATION_INSUFFICIENT_EVIDENCE


def test_absent_metrics_are_reported_but_never_fail_the_gate():
    partial = {
        key: value for key, value in REPORTED_WARM.items()
        if key not in {
            "mean_distance_travelled_m", "mean_absolute_action",
            "nontrivial_action_fraction",
        }
    }
    verdict = evaluate_competence_gate(partial)
    assert verdict.passed
    assert set(verdict.unavailable_metrics) == {
        "mean_distance_travelled_m", "mean_absolute_action",
        "nontrivial_action_fraction",
    }


def test_baseline_collapse_uses_the_mandated_minimums():
    assert not has_collapsed_below_baseline(REPORTED_WARM)
    collapsed = dict(REPORTED_WARM)
    collapsed.update({
        "first_gate_crossing_rate": 0.5,
        "target_switch_rate": 0.4,
        "universal_transition_success_rate": 0.05,
    })
    assert has_collapsed_below_baseline(collapsed)
    assert set(baseline_collapse_criteria(collapsed)) == {
        "first_gate_crossing", "target_switch", "universal_transition_success",
    }


def test_threshold_overrides_reject_unknown_keys():
    tuned = thresholds_from_mapping({"min_universal_transition_success_rate": 0.3})
    assert tuned.min_universal_transition_success_rate == pytest.approx(0.3)
    assert tuned.min_first_gate_crossing_rate == pytest.approx(
        DEFAULT_COMPETENCE_THRESHOLDS.min_first_gate_crossing_rate
    )
    with pytest.raises(ValueError):
        thresholds_from_mapping({"min_unknown_metric": 1.0})


# ---------------------------------------------------------------- aggregation


def _episode(**overrides):
    row = {
        "episode_type": "transition_focus",
        "universal_transition_success": True,
        "first_gate_crossed": True,
        "correct_target_switch": True,
        "new_target_acquired_and_aligned": True,
        "new_target_range_decreasing": True,
        "previous_gate_return": False,
        "missed_gate_dnf": 0,
        "collision_events": 0,
        "out_of_bounds_events": 0,
        "wrong_direction_events": 0,
        "acquisition_timeout": False,
        "acquisition_time_s": 0.4,
        "action_jerk": 0.1,
        "action_source": "ppo_policy",
        "gates_completed": 1,
        "collision_entries": 0,
        "collision_contact_frames": 0,
        "action_steps": 100,
        "nontrivial_action_steps": 90,
        "mean_absolute_action_per_axis": [0.2, 0.1, 0.1, 0.1],
        "distance_travelled_m": 6.0,
    }
    row.update(overrides)
    return row


def test_aggregate_reports_every_required_anti_inactivity_metric():
    metrics = aggregate_transition_benchmark([_episode(), _episode(gates_completed=2)])
    for name in (
        "first_gate_crossing_rate", "target_switch_rate",
        "universal_transition_success_rate", "completed_gate_count",
        "fraction_of_episodes_reaching_first_gate", "mean_distance_travelled_m",
        "mean_absolute_action_per_axis", "mean_absolute_action",
        "nontrivial_action_fraction", "acquisition_timeout_rate",
        "collision_episodes", "collision_entries", "collision_contact_frames",
        "missed_gate_dnf", "wrong_direction_events", "previous_gate_returns",
        "mean_action_jerk",
    ):
        assert name in metrics, name
    assert metrics["completed_gate_count"] == 3
    assert metrics["mean_completed_gates_per_episode"] == pytest.approx(1.5)
    assert metrics["fraction_of_episodes_reaching_first_gate"] == pytest.approx(1.0)
    assert metrics["mean_distance_travelled_m"] == pytest.approx(6.0)
    assert metrics["nontrivial_action_fraction"] == pytest.approx(0.9)
    assert metrics["mean_absolute_action_per_axis"]["surge"] == pytest.approx(0.2)
    assert metrics["mean_absolute_action"] == pytest.approx(0.125)
    assert metrics["nontrivial_action_threshold"] == NONTRIVIAL_ACTION_THRESHOLD


def test_aggregate_separates_collision_entries_from_contact_frames():
    metrics = aggregate_transition_benchmark([
        _episode(collision_events=3, collision_entries=1, collision_contact_frames=180),
        _episode(collision_events=0, collision_entries=0, collision_contact_frames=0),
    ])
    assert metrics["collision_episodes"] == 1
    assert metrics["collision_entries"] == 1
    assert metrics["collision_contact_frames"] == 180


def test_aggregate_omits_activity_metrics_for_pre_existing_reports():
    legacy = _episode()
    for key in (
        "distance_travelled_m", "action_steps", "nontrivial_action_steps",
        "mean_absolute_action_per_axis", "collision_entries",
        "collision_contact_frames",
    ):
        legacy.pop(key)
    metrics = aggregate_transition_benchmark([legacy])
    assert metrics["mean_distance_travelled_m"] is None
    assert metrics["mean_absolute_action"] is None
    assert metrics["nontrivial_action_fraction"] is None
    assert metrics["collision_entries"] is None
    # Participation evidence is still derivable from stored per-episode rows.
    assert metrics["completed_gate_count"] == 1
    assert metrics["fraction_of_episodes_reaching_first_gate"] == pytest.approx(1.0)


def test_aggregation_stays_deterministic():
    rows = [_episode(), _episode(gates_completed=0, first_gate_crossed=False)]
    assert aggregate_transition_benchmark(rows) == aggregate_transition_benchmark(
        list(rows)
    )


# --------------------------------------------------------- frozen A/B evidence


# The heavy A/B evidence lives under the ignored results/rl/ tree; the compact,
# auditable copy of the selection is tracked here.
PUBLIC_ROOT = Path("results/rl_public/universal_transition_selection")


def test_corrected_ab_report_selects_the_competent_warm_start():
    report = json.loads(
        (PUBLIC_ROOT / "ab_comparison_corrected.json").read_text(encoding="utf-8")
    )
    assert report["selected_initialization"] == "selective_warm_start"
    assert report["selection"]["rejected_candidates"]["scratch"] == (
        CLASSIFICATION_DEGENERATE_INACTIVE
    )
    assert report["stages"] == [
        "competence_qualification", "safety_ranking", "final_selection",
    ]
    warm = report["arms"]["selective_warm_start"]["recomputed_metrics"]
    assert warm["completed_gate_count"] == 946
    assert warm["universal_transition_success_rate"] == pytest.approx(0.314)
    scratch = report["arms"]["scratch"]["recomputed_metrics"]
    assert scratch["completed_gate_count"] == 15
    assert scratch["universal_transition_success_rate"] == pytest.approx(0.0)


def test_original_ranking_is_preserved_unmodified_beside_the_correction():
    original = (PUBLIC_ROOT / "ab_comparison_original.md").read_text(encoding="utf-8")
    assert "Selected initialization: **scratch**" in original


def test_the_gate_rejects_the_originally_selected_arm():
    report = json.loads(
        (PUBLIC_ROOT / "ab_comparison_corrected.json").read_text(encoding="utf-8")
    )
    superseded = report["supersedes"]
    assert superseded["selected_initialization"] == "scratch"
    verdict = evaluate_competence_gate(
        report["arms"]["scratch"]["gate_metrics"]
    )
    assert not verdict.passed
    assert verdict.classification == CLASSIFICATION_DEGENERATE_INACTIVE
