"""Tests for held-out policy evaluation through the unchanged referee."""

import pytest

from marine_race_arena.learning.evaluate_policy import (
    EVALUATION_END_REASONS,
    EvalReport,
    EvalResult,
    derive_evaluation_end_reason,
    evaluate_controller,
)
from marine_race_arena.participants.controller_loader import ControllerLoader

STAGE1 = "marine_race_arena/tracks/training/stage1_single_gate.json"


def _rule_factory():
    return ControllerLoader().load("rule_gate_center_then_commit")


def test_evaluate_rule_controller_report_structure():
    report = evaluate_controller(
        STAGE1,
        _rule_factory,
        seeds=[0, 1],
        label="center_then_commit",
        adapter="fallback",
        allow_fallback=True,
        duration_s=3.0,
        dt=0.1,
    )
    assert report.n == 2
    assert 0.0 <= report.completion_rate <= 1.0
    for r in report.results:
        assert r.expected_gates == 1  # single-gate stage
        assert r.completed_gates >= 0
        assert r.status is not None
    s = report.summary()
    for key in (
        "completion_rate", "completion_rate_wilson95_low", "completion_rate_wilson95_high",
        "mean_gates", "mean_collisions", "episodes",
        "mean_missed_gate_attempts", "mean_wrong_direction_crossings", "mean_inference_time_ms",
    ):
        assert key in s
    assert s["episodes"] == 2
    # Per-seed rows expose the corrected, independent metrics and provenance.
    for r in report.results:
        assert hasattr(r, "missed_gate_attempts") and hasattr(r, "wrong_direction_crossings")
        assert r.adapter_used == "fallback"
        assert r.inference_time_ms is not None and r.inference_time_ms >= 0.0
        assert r.wall_s is not None


def test_missed_gate_and_wrong_direction_are_independent():
    # A row with missed-gate attempts but no wrong-direction crossings, and vice versa.
    a = EvalResult(0, "DNF", False, 0, 1, None, None, 0, 0, 0, 0, missed_gate_attempts=2, wrong_direction_crossings=0)
    b = EvalResult(1, "RUNNING", False, 0, 1, None, None, 0, 0, 0, 0, missed_gate_attempts=0, wrong_direction_crossings=3)
    assert a.missed_gate_attempts == 2 and a.wrong_direction_crossings == 0
    assert b.missed_gate_attempts == 0 and b.wrong_direction_crossings == 3
    report = EvalReport(track="t", label="x", results=[a, b])
    s = report.summary()
    assert s["mean_missed_gate_attempts"] == pytest.approx(1.0)
    assert s["mean_wrong_direction_crossings"] == pytest.approx(1.5)


def test_wilson_interval_brackets_the_point_estimate():
    report = EvalReport(track="t", label="x")
    report.results = [EvalResult(i, "FINISHED", True, 1, 1, 1.0, 1.0, 0, 0, 0, 0, 0) for i in range(20)]
    ci = report.wilson_interval()
    assert ci["low"] <= report.completion_rate <= ci["high"] + 1e-9  # boundary (p=1) float tolerance
    assert 0.8 < ci["low"] < 1.0  # 20/20 gives a tight lower bound well above 0.8, below 1


def test_report_aggregations_on_synthetic_results():
    report = EvalReport(track="t", label="x")
    report.results = [
        EvalResult(0, "FINISHED", True, 3, 3, 100.0, 100.0, 0, 0, 0, 0, 0),
        EvalResult(1, "FINISHED", True, 3, 3, 120.0, 130.0, 1, 0, 0, 0, 0),
        EvalResult(2, "TIMEOUT", False, 1, 3, None, None, 2, 0, 1, 0, 0),
    ]
    assert report.completion_rate == pytest.approx(2 / 3)
    assert report.mean_gates == pytest.approx((3 + 3 + 1) / 3)
    # finished-only time excludes the unfinished (None) run
    assert report.mean_official_time_finished == pytest.approx((100.0 + 120.0) / 2)
    assert report.mean_collisions == pytest.approx((0 + 1 + 2) / 3)


@pytest.mark.parametrize(
    "referee_status, truncated, expected",
    [
        ("FINISHED", False, "FINISHED"),          # successful finish
        ("RUNNING", False, "TIME_LIMIT"),          # race duration expired, referee still running
        ("NOT_STARTED", False, "TIME_LIMIT"),      # never released, window closed
        ("RUNNING", True, "MAX_STEPS"),            # step-wise runner truncation
        ("DNF", False, "REFEREE_TERMINAL"),        # terminal referee failure
        ("DSQ", False, "REFEREE_TERMINAL"),
        ("STUCK", False, "REFEREE_TERMINAL"),
        ("TIMEOUT", False, "REFEREE_TERMINAL"),    # referee gate/stuck timeout (not runner TIME_LIMIT)
        ("CONTROLLER_ERROR", False, "CONTROLLER_ERROR"),  # controller exception -> referee marks it
        ("MANUAL_STOP", False, "MANUAL_STOP"),
        ("SOMETHING_ELSE", False, "UNKNOWN"),      # defensive default
    ],
)
def test_derive_evaluation_end_reason(referee_status, truncated, expected):
    reason = derive_evaluation_end_reason(referee_status, truncated_by_max_steps=truncated)
    assert reason == expected
    assert reason in EVALUATION_END_REASONS


def test_end_reason_accepts_enum_like_status():
    class _S:
        value = "FINISHED"

    assert derive_evaluation_end_reason(_S()) == "FINISHED"


def test_eval_result_backward_compatible_referee_status():
    # Old-style construction (no referee_status/evaluation_end_reason) still works;
    # referee_status mirrors the legacy `status` field.
    r = EvalResult(0, "FINISHED", True, 1, 1, 1.0, 1.0, 0, 0, 0, 0, 0)
    assert r.referee_status == "FINISHED"
    assert r.evaluation_end_reason == "UNKNOWN"  # unset until derived by the runner
    # New-style construction carries both explicitly.
    r2 = EvalResult(1, "RUNNING", False, 0, 1, None, None, 0, 0, 0, 0, 0,
                    referee_status="RUNNING", evaluation_end_reason="TIME_LIMIT")
    assert r2.referee_status == "RUNNING" and r2.evaluation_end_reason == "TIME_LIMIT"


def test_summary_reports_end_reason_breakdown():
    report = EvalReport(track="t", label="x")
    report.results = [
        EvalResult(0, "FINISHED", True, 1, 1, 1.0, 1.0, 0, 0, 0, 0, 0,
                   referee_status="FINISHED", evaluation_end_reason="FINISHED"),
        EvalResult(1, "RUNNING", False, 0, 1, None, None, 0, 0, 0, 0, 0,
                   referee_status="RUNNING", evaluation_end_reason="TIME_LIMIT"),
    ]
    s = report.summary()
    assert s["end_reason_counts"] == {"FINISHED": 1, "TIME_LIMIT": 1}
    assert s["referee_status_counts"] == {"FINISHED": 1, "RUNNING": 1}


def test_evaluate_populates_end_reason_from_referee(tmp_path):
    # A real fallback run: the rule controller finishes or times out; either way the
    # per-seed row carries a referee_status and a documented evaluation_end_reason.
    report = evaluate_controller(
        STAGE1, _rule_factory, seeds=[0], adapter="fallback", allow_fallback=True, duration_s=3.0
    )
    r = report.results[0]
    assert r.referee_status == r.status
    assert r.evaluation_end_reason in EVALUATION_END_REASONS
    # Consistency: a FINISHED referee status must map to a FINISHED end reason.
    if r.referee_status == "FINISHED":
        assert r.evaluation_end_reason == "FINISHED"


def test_evaluation_seeds_are_independent_runs():
    """Each seed gets a fresh controller and adapter (no state carryover)."""
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return ControllerLoader().load("rule_gate_center_then_commit")

    evaluate_controller(STAGE1, factory, seeds=[5, 6, 7], adapter="fallback", allow_fallback=True, duration_s=2.0)
    assert calls["n"] == 3
