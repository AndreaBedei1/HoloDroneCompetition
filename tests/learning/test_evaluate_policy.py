"""Tests for held-out policy evaluation through the unchanged referee."""

import pytest

from marine_race_arena.learning.evaluate_policy import EvalReport, EvalResult, evaluate_controller
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


def test_evaluation_seeds_are_independent_runs():
    """Each seed gets a fresh controller and adapter (no state carryover)."""
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return ControllerLoader().load("rule_gate_center_then_commit")

    evaluate_controller(STAGE1, factory, seeds=[5, 6, 7], adapter="fallback", allow_fallback=True, duration_s=2.0)
    assert calls["n"] == 3
