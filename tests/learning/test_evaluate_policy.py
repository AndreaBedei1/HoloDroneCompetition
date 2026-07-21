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
    for key in ("completion_rate", "mean_gates", "mean_collisions", "episodes"):
        assert key in s
    assert s["episodes"] == 2


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
