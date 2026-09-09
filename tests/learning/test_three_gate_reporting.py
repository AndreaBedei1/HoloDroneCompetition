"""Regression tests for "not evaluated" vs "evaluated and failed" reporting.

The reliability-first long run reported ``three_gate_success: 0.0`` while its C3
evaluation suite contained no three-gate case at all: three-gate cases only enter
the suite from stage C5. A rate of ``0.0`` cannot express "never measured", so the
status was read as a total failure. These tests pin the distinction.
"""

import pytest

from marine_race_arena.learning.longrun_evaluation import (
    _evaluation_cases,
    aggregate_evaluation,
    category_evaluated,
    category_not_evaluated_reason,
    official_evaluation_unlocked,
    rate_or,
)
from marine_race_arena.learning.longrun_monitor import (
    _three_gate_status_fields,
    evaluation_regression_reasons,
)


def _rows(*categories):
    return [{"category": category, "finished": True} for category in categories]


def test_three_gate_cases_are_absent_from_the_stage_c3_full_suite(tmp_path):
    """The historical cause: the C3 suite simply has no three-gate case."""
    cases = _evaluation_cases(
        "C3", mode="full", seeds=list(range(25300, 25330)), output_dir=tmp_path
    )
    assert {case["category"] for case in cases} == {
        "retention", "straight", "left", "right", "current",
    }
    assert not [case for case in cases if case["category"] == "three_gate"]


def test_three_gate_cases_enter_the_suite_at_stage_c5(tmp_path):
    cases = _evaluation_cases(
        "C5", mode="full", seeds=list(range(25300, 25330)), output_dir=tmp_path
    )
    assert [case for case in cases if case["category"] == "three_gate"]


def test_unevaluated_category_reports_none_not_zero():
    aggregate = aggregate_evaluation(_rows("retention", "straight", "left", "right"))
    assert aggregate["three_gate_completion_rate"] is None
    assert aggregate["three_gate_evaluated"] is False
    assert aggregate["three_gate_n"] == 0
    assert aggregate["not_evaluated_categories"] == ["three_gate", "six_gate"]
    assert category_evaluated(aggregate, "three_gate") is False


def test_evaluated_and_failed_category_reports_zero_not_none():
    rows = _rows("retention", "straight", "left", "right")
    rows.append({"category": "three_gate", "finished": False})
    aggregate = aggregate_evaluation(rows)
    assert aggregate["three_gate_completion_rate"] == 0.0
    assert aggregate["three_gate_evaluated"] is True
    assert aggregate["three_gate_n"] == 1
    assert category_evaluated(aggregate, "three_gate") is True


def test_status_fields_separate_not_evaluated_from_failed():
    unevaluated = aggregate_evaluation(_rows("retention", "straight"))
    fields = _three_gate_status_fields(unevaluated, "C3")
    assert fields["three_gate_status"] == "not_evaluated"
    assert fields["three_gate_success"] is None
    assert fields["three_gate_evaluated"] is False
    assert "C5" in fields["three_gate_note"]

    failed = aggregate_evaluation(
        _rows("retention", "straight") + [{"category": "three_gate", "finished": False}]
    )
    fields = _three_gate_status_fields(failed, "C5")
    assert fields["three_gate_status"] == "failed"
    assert fields["three_gate_success"] == 0.0
    assert fields["three_gate_evaluated"] is True
    assert fields["three_gate_note"] is None

    passed = aggregate_evaluation(
        _rows("retention", "straight", "three_gate")
    )
    assert _three_gate_status_fields(passed, "C5")["three_gate_status"] == "passed"


def test_partial_three_gate_success_is_reported_as_partial():
    rows = _rows("retention", "three_gate") + [
        {"category": "three_gate", "finished": False}
    ]
    fields = _three_gate_status_fields(aggregate_evaluation(rows), "C5")
    assert fields["three_gate_status"] == "partial"
    assert fields["three_gate_success"] == pytest.approx(0.5)


def test_official_circuits_never_unlock_on_an_unmeasured_three_gate_rate():
    rows = _rows("straight", "straight", "left", "right")
    aggregate = aggregate_evaluation(rows)
    assert aggregate["straight_completion_rate"] == 1.0
    assert official_evaluation_unlocked(aggregate) is False

    with_three_gate = aggregate_evaluation(rows + _rows("three_gate"))
    assert official_evaluation_unlocked(with_three_gate) is True


def test_not_evaluated_reason_names_the_stage_that_introduces_the_category():
    assert "C5" in category_not_evaluated_reason("three_gate", "C3")
    assert "C6" in category_not_evaluated_reason("six_gate", "C3")
    assert category_not_evaluated_reason("three_gate", "C5") is None
    assert category_not_evaluated_reason("left", "C3") is None


def test_rate_or_maps_not_evaluated_to_the_callers_default():
    metrics = {"three_gate_completion_rate": None, "left_completion_rate": 0.5}
    assert rate_or(metrics, "three_gate_completion_rate", 1.0) == 1.0
    assert rate_or(metrics, "three_gate_completion_rate", 0.0) == 0.0
    assert rate_or(metrics, "left_completion_rate", 1.0) == 0.5
    assert rate_or(metrics, "missing_key", 0.25) == 0.25


class _Rollback:
    enabled = True
    completion_drop = 0.1
    single_gate_floor = 0.9
    straight_floor = 0.85
    directional_floor = 0.7
    maximum_safety_episodes = 0
    maximum_previous_gate_returns = 0


def test_rollback_gate_skips_floors_for_categories_that_were_not_evaluated():
    incumbent = {"completion_rate": 1.0}
    # A light suite that only measured turns must not look like a retention collapse.
    report = aggregate_evaluation(_rows("left", "right"))
    report["completion_rate"] = 1.0
    reasons = evaluation_regression_reasons(report, incumbent, _Rollback())
    assert "single_gate_retention_floor" not in reasons
    assert "straight_retention_floor" not in reasons


def test_rollback_gate_still_fires_on_a_measured_retention_collapse():
    incumbent = {"completion_rate": 1.0}
    report = aggregate_evaluation(
        [{"category": "retention", "finished": False},
         {"category": "straight", "finished": False}]
    )
    report["completion_rate"] = 1.0
    reasons = evaluation_regression_reasons(report, incumbent, _Rollback())
    assert "single_gate_retention_floor" in reasons
    assert "straight_retention_floor" in reasons


def test_legacy_reports_without_category_metrics_still_read_as_evaluated():
    """Reports written before this schema always emitted a float; keep them usable."""
    legacy = {
        "single_gate_completion_rate": 1.0,
        "straight_completion_rate": 1.0,
        "left_completion_rate": 0.9,
        "right_completion_rate": 0.9,
    }
    assert category_evaluated(legacy, "retention") is True
    assert category_evaluated(legacy, "straight") is True
    assert category_evaluated(legacy, "three_gate") is False


# --------------------------------------------------------------------------- #
# Run audit
# --------------------------------------------------------------------------- #
def _write_run(tmp_path, evaluations, status):
    import json as _json

    (tmp_path / "evaluations").mkdir(parents=True, exist_ok=True)
    for name, report in evaluations.items():
        (tmp_path / "evaluations" / name).write_text(
            _json.dumps(report), encoding="utf-8"
        )
    (tmp_path / "status.json").write_text(_json.dumps(status), encoding="utf-8")
    return tmp_path


def test_audit_reports_never_evaluated_when_no_case_ever_ran(tmp_path):
    from marine_race_arena.learning.three_gate_audit import audit_run

    run = _write_run(
        tmp_path,
        {
            "full_000001000.json": {
                "timesteps": 1000, "stage": "C3", "mode": "full", "n_eval": 20,
                "three_gate_completion_rate": 0.0,
                "rows": [{"category": "left", "finished": True}] * 20,
            }
        },
        {"three_gate_success": 0.0, "curriculum_stage": "C3"},
    )
    audit = audit_run(run)
    assert audit["verdict"] == "never_evaluated"
    assert audit["episodes_of_category"] == 0
    assert audit["reported_status_was_misleading"] is True
    assert "C5" in audit["reason"]


def test_audit_reports_evaluated_and_failed_when_cases_ran_and_lost(tmp_path):
    from marine_race_arena.learning.three_gate_audit import audit_run

    run = _write_run(
        tmp_path,
        {
            "full_000002000.json": {
                "timesteps": 2000, "stage": "C5", "mode": "full", "n_eval": 3,
                "three_gate_completion_rate": 0.0,
                "rows": [
                    {"category": "left", "finished": True},
                    {"category": "three_gate", "finished": False},
                    {"category": "three_gate", "finished": False},
                ],
            }
        },
        {"three_gate_success": 0.0, "curriculum_stage": "C5"},
    )
    audit = audit_run(run)
    assert audit["verdict"] == "evaluated_and_failed"
    assert audit["episodes_of_category"] == 2
    assert audit["reported_status_was_misleading"] is False
