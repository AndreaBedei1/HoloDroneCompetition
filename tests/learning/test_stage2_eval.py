"""Tests for Stage-2 robustness metrics and best-model selection (numpy-only)."""

import pytest

from marine_race_arena.learning.stage2_eval import (
    aggregate_stage2,
    is_extreme_corner,
    stage2_best_metric_key,
    stage2_is_better,
)


def _row(finished, lat=0.0, yaw=0.0, oob=0, wrong=0, coll=0, gates=1, time_s=10.0, penalty=0.0):
    return {
        "finished": finished, "referee_status": "FINISHED" if finished else "RUNNING",
        "evaluation_end_reason": "FINISHED" if finished else "TIME_LIMIT",
        "completed_gates": gates, "collision_events": coll, "out_of_bounds_events": oob,
        "out_of_bounds_episode": oob > 0, "wrong_direction_crossings": wrong, "missed_gate_attempts": 0,
        "raw_completion_time_s": (time_s if finished else None), "penalty_s": penalty,
        "penalized_completion_time_s": (time_s + penalty if finished else None),
        "time_s": (time_s if finished else None),
        "action_saturation": 0.0, "action_smoothness": 0.0,
        "mean_abs_action": {"surge": 0.3, "sway": 0.0, "heave": 0.0, "yaw": 0.1},
        "inference_ms": 40.0,
        "applied_randomization": {"lateral_offset_m": lat, "yaw_offset_deg": yaw},
        "is_extreme_corner": is_extreme_corner({"lateral_offset_m": lat, "yaw_offset_deg": yaw}),
        "actions_finite": True,
    }


def test_is_extreme_corner():
    assert is_extreme_corner({"lateral_offset_m": 0.95, "yaw_offset_deg": 14.0})
    assert not is_extreme_corner({"lateral_offset_m": 0.95, "yaw_offset_deg": 5.0})   # yaw too small
    assert not is_extreme_corner({"lateral_offset_m": 0.5, "yaw_offset_deg": 14.0})   # lateral too small
    assert not is_extreme_corner(None)


def test_aggregate_splits_interior_and_extreme():
    rows = [
        _row(True, lat=0.1, yaw=1.0),    # interior success
        _row(True, lat=0.2, yaw=-2.0),   # interior success
        _row(False, lat=0.95, yaw=14.0, oob=3),  # extreme failure (OOB)
        _row(True, lat=0.9, yaw=13.0),   # extreme success
    ]
    agg = aggregate_stage2(rows)
    assert agg["n_eval"] == 4 and agg["completion_rate"] == 0.75
    assert agg["interior_n"] == 2 and agg["interior_completion"] == 1.0
    assert agg["extreme_n"] == 2 and agg["extreme_completion"] == 0.5
    assert agg["oob_episodes"] == 1
    assert agg["end_reason_counts"]["FINISHED"] == 3 and agg["end_reason_counts"]["TIME_LIMIT"] == 1


def test_stage2_best_key_prefers_robustness_over_speed():
    # Same completion, but A is faster while B is more robust at the extreme corners.
    fast_but_fragile = aggregate_stage2([_row(True, lat=0.9, yaw=13.0, time_s=8.0),
                                         _row(False, lat=0.95, yaw=14.0, oob=2)])
    robust = aggregate_stage2([_row(True, lat=0.9, yaw=13.0, time_s=11.0),
                               _row(True, lat=0.95, yaw=14.0, time_s=11.0)])
    assert stage2_is_better(robust, fast_but_fragile)  # higher extreme completion wins
    assert not stage2_is_better(fast_but_fragile, robust)


def test_stage2_best_key_breaks_ties_on_oob_then_time():
    a = aggregate_stage2([_row(True, lat=0.9, yaw=13.0), _row(True, lat=0.1, yaw=1.0, oob=1)])
    b = aggregate_stage2([_row(True, lat=0.9, yaw=13.0), _row(True, lat=0.1, yaw=1.0, oob=0)])
    # equal completion + extreme; b has fewer OOB episodes -> b wins
    assert stage2_is_better(b, a) and not stage2_is_better(a, b)


def test_penalty_metrics_are_distinct_from_completion_time():
    # A finished episode with a 3s penalty: raw 10s, penalized 13s; penalty is not the time.
    agg = aggregate_stage2([_row(True, lat=0.1, yaw=1.0, time_s=10.0, penalty=3.0),
                            _row(False, lat=0.1, yaw=1.0)])  # unfinished -> null times
    assert agg["mean_time_finished"] == 10.0
    assert agg["mean_penalized_time_finished"] == 13.0
    assert agg["mean_penalty_s"] == 1.5  # (3 + 0) / 2 rows
    # An unfinished episode has no completion time or penalized time (null, not fabricated).
    rows = [_row(False, lat=0.1, yaw=1.0)]
    assert rows[0]["raw_completion_time_s"] is None and rows[0]["penalized_completion_time_s"] is None


def test_best_key_none_extreme_ranks_below_measured():
    no_extreme = aggregate_stage2([_row(True, lat=0.1, yaw=1.0)])  # extreme_completion is None
    with_extreme = aggregate_stage2([_row(True, lat=0.1, yaw=1.0), _row(True, lat=0.9, yaw=13.0)])
    assert stage2_is_better(with_extreme, no_extreme)
