from __future__ import annotations

import json
from pathlib import Path

import pytest

from marine_race_arena.scripts.run_benchmark import (
    BenchmarkRunResult,
    aggregate_run_results,
    _build_run_metadata,
    write_aggregate_outputs,
)


def test_benchmark_aggregation_from_fake_summaries(tmp_path: Path) -> None:
    finished_summary = _write_summary(
        tmp_path / "finished_summary.json",
        {
            "participants": [
                {
                    "rank": 1,
                    "participant_id": "p1",
                    "status": "FINISHED",
                    "official_time_s": 10.0,
                    "penalized_time_s": 12.0,
                    "completed_gates": 12,
                    "collisions": 1,
                    "obstacle_collisions": 2,
                    "out_of_bounds_events": 0,
                    "stuck_events": 0,
                }
            ]
        },
    )
    dnf_summary = _write_summary(
        tmp_path / "dnf_summary.json",
        {
            "participants": [
                {
                    "rank": 1,
                    "participant_id": "p1",
                    "status": "DNF",
                    "official_time_s": None,
                    "penalized_time_s": None,
                    "completed_gates": 4,
                    "collisions": 3,
                    "obstacle_collisions": 1,
                    "out_of_bounds_events": 1,
                    "stuck_events": 1,
                }
            ]
        },
    )
    manual_summary = _write_summary(
        tmp_path / "manual_summary.json",
        {
            "participants": [
                {
                    "rank": 1,
                    "participant_id": "p1",
                    "status": "MANUAL_STOP",
                    "completed_gates": 2,
                }
            ]
        },
    )
    dnf_events = tmp_path / "dnf_events.jsonl"
    dnf_events.write_text(json.dumps({"event": "dnf", "reason": "missed_gate"}) + "\n", encoding="utf-8")

    aggregate, rows = aggregate_run_results(
        [
            BenchmarkRunResult(0, tmp_path / "run0", 0, {"task": "clean_gate"}, finished_summary),
            BenchmarkRunResult(1, tmp_path / "run1", 0, {"task": "clean_gate"}, dnf_summary, dnf_events),
            BenchmarkRunResult(2, tmp_path / "run2", 0, {"task": "clean_gate"}, manual_summary),
        ]
    )

    assert aggregate["number_of_runs"] == 3
    assert aggregate["completion_rate"] == pytest.approx(1 / 3)
    assert aggregate["mean_official_time_s"] == 10.0
    assert aggregate["std_official_time_s"] == 0.0
    assert aggregate["mean_penalized_time_s"] == 12.0
    assert aggregate["mean_completed_gates"] == pytest.approx(6.0)
    assert aggregate["mean_collision_events"] == pytest.approx(4 / 3)
    assert aggregate["mean_obstacle_collision_events"] == pytest.approx(1.0)
    assert aggregate["mean_out_of_bounds_events"] == pytest.approx(1 / 3)
    assert aggregate["mean_stuck_events"] == pytest.approx(1 / 3)
    assert aggregate["total_dnf"] == 1
    assert aggregate["dnf_reasons"] == {"missed_gate": 1}
    assert aggregate["manual_stop_count"] == 1
    assert aggregate["controller_error_count"] == 0
    assert rows[1]["dnf_reason"] == "missed_gate"

    csv_path, json_path = write_aggregate_outputs(tmp_path / "aggregate", aggregate, rows)

    assert csv_path.name == "benchmark_summary.csv"
    assert json_path.name == "benchmark_summary.json"
    assert csv_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8"))["aggregate"]["number_of_runs"] == 3


def test_benchmark_aggregation_handles_missing_metrics_safely(tmp_path: Path) -> None:
    empty_summary = _write_summary(tmp_path / "empty_summary.json", {})
    partial_summary = _write_summary(
        tmp_path / "partial_summary.json",
        {"participants": [{"status": "CONTROLLER_ERROR"}]},
    )

    aggregate, rows = aggregate_run_results(
        [
            BenchmarkRunResult(0, tmp_path / "run0", 0, {}, empty_summary),
            BenchmarkRunResult(1, tmp_path / "run1", 0, {}, partial_summary),
            BenchmarkRunResult(2, tmp_path / "run2", 1, {}),
        ]
    )

    assert aggregate["number_of_runs"] == 3
    assert aggregate["completion_rate"] == 0.0
    assert aggregate["mean_official_time_s"] is None
    assert aggregate["std_official_time_s"] is None
    assert aggregate["mean_penalized_time_s"] is None
    assert aggregate["mean_completed_gates"] == 0.0
    assert aggregate["manual_stop_count"] == 0
    assert aggregate["controller_error_count"] == 1
    assert rows[0]["status"] == "UNKNOWN"
    assert rows[2]["status"] == "RUN_FAILED"


def test_benchmark_metadata_records_motion_compensation() -> None:
    args = _Args(
        benchmark_task="clean_gate",
        track=str(Path("marine_race_arena/tracks/marine_race_horseshoe_bay.json")),
        controller="rule_gate_baseline",
        controller_class=None,
        adapter="fallback",
        allow_fallback=False,
        obstacles="none",
        obstacle_density=None,
        obstacle_physics=None,
        current_profile="none",
        motion_compensation="none",
        gate_timeout_s=180.0,
        duration=120.0,
        dt=0.1,
        official=False,
        print_beacon_targets=False,
    )

    metadata = _build_run_metadata(args, seed=0, controller_role="automatic")

    assert metadata["motion_compensation"] == "none"
    assert metadata["gate_timeout_s"] == 180.0


def _write_summary(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class _Args:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)
