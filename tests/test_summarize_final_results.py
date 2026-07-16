from __future__ import annotations

from pathlib import Path

import pytest

from summarize_final_results import (
    _markdown,
    _write_outputs,
    analyze_progress_events,
    build_manifest_row,
)
from validate_final_matrix import expected_runs


def test_final_matrix_implies_124_participant_records() -> None:
    count = 0
    for run in expected_runs():
        if run.metadata_path.startswith("coordination/"):
            count += 3
        elif run.metadata_path.startswith("fleet_gap90/"):
            count += 2
        else:
            count += 1

    assert count == 124


def test_progress_analysis_distinguishes_early_delayed_and_missed() -> None:
    events = [
        {"event": "controller_local_state", "participant_id": "rov", "time_s": 0.5, "local_completed": 1},
        {"event": "gate_passed", "participant_id": "rov", "time_s": 1.0},
        {"event": "gate_passed", "participant_id": "rov", "time_s": 2.0},
        {"event": "controller_local_state", "participant_id": "rov", "time_s": 2.5, "local_completed": 2},
        {"event": "gate_passed", "participant_id": "rov", "time_s": 3.0},
    ]

    report = analyze_progress_events(
        events,
        participant_ids=["rov"],
        expected_gates_per_rover=3,
        tolerance_s=0.1,
    )

    assert report["totals"]["false_local_advancements"] == 1
    assert report["totals"]["delayed_local_advancements"] == 1
    assert report["totals"]["missed_local_advancements"] == 1


def test_progress_analysis_detects_team_finish_order_inversion() -> None:
    events = [
        {"event": "gate_passed", "participant_id": "rov_a", "sequence_index": 0, "time_s": 2.0},
        {"event": "gate_passed", "participant_id": "rov_b", "sequence_index": 0, "time_s": 3.0},
        {
            "event": "controller_local_state",
            "participant_id": "rov_b",
            "time_s": 3.5,
            "local_completed": 1,
            "local_status": "FINISHED",
        },
        {
            "event": "controller_local_state",
            "participant_id": "rov_a",
            "time_s": 4.0,
            "local_completed": 1,
            "local_status": "FINISHED",
        },
    ]

    report = analyze_progress_events(
        events,
        participant_ids=["rov_a", "rov_b"],
        expected_gates_per_rover=1,
        tolerance_s=0.1,
    )

    assert report["totals"]["finish_order_pair_comparisons"] == 1
    assert report["totals"]["finish_order_inversions"] == 1
    assert report["finish_order_pairs"][0]["classification"] == "inversion"


def test_team_manifest_uses_unique_team_proximity_count(tmp_path: Path) -> None:
    root = tmp_path / "results"
    run = root / "fleet" / "run"
    run.mkdir(parents=True)
    metadata_path = run / "benchmark_metadata.json"
    summary_path = run / "race_summary.json"
    event_path = run / "race.jsonl"
    metadata_path.write_text("{}", encoding="utf-8")
    summary_path.write_text("{}", encoding="utf-8")
    event_path.write_text("", encoding="utf-8")
    metadata = {
        "track": "track.json",
        "controller": "rule_gate_baseline",
        "current_profile_requested": "none",
        "num_rovers": 2,
        "staggered_start": True,
        "start_gap_s": 90.0,
        "seed": 0,
    }
    participants = [
        {
            "participant_id": "rov_1",
            "status": "FINISHED",
            "completed_gates": 12,
            "collisions": 2,
            "involved_inter_vehicle_collisions": 1,
            "official_time_s": 10.0,
            "penalized_time_s": 20.0,
        },
        {
            "participant_id": "rov_2",
            "status": "FINISHED",
            "completed_gates": 12,
            "collisions": 3,
            "involved_inter_vehicle_collisions": 1,
            "official_time_s": 11.0,
            "penalized_time_s": 21.0,
        },
    ]
    summary = {
        "participants": participants,
        "team_summary": {
            "rover_count": 2,
            "total_completed_gates": 24,
            "expected_total_gates": 24,
            "all_rovers_finished": True,
            "team_elapsed_time_s": 101.0,
            "team_penalized_time_s": 111.0,
            "total_gate_collisions": 5,
            "total_obstacle_collisions": 0,
            "total_inter_vehicle_collisions": 1,
            "total_collisions": 6,
            "total_penalties_s": 10.0,
        },
    }

    row = build_manifest_row(
        results_root=root,
        metadata_path=metadata_path,
        metadata=metadata,
        summary_path=summary_path,
        event_path=event_path,
        summary=summary,
        track_name="Test Track",
        expected_gates_per_rover=12,
    )

    assert row["proximity_events"] == 1
    assert row["gate_world_collisions"] == 5
    assert row["official_time_s"] is None
    assert row["penalized_time_s"] is None
    assert row["team_elapsed_time_s"] == 101.0
    assert row["team_penalized_time_s"] == 111.0
    assert row["start_gap_s"] == 90.0
    rendered = _markdown(
        {
            "run_count": 1,
            "expected_run_count": 78,
            "complete_matrix": False,
            "runs": [row],
            "local_vs_referee": {"overall": {}},
        }
    )
    assert "101.000" in rendered
    assert "111.000" in rendered
    output = tmp_path / "output"
    report = {
        "run_count": 1,
        "expected_run_count": 78,
        "complete_matrix": False,
        "runs": [row],
        "aggregates": {"runs": [], "participants": []},
        "local_vs_referee": {"overall": {}},
    }
    _write_outputs(report, output)
    assert len((output / "complete_experiment_manifest.csv").read_text().splitlines()) == 2
    participant_lines = (output / "complete_experiment_participants.csv").read_text().splitlines()
    assert len(participant_lines) == 3
    assert participant_lines[0].split(",").count("controller") == 1

    heterogeneous = dict(metadata)
    heterogeneous["controllers"] = ["leader_controller", "follower_controller"]
    reversed_summary = dict(summary)
    reversed_summary["participants"] = list(reversed(participants))
    reversed_row = build_manifest_row(
        results_root=root,
        metadata_path=metadata_path,
        metadata=heterogeneous,
        summary_path=summary_path,
        event_path=event_path,
        summary=reversed_summary,
        track_name="Test Track",
        expected_gates_per_rover=12,
    )
    by_id = {participant["participant_id"]: participant for participant in reversed_row["participants"]}
    assert by_id["rov_1"]["release_index"] == 0
    assert by_id["rov_1"]["controller"] == "leader_controller"
    assert by_id["rov_2"]["release_index"] == 1
    assert by_id["rov_2"]["controller"] == "follower_controller"

    with pytest.raises(ValueError, match="team_summary"):
        build_manifest_row(
            results_root=root,
            metadata_path=metadata_path,
            metadata=metadata,
            summary_path=summary_path,
            event_path=event_path,
            summary={"participants": participants},
            track_name="Test Track",
            expected_gates_per_rover=12,
        )


def test_single_rover_default_gap_is_not_reported_as_active(tmp_path: Path) -> None:
    root = tmp_path / "results"
    run = root / "clean" / "run"
    run.mkdir(parents=True)
    metadata_path = run / "benchmark_metadata.json"
    summary_path = run / "race_summary.json"
    event_path = run / "race.jsonl"
    for path in (metadata_path, summary_path):
        path.write_text("{}", encoding="utf-8")
    event_path.write_text("", encoding="utf-8")
    row = build_manifest_row(
        results_root=root,
        metadata_path=metadata_path,
        metadata={
            "track": "track.json",
            "controller": "rule_gate_baseline",
            "num_rovers": 1,
            "staggered_start": False,
            "start_gap_s": 20.0,
            "seed": 0,
        },
        summary_path=summary_path,
        event_path=event_path,
        summary={
            "participants": [
                {
                    "participant_id": "rov",
                    "status": "TIMEOUT",
                    "completed_gates": 5,
                }
            ]
        },
        track_name="Test Track",
        expected_gates_per_rover=12,
    )

    assert row["start_gap_s"] is None
    assert row["fleet_configuration"]["start_gap_s"] is None
    assert row["status"] == "TIMEOUT"
    assert row["official_time_s"] is None
    assert row["penalized_time_s"] is None
