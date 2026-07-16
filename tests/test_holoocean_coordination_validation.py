"""Unit checks for coordination-validation provenance metadata."""

from argparse import Namespace
from pathlib import Path

import pytest

from marine_race_arena.scripts.run_holoocean_coordination_validation import (
    _artifact_contract_errors,
    _reproduction_argv,
    _runtime_contract,
    run_validation,
)


def test_holoocean_runtime_contract_is_explicit_and_disallows_fallback():
    assert _runtime_contract("holoocean") == {
        "adapter": "holoocean",
        "fallback_used": False,
        "fallback_allowed": False,
        "controller_observation_contract": "onboard_only_v1",
    }


def test_runtime_contract_marks_fallback_if_ever_supplied():
    assert _runtime_contract("fallback")["fallback_used"] is True


def _args(output_dir: Path) -> Namespace:
    return Namespace(
        output_dir=str(output_dir),
        track="marine_race_arena/tracks/marine_race_horseshoe_bay.json",
        inter_vehicle_modes=["diagnostic"],
        seeds=[0],
        conditions=["leader_follower"],
        team_size=1,
        min_gate_gap=2,
        start_gap_s=8.0,
        lateral_offset_m=1.5,
        duration_s=560.0,
        dt=0.033,
        comms_packet_loss_prob=0.0,
        team_id="test_team",
        headless=True,
        log_participant_states=True,
        invocation_argv=["--seeds", "0"],
    )


def test_reproduction_argv_isolates_one_run(tmp_path):
    args = _args(tmp_path / "output")
    argv = _reproduction_argv(
        args=args,
        seed=2,
        condition="leader_follower",
        mode="diagnostic",
        output_dir=tmp_path / "reproduction",
    )
    assert argv[argv.index("--seeds") + 1] == "2"
    assert argv[argv.index("--conditions") + 1] == "leader_follower"
    assert argv[argv.index("--inter-vehicle-modes") + 1] == "diagnostic"
    assert argv[argv.index("--output-dir") + 1] == str(tmp_path / "reproduction")
    assert "--log-participant-states" in argv


def test_artifact_contract_audit_requires_latency_count_match():
    summary = {
        **_runtime_contract("holoocean"),
        "local_progress": {},
        "comms": {
            "messages_delivered": 2,
            "delivery_latency_s": {"count": 1},
        },
    }
    runs = {
        "diagnostic": {
            "0": {"leader_follower": {"ok": True, "summary": summary}}
        }
    }
    assert _artifact_contract_errors(runs) == [
        "diagnostic/seed_0/leader_follower: latency/delivery counts differ"
    ]


def test_artifact_contract_audit_rejects_local_referee_progress_mismatch():
    summary = {
        **_runtime_contract("holoocean"),
        "participants": [
            {
                "participant_id": "bluerov2_01",
                "completed_gates": 11,
                "status": "RUNNING",
            }
        ],
        "local_progress": {
            "bluerov2_01": {
                "local_completed": 12,
                "advancements": 12,
                "status": "FINISHED",
            }
        },
    }
    runs = {
        "diagnostic": {"2": {"no_coordination": {"ok": True, "summary": summary}}}
    }

    assert _artifact_contract_errors(runs) == [
        "diagnostic/seed_2/no_coordination/bluerov2_01: "
        "local_completed=12, referee_completed=11",
        "diagnostic/seed_2/no_coordination/bluerov2_01: "
        "local/referee FINISHED status differs",
    ]


def test_run_validation_writes_provenance_and_rejects_mixed_output(monkeypatch, tmp_path):
    output_dir = tmp_path / "coordination"
    args = _args(output_dir)

    def fake_simulation(**kwargs):
        summary_path = kwargs["output_dir"] / "fake_summary.json"
        return {
            "ok": True,
            "summary": {
                **_runtime_contract("holoocean"),
                "participants": [
                    {
                        "participant_id": "bluerov2_01",
                        "completed_gates": 1,
                        "status": "FINISHED",
                        "collisions": 0,
                        "obstacle_collisions": 0,
                        "involved_inter_vehicle_collisions": 0,
                        "out_of_bounds_events": 0,
                        "stuck_events": 0,
                        "penalties_s": 0.0,
                    }
                ],
                "team_summary": {
                    "all_rovers_finished": True,
                    "rover_count": 1,
                    "expected_total_gates": 1,
                },
                "local_progress": {
                    "bluerov2_01": {
                        "status": "FINISHED",
                        "local_completed": 1,
                        "advancements": 1,
                    }
                },
                "comms": {
                    "messages_sent": 0,
                    "messages_delivered": 0,
                    "dropped_rate_limited": 0,
                    "dropped_oversized": 0,
                    "dropped_out_of_range": 0,
                    "dropped_packet_loss": 0,
                    "delivery_latency_s": {
                        "count": 0,
                        "min": None,
                        "mean": None,
                        "p50": None,
                        "p95": None,
                        "max": None,
                    },
                },
            },
            "summary_path": str(summary_path),
            "event_path": str(kwargs["output_dir"] / "fake.jsonl"),
        }

    monkeypatch.setattr(
        "marine_race_arena.scripts.run_holoocean_coordination_validation.simulate_holoocean_fleet",
        fake_simulation,
    )
    report = run_validation(args)
    result = report["runs"]["diagnostic"]["0"]["leader_follower"]
    assert report["all_runs_executed"] is True
    assert report["all_runs_executed"] is True
    assert report["all_progress_consistent"] is True
    assert report["artifact_contract_audit"] == {"ok": True, "errors": []}
    assert result["scientific_outcome"] == {
        "all_rovers_finished": True,
        "progress_consistent": True,
        "clean_finish": True,
    }
    assert result["metadata"]["source_tree_sha256"]
    assert result["metadata"]["reproduction_command"]
    assert result["summary"]["experiment_metadata"]["run_ok"] is True
    assert Path(result["metadata_path"]).is_file()
    assert Path(result["summary_path"]).is_file()

    with pytest.raises(ValueError, match="non-empty output directory"):
        run_validation(args)
