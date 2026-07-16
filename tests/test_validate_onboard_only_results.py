"""Tests for the global onboard-only result artifact audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pytest

from marine_race_arena.scripts.validate_onboard_only_results import (
    _source_tree_sha256,
    audit_results,
    main,
)


def test_coordination_clean_run_passes_all_three_audit_layers(tmp_path: Path) -> None:
    _write_run(tmp_path, kind="coordination")

    report = audit_results(tmp_path, project_root=tmp_path)

    assert report["audit_pass"] is True
    assert report["summary"] == {
        "run_count": 1,
        "execution_failures": 0,
        "artifact_contract_failures": 0,
        "progress_mismatch_failures": 0,
        "referee_finished_runs": 1,
        "clean_runs": 1,
    }
    run = report["runs"][0]
    assert run["kind"] == "coordination"
    assert run["execution"] == {"ok": True, "errors": []}
    assert run["artifact_contract"]["ok"] is True
    assert run["scientific"]["progress_consistent"] is True
    assert run["scientific"]["all_referee_finished"] is True
    assert run["scientific"]["clean_finish"] is True


def test_coherent_benchmark_dnf_and_collision_remain_valid_scientific_results(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        kind="benchmark",
        referee_status="DNF",
        local_status="RUNNING",
        collisions=2,
        penalties_s=10.0,
    )

    report = audit_results(tmp_path, project_root=tmp_path)

    run = report["runs"][0]
    assert report["audit_pass"] is True
    assert run["execution"]["ok"] is True
    assert run["artifact_contract"]["ok"] is True
    assert run["scientific"]["progress_consistent"] is True
    assert run["scientific"]["all_referee_finished"] is False
    assert run["scientific"]["clean_finish"] is False
    assert run["scientific"]["totals"]["gate_world_collisions"] == 2
    assert run["scientific"]["totals"]["penalties_s"] == 10.0


@pytest.mark.parametrize(
    ("referee_completed", "local_completed", "referee_status", "local_status", "needle"),
    [
        (0, 1, "RUNNING", "FINISHED", "local_completed=1"),
        (1, 0, "DNF", "RUNNING", "local_completed=0"),
        (1, 1, "FINISHED", "RUNNING", "FINISHED parity differs"),
    ],
)
def test_summary_progress_mismatches_are_blocking(
    tmp_path: Path,
    referee_completed: int,
    local_completed: int,
    referee_status: str,
    local_status: str,
    needle: str,
) -> None:
    _write_run(
        tmp_path,
        referee_completed=referee_completed,
        local_completed=local_completed,
        referee_status=referee_status,
        local_status=local_status,
    )

    report = audit_results(tmp_path, project_root=tmp_path)

    run = report["runs"][0]
    assert run["execution"]["ok"] is True
    assert run["artifact_contract"]["ok"] is True
    assert run["scientific"]["progress_consistent"] is False
    assert report["audit_pass"] is False
    assert any(needle in error for error in run["scientific"]["progress_errors"])


def test_participant_set_and_advancement_count_must_match(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    summary_path = run_dir / "race_summary.json"
    summary = _read_json(summary_path)
    local = summary["local_progress"].pop("bluerov2_01")
    local["advancements"] = 2
    summary["local_progress"]["wrong_rover"] = local
    _write_json(summary_path, summary)

    report = audit_results(tmp_path, project_root=tmp_path)

    errors = report["runs"][0]["scientific"]["progress_errors"]
    assert report["audit_pass"] is False
    assert any("Participant sets differ" in error for error in errors)


def test_local_advancements_must_equal_local_completed(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    summary_path = run_dir / "race_summary.json"
    summary = _read_json(summary_path)
    summary["local_progress"]["bluerov2_01"]["advancements"] = 2
    _write_json(summary_path, summary)

    report = audit_results(tmp_path, project_root=tmp_path)

    errors = report["runs"][0]["scientific"]["progress_errors"]
    assert report["audit_pass"] is False
    assert any("advancements=2, local_completed=1" in error for error in errors)


def test_event_order_requires_referee_pass_no_later_than_local_plus_dt(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        referee_gate_times=[2.0],
        local_advancement_times=[1.0],
        dt=0.1,
    )

    report = audit_results(tmp_path, project_root=tmp_path)

    run = report["runs"][0]
    assert run["execution"]["ok"] is True
    assert run["artifact_contract"]["ok"] is True
    assert run["scientific"]["progress_consistent"] is False
    assert any(
        "precedes referee gate_passed" in error
        for error in run["scientific"]["progress_errors"]
    )


def test_local_advancement_without_referee_event_is_blocking(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        referee_completed=1,
        local_completed=1,
        referee_gate_times=[],
        local_advancement_times=[1.0],
    )

    report = audit_results(tmp_path, project_root=tmp_path)

    errors = report["runs"][0]["scientific"]["progress_errors"]
    assert report["audit_pass"] is False
    assert any("has no corresponding referee" in error for error in errors)
    assert any("event log has 0 gate_passed" in error for error in errors)


@pytest.mark.parametrize(
    "mutation",
    ["adapter", "fallback", "official", "contract", "track_hash"],
)
def test_runtime_and_track_contract_violations_are_blocking(
    tmp_path: Path,
    mutation: str,
) -> None:
    run_dir = _write_run(tmp_path)
    metadata_path = run_dir / "experiment_metadata.json"
    summary_path = run_dir / "race_summary.json"
    metadata = _read_json(metadata_path)
    summary = _read_json(summary_path)

    if mutation == "adapter":
        metadata["actual_adapter"] = "fallback"
        summary["adapter"] = "fallback"
    elif mutation == "fallback":
        metadata["fallback_used"] = True
        summary["fallback_used"] = True
    elif mutation == "official":
        metadata["official"] = False
        summary["validation_setup"]["official"] = False
    elif mutation == "contract":
        metadata["controller_observation_contract"] = "privileged"
        summary["controller_observation_contract"] = "privileged"
    else:
        metadata["track_sha256"] = "0" * 64
        summary["experiment_metadata"]["track_sha256"] = "0" * 64
    _write_json(metadata_path, metadata)
    _write_json(summary_path, summary)

    report = audit_results(tmp_path, project_root=tmp_path)

    run = report["runs"][0]
    assert run["execution"]["ok"] is True
    assert run["artifact_contract"]["ok"] is False
    assert run["scientific"]["progress_consistent"] is True
    assert report["audit_pass"] is False


def test_source_tree_hash_is_checked_when_present(tmp_path: Path) -> None:
    matching_source_hash = _source_tree_sha256(tmp_path)
    run_dir = _write_run(tmp_path, source_tree_sha256=matching_source_hash)

    matching = audit_results(tmp_path, project_root=tmp_path)
    assert matching["runs"][0]["artifact_contract"]["ok"] is True

    metadata_path = run_dir / "experiment_metadata.json"
    summary_path = run_dir / "race_summary.json"
    metadata = _read_json(metadata_path)
    summary = _read_json(summary_path)
    metadata["source_tree_sha256"] = "f" * 64
    summary["experiment_metadata"]["source_tree_sha256"] = "f" * 64
    _write_json(metadata_path, metadata)
    _write_json(summary_path, summary)

    stale = audit_results(tmp_path, project_root=tmp_path)
    errors = stale["runs"][0]["artifact_contract"]["errors"]
    assert stale["audit_pass"] is False
    assert any("Source-tree SHA-256 mismatch" in error for error in errors)


def test_cli_writes_both_manifests_and_exit_code_ignores_coherent_dnf(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        kind="benchmark",
        referee_status="DNF",
        local_status="RUNNING",
    )
    json_output = tmp_path / "audit" / "manifest.json"
    markdown_output = tmp_path / "audit" / "manifest.md"

    return_code = main(
        [
            str(tmp_path),
            "--project-root",
            str(tmp_path),
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )

    assert return_code == 0
    assert _read_json(json_output)["audit_pass"] is True
    assert "Overall audit: **PASS**" in markdown_output.read_text(encoding="utf-8")


def test_missing_or_failed_execution_is_blocking(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, kind="benchmark")
    metadata_path = run_dir / "benchmark_metadata.json"
    metadata = _read_json(metadata_path)
    metadata["return_code"] = 1
    _write_json(metadata_path, metadata)
    (run_dir / "race_events.jsonl").unlink()

    report = audit_results(tmp_path, project_root=tmp_path)

    assert report["audit_pass"] is False
    assert report["runs"][0]["execution"]["ok"] is False


def _write_run(
    root: Path,
    *,
    kind: str = "coordination",
    referee_status: str = "FINISHED",
    local_status: str = "FINISHED",
    referee_completed: int = 1,
    local_completed: int = 1,
    referee_gate_times: Optional[List[float]] = None,
    local_advancement_times: Optional[List[float]] = None,
    collisions: int = 0,
    penalties_s: float = 0.0,
    dt: float = 0.1,
    source_tree_sha256: Optional[str] = None,
) -> Path:
    run_dir = root / ("coordination_run" if kind == "coordination" else "benchmark_run")
    run_dir.mkdir(parents=True)
    track_path = root / "track.json"
    track_path.write_text('{"track": "test"}\n', encoding="utf-8")
    track_sha256 = hashlib.sha256(track_path.read_bytes()).hexdigest()
    metadata_name = (
        "experiment_metadata.json" if kind == "coordination" else "benchmark_metadata.json"
    )
    metadata: Dict[str, Any] = {
        "seed": 0,
        "track": str(track_path),
        "track_sha256": track_sha256,
        "actual_adapter": "holoocean",
        "fallback_used": False,
        "official": True,
        "controller_observation_contract": "onboard_only_v1",
        "dt": dt,
    }
    if kind == "coordination":
        metadata.update({"run_ok": True, "fallback_allowed": False, "condition": "test"})
        local_key = "local_progress"
    else:
        metadata.update(
            {
                "return_code": 0,
                "adapter": "holoocean",
                "allow_fallback": False,
                "fallback_disabled": True,
                "controller": "rule_gate_baseline",
            }
        )
        local_key = "controller_local_progress"
    if source_tree_sha256 is not None:
        metadata["source_tree_sha256"] = source_tree_sha256

    summary: Dict[str, Any] = {
        "adapter": "holoocean",
        "fallback_used": False,
        "controller_observation_contract": "onboard_only_v1",
        "participants": [
            {
                "participant_id": "bluerov2_01",
                "status": referee_status,
                "completed_gates": referee_completed,
                "official_time_s": 2.0 if referee_status == "FINISHED" else None,
                "penalties_s": penalties_s,
                "collisions": collisions,
                "obstacle_collisions": 0,
                "involved_inter_vehicle_collisions": 0,
                "out_of_bounds_events": 0,
                "stuck_events": 0,
            }
        ],
        local_key: {
            "bluerov2_01": {
                "status": local_status,
                "local_completed": local_completed,
                "advancements": local_completed,
            }
        },
    }
    if kind == "coordination":
        summary["fallback_allowed"] = False
        summary["validation_setup"] = {"official": True, "dt": dt, "track": str(track_path)}
        summary["experiment_metadata"] = dict(metadata)

    _write_json(run_dir / metadata_name, metadata)
    _write_json(run_dir / "race_summary.json", summary)
    referee_times = (
        referee_gate_times
        if referee_gate_times is not None
        else [1.0 + index for index in range(referee_completed)]
    )
    local_times = (
        local_advancement_times
        if local_advancement_times is not None
        else [1.5 + index for index in range(local_completed)]
    )
    events: List[Mapping[str, Any]] = [
        {
            "event": "controller_local_state",
            "participant_id": "bluerov2_01",
            "time_s": 0.0,
            "local_completed": 0,
        }
    ]
    events.extend(
        {
            "event": "gate_passed",
            "participant_id": "bluerov2_01",
            "time_s": time_s,
            "sequence_index": index,
            "gate_id": f"G{index + 1:02d}",
        }
        for index, time_s in enumerate(referee_times)
    )
    events.extend(
        {
            "event": "controller_local_state",
            "participant_id": "bluerov2_01",
            "time_s": time_s,
            "local_completed": index + 1,
        }
        for index, time_s in enumerate(local_times)
    )
    events.sort(key=lambda event: float(event["time_s"]))
    (run_dir / "race_events.jsonl").write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
    return run_dir


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
