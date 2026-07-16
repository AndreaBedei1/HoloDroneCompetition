from __future__ import annotations

import json

from validate_final_matrix import expected_runs, validate_matrix


def test_expected_final_manifest_has_78_unique_runs() -> None:
    runs = expected_runs()

    assert len(runs) == 78
    assert len({run.metadata_path for run in runs}) == 78


def test_empty_result_root_fails_with_all_runs_missing(tmp_path) -> None:
    report = validate_matrix(tmp_path, project_root=tmp_path)

    assert report["coverage_pass"] is False
    assert report["matrix_pass"] is False
    assert report["expected_run_count"] == 78
    assert report["discovered_run_count"] == 0
    assert len(report["missing_metadata"]) == 78


def _write_metadata(root, relative_path: str, metadata: dict) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata), encoding="utf-8")


def test_coordination_track_and_controller_mismatches_are_rejected(tmp_path) -> None:
    expected = next(run for run in expected_runs() if run.metadata_path.startswith("coordination/main/gap_8"))
    metadata = expected.expected_fields
    metadata["track"] = "marine_race_arena/tracks/wrong.json"
    metadata["controllers"] = ["wrong"]
    metadata["source_tree_sha256"] = "test-hash"
    _write_metadata(tmp_path, expected.metadata_path, metadata)

    report = validate_matrix(tmp_path, project_root=tmp_path)

    assert any("track=" in error and "wrong.json" in error for error in report["errors"])
    assert any("controllers=" in error and "wrong" in error for error in report["errors"])


def test_unexpected_metadata_and_duplicate_conditions_are_rejected(tmp_path) -> None:
    benchmark_runs = [run for run in expected_runs() if run.metadata_path.startswith("clean/horseshoe")][:2]
    duplicate = benchmark_runs[0].expected_fields
    duplicate["source_tree_sha256"] = "test-hash"
    for run in benchmark_runs:
        _write_metadata(tmp_path, run.metadata_path, duplicate)
    _write_metadata(tmp_path, "unexpected/benchmark_metadata.json", {})

    report = validate_matrix(tmp_path, project_root=tmp_path)

    assert report["unexpected_metadata"] == ["unexpected/benchmark_metadata.json"]
    assert any("Duplicate scientific condition" in error for error in report["errors"])


def test_complete_metadata_coverage_is_distinct_from_artifact_audit(tmp_path) -> None:
    for expected in expected_runs():
        metadata = dict(expected.expected_fields)
        metadata["source_tree_sha256"] = "test-hash"
        _write_metadata(tmp_path, expected.metadata_path, metadata)

    report = validate_matrix(tmp_path, project_root=tmp_path)

    assert report["coverage_pass"] is True
    assert report["coverage_errors"] == []
    assert report["matrix_pass"] is False
    assert report["artifact_audit"]["audit_pass"] is False
