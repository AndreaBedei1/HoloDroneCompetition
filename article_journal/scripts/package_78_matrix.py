"""Build and verify the compact release package for the frozen 78-run matrix.

The raw matrix remains outside Git because it contains large per-run trees.  The
committed package is derived only from its complete experiment manifest, which
already contains the per-run metrics needed to reproduce the manuscript tables.
No simulator is launched by this script.

Build from a checkout that has the raw matrix available::

    python article_journal/scripts/package_78_matrix.py build \
      --source C:/path/to/results/onboard_only_validation/final_20260715

Verify the committed package in any checkout::

    python article_journal/scripts/package_78_matrix.py verify
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


PACKAGE = Path(__file__).resolve().parents[1] / "scientific_release" / "matrix_78_20260715"
RUN_FIELDS = [
    "run_id", "experiment", "experiment_variant", "kind", "track", "track_file",
    "controller", "controllers", "condition", "current_profile", "seed", "status",
    "expected_gates", "completed_gates", "official_time_s", "penalized_time_s",
    "penalties_s", "gate_world_collisions", "obstacle_collisions",
    "out_of_bounds_events", "proximity_events", "stuck_events", "all_rovers_finished",
    "start_gap_s", "min_gate_gap_configured", "min_gate_gap_effective",
    "team_elapsed_time_s", "team_penalized_time_s", "actual_adapter", "fallback_used",
    "source_tree_sha256", "summary_path", "event_path", "reproduction_command",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def scalar(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def normalize_run(run: Dict[str, Any]) -> Dict[str, Any]:
    execution = run.get("execution") or {}
    values = {
        "run_id": run.get("run_id"),
        "experiment": run.get("experiment"),
        "experiment_variant": run.get("experiment_variant"),
        "kind": run.get("kind"),
        "track": run.get("track"),
        "track_file": run.get("track_file"),
        "controller": run.get("controller"),
        "controllers": run.get("controllers"),
        "condition": run.get("condition"),
        "current_profile": run.get("current_profile"),
        "seed": run.get("seed"),
        "status": run.get("status"),
        "expected_gates": run.get("expected_gates"),
        "completed_gates": run.get("completed_gates"),
        "official_time_s": run.get("official_time_s"),
        "penalized_time_s": run.get("penalized_time_s"),
        "penalties_s": run.get("penalties_s"),
        "gate_world_collisions": run.get("gate_world_collisions"),
        "obstacle_collisions": run.get("obstacle_collisions"),
        "out_of_bounds_events": run.get("out_of_bounds_events"),
        "proximity_events": run.get("proximity_events"),
        "stuck_events": run.get("stuck_events"),
        "all_rovers_finished": run.get("all_rovers_finished"),
        "start_gap_s": run.get("start_gap_s"),
        "min_gate_gap_configured": run.get("min_gate_gap_configured"),
        "min_gate_gap_effective": run.get("min_gate_gap_effective"),
        "team_elapsed_time_s": run.get("team_elapsed_time_s"),
        "team_penalized_time_s": run.get("team_penalized_time_s"),
        "actual_adapter": execution.get("actual_adapter"),
        "fallback_used": execution.get("fallback_used"),
        "source_tree_sha256": run.get("source_tree_sha256"),
        "summary_path": run.get("summary_path"),
        "event_path": run.get("event_path"),
        "reproduction_command": run.get("reproduction_command"),
    }
    return {field: scalar(values[field]) for field in RUN_FIELDS}


def canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def build(source: Path) -> None:
    source_manifest = source / "complete_experiment_manifest.json"
    raw = read_json(source_manifest)
    runs = [normalize_run(run) for run in raw["runs"]]
    if len(runs) != 78 or raw.get("run_count") != 78 or not raw.get("complete_matrix"):
        raise SystemExit("source manifest is not the expected complete 78-run matrix")
    PACKAGE.mkdir(parents=True, exist_ok=True)
    runs_json = {"schema_version": "mra_78_run_rows_v1", "fields": RUN_FIELDS, "runs": runs}
    write_bytes(PACKAGE / "runs.json", canonical_json(runs_json))
    with (PACKAGE / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(runs)
    manifest = {
        "schema_version": "mra_78_run_release_v1",
        "source_manifest": "results/onboard_only_validation/final_20260715/complete_experiment_manifest.json",
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_tree_sha256": raw.get("source_tree_sha256"),
        "generated_at": raw.get("generated_at"),
        "expected_run_count": 78,
        "run_count": len(runs),
        "participant_record_count": raw.get("participant_record_count"),
        "coverage_pass": bool(raw.get("coverage_pass")),
        "matrix_audit_pass": bool(raw.get("matrix_audit_pass")),
        "onboard_audit_pass": bool(raw.get("onboard_audit_pass")),
        "notes": [
            "Per-run rows are a lossless compact projection of the committed source manifest.",
            "Raw per-run logs remain outside Git and are not modified by the packaging step.",
            "The package supports the manuscript tables; it does not claim physical validation.",
        ],
    }
    write_bytes(PACKAGE / "manifest.json", canonical_json(manifest))
    sums: List[str] = []
    for path in sorted(PACKAGE.iterdir()):
        if path.name in {"SHA256SUMS", "README.md"}:
            continue
        sums.append(f"{sha256_file(path)}  {path.name}")
    write_bytes(PACKAGE / "SHA256SUMS", ("\n".join(sums) + "\n").encode("ascii"))


def verify() -> None:
    manifest = read_json(PACKAGE / "manifest.json")
    rows = read_json(PACKAGE / "runs.json")
    if manifest.get("run_count") != 78 or len(rows.get("runs", [])) != 78:
        raise SystemExit("FAIL: package does not contain exactly 78 rows")
    if rows.get("fields") != RUN_FIELDS:
        raise SystemExit("FAIL: run field schema mismatch")
    with (PACKAGE / "runs.csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != 78 or [r["run_id"] for r in csv_rows] != [r["run_id"] for r in rows["runs"]]:
        raise SystemExit("FAIL: CSV and JSON rows differ")
    for line in (PACKAGE / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        if sha256_file(PACKAGE / name) != digest:
            raise SystemExit(f"FAIL: hash mismatch for {name}")
    print(f"PASS: 78 rows, {manifest.get('participant_record_count')} participant records, hashes verified")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--source", type=Path, required=True)
    sub.add_parser("verify")
    args = parser.parse_args()
    if args.command == "build":
        build(args.source.resolve())
        verify()
    else:
        verify()


if __name__ == "__main__":
    main()
