#!/usr/bin/env python3
"""Validate the exact 78-run article matrix and its onboard-only artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from marine_race_arena.scripts.validate_onboard_only_results import audit_results


CONTROLLERS = ("rule_gate_baseline", "rule_gate_center_then_commit")
TRACKS = {
    "horseshoe": ("marine_race_arena/tracks/marine_race_horseshoe_bay.json", 560.0),
    "vertical": ("marine_race_arena/tracks/marine_race_vertical_serpent.json", 900.0),
    "mixed": ("marine_race_arena/tracks/marine_race_mixed_endurance.json", 1300.0),
}


@dataclass(frozen=True)
class ExpectedRun:
    metadata_path: str
    fields: tuple[tuple[str, Any], ...]

    @property
    def expected_fields(self) -> dict[str, Any]:
        return dict(self.fields)


def expected_runs() -> list[ExpectedRun]:
    runs: list[ExpectedRun] = []
    seeds = range(5)

    for track_key, (track, duration) in TRACKS.items():
        for controller in CONTROLLERS:
            group = Path("clean") / track_key / controller
            for seed in seeds:
                runs.append(
                    _benchmark_run(
                        group,
                        seed,
                        task="clean_gate",
                        track=track,
                        controller=controller,
                        duration=duration,
                        current_profile="none",
                        num_rovers=1,
                    )
                )

    track, duration = TRACKS["horseshoe"]
    for controller in CONTROLLERS:
        for profile in ("medium", "strong"):
            group = Path("currents") / "horseshoe" / controller / profile
            for seed in seeds:
                runs.append(
                    _benchmark_run(
                        group,
                        seed,
                        task="current_gate",
                        track=track,
                        controller=controller,
                        duration=duration,
                        current_profile=profile,
                        num_rovers=1,
                    )
                )

    for controller in CONTROLLERS:
        group = Path("fleet_gap90") / controller
        for seed in seeds:
            runs.append(
                _benchmark_run(
                    group,
                    seed,
                    task="clean_gate",
                    track=track,
                    controller=controller,
                    duration=duration,
                    current_profile="none",
                    num_rovers=2,
                    start_gap_s=90.0,
                )
            )

    for gap_label, gap in (("gap_8", 8.0), ("gap_0", 0.0)):
        for seed in range(3):
            for condition in ("no_coordination", "leader_follower"):
                runs.append(
                    _coordination_run(
                        Path("coordination") / "main" / gap_label,
                        seed,
                        condition=condition,
                        start_gap_s=gap,
                        min_gate_gap=2,
                    )
                )
            runs.append(
                _coordination_run(
                    Path("coordination") / "min_gate_gap_1" / gap_label,
                    seed,
                    condition="leader_follower",
                    start_gap_s=gap,
                    min_gate_gap=1,
                )
            )

    if len(runs) != 78 or len({run.metadata_path for run in runs}) != 78:
        raise AssertionError("Internal expected-run manifest is not exactly 78 unique runs.")
    return runs


def _benchmark_run(
    group: Path,
    seed: int,
    *,
    task: str,
    track: str,
    controller: str,
    duration: float,
    current_profile: str,
    num_rovers: int,
    start_gap_s: float = 20.0,
) -> ExpectedRun:
    metadata = group / "runs" / f"run_{seed + 1:03d}_seed_{seed}" / "benchmark_metadata.json"
    fields: dict[str, Any] = {
        "seed": seed,
        "task": task,
        "track": track,
        "controller": controller,
        "duration_s": duration,
        "dt": 0.033,
        "official": True,
        "current_profile_requested": current_profile,
        "obstacles_requested": "none",
        "motion_compensation": "none",
        "num_rovers": num_rovers,
        "staggered_start": num_rovers > 1,
        "start_gap_s": start_gap_s,
    }
    if current_profile != "none":
        fields.update(
            {
                "physical_current_coupling_active": True,
                "current_result_acceptable": True,
            }
        )
    if num_rovers > 1:
        fields.update(
            {
                "staggered_lateral_offset_m": 3.0,
                "inter_vehicle_collision_mode": "diagnostic",
                "inter_vehicle_collision_xy_threshold_m": 0.8,
                "inter_vehicle_collision_z_threshold_m": 0.75,
                "inter_vehicle_collision_release_threshold_m": 1.05,
                "inter_vehicle_collision_cooldown_s": 1.0,
                "team_id": "fleet_01",
            }
        )
    return ExpectedRun(
        metadata.as_posix(),
        tuple(fields.items()),
    )


def _coordination_run(
    group: Path,
    seed: int,
    *,
    condition: str,
    start_gap_s: float,
    min_gate_gap: int,
) -> ExpectedRun:
    metadata = group / "diagnostic" / f"seed_{seed}" / condition / "experiment_metadata.json"
    controllers = ["rule_gate_baseline", *(["rule_gate_center_then_commit"] * 2)]
    if condition == "leader_follower":
        controllers = [f"leader_follower({controller})" for controller in controllers]
    return ExpectedRun(
        metadata.as_posix(),
        tuple(
            {
                "seed": seed,
                "experiment": "holoocean_coordination_validation",
                "track": TRACKS["horseshoe"][0],
                "controllers": controllers,
                "condition": condition,
                "inter_vehicle_collision_mode": "diagnostic",
                "team_size": 3,
                "start_gap_s": start_gap_s,
                "lateral_offset_m": 1.5,
                "min_gate_gap": min_gate_gap,
                "comms_packet_loss_prob": 0.0,
                "duration_s": 560.0,
                "dt": 0.033,
                "official": True,
                "current_profile": "none",
                "obstacles": "none",
            }.items()
        ),
    )


def validate_matrix(results_root: str | Path, *, project_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(results_root).resolve()
    expected = expected_runs()
    expected_paths = {run.metadata_path: run for run in expected}
    discovered_paths = {
        path.relative_to(root).as_posix(): path
        for filename in ("benchmark_metadata.json", "experiment_metadata.json")
        for path in root.rglob(filename)
        if path.is_file()
    } if root.is_dir() else {}

    errors: list[str] = []
    missing = sorted(set(expected_paths) - set(discovered_paths))
    unexpected = sorted(set(discovered_paths) - set(expected_paths))
    errors.extend(f"Missing expected metadata: {path}" for path in missing)
    errors.extend(f"Unexpected metadata: {path}" for path in unexpected)

    rows: list[dict[str, Any]] = []
    scientific_keys: set[tuple[Any, ...]] = set()
    source_hashes: set[str] = set()
    for relative_path in sorted(set(expected_paths) & set(discovered_paths)):
        path = discovered_paths[relative_path]
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"Unreadable metadata {relative_path}: {type(exc).__name__}: {exc}")
            continue
        if not isinstance(metadata, dict):
            errors.append(f"Metadata is not an object: {relative_path}")
            continue
        source_hash = metadata.get("source_tree_sha256")
        if isinstance(source_hash, str) and source_hash:
            source_hashes.add(source_hash)

        mismatches = []
        for field, expected_value in expected_paths[relative_path].expected_fields.items():
            actual = metadata.get(field)
            if actual != expected_value:
                mismatches.append(f"{field}={actual!r}, expected {expected_value!r}")
        errors.extend(f"{relative_path}: {mismatch}" for mismatch in mismatches)

        key = (
            metadata.get("experiment", "benchmark"),
            metadata.get("task"),
            metadata.get("track"),
            metadata.get("controller"),
            tuple(metadata.get("controllers") or []),
            metadata.get("condition"),
            metadata.get("current_profile_requested", metadata.get("current_profile")),
            metadata.get("num_rovers", metadata.get("team_size")),
            metadata.get("start_gap_s"),
            metadata.get("min_gate_gap"),
            metadata.get("seed"),
        )
        if key in scientific_keys:
            errors.append(f"Duplicate scientific condition: {key!r}")
        scientific_keys.add(key)
        rows.append(
            {
                "metadata_path": relative_path,
                "seed": metadata.get("seed"),
                "condition": metadata.get("condition"),
                "controller": metadata.get("controller"),
                "current_profile": metadata.get(
                    "current_profile_requested", metadata.get("current_profile")
                ),
                "field_mismatches": mismatches,
            }
        )

    if len(source_hashes) != 1:
        errors.append(f"Expected one source_tree_sha256 across the matrix, found {sorted(source_hashes)}")

    coverage_errors = list(errors)
    coverage_pass = not coverage_errors

    artifact_audit = audit_results(root, project_root=project_root)
    if not artifact_audit.get("audit_pass"):
        errors.append("The onboard-only execution/artifact/progress audit failed.")

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "results_root": str(root),
        "coverage_pass": coverage_pass,
        "matrix_pass": not errors,
        "expected_run_count": 78,
        "discovered_run_count": len(discovered_paths),
        "matched_run_count": len(rows),
        "source_tree_sha256": next(iter(source_hashes), None),
        "coverage_errors": coverage_errors,
        "errors": errors,
        "missing_metadata": missing,
        "unexpected_metadata": unexpected,
        "runs": rows,
        "artifact_audit": artifact_audit,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Final 78-run matrix audit",
        "",
        f"Overall matrix audit: **{'PASS' if report.get('matrix_pass') else 'FAIL'}**",
        f"Exact 78-run coverage and metadata: **{'PASS' if report.get('coverage_pass') else 'FAIL'}**",
        f"Onboard execution/artifact/progress audit: **{'PASS' if (report.get('artifact_audit') or {}).get('audit_pass') else 'FAIL'}**",
        "",
        f"- Expected runs: {report.get('expected_run_count')}",
        f"- Discovered runs: {report.get('discovered_run_count')}",
        f"- Matched runs: {report.get('matched_run_count')}",
        f"- Source fingerprint: `{report.get('source_tree_sha256')}`",
        "",
    ]
    errors = report.get("errors") or []
    if errors:
        lines.extend(["## Errors", ""])
        lines.extend(f"- {error}" for error in errors)
        lines.append("")
    lines.extend(
        [
            "| Metadata | Seed | Controller | Condition | Current | Fields |",
            "|---|---:|---|---|---|---:|",
        ]
    )
    for row in report.get("runs") or []:
        lines.append(
            "| {metadata_path} | {seed} | {controller} | {condition} | {current_profile} | {fields} |".format(
                **row,
                fields="ok" if not row.get("field_mismatches") else "mismatch",
            )
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--markdown-output", default=None)
    args = parser.parse_args(argv)

    root = Path(args.results_root)
    report = validate_matrix(root, project_root=args.project_root)
    json_path = Path(args.json_output) if args.json_output else root / "final_matrix_audit.json"
    markdown_path = (
        Path(args.markdown_output)
        if args.markdown_output
        else root / "final_matrix_audit.md"
    )
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(_markdown(report))
    print(f"Wrote {json_path} and {markdown_path}")
    return 0 if report["matrix_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
