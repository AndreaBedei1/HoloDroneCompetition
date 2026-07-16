"""Audit onboard-only HoloOcean result artifacts without changing their outcome.

The validator deliberately separates three questions:

``execution``
    Did the run complete and produce readable metadata, summary, and JSONL?
``artifact_contract``
    Was it an official onboard-only HoloOcean run with fallback disabled, and
    do the recorded provenance hashes still match the files being audited?
``scientific``
    What happened in the race, and does controller-local course progress agree
    with the independent referee?  A coherent DNF or collision remains a valid
    scientific result.  A local/referee progress mismatch does not.

Both coordination-validation outputs (``experiment_metadata.json``) and normal
benchmark outputs (``benchmark_metadata.json``) are supported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
EXPECTED_ADAPTER = "holoocean"
EXPECTED_OBSERVATION_CONTRACT = "onboard_only_v1"
METADATA_FILENAMES = ("experiment_metadata.json", "benchmark_metadata.json")
DEFAULT_JSON_NAME = "onboard_only_audit.json"
DEFAULT_MARKDOWN_NAME = "onboard_only_audit.md"


def audit_results(
    results_root: Path | str,
    *,
    project_root: Path | str | None = None,
) -> Dict[str, Any]:
    """Audit every supported run below ``results_root``.

    The returned ``audit_pass`` is false only for execution errors, artifact
    contract errors, or controller-local/referee progress mismatches.  Race
    outcomes such as a coherent DNF, collision, or penalty are reported under
    ``scientific`` but do not by themselves fail the audit.
    """

    root = Path(results_root).resolve()
    repo_root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    current_source_sha256 = _source_tree_sha256(repo_root)
    metadata_paths = _discover_metadata(root)
    runs = [
        _audit_run(
            metadata_path,
            results_root=root,
            project_root=repo_root,
            current_source_sha256=current_source_sha256,
        )
        for metadata_path in metadata_paths
    ]

    discovery_errors: List[str] = []
    if not root.is_dir():
        discovery_errors.append(f"Results root does not exist or is not a directory: {root}")
    elif not metadata_paths:
        discovery_errors.append(
            "No experiment_metadata.json or benchmark_metadata.json files were found."
        )

    execution_failures = sum(not run["execution"]["ok"] for run in runs)
    artifact_failures = sum(not run["artifact_contract"]["ok"] for run in runs)
    progress_failures = sum(
        not run["scientific"]["progress_consistent"] for run in runs
    )
    referee_finished_runs = sum(
        bool(run["scientific"]["all_referee_finished"]) for run in runs
    )
    clean_runs = sum(bool(run["scientific"]["clean_finish"]) for run in runs)
    audit_pass = not discovery_errors and all(run["audit_pass"] for run in runs)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "results_root": str(root),
        "project_root": str(repo_root),
        "current_source_tree_sha256": current_source_sha256,
        "audit_pass": audit_pass,
        "discovery_errors": discovery_errors,
        "summary": {
            "run_count": len(runs),
            "execution_failures": execution_failures,
            "artifact_contract_failures": artifact_failures,
            "progress_mismatch_failures": progress_failures,
            "referee_finished_runs": referee_finished_runs,
            "clean_runs": clean_runs,
        },
        "runs": runs,
    }


def write_manifest(
    report: Mapping[str, Any],
    *,
    json_path: Path | str,
    markdown_path: Path | str,
) -> Tuple[Path, Path]:
    """Write the machine-readable and human-readable audit manifests."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_output.write_text(_markdown(report), encoding="utf-8")
    return json_output, markdown_output


def _discover_metadata(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    paths = {
        path.resolve()
        for filename in METADATA_FILENAMES
        for path in root.rglob(filename)
        if path.is_file()
    }
    return sorted(paths, key=lambda path: path.as_posix())


def _audit_run(
    metadata_path: Path,
    *,
    results_root: Path,
    project_root: Path,
    current_source_sha256: str,
) -> Dict[str, Any]:
    kind = (
        "coordination"
        if metadata_path.name == "experiment_metadata.json"
        else "benchmark"
    )
    run_dir = metadata_path.parent
    run_id = _relative_label(run_dir, results_root)
    execution_errors: List[str] = []
    artifact_errors: List[str] = []
    artifact_warnings: List[str] = []
    progress_errors: List[str] = []

    metadata = _load_json_object(metadata_path, execution_errors, "metadata")
    if metadata is None:
        return _failed_run_record(
            run_id=run_id,
            kind=kind,
            metadata_path=metadata_path,
            execution_errors=execution_errors,
        )

    _audit_execution_status(metadata, kind, execution_errors)
    summary_path = _locate_run_artifact(
        metadata=metadata,
        metadata_key="summary_path",
        run_dir=run_dir,
        project_root=project_root,
        pattern="*_summary.json",
        label="summary",
        errors=execution_errors,
    )
    event_path = _locate_run_artifact(
        metadata=metadata,
        metadata_key="event_path",
        run_dir=run_dir,
        project_root=project_root,
        pattern="*.jsonl",
        label="event log",
        errors=execution_errors,
    )

    summary = (
        _load_json_object(summary_path, execution_errors, "summary")
        if summary_path is not None
        else None
    )
    events = (
        _load_jsonl(event_path, execution_errors)
        if event_path is not None
        else None
    )

    if summary is not None:
        _audit_artifact_contract(
            metadata=metadata,
            summary=summary,
            project_root=project_root,
            current_source_sha256=current_source_sha256,
            errors=artifact_errors,
            warnings=artifact_warnings,
        )
        dt = _finite_float(metadata.get("dt"))
        if dt is None:
            setup = summary.get("validation_setup")
            if isinstance(setup, Mapping):
                dt = _finite_float(setup.get("dt"))
        if dt is None or dt < 0.0:
            artifact_errors.append("Missing or invalid non-negative dt for event ordering audit.")
            dt = 0.0
        scientific = _audit_scientific_outcome(
            summary=summary,
            events=events,
            dt=dt,
            progress_errors=progress_errors,
        )
    else:
        scientific = _empty_scientific_result(progress_errors)
        progress_errors.append("Progress audit unavailable because the summary is missing.")
        scientific["progress_consistent"] = False
        scientific["progress_errors"] = progress_errors

    if events is None:
        progress_errors.append("Progress audit unavailable because the event log is missing or invalid.")
        scientific["progress_consistent"] = False
        scientific["progress_errors"] = progress_errors

    execution_ok = not execution_errors
    artifact_ok = not artifact_errors
    progress_ok = bool(scientific["progress_consistent"])
    return {
        "run_id": run_id,
        "kind": kind,
        "metadata_path": str(metadata_path),
        "summary_path": str(summary_path) if summary_path is not None else None,
        "event_path": str(event_path) if event_path is not None else None,
        "seed": metadata.get("seed"),
        "condition": metadata.get("condition"),
        "controller": metadata.get("controller"),
        "track": metadata.get("track"),
        "execution": {"ok": execution_ok, "errors": execution_errors},
        "artifact_contract": {
            "ok": artifact_ok,
            "errors": artifact_errors,
            "warnings": artifact_warnings,
        },
        "scientific": scientific,
        "audit_pass": execution_ok and artifact_ok and progress_ok,
    }


def _failed_run_record(
    *,
    run_id: str,
    kind: str,
    metadata_path: Path,
    execution_errors: List[str],
) -> Dict[str, Any]:
    progress_errors = ["Progress audit unavailable because metadata could not be read."]
    scientific = _empty_scientific_result(progress_errors)
    return {
        "run_id": run_id,
        "kind": kind,
        "metadata_path": str(metadata_path),
        "summary_path": None,
        "event_path": None,
        "seed": None,
        "condition": None,
        "controller": None,
        "track": None,
        "execution": {"ok": False, "errors": execution_errors},
        "artifact_contract": {
            "ok": False,
            "errors": ["Artifact contract audit unavailable because metadata is invalid."],
            "warnings": [],
        },
        "scientific": scientific,
        "audit_pass": False,
    }


def _audit_execution_status(
    metadata: Mapping[str, Any],
    kind: str,
    errors: List[str],
) -> None:
    if kind == "coordination":
        if metadata.get("run_ok") is not True:
            errors.append(
                f"Coordination run_ok={metadata.get('run_ok')!r}; expected true."
            )
    else:
        return_code = metadata.get("return_code")
        if return_code is None:
            errors.append("Benchmark metadata is missing return_code.")
        elif _integer(return_code) != 0:
            errors.append(f"Benchmark return_code={return_code!r}; expected 0.")


def _locate_run_artifact(
    *,
    metadata: Mapping[str, Any],
    metadata_key: str,
    run_dir: Path,
    project_root: Path,
    pattern: str,
    label: str,
    errors: List[str],
) -> Optional[Path]:
    recorded = metadata.get(metadata_key)
    recorded_path: Optional[Path] = None
    if isinstance(recorded, str) and recorded.strip():
        recorded_path = _resolve_recorded_path(recorded, run_dir, project_root)
        if not recorded_path.is_file():
            errors.append(f"Recorded {label} does not exist: {recorded}")
            recorded_path = None

    candidates = sorted(path.resolve() for path in run_dir.glob(pattern) if path.is_file())
    if len(candidates) > 1:
        errors.append(
            f"Run directory contains {len(candidates)} candidate {label} files; expected one."
        )
    if recorded_path is not None:
        return recorded_path.resolve()
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        errors.append(f"Missing {label} file matching {pattern!r} in {run_dir}.")
    return None


def _resolve_recorded_path(value: str, run_dir: Path, project_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    project_candidate = project_root / path
    if project_candidate.exists():
        return project_candidate
    return run_dir / path


def _load_json_object(
    path: Optional[Path],
    errors: List[str],
    label: str,
) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"Cannot read {label} {path}: {type(exc).__name__}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{label.capitalize()} must contain a JSON object: {path}")
        return None
    return value


def _load_jsonl(path: Path, errors: List[str]) -> Optional[List[Dict[str, Any]]]:
    events: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(
                        f"Invalid JSONL at {path}:{line_number}: {exc.msg}."
                    )
                    return None
                if not isinstance(value, dict):
                    errors.append(f"JSONL entry at {path}:{line_number} is not an object.")
                    return None
                events.append(value)
    except (OSError, UnicodeError) as exc:
        errors.append(f"Cannot read event log {path}: {type(exc).__name__}: {exc}")
        return None
    if not events:
        errors.append(f"Event log is empty: {path}")
        return None
    return events


def _audit_artifact_contract(
    *,
    metadata: Mapping[str, Any],
    summary: Mapping[str, Any],
    project_root: Path,
    current_source_sha256: str,
    errors: List[str],
    warnings: List[str],
) -> None:
    embedded = summary.get("experiment_metadata")
    embedded_metadata = embedded if isinstance(embedded, Mapping) else {}
    setup = summary.get("validation_setup")
    validation_setup = setup if isinstance(setup, Mapping) else {}

    adapter_values = _present_values(
        ("metadata.actual_adapter", metadata.get("actual_adapter")),
        ("summary.adapter", summary.get("adapter")),
    )
    if not adapter_values:
        errors.append("Missing actual simulator adapter in metadata and summary.")
    for label, value in adapter_values:
        if value != EXPECTED_ADAPTER:
            errors.append(f"{label}={value!r}; expected {EXPECTED_ADAPTER!r}.")

    fallback_used_values = _present_values(
        ("metadata.fallback_used", metadata.get("fallback_used")),
        ("summary.fallback_used", summary.get("fallback_used")),
    )
    if not fallback_used_values:
        errors.append("Missing fallback_used in metadata and summary.")
    for label, value in fallback_used_values:
        if value is not False:
            errors.append(f"{label}={value!r}; expected false.")

    fallback_policy_values: List[Tuple[str, Any]] = []
    if "fallback_allowed" in metadata:
        fallback_policy_values.append(
            ("metadata.fallback_allowed", metadata.get("fallback_allowed"))
        )
    if "allow_fallback" in metadata:
        fallback_policy_values.append(
            ("metadata.allow_fallback", metadata.get("allow_fallback"))
        )
    if "fallback_disabled" in metadata:
        disabled = metadata.get("fallback_disabled")
        fallback_policy_values.append(
            (
                "metadata.fallback_disabled (inverted)",
                not disabled if isinstance(disabled, bool) else disabled,
            )
        )
    if "fallback_allowed" in summary:
        fallback_policy_values.append(
            ("summary.fallback_allowed", summary.get("fallback_allowed"))
        )
    if not fallback_policy_values:
        errors.append("Missing an explicit fallback-disabled policy.")
    for label, allowed in fallback_policy_values:
        if allowed is not False:
            errors.append(f"{label} implies fallback is allowed: {allowed!r}.")

    contract_values = _present_values(
        (
            "metadata.controller_observation_contract",
            metadata.get("controller_observation_contract"),
        ),
        (
            "summary.controller_observation_contract",
            summary.get("controller_observation_contract"),
        ),
    )
    if not contract_values:
        errors.append("Missing controller observation contract.")
    for label, value in contract_values:
        if value != EXPECTED_OBSERVATION_CONTRACT:
            errors.append(
                f"{label}={value!r}; expected {EXPECTED_OBSERVATION_CONTRACT!r}."
            )

    official_values = _present_values(
        ("metadata.official", metadata.get("official")),
        ("validation_setup.official", validation_setup.get("official")),
    )
    if not official_values:
        errors.append("Missing official-mode provenance.")
    for label, value in official_values:
        if value is not True:
            errors.append(f"{label}={value!r}; expected true.")

    track_sha_values = _present_values(
        ("metadata.track_sha256", metadata.get("track_sha256")),
        (
            "summary.experiment_metadata.track_sha256",
            embedded_metadata.get("track_sha256"),
        ),
    )
    if track_sha_values:
        unique_track_hashes = {str(value).lower() for _, value in track_sha_values}
        if len(unique_track_hashes) != 1:
            errors.append(f"Recorded track hashes disagree: {sorted(unique_track_hashes)}")
        track_value = metadata.get("track") or validation_setup.get("track") or summary.get(
            "track_file"
        )
        if not isinstance(track_value, str) or not track_value.strip():
            errors.append("Track hash is present but the track path is missing.")
        else:
            track_path = _resolve_project_path(track_value, project_root)
            actual_track_sha256 = _file_sha256(track_path)
            if actual_track_sha256 is None:
                errors.append(f"Cannot read recorded track file: {track_path}")
            elif actual_track_sha256.lower() not in unique_track_hashes:
                errors.append(
                    "Track SHA-256 mismatch: "
                    f"recorded={sorted(unique_track_hashes)}, actual={actual_track_sha256}."
                )
    else:
        warnings.append("No track_sha256 was recorded; track content could not be verified.")

    source_sha_values = _present_values(
        ("metadata.source_tree_sha256", metadata.get("source_tree_sha256")),
        (
            "summary.experiment_metadata.source_tree_sha256",
            embedded_metadata.get("source_tree_sha256"),
        ),
    )
    if source_sha_values:
        unique_source_hashes = {str(value).lower() for _, value in source_sha_values}
        if len(unique_source_hashes) != 1:
            errors.append(
                f"Recorded source-tree hashes disagree: {sorted(unique_source_hashes)}"
            )
        if current_source_sha256.lower() not in unique_source_hashes:
            errors.append(
                "Source-tree SHA-256 mismatch: "
                f"recorded={sorted(unique_source_hashes)}, current={current_source_sha256}."
            )
    else:
        warnings.append(
            "No source_tree_sha256 was recorded; exact dirty-worktree provenance is unavailable."
        )


def _audit_scientific_outcome(
    *,
    summary: Mapping[str, Any],
    events: Optional[Sequence[Mapping[str, Any]]],
    dt: float,
    progress_errors: List[str],
) -> Dict[str, Any]:
    participants_value = summary.get("participants")
    participant_rows: List[Dict[str, Any]] = []
    referee: Dict[str, Mapping[str, Any]] = {}
    if not isinstance(participants_value, list):
        progress_errors.append("Summary participants must be a list.")
        participants_value = []
    for value in participants_value:
        if not isinstance(value, Mapping):
            progress_errors.append("Summary contains a non-object participant entry.")
            continue
        participant_id = value.get("participant_id")
        if not isinstance(participant_id, str) or not participant_id:
            progress_errors.append("Summary participant is missing participant_id.")
            continue
        if participant_id in referee:
            progress_errors.append(f"Duplicate referee participant {participant_id!r}.")
            continue
        referee[participant_id] = value

    local_value = summary.get("local_progress")
    if not isinstance(local_value, Mapping):
        local_value = summary.get("controller_local_progress")
    local = local_value if isinstance(local_value, Mapping) else {}
    if not isinstance(local_value, Mapping):
        progress_errors.append(
            "Summary is missing local_progress/controller_local_progress."
        )

    referee_ids = set(referee)
    local_ids = {str(key) for key in local}
    if referee_ids != local_ids:
        progress_errors.append(
            "Participant sets differ: "
            f"referee_only={sorted(referee_ids - local_ids)}, "
            f"local_only={sorted(local_ids - referee_ids)}."
        )

    final_counts: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
    for participant_id in sorted(referee_ids | local_ids):
        referee_row = referee.get(participant_id, {})
        local_row_value = local.get(participant_id)
        local_row = local_row_value if isinstance(local_row_value, Mapping) else {}
        if participant_id in local and not isinstance(local_row_value, Mapping):
            progress_errors.append(
                f"Local progress for {participant_id} must be an object."
            )

        completed = _integer(referee_row.get("completed_gates"))
        local_completed = _integer(local_row.get("local_completed"))
        advancements = _integer(local_row.get("advancements"))
        final_counts[participant_id] = (completed, local_completed)
        if completed is None:
            progress_errors.append(
                f"{participant_id}: invalid referee completed_gates={referee_row.get('completed_gates')!r}."
            )
        if local_completed is None:
            progress_errors.append(
                f"{participant_id}: invalid local_completed={local_row.get('local_completed')!r}."
            )
        if completed is not None and local_completed is not None and completed != local_completed:
            progress_errors.append(
                f"{participant_id}: local_completed={local_completed}, "
                f"referee completed_gates={completed}."
            )
        if advancements is None:
            progress_errors.append(
                f"{participant_id}: invalid advancements={local_row.get('advancements')!r}."
            )
        elif local_completed is not None and advancements != local_completed:
            progress_errors.append(
                f"{participant_id}: advancements={advancements}, "
                f"local_completed={local_completed}."
            )

        referee_status = str(referee_row.get("status") or "UNKNOWN")
        local_status = str(local_row.get("status") or "UNKNOWN")
        if (referee_status == "FINISHED") != (local_status == "FINISHED"):
            progress_errors.append(
                f"{participant_id}: FINISHED parity differs "
                f"(referee={referee_status}, local={local_status})."
            )

        participant_rows.append(
            {
                "participant_id": participant_id,
                "referee_status": referee_status,
                "referee_completed_gates": completed,
                "local_status": local_status,
                "local_completed": local_completed,
                "local_advancements": advancements,
                "official_time_s": referee_row.get("official_time_s"),
                "penalties_s": _number_or_zero(referee_row.get("penalties_s")),
                "gate_world_collisions": _integer_or_zero(referee_row.get("collisions")),
                "obstacle_collisions": _integer_or_zero(
                    referee_row.get("obstacle_collisions")
                ),
                "inter_vehicle_events": _integer_or_zero(
                    referee_row.get("involved_inter_vehicle_collisions")
                ),
                "out_of_bounds_events": _integer_or_zero(
                    referee_row.get("out_of_bounds_events")
                ),
                "stuck_events": _integer_or_zero(referee_row.get("stuck_events")),
            }
        )

    if events is not None:
        _audit_progress_events(
            events=events,
            participant_ids=referee_ids,
            final_counts=final_counts,
            dt=dt,
            errors=progress_errors,
        )

    all_referee_finished = bool(participant_rows) and all(
        row["referee_status"] == "FINISHED" for row in participant_rows
    )
    total_gate_world_collisions = sum(
        row["gate_world_collisions"] for row in participant_rows
    )
    total_obstacle_collisions = sum(
        row["obstacle_collisions"] for row in participant_rows
    )
    total_inter_vehicle_events = sum(
        row["inter_vehicle_events"] for row in participant_rows
    )
    total_out_of_bounds_events = sum(
        row["out_of_bounds_events"] for row in participant_rows
    )
    total_stuck_events = sum(row["stuck_events"] for row in participant_rows)
    total_penalties_s = sum(row["penalties_s"] for row in participant_rows)
    clean_finish = all_referee_finished and not any(
        (
            total_gate_world_collisions,
            total_obstacle_collisions,
            total_inter_vehicle_events,
            total_out_of_bounds_events,
            total_stuck_events,
        )
    ) and math.isclose(total_penalties_s, 0.0, abs_tol=1e-9)

    return {
        "progress_consistent": not progress_errors,
        "progress_errors": progress_errors,
        "all_referee_finished": all_referee_finished,
        "clean_finish": clean_finish,
        "participants": participant_rows,
        "totals": {
            "gate_world_collisions": total_gate_world_collisions,
            "obstacle_collisions": total_obstacle_collisions,
            "inter_vehicle_events": total_inter_vehicle_events,
            "out_of_bounds_events": total_out_of_bounds_events,
            "stuck_events": total_stuck_events,
            "penalties_s": total_penalties_s,
        },
    }


def _audit_progress_events(
    *,
    events: Sequence[Mapping[str, Any]],
    participant_ids: set[str],
    final_counts: Mapping[str, Tuple[Optional[int], Optional[int]]],
    dt: float,
    errors: List[str],
) -> None:
    official_times: Dict[str, List[float]] = {participant_id: [] for participant_id in participant_ids}
    local_times: Dict[str, Dict[int, float]] = {
        participant_id: {} for participant_id in participant_ids
    }
    last_local: Dict[str, int] = {participant_id: 0 for participant_id in participant_ids}

    for event in events:
        event_type = event.get("event")
        participant_id = event.get("participant_id")
        if event_type not in {"gate_passed", "controller_local_state"}:
            continue
        if not isinstance(participant_id, str) or participant_id not in participant_ids:
            errors.append(
                f"{event_type}: unknown or missing participant_id={participant_id!r}."
            )
            continue
        time_s = _finite_float(event.get("time_s"))
        if time_s is None:
            errors.append(f"{participant_id}: {event_type} has invalid time_s.")
            continue

        if event_type == "gate_passed":
            expected_sequence_index = len(official_times[participant_id])
            sequence_index = _integer(event.get("sequence_index"))
            if sequence_index != expected_sequence_index:
                errors.append(
                    f"{participant_id}: gate_passed sequence_index={sequence_index!r}, "
                    f"expected {expected_sequence_index}."
                )
            official_times[participant_id].append(time_s)
            continue

        local_completed = _integer(event.get("local_completed"))
        if local_completed is None or local_completed < 0:
            errors.append(
                f"{participant_id}: controller_local_state has invalid local_completed."
            )
            continue
        previous = last_local[participant_id]
        if local_completed < previous:
            errors.append(
                f"{participant_id}: local_completed regressed from {previous} "
                f"to {local_completed}."
            )
            continue
        if local_completed > previous + 1:
            errors.append(
                f"{participant_id}: local_completed jumped from {previous} "
                f"to {local_completed}."
            )
        for advancement in range(previous + 1, local_completed + 1):
            local_times[participant_id].setdefault(advancement, time_s)
        last_local[participant_id] = local_completed

    for participant_id in sorted(participant_ids):
        referee_completed, local_completed = final_counts.get(participant_id, (None, None))
        if referee_completed is not None and len(official_times[participant_id]) != referee_completed:
            errors.append(
                f"{participant_id}: event log has {len(official_times[participant_id])} "
                f"gate_passed events, summary reports {referee_completed}."
            )
        if local_completed is None:
            continue
        if last_local[participant_id] != local_completed:
            errors.append(
                f"{participant_id}: event log ends at local_completed={last_local[participant_id]}, "
                f"summary reports {local_completed}."
            )
        for advancement in range(1, local_completed + 1):
            local_time = local_times[participant_id].get(advancement)
            if local_time is None:
                errors.append(
                    f"{participant_id}: missing controller-local advancement {advancement} in JSONL."
                )
                continue
            if len(official_times[participant_id]) < advancement:
                errors.append(
                    f"{participant_id}: local advancement {advancement} at {local_time:.6f}s "
                    "has no corresponding referee gate_passed event."
                )
                continue
            referee_time = official_times[participant_id][advancement - 1]
            if referee_time > local_time + dt + 1e-9:
                errors.append(
                    f"{participant_id}: local advancement {advancement} at {local_time:.6f}s "
                    f"precedes referee gate_passed at {referee_time:.6f}s by more than dt={dt:.6f}s."
                )


def _empty_scientific_result(progress_errors: List[str]) -> Dict[str, Any]:
    return {
        "progress_consistent": False,
        "progress_errors": progress_errors,
        "all_referee_finished": False,
        "clean_finish": False,
        "participants": [],
        "totals": {
            "gate_world_collisions": 0,
            "obstacle_collisions": 0,
            "inter_vehicle_events": 0,
            "out_of_bounds_events": 0,
            "stuck_events": 0,
            "penalties_s": 0.0,
        },
    }


def _present_values(*items: Tuple[str, Any]) -> List[Tuple[str, Any]]:
    return [(label, value) for label, value in items if value is not None]


def _resolve_project_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _file_sha256(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _source_tree_sha256(project_root: Path) -> str:
    digest = hashlib.sha256()
    source_root = project_root / "marine_race_arena"
    paths = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".py", ".json"}
    ) if source_root.is_dir() else []
    for path in paths:
        digest.update(path.relative_to(project_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _integer(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(numeric) or not math.isclose(numeric, converted, abs_tol=1e-9):
        return None
    return converted


def _integer_or_zero(value: Any) -> int:
    converted = _integer(value)
    return 0 if converted is None else converted


def _finite_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


def _number_or_zero(value: Any) -> float:
    converted = _finite_float(value)
    return 0.0 if converted is None else converted


def _relative_label(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return str(path)
    value = relative.as_posix()
    return value if value != "." else path.name


def _markdown(report: Mapping[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    lines = [
        "# Onboard-only result audit",
        "",
        f"Overall audit: **{'PASS' if report.get('audit_pass') else 'FAIL'}**",
        "",
        f"- Runs: {summary.get('run_count', 0)}",
        f"- Execution failures: {summary.get('execution_failures', 0)}",
        f"- Artifact-contract failures: {summary.get('artifact_contract_failures', 0)}",
        f"- Progress mismatches: {summary.get('progress_mismatch_failures', 0)}",
        f"- Referee-finished runs: {summary.get('referee_finished_runs', 0)}",
        f"- Clean runs: {summary.get('clean_runs', 0)}",
        "",
        "| Run | Kind | Execution | Artifact | Progress | Referee finished | Clean |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    runs = report.get("runs") if isinstance(report.get("runs"), list) else []
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        execution = run.get("execution") if isinstance(run.get("execution"), Mapping) else {}
        artifact = (
            run.get("artifact_contract")
            if isinstance(run.get("artifact_contract"), Mapping)
            else {}
        )
        scientific = (
            run.get("scientific") if isinstance(run.get("scientific"), Mapping) else {}
        )
        lines.append(
            "| {run} | {kind} | {execution} | {artifact} | {progress} | {finished} | {clean} |".format(
                run=run.get("run_id"),
                kind=run.get("kind"),
                execution=_yes_no(execution.get("ok")),
                artifact=_yes_no(artifact.get("ok")),
                progress=_yes_no(scientific.get("progress_consistent")),
                finished=_yes_no(scientific.get("all_referee_finished")),
                clean=_yes_no(scientific.get("clean_finish")),
            )
        )

    discovery_errors = report.get("discovery_errors")
    if isinstance(discovery_errors, list) and discovery_errors:
        lines.extend(["", "## Discovery errors", ""])
        lines.extend(f"- {error}" for error in discovery_errors)

    for run in runs:
        if not isinstance(run, Mapping):
            continue
        execution = run.get("execution") if isinstance(run.get("execution"), Mapping) else {}
        artifact = (
            run.get("artifact_contract")
            if isinstance(run.get("artifact_contract"), Mapping)
            else {}
        )
        scientific = (
            run.get("scientific") if isinstance(run.get("scientific"), Mapping) else {}
        )
        errors = [
            *(execution.get("errors") or []),
            *(artifact.get("errors") or []),
            *(scientific.get("progress_errors") or []),
        ]
        warnings = artifact.get("warnings") or []
        if errors or warnings:
            lines.extend(["", f"## {run.get('run_id')}", ""])
            lines.extend(f"- ERROR: {error}" for error in errors)
            lines.extend(f"- WARNING: {warning}" for warning in warnings)
    lines.append("")
    return "\n".join(lines)


def _yes_no(value: Any) -> str:
    return "yes" if value is True else "no"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", help="Root containing coordination and/or benchmark runs.")
    parser.add_argument(
        "--project-root",
        default=None,
        help="Repository root used to resolve tracks and calculate the source-tree hash.",
    )
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--markdown-output", default=None)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    results_root = Path(args.results_root)
    report = audit_results(results_root, project_root=args.project_root)
    json_path = Path(args.json_output) if args.json_output else results_root / DEFAULT_JSON_NAME
    markdown_path = (
        Path(args.markdown_output)
        if args.markdown_output
        else results_root / DEFAULT_MARKDOWN_NAME
    )
    written_json, written_markdown = write_manifest(
        report,
        json_path=json_path,
        markdown_path=markdown_path,
    )
    print(_markdown(report))
    print(f"Wrote {written_json} and {written_markdown}")
    return 0 if report["audit_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
