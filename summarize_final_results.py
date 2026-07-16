#!/usr/bin/env python3
"""Build the complete 78-run manifest and local-versus-referee analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from validate_final_matrix import expected_runs, validate_matrix


MANIFEST_JSON = "complete_experiment_manifest.json"
MANIFEST_CSV = "complete_experiment_manifest.csv"
MANIFEST_MARKDOWN = "complete_experiment_manifest.md"
PARTICIPANTS_CSV = "complete_experiment_participants.csv"
AGGREGATES_JSON = "aggregated_results.json"
PROGRESS_JSON = "local_vs_referee_analysis.json"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int:
    number = _number(value)
    return 0 if number is None else int(number)


def _track_info(metadata: Mapping[str, Any], project_root: Path) -> tuple[str, int]:
    track_value = str(metadata.get("track") or "")
    track_path = Path(track_value)
    if not track_path.is_absolute():
        track_path = project_root / track_path
    track = _load_json(track_path)
    name = str((track.get("race") or {}).get("name") or track_path.stem)
    laps = _integer((track.get("race") or {}).get("laps")) or 1
    gate_sequence = (track.get("track") or {}).get("gate_sequence")
    if not isinstance(gate_sequence, list) or not gate_sequence:
        raise ValueError(f"Track {track_path} has no ordered gate_sequence")
    return name, len(gate_sequence) * laps


def _artifact_path(
    metadata_path: Path,
    metadata: Mapping[str, Any],
    key: str,
    pattern: str,
    project_root: Path,
    results_root: Path,
) -> Path:
    recorded = metadata.get(key)
    run_dir = metadata_path.parent.resolve()
    candidates: list[Path] = []
    if isinstance(recorded, str) and recorded:
        path = Path(recorded)
        recorded_path = (path if path.is_absolute() else project_root / path).resolve()
        if not recorded_path.is_file():
            raise ValueError(f"Recorded {key} does not exist: {recorded_path}")
        candidates.append(recorded_path)
    globbed = sorted(run_dir.glob(pattern))
    candidates.extend(globbed)
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        label = str(resolved).casefold()
        if label not in seen:
            seen.add(label)
            unique.append(resolved)
    existing = [path for path in unique if path.is_file()]
    if len(existing) != 1 or len(globbed) != 1:
        raise ValueError(
            f"Expected exactly one {key} for {metadata_path}, found "
            f"{[str(path) for path in existing]}"
        )
    artifact = existing[0]
    if artifact.parent != run_dir:
        raise ValueError(f"Recorded {key} leaves its run directory: {artifact}")
    try:
        artifact.relative_to(results_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Recorded {key} leaves the results root: {artifact}") from exc
    if artifact != globbed[0].resolve():
        raise ValueError(
            f"Recorded {key} disagrees with the unique run artifact: "
            f"recorded={artifact}, discovered={globbed[0].resolve()}"
        )
    return artifact


def _family(relative_metadata_path: str) -> tuple[str, str | None]:
    if relative_metadata_path.startswith("clean/"):
        return "clean", None
    if relative_metadata_path.startswith("currents/"):
        return "currents", None
    if relative_metadata_path.startswith("fleet_gap90/"):
        return "fleet_gap90", None
    if relative_metadata_path.startswith("coordination/main/"):
        return "coordination", "main"
    if relative_metadata_path.startswith("coordination/min_gate_gap_1/"):
        return "coordination", "min_gate_gap_1"
    return "unknown", None


def _participant_rows(
    summary: Mapping[str, Any], expected_gates_per_rover: int
) -> list[dict[str, Any]]:
    values = summary.get("participants")
    if not isinstance(values, list):
        return []
    rows: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, Mapping):
            continue
        rows.append(
            {
                "participant_id": value.get("participant_id"),
                "rank": value.get("rank"),
                "status": value.get("status"),
                "completed_gates": _integer(value.get("completed_gates")),
                "expected_gates": expected_gates_per_rover,
                "gate_world_collisions": max(
                    0,
                    _integer(value.get("collisions"))
                    - _integer(value.get("obstacle_collisions")),
                ),
                "obstacle_collisions": _integer(value.get("obstacle_collisions")),
                "proximity_events_involving_rover": _integer(
                    value.get("involved_inter_vehicle_collisions")
                ),
                "out_of_bounds_events": _integer(value.get("out_of_bounds_events")),
                "stuck_events": _integer(value.get("stuck_events")),
                "official_time_s": _number(value.get("official_time_s")),
                "penalized_time_s": _number(value.get("penalized_time_s")),
                "penalties_s": _number(value.get("penalties_s")) or 0.0,
                "release_time_s": _number(value.get("release_time_s")),
            }
        )
    return rows


def build_manifest_row(
    *,
    results_root: Path,
    metadata_path: Path,
    metadata: Mapping[str, Any],
    summary_path: Path,
    event_path: Path,
    summary: Mapping[str, Any],
    track_name: str,
    expected_gates_per_rover: int,
) -> dict[str, Any]:
    relative_metadata = metadata_path.relative_to(results_root).as_posix()
    participants = _participant_rows(summary, expected_gates_per_rover)
    if not participants:
        raise ValueError(f"No participant summaries in {summary_path}")

    num_rovers = _integer(metadata.get("num_rovers", metadata.get("team_size"))) or len(
        participants
    )
    if len(participants) != num_rovers:
        raise ValueError(
            f"Participant-count mismatch in {summary_path}: "
            f"metadata={num_rovers}, summary={len(participants)}"
        )
    team = summary.get("team_summary")
    if num_rovers > 1 and not isinstance(team, Mapping):
        raise ValueError(f"Multi-rover summary has no team_summary: {summary_path}")
    if isinstance(team, Mapping):
        required_team_fields = {
            "rover_count",
            "expected_total_gates",
            "total_completed_gates",
            "all_rovers_finished",
            "team_elapsed_time_s",
            "total_gate_collisions",
            "total_obstacle_collisions",
            "total_inter_vehicle_collisions",
            "total_collisions",
            "total_penalties_s",
            "team_penalized_time_s",
        }
        missing_team_fields = sorted(required_team_fields - set(team))
        if missing_team_fields:
            raise ValueError(
                f"Incomplete team_summary in {summary_path}: missing {missing_team_fields}"
            )
        if _integer(team.get("rover_count")) != len(participants):
            raise ValueError(
                f"Team rover-count mismatch in {summary_path}: "
                f"team={team.get('rover_count')}, participants={len(participants)}"
            )
        completed_gates = _integer(team.get("total_completed_gates"))
        participant_completed = sum(row["completed_gates"] for row in participants)
        if completed_gates != participant_completed:
            raise ValueError(
                f"Team completed-gate mismatch in {summary_path}: "
                f"team={completed_gates}, participants={participant_completed}"
            )
        expected_gates = _integer(team.get("expected_total_gates"))
        calculated_expected = expected_gates_per_rover * len(participants)
        if expected_gates != calculated_expected:
            raise ValueError(
                f"Team expected-gate mismatch in {summary_path}: "
                f"summary={expected_gates}, calculated={calculated_expected}"
            )
        all_finished = bool(team.get("all_rovers_finished"))
        participants_finished = all(row["status"] == "FINISHED" for row in participants)
        if all_finished != participants_finished:
            raise ValueError(
                f"Team finish-state mismatch in {summary_path}: "
                f"team={all_finished}, participants={participants_finished}"
            )
        status = "FINISHED" if all_finished else (
            f"{sum(row['status'] == 'FINISHED' for row in participants)}/"
            f"{len(participants)} FINISHED"
        )
        official_time = None
        penalized_time = None
        team_elapsed_time = _number(team.get("team_elapsed_time_s"))
        team_penalized_time = _number(team.get("team_penalized_time_s"))
        gate_world_collisions = _integer(team.get("total_gate_collisions"))
        obstacle_collisions = _integer(team.get("total_obstacle_collisions"))
        proximity_events = _integer(team.get("total_inter_vehicle_collisions"))
        penalties_s = _number(team.get("total_penalties_s")) or 0.0
        participant_gate_collisions = sum(
            row["gate_world_collisions"] for row in participants
        )
        participant_obstacle_collisions = sum(
            row["obstacle_collisions"] for row in participants
        )
        participant_proximity_involvements = sum(
            row["proximity_events_involving_rover"] for row in participants
        )
        if gate_world_collisions != participant_gate_collisions:
            raise ValueError(
                f"Team gate/world-collision mismatch in {summary_path}: "
                f"team={gate_world_collisions}, participants={participant_gate_collisions}"
            )
        if obstacle_collisions != participant_obstacle_collisions:
            raise ValueError(
                f"Team obstacle-collision mismatch in {summary_path}: "
                f"team={obstacle_collisions}, participants={participant_obstacle_collisions}"
            )
        if participant_proximity_involvements != 2 * proximity_events:
            raise ValueError(
                f"Team proximity-event mismatch in {summary_path}: "
                f"team={proximity_events}, participant involvements="
                f"{participant_proximity_involvements}"
            )
        total_collisions = _integer(team.get("total_collisions"))
        calculated_total_collisions = (
            gate_world_collisions + obstacle_collisions + proximity_events
        )
        if total_collisions != calculated_total_collisions:
            raise ValueError(
                f"Team total-collision mismatch in {summary_path}: "
                f"team={total_collisions}, calculated={calculated_total_collisions}"
            )
        if all_finished:
            if team_elapsed_time is None or team_penalized_time is None:
                raise ValueError(f"Finished team has missing time in {summary_path}")
            if not math.isclose(
                team_penalized_time,
                team_elapsed_time + penalties_s,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    f"Team penalized-time mismatch in {summary_path}: "
                    f"penalized={team_penalized_time}, elapsed={team_elapsed_time}, "
                    f"penalties={penalties_s}"
                )
    else:
        participant = participants[0]
        completed_gates = participant["completed_gates"]
        expected_gates = participant["expected_gates"]
        all_finished = participant["status"] == "FINISHED"
        status = str(participant["status"])
        official_time = participant["official_time_s"]
        penalized_time = participant["penalized_time_s"]
        team_elapsed_time = None
        team_penalized_time = None
        gate_world_collisions = participant["gate_world_collisions"]
        obstacle_collisions = participant["obstacle_collisions"]
        proximity_events = 0
        penalties_s = participant["penalties_s"]

    controllers_value = metadata.get("controllers")
    controllers = (
        [str(value) for value in controllers_value]
        if isinstance(controllers_value, list)
        else [str(metadata.get("controller"))]
    )
    coordinated_experiment = relative_metadata.startswith("coordination/")
    staggered_active = coordinated_experiment or bool(metadata.get("staggered_start"))
    start_gap = _number(metadata.get("start_gap_s")) if staggered_active else None
    lateral_offset = (
        _number(metadata.get("staggered_lateral_offset_m", metadata.get("lateral_offset_m")))
        if staggered_active
        else None
    )
    condition = metadata.get("condition")
    min_gate_gap_configured = metadata.get("min_gate_gap")
    min_gate_gap_effective = (
        min_gate_gap_configured if condition == "leader_follower" else None
    )
    current_profile = metadata.get(
        "current_profile_requested", metadata.get("current_profile", "none")
    )
    if not isinstance(controllers_value, list) and num_rovers > 1:
        controllers = controllers * num_rovers
    if len(controllers) != len(participants):
        raise ValueError(
            f"Controller/participant count mismatch in {metadata_path}: "
            f"controllers={len(controllers)}, participants={len(participants)}"
        )
    # Referee summaries are ranking-ordered, while metadata controllers are
    # creation/release-ordered.  The frozen runners create zero-padded IDs in
    # that same release order, so bind through the sorted IDs instead of row
    # position; rank remains an independent outcome field.
    ordered_participant_ids = sorted(str(row["participant_id"]) for row in participants)
    controller_by_id = dict(zip(ordered_participant_ids, controllers))
    release_index_by_id = {
        participant_id: index for index, participant_id in enumerate(ordered_participant_ids)
    }
    for participant in participants:
        participant_id = str(participant["participant_id"])
        participant["release_index"] = release_index_by_id[participant_id]
        participant["controller"] = controller_by_id[participant_id]
    fleet_configuration = {
        "enabled": num_rovers > 1,
        "num_rovers": num_rovers,
        "team_id": team.get("team_id") if isinstance(team, Mapping) else None,
        "start_gap_s": start_gap,
        "lateral_offset_m": lateral_offset,
        "condition": condition,
        "min_gate_gap_configured": min_gate_gap_configured,
        "min_gate_gap_effective": min_gate_gap_effective,
        "inter_vehicle_collision_mode": metadata.get("inter_vehicle_collision_mode"),
        "coordination_enabled": condition == "leader_follower",
        "comms_enabled": condition == "leader_follower",
        "comms_packet_loss_prob": metadata.get("comms_packet_loss_prob"),
    }
    current = {
        "requested_profile": current_profile,
        "resolved_profile": metadata.get("current_profile", current_profile),
        "physical_coupling_active": metadata.get("physical_current_coupling_active"),
        "result_acceptable": metadata.get("current_result_acceptable"),
    }
    execution_ok = (
        bool(metadata.get("run_ok"))
        if "run_ok" in metadata
        else _integer(metadata.get("return_code")) == 0
    )
    experiment, experiment_variant = _family(relative_metadata)
    return {
        "run_id": metadata_path.parent.relative_to(results_root).as_posix(),
        "kind": "coordination" if coordinated_experiment else "benchmark",
        "experiment": experiment,
        "experiment_variant": experiment_variant,
        "track": track_name,
        "track_file": metadata.get("track"),
        "controller": "; ".join(controllers),
        "controllers": controllers,
        "fleet_configuration": fleet_configuration,
        "current_profile": current_profile,
        "current": current,
        "start_gap_s": start_gap,
        "seed": _integer(metadata.get("seed")),
        "condition": condition,
        "min_gate_gap_configured": min_gate_gap_configured,
        "min_gate_gap_effective": min_gate_gap_effective,
        "result_path": metadata_path.parent.relative_to(results_root).as_posix(),
        "metadata_path": relative_metadata,
        "summary_path": summary_path.relative_to(results_root).as_posix(),
        "event_path": event_path.relative_to(results_root).as_posix(),
        "reproduction_command": metadata.get("reproduction_command"),
        "status": status,
        "all_rovers_finished": all_finished,
        "completed_gates": completed_gates,
        "expected_gates": expected_gates,
        "gate_world_collisions": gate_world_collisions,
        "obstacle_collisions": obstacle_collisions,
        "proximity_events": proximity_events,
        "out_of_bounds_events": sum(row["out_of_bounds_events"] for row in participants),
        "stuck_events": sum(row["stuck_events"] for row in participants),
        "official_time_s": official_time,
        "penalized_time_s": penalized_time,
        "team_elapsed_time_s": team_elapsed_time,
        "team_penalized_time_s": team_penalized_time,
        "penalties_s": penalties_s,
        "source_tree_sha256": metadata.get("source_tree_sha256"),
        "execution": {
            "ok": execution_ok,
            "return_code": metadata.get("return_code"),
            "error": metadata.get("error"),
            "actual_adapter": metadata.get("actual_adapter"),
            "fallback_used": metadata.get("fallback_used"),
            "controller_observation_contract": metadata.get(
                "controller_observation_contract"
            ),
        },
        "participants": participants,
    }


def analyze_progress_events(
    events: Iterable[Mapping[str, Any]],
    *,
    participant_ids: Sequence[str],
    expected_gates_per_rover: int,
    tolerance_s: float,
    summary_referee_completed: Mapping[str, int] | None = None,
    summary_local_completed: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    referee_times: dict[str, list[float]] = defaultdict(list)
    local_times: dict[str, list[float]] = defaultdict(list)
    previous_local: dict[str, int] = defaultdict(int)
    local_finish_time: dict[str, float] = {}
    event_errors: dict[str, list[str]] = defaultdict(list)

    for event in events:
        participant_id = event.get("participant_id")
        if not isinstance(participant_id, str):
            continue
        time_s = _number(event.get("time_s"))
        if time_s is None:
            continue
        if event.get("event") == "gate_passed":
            expected_index = len(referee_times[participant_id]) % expected_gates_per_rover
            sequence_index = _integer(event.get("sequence_index"))
            if sequence_index != expected_index:
                event_errors[participant_id].append(
                    f"gate_passed sequence_index={sequence_index}, expected {expected_index}"
                )
            referee_times[participant_id].append(time_s)
        elif event.get("event") == "controller_local_state":
            completed = _integer(event.get("local_completed"))
            previous = previous_local[participant_id]
            if completed < previous:
                event_errors[participant_id].append(
                    f"local_completed regressed from {previous} to {completed}"
                )
            if completed > previous:
                if completed > previous + 1:
                    event_errors[participant_id].append(
                        f"local_completed jumped from {previous} to {completed}"
                    )
                local_times[participant_id].extend([time_s] * (completed - previous))
                previous_local[participant_id] = completed
            if completed >= expected_gates_per_rover or event.get("local_status") == "FINISHED":
                local_finish_time.setdefault(participant_id, time_s)

    comparisons: list[dict[str, Any]] = []
    participant_reports: list[dict[str, Any]] = []
    totals = {
        "expected_advancements": 0,
        "referee_advancements": 0,
        "local_advancements": 0,
        "matched_advancements": 0,
        "false_local_advancements": 0,
        "missed_local_advancements": 0,
        "delayed_local_advancements": 0,
        "local_finish_before_referee": 0,
        "referee_finish_before_local": 0,
        "finish_same_tick": 0,
        "finish_order_pair_comparisons": 0,
        "finish_order_inversions": 0,
        "finish_order_ties": 0,
        "event_consistency_errors": 0,
    }
    delays: list[float] = []

    for participant_id in participant_ids:
        referee = referee_times.get(participant_id, [])
        local = local_times.get(participant_id, [])
        totals["expected_advancements"] += expected_gates_per_rover
        totals["referee_advancements"] += len(referee)
        totals["local_advancements"] += len(local)
        false_count = 0
        missed_count = 0
        delayed_count = 0
        participant_delays: list[float] = []
        errors = list(event_errors.get(participant_id, []))
        for ordinal in range(1, max(len(referee), len(local), expected_gates_per_rover) + 1):
            referee_time = referee[ordinal - 1] if ordinal <= len(referee) else None
            local_time = local[ordinal - 1] if ordinal <= len(local) else None
            delay = None
            classification = "not_reached"
            if referee_time is not None and local_time is not None:
                delay = local_time - referee_time
                delays.append(delay)
                participant_delays.append(delay)
                totals["matched_advancements"] += 1
                if delay < -tolerance_s:
                    classification = "false_early_local_advancement"
                    false_count += 1
                elif delay > tolerance_s:
                    classification = "delayed_local_advancement"
                    delayed_count += 1
                else:
                    classification = "same_tick"
            elif local_time is not None:
                classification = "false_unmatched_local_advancement"
                false_count += 1
            elif referee_time is not None:
                classification = "missed_local_advancement"
                missed_count += 1
            comparisons.append(
                {
                    "participant_id": participant_id,
                    "gate_ordinal": ordinal,
                    "referee_time_s": referee_time,
                    "local_time_s": local_time,
                    "delay_s": delay,
                    "classification": classification,
                }
            )

        totals["false_local_advancements"] += false_count
        totals["missed_local_advancements"] += missed_count
        totals["delayed_local_advancements"] += delayed_count
        if summary_referee_completed is not None:
            summary_count = _integer(summary_referee_completed.get(participant_id))
            if len(referee) != summary_count:
                errors.append(
                    f"JSONL referee count {len(referee)} != summary count {summary_count}"
                )
        if summary_local_completed is not None:
            summary_count = _integer(summary_local_completed.get(participant_id))
            if len(local) != summary_count:
                errors.append(
                    f"JSONL local count {len(local)} != summary count {summary_count}"
                )
        totals["event_consistency_errors"] += len(errors)
        referee_finish = (
            referee[expected_gates_per_rover - 1]
            if len(referee) >= expected_gates_per_rover
            else None
        )
        local_finish = local_finish_time.get(participant_id)
        finish_order = "not_comparable"
        if referee_finish is not None and local_finish is not None:
            finish_delta = local_finish - referee_finish
            if finish_delta < -tolerance_s:
                finish_order = "local_before_referee"
                totals["local_finish_before_referee"] += 1
            elif finish_delta > tolerance_s:
                finish_order = "referee_before_local"
                totals["referee_finish_before_local"] += 1
            else:
                finish_order = "same_tick"
                totals["finish_same_tick"] += 1
        else:
            finish_delta = None
        participant_reports.append(
            {
                "participant_id": participant_id,
                "expected_advancements": expected_gates_per_rover,
                "referee_advancements": len(referee),
                "local_advancements": len(local),
                "false_local_advancements": false_count,
                "missed_local_advancements": missed_count,
                "delayed_local_advancements": delayed_count,
                "advancement_delay_s": _stats(participant_delays),
                "referee_finish_time_s": referee_finish,
                "local_finish_time_s": local_finish,
                "finish_delta_s": finish_delta,
                "finish_order": finish_order,
                "consistent": false_count == 0 and missed_count == 0 and not errors,
                "errors": errors,
            }
        )

    finish_order_pairs: list[dict[str, Any]] = []
    for left_index, participant_a in enumerate(participant_ids):
        for participant_b in participant_ids[left_index + 1 :]:
            referee_a = (
                referee_times[participant_a][expected_gates_per_rover - 1]
                if len(referee_times[participant_a]) >= expected_gates_per_rover
                else None
            )
            referee_b = (
                referee_times[participant_b][expected_gates_per_rover - 1]
                if len(referee_times[participant_b]) >= expected_gates_per_rover
                else None
            )
            local_a = local_finish_time.get(participant_a)
            local_b = local_finish_time.get(participant_b)
            classification = "not_comparable"
            referee_delta = None
            local_delta = None
            if None not in (referee_a, referee_b, local_a, local_b):
                referee_delta = float(referee_a) - float(referee_b)
                local_delta = float(local_a) - float(local_b)
                referee_order = (
                    0
                    if abs(referee_delta) <= tolerance_s
                    else (-1 if referee_delta < 0.0 else 1)
                )
                local_order = (
                    0
                    if abs(local_delta) <= tolerance_s
                    else (-1 if local_delta < 0.0 else 1)
                )
                totals["finish_order_pair_comparisons"] += 1
                if referee_order == 0 or local_order == 0:
                    classification = "tie_within_tolerance"
                    totals["finish_order_ties"] += 1
                elif referee_order != local_order:
                    classification = "inversion"
                    totals["finish_order_inversions"] += 1
                else:
                    classification = "same_order"
            finish_order_pairs.append(
                {
                    "participant_a": participant_a,
                    "participant_b": participant_b,
                    "referee_finish_delta_s": referee_delta,
                    "local_finish_delta_s": local_delta,
                    "classification": classification,
                }
            )

    return {
        "tolerance_s": tolerance_s,
        "totals": totals,
        "advancement_delay_s": _stats(delays),
        "participants": participant_reports,
        "gate_comparisons": comparisons,
        "finish_order_pairs": finish_order_pairs,
        "errors": [
            f"{row['participant_id']}: {error}"
            for row in participant_reports
            for error in row["errors"]
        ],
    }


def _read_events(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Non-object event at {path}:{line_number}")
            yield value


def _stats(values: Iterable[Any]) -> dict[str, Any]:
    numbers = [value for value in (_number(item) for item in values) if value is not None]
    if not numbers:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    ordered = sorted(numbers)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(numbers),
        "mean": round(statistics.fmean(numbers), 6),
        "std": round(statistics.pstdev(numbers), 6),
        "median": round(statistics.median(numbers), 6),
        "p95": round(ordered[p95_index], 6),
        "min": round(min(numbers), 6),
        "max": round(max(numbers), 6),
    }


def _aggregate_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("experiment"),
        row.get("track"),
        row.get("controller"),
        row.get("current_profile"),
        row.get("start_gap_s"),
        row.get("condition"),
        row.get("min_gate_gap_effective"),
    )


def aggregate_results(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_aggregate_key(row)].append(row)
    aggregates: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        finished = [row for row in group if row.get("all_rovers_finished")]
        aggregates.append(
            {
                "experiment": key[0],
                "track": key[1],
                "controller": key[2],
                "current_profile": key[3],
                "start_gap_s": key[4],
                "condition": key[5],
                "min_gate_gap_effective": key[6],
                "runs": len(group),
                "finished_runs": len(finished),
                "completion_rate": len(finished) / len(group),
                "completed_gates": _stats(row.get("completed_gates") for row in group),
                "gate_world_collisions": _stats(
                    row.get("gate_world_collisions") for row in group
                ),
                "obstacle_collisions": _stats(row.get("obstacle_collisions") for row in group),
                "proximity_events": _stats(row.get("proximity_events") for row in group),
                "out_of_bounds_events": _stats(
                    row.get("out_of_bounds_events") for row in group
                ),
                "stuck_events": _stats(row.get("stuck_events") for row in group),
                "official_time_s_finished_only": _stats(
                    row.get("official_time_s") for row in finished
                ),
                "penalized_time_s_finished_only": _stats(
                    row.get("penalized_time_s") for row in finished
                ),
                "team_elapsed_time_s_finished_only": _stats(
                    row.get("team_elapsed_time_s") for row in finished
                ),
                "team_penalized_time_s_finished_only": _stats(
                    row.get("team_penalized_time_s") for row in finished
                ),
            }
        )
    return aggregates


def aggregate_participants(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        for participant in row.get("participants") or []:
            key = (
                row.get("experiment"),
                row.get("experiment_variant"),
                row.get("track"),
                row.get("current_profile"),
                row.get("start_gap_s"),
                row.get("condition"),
                row.get("min_gate_gap_effective"),
                participant.get("release_index"),
                participant.get("controller"),
            )
            groups[key].append(participant)
    aggregates: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        finished = [participant for participant in group if participant.get("status") == "FINISHED"]
        aggregates.append(
            {
                "experiment": key[0],
                "experiment_variant": key[1],
                "track": key[2],
                "current_profile": key[3],
                "start_gap_s": key[4],
                "condition": key[5],
                "min_gate_gap_effective": key[6],
                "release_index": key[7],
                "controller": key[8],
                "runs": len(group),
                "finished_runs": len(finished),
                "completed_gates": _stats(
                    participant.get("completed_gates") for participant in group
                ),
                "gate_world_collisions": _stats(
                    participant.get("gate_world_collisions") for participant in group
                ),
                "proximity_event_involvements": _stats(
                    participant.get("proximity_events_involving_rover")
                    for participant in group
                ),
                "out_of_bounds_events": _stats(
                    participant.get("out_of_bounds_events") for participant in group
                ),
                "stuck_events": _stats(
                    participant.get("stuck_events") for participant in group
                ),
                "official_time_s_finished_only": _stats(
                    participant.get("official_time_s") for participant in finished
                ),
                "penalized_time_s_finished_only": _stats(
                    participant.get("penalized_time_s") for participant in finished
                ),
            }
        )
    return aggregates


def _progress_totals(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total_keys = (
        "expected_advancements",
        "referee_advancements",
        "local_advancements",
        "matched_advancements",
        "false_local_advancements",
        "missed_local_advancements",
        "delayed_local_advancements",
        "local_finish_before_referee",
        "referee_finish_before_local",
        "finish_same_tick",
        "finish_order_pair_comparisons",
        "finish_order_inversions",
        "finish_order_ties",
        "event_consistency_errors",
    )
    totals = {
        key: sum(_integer((report.get("totals") or {}).get(key)) for report in reports)
        for key in total_keys
    }
    delays = [
        comparison.get("delay_s")
        for report in reports
        for comparison in report.get("gate_comparisons") or []
        if isinstance(comparison, Mapping)
    ]
    totals["advancement_delay_s"] = _stats(delays)
    totals["progress_reliable"] = (
        totals["false_local_advancements"] == 0
        and totals["missed_local_advancements"] == 0
        and totals["event_consistency_errors"] == 0
        and totals["finish_order_inversions"] == 0
    )
    return totals


def _progress_report(run_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_track: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for report in run_reports:
        by_track[str(report.get("track"))].append(report)
    return {
        "definition": {
            "false_local_advancement": (
                "A local increment precedes its same-ordinal referee crossing by more than "
                "one configured dt, or has no corresponding referee crossing."
            ),
            "missed_local_advancement": (
                "A referee crossing has no same-ordinal local increment in the run."
            ),
            "delay_s": "controller-local advancement time minus referee gate-crossing time.",
        },
        "overall": _progress_totals(run_reports),
        "by_track": {
            track: _progress_totals(reports) for track, reports in sorted(by_track.items())
        },
        "runs": list(run_reports),
    }


def build_reports(
    results_root: str | Path,
    *,
    project_root: str | Path | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    root = Path(results_root).resolve()
    repo = Path(project_root).resolve() if project_root else Path(__file__).resolve().parent
    matrix_validation = (
        validate_matrix(root, project_root=repo) if require_complete else None
    )
    if matrix_validation is not None and not matrix_validation.get("coverage_pass"):
        raise ValueError(
            "Final matrix coverage/metadata validation failed: "
            + "; ".join(matrix_validation.get("coverage_errors") or [])
        )
    expected = {run.metadata_path for run in expected_runs()}
    discovered = {
        path.relative_to(root).as_posix(): path
        for filename in ("benchmark_metadata.json", "experiment_metadata.json")
        for path in root.rglob(filename)
        if path.is_file()
    }
    missing = sorted(expected - set(discovered))
    unexpected = sorted(set(discovered) - expected)
    if require_complete and (missing or unexpected):
        raise ValueError(
            f"Final matrix is incomplete: missing={len(missing)}, unexpected={len(unexpected)}"
        )

    rows: list[dict[str, Any]] = []
    run_progress: list[dict[str, Any]] = []
    audit_by_metadata = {
        str(Path(run["metadata_path"]).resolve()).casefold(): run
        for run in ((matrix_validation or {}).get("artifact_audit") or {}).get("runs", [])
        if isinstance(run, Mapping) and isinstance(run.get("metadata_path"), str)
    }
    for relative_path in sorted(expected & set(discovered)):
        metadata_path = discovered[relative_path]
        metadata = _load_json(metadata_path)
        if require_complete and not metadata.get("completed_at"):
            raise ValueError(f"Run metadata has no completion timestamp: {metadata_path}")
        summary_path = _artifact_path(
            metadata_path,
            metadata,
            "summary_path",
            "*_summary.json",
            repo,
            root,
        )
        event_path = _artifact_path(
            metadata_path, metadata, "event_path", "*.jsonl", repo, root
        )
        summary = _load_json(summary_path)
        track_name, expected_gates_per_rover = _track_info(metadata, repo)
        row = build_manifest_row(
            results_root=root,
            metadata_path=metadata_path,
            metadata=metadata,
            summary_path=summary_path,
            event_path=event_path,
            summary=summary,
            track_name=track_name,
            expected_gates_per_rover=expected_gates_per_rover,
        )
        audit = audit_by_metadata.get(str(metadata_path.resolve()).casefold())
        if audit is not None:
            row["audit"] = {
                "execution_ok": bool((audit.get("execution") or {}).get("ok")),
                "artifact_contract_ok": bool(
                    (audit.get("artifact_contract") or {}).get("ok")
                ),
                "progress_consistent": bool(
                    (audit.get("scientific") or {}).get("progress_consistent")
                ),
                "clean_finish": bool((audit.get("scientific") or {}).get("clean_finish")),
            }
        rows.append(row)
        local_snapshot = summary.get("controller_local_progress")
        if not isinstance(local_snapshot, Mapping):
            local_snapshot = summary.get("local_progress")
        if not isinstance(local_snapshot, Mapping):
            local_snapshot = {}
        progress = analyze_progress_events(
            _read_events(event_path),
            participant_ids=[str(value["participant_id"]) for value in row["participants"]],
            expected_gates_per_rover=expected_gates_per_rover,
            tolerance_s=_number(metadata.get("dt")) or 0.033,
            summary_referee_completed={
                str(value["participant_id"]): _integer(value["completed_gates"])
                for value in row["participants"]
            },
            summary_local_completed={
                str(participant_id): _integer((value or {}).get("local_completed"))
                for participant_id, value in local_snapshot.items()
                if isinstance(value, Mapping)
            },
        )
        progress.update(
            {
                "experiment": row["experiment"],
                "track": row["track"],
                "controller": row["controller"],
                "condition": row["condition"],
                "start_gap_s": row["start_gap_s"],
                "seed": row["seed"],
                "event_path": row["event_path"],
            }
        )
        row["progress"] = {
            "totals": progress["totals"],
            "advancement_delay_s": progress["advancement_delay_s"],
            "participants": progress["participants"],
            "finish_order_pairs": progress["finish_order_pairs"],
            "errors": progress["errors"],
        }
        run_progress.append(progress)

    aggregates = {
        "runs": aggregate_results(rows),
        "participants": aggregate_participants(rows),
    }
    progress = _progress_report(run_progress)
    participant_record_count = sum(len(row["participants"]) for row in rows)
    if require_complete and participant_record_count != 124:
        raise ValueError(
            f"Expected 124 participant records across the matrix, found "
            f"{participant_record_count}"
        )
    source_hashes = sorted(
        {str(row["source_tree_sha256"]) for row in rows if row.get("source_tree_sha256")}
    )
    missing_source_hashes = [row["run_id"] for row in rows if not row.get("source_tree_sha256")]
    if require_complete and (missing_source_hashes or len(source_hashes) != 1):
        raise ValueError(
            "Expected one non-empty source fingerprint across every run: "
            f"hashes={source_hashes}, missing={missing_source_hashes}"
        )
    report = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "results_root": str(root),
        "complete_matrix": not missing and not unexpected and len(rows) == 78,
        "expected_run_count": 78,
        "run_count": len(rows),
        "missing_metadata": missing,
        "unexpected_metadata": unexpected,
        "participant_record_count": participant_record_count,
        "coverage_pass": (matrix_validation or {}).get("coverage_pass"),
        "matrix_audit_pass": (matrix_validation or {}).get("matrix_pass"),
        "onboard_audit_pass": ((matrix_validation or {}).get("artifact_audit") or {}).get(
            "audit_pass"
        ),
        "source_tree_sha256": source_hashes[0] if len(source_hashes) == 1 else source_hashes,
        "runs": rows,
        "aggregates": aggregates,
        "local_vs_referee": progress,
    }
    return report


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _write_outputs(report: Mapping[str, Any], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(root / MANIFEST_JSON, json.dumps(report, indent=2, sort_keys=True))
    rows = report.get("runs") or []
    csv_fields = [
        "experiment",
        "track",
        "controller",
        "fleet_configuration",
        "current_profile",
        "start_gap_s",
        "seed",
        "condition",
        "min_gate_gap_configured",
        "min_gate_gap_effective",
        "result_path",
        "status",
        "completed_gates",
        "expected_gates",
        "gate_world_collisions",
        "obstacle_collisions",
        "proximity_events",
        "out_of_bounds_events",
        "stuck_events",
        "official_time_s",
        "penalized_time_s",
        "team_elapsed_time_s",
        "team_penalized_time_s",
        "source_tree_sha256",
    ]
    _atomic_write_csv(
        root / MANIFEST_CSV,
        csv_fields,
        ({key: _csv_value(row.get(key)) for key in csv_fields} for row in rows),
    )
    participant_fields = [
        "run_id",
        "experiment",
        "experiment_variant",
        "track",
        "seed",
        "condition",
        "participant_id",
        "release_index",
        "controller",
        "rank",
        "status",
        "completed_gates",
        "expected_gates",
        "gate_world_collisions",
        "obstacle_collisions",
        "proximity_events_involving_rover",
        "out_of_bounds_events",
        "stuck_events",
        "official_time_s",
        "penalized_time_s",
        "penalties_s",
        "release_time_s",
    ]
    participant_rows = []
    for row in rows:
        shared = {key: row.get(key) for key in participant_fields if key != "participant_id"}
        for participant in row.get("participants") or []:
            participant_rows.append({**shared, **participant})
    _atomic_write_csv(root / PARTICIPANTS_CSV, participant_fields, participant_rows)
    _atomic_write_text(
        root / AGGREGATES_JSON,
        json.dumps(report.get("aggregates"), indent=2, sort_keys=True),
    )
    _atomic_write_text(
        root / PROGRESS_JSON,
        json.dumps(report.get("local_vs_referee"), indent=2, sort_keys=True),
    )
    _atomic_write_text(root / MANIFEST_MARKDOWN, _markdown(report))


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            if value and not value.endswith("\n"):
                handle.write("\n")
            handle.flush()
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_csv(
    path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _format_time(value: Any) -> str:
    number = _number(value)
    return "-" if number is None else f"{number:.3f}"


def _markdown(report: Mapping[str, Any]) -> str:
    progress = (report.get("local_vs_referee") or {}).get("overall") or {}
    lines = [
        "# Complete onboard-only experiment manifest",
        "",
        f"Runs: **{report.get('run_count')}/{report.get('expected_run_count')}**",
        f"Exact matrix coverage: **{'PASS' if report.get('complete_matrix') else 'FAIL'}**",
        f"False local advancements: **{progress.get('false_local_advancements', 0)}**",
        f"Missed local advancements: **{progress.get('missed_local_advancements', 0)}**",
        "",
        "| Experiment | Track | Controller(s) | Current | Gap | Seed | Condition | Status | Gates | Coll. | IV | Official/team elapsed | Penalized/team penalized | Path |",
        "|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report.get("runs") or []:
        display = dict(row)
        display.update(
            {
                "gap": "-" if row.get("start_gap_s") is None else row.get("start_gap_s"),
                "condition": row.get("condition") or "-",
                "official": _format_time(
                    row.get("official_time_s")
                    if row.get("official_time_s") is not None
                    else row.get("team_elapsed_time_s")
                ),
                "penalized": _format_time(
                    row.get("penalized_time_s")
                    if row.get("penalized_time_s") is not None
                    else row.get("team_penalized_time_s")
                ),
            }
        )
        lines.append(
            "| {experiment} | {track} | `{controller}` | {current_profile} | {gap} | {seed} | "
            "{condition} | {status} | {completed_gates}/{expected_gates} | "
            "{gate_world_collisions} | {proximity_events} | {official} | {penalized} | `{result_path}` |".format(
                **display
            )
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.results_root).resolve()
    try:
        report = build_reports(
            root,
            project_root=args.project_root,
            require_complete=not args.allow_partial,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    _write_outputs(report, root)
    print(_markdown(report))
    print(
        f"Wrote {MANIFEST_JSON}, {MANIFEST_CSV}, {MANIFEST_MARKDOWN}, "
        f"{PARTICIPANTS_CSV}, {AGGREGATES_JSON}, and {PROGRESS_JSON} under {root}"
    )
    return 0 if report.get("complete_matrix") and report.get("matrix_audit_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
