"""Analyze collision events from a staggered multi-rover smoke run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from marine_race_arena.arena.arena_builder import ArenaBuilder
from marine_race_arena.config.loader import load_track_config
from marine_race_arena.config.schema import Vector3


CSV_FIELDS = [
    "participant_id",
    "collision_time_s",
    "completed_gates",
    "target_gate_id",
    "position",
    "nearest_gate_id",
    "distance_to_nearest_gate_center_m",
    "min_distance_to_other_rover_m",
    "nearest_other_rover_id",
    "other_rover_distances_m",
    "other_rover_states",
    "release_state_all_rovers",
]


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    event_path = Path(args.event_jsonl)
    summary_path = Path(args.summary_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _read_json(summary_path)
    track_path = Path(args.track) if args.track else Path(str(summary.get("track_file", "")))
    if not track_path:
        raise SystemExit("A track path is required when summary JSON does not include track_file.")
    events = _read_jsonl(event_path)
    analysis = analyze_events(events, summary, track_path)

    csv_path = output_dir / "collision_analysis.csv"
    json_path = output_dir / "collision_analysis.json"
    report_path = output_dir / "collision_analysis_report.md"
    _write_csv(csv_path, analysis["collisions"])
    _write_json(json_path, analysis)
    _write_report(report_path, analysis, event_path, summary_path, track_path)
    print(f"Collision analysis CSV: {csv_path}")
    print(f"Collision analysis JSON: {json_path}")
    print(f"Collision analysis report: {report_path}")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--track", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser


def analyze_events(
    events: list[dict[str, Any]],
    summary: Mapping[str, Any],
    track_path: Path,
) -> dict[str, Any]:
    config = load_track_config(track_path, debug=True)
    arena = ArenaBuilder(config).build()
    gate_centers = {gate_id: gate.center for gate_id, gate in arena.gate_map.items()}
    gate_sequence = list(config.track.gate_sequence)
    participant_ids = _participant_ids(summary, events)
    release_times = _release_times(summary, events, participant_ids)
    finish_times = _finish_times(events)
    gate_timelines = _gate_timelines(events, participant_ids)
    state_by_time = _participant_states_by_time(events)
    collision_rows = []

    for event in events:
        if event.get("event") != "collision":
            continue
        participant_id = str(event.get("participant_id") or "")
        time_s = _safe_float(event.get("time_s"), 0.0)
        time_key = _time_key(time_s)
        states_at_time = state_by_time.get(time_key, {})
        participant_state = states_at_time.get(participant_id, {})
        completed_gates = int(
            _safe_float(
                participant_state.get("completed_gates"),
                _completed_gates_at(gate_timelines.get(participant_id, []), time_s),
            )
        )
        target_gate_id = str(participant_state.get("target_gate_id") or _target_gate(gate_sequence, completed_gates))
        position = _vector3(event.get("position"))
        nearest_gate_id, nearest_gate_distance = _nearest_gate(position, gate_centers)
        other_distances, nearest_other_id, nearest_other_distance = _other_rover_distances(
            participant_id,
            position,
            states_at_time,
        )
        other_states = {
            other_id: _status_at_time(other_id, time_s, release_times, finish_times, states_at_time)
            for other_id in participant_ids
            if other_id != participant_id
        }
        release_state_all = {
            other_id: _release_state_at_time(other_id, time_s, release_times, finish_times, states_at_time)
            for other_id in participant_ids
        }
        collision_rows.append(
            {
                "participant_id": participant_id,
                "collision_time_s": round(time_s, 6),
                "completed_gates": completed_gates,
                "target_gate_id": target_gate_id,
                "position": position,
                "nearest_gate_id": nearest_gate_id,
                "distance_to_nearest_gate_center_m": nearest_gate_distance,
                "min_distance_to_other_rover_m": nearest_other_distance,
                "nearest_other_rover_id": nearest_other_id,
                "other_rover_distances_m": other_distances,
                "other_rover_states": other_states,
                "release_state_all_rovers": release_state_all,
            }
        )

    return {
        "summary": {
            "track": str(track_path),
            "event_path": None,
            "participant_count": len(participant_ids),
            "collision_count": len(collision_rows),
            "state_telemetry_available": bool(state_by_time),
        },
        "participants": _summary_participants(summary),
        "aggregates": _aggregates(collision_rows),
        "collisions": collision_rows,
    }


def _participant_ids(summary: Mapping[str, Any], events: Iterable[Mapping[str, Any]]) -> list[str]:
    ids = []
    participants = summary.get("participants", [])
    if isinstance(participants, list):
        for participant in participants:
            if isinstance(participant, Mapping) and participant.get("participant_id") is not None:
                ids.append(str(participant["participant_id"]))
    for event in events:
        participant_id = event.get("participant_id")
        if participant_id is not None:
            ids.append(str(participant_id))
    return sorted(set(ids))


def _release_times(
    summary: Mapping[str, Any],
    events: Iterable[Mapping[str, Any]],
    participant_ids: Iterable[str],
) -> dict[str, float | None]:
    release_times = {participant_id: None for participant_id in participant_ids}
    participants = summary.get("participants", [])
    if isinstance(participants, list):
        for participant in participants:
            if not isinstance(participant, Mapping):
                continue
            participant_id = participant.get("participant_id")
            if participant_id is not None and participant.get("release_time_s") is not None:
                release_times[str(participant_id)] = _safe_float(participant.get("release_time_s"), 0.0)
    for event in events:
        if event.get("event") == "participant_released" and event.get("participant_id") is not None:
            release_times[str(event["participant_id"])] = _safe_float(event.get("release_time_s", event.get("time_s")), 0.0)
    return release_times


def _finish_times(events: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    finish_times = {}
    for event in events:
        if event.get("event") == "race_finish" and event.get("participant_id") is not None:
            finish_times[str(event["participant_id"])] = _safe_float(event.get("time_s"), 0.0)
    return finish_times


def _gate_timelines(
    events: Iterable[Mapping[str, Any]],
    participant_ids: Iterable[str],
) -> dict[str, list[tuple[float, int]]]:
    timelines = {participant_id: [(0.0, 0)] for participant_id in participant_ids}
    counts = {participant_id: 0 for participant_id in participant_ids}
    for event in events:
        if event.get("event") != "gate_passed" or event.get("participant_id") is None:
            continue
        participant_id = str(event["participant_id"])
        counts[participant_id] = counts.get(participant_id, 0) + 1
        timelines.setdefault(participant_id, [(0.0, 0)]).append(
            (_safe_float(event.get("time_s"), 0.0), counts[participant_id])
        )
    return timelines


def _participant_states_by_time(events: Iterable[Mapping[str, Any]]) -> dict[float, dict[str, dict[str, Any]]]:
    states: dict[float, dict[str, dict[str, Any]]] = {}
    for event in events:
        if event.get("event") != "participant_state" or event.get("participant_id") is None:
            continue
        time_key = _time_key(_safe_float(event.get("time_s"), 0.0))
        states.setdefault(time_key, {})[str(event["participant_id"])] = dict(event)
    return states


def _completed_gates_at(timeline: list[tuple[float, int]], time_s: float) -> int:
    completed = 0
    for event_time, count in timeline:
        if event_time <= time_s + 1e-9:
            completed = count
        else:
            break
    return completed


def _target_gate(gate_sequence: list[str], completed_gates: int) -> str | None:
    if not gate_sequence:
        return None
    if completed_gates >= len(gate_sequence):
        return gate_sequence[-1]
    return gate_sequence[max(0, completed_gates)]


def _nearest_gate(position: Vector3 | None, gate_centers: Mapping[str, Vector3]) -> tuple[str | None, float | None]:
    if position is None:
        return None, None
    nearest_id = None
    nearest_distance = math.inf
    for gate_id, center in gate_centers.items():
        distance = _distance(position, center)
        if distance < nearest_distance:
            nearest_id = gate_id
            nearest_distance = distance
    return nearest_id, round(nearest_distance, 4) if math.isfinite(nearest_distance) else None


def _other_rover_distances(
    participant_id: str,
    position: Vector3 | None,
    states_at_time: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, float], str | None, float | None]:
    if position is None:
        return {}, None, None
    distances = {}
    nearest_id = None
    nearest_distance = math.inf
    for other_id, state in states_at_time.items():
        if other_id == participant_id:
            continue
        other_position = _vector3(state.get("position"))
        if other_position is None:
            continue
        distance = _distance(position, other_position)
        distances[other_id] = round(distance, 4)
        if distance < nearest_distance:
            nearest_id = other_id
            nearest_distance = distance
    if not distances:
        return {}, None, None
    return distances, nearest_id, round(nearest_distance, 4)


def _status_at_time(
    participant_id: str,
    time_s: float,
    release_times: Mapping[str, float | None],
    finish_times: Mapping[str, float],
    states_at_time: Mapping[str, Mapping[str, Any]],
) -> str:
    state = states_at_time.get(participant_id, {})
    if state.get("status"):
        return str(state["status"])
    release_time = release_times.get(participant_id)
    if release_time is None or time_s < release_time:
        return "WAITING"
    if participant_id in finish_times and time_s >= finish_times[participant_id]:
        return "FINISHED"
    return "RUNNING"


def _release_state_at_time(
    participant_id: str,
    time_s: float,
    release_times: Mapping[str, float | None],
    finish_times: Mapping[str, float],
    states_at_time: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    release_time = release_times.get(participant_id)
    return {
        "status": _status_at_time(participant_id, time_s, release_times, finish_times, states_at_time),
        "release_time_s": release_time,
        "released": release_time is not None and time_s >= release_time,
        "finished": participant_id in finish_times and time_s >= finish_times[participant_id],
    }


def _summary_participants(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    participants = summary.get("participants", [])
    if not isinstance(participants, list):
        return rows
    for participant in participants:
        if not isinstance(participant, Mapping):
            continue
        rows.append(
            {
                "participant_id": participant.get("participant_id"),
                "status": participant.get("status"),
                "completed_gates": participant.get("completed_gates"),
                "official_time_s": participant.get("official_time_s"),
                "collisions": participant.get("collisions"),
                "stuck_events": participant.get("stuck_events"),
                "out_of_bounds_events": participant.get("out_of_bounds_events"),
            }
        )
    return rows


def _aggregates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_participant: dict[str, int] = {}
    by_nearest_gate: dict[str, int] = {}
    near_gate = 0
    min_other_distances = []
    simultaneous_collision_times: dict[str, set[str]] = {}
    for row in rows:
        participant_id = str(row.get("participant_id") or "")
        by_participant[participant_id] = by_participant.get(participant_id, 0) + 1
        gate_id = str(row.get("nearest_gate_id") or "unknown")
        by_nearest_gate[gate_id] = by_nearest_gate.get(gate_id, 0) + 1
        distance_to_gate = row.get("distance_to_nearest_gate_center_m")
        if isinstance(distance_to_gate, (int, float)) and distance_to_gate <= 2.5:
            near_gate += 1
        min_other = row.get("min_distance_to_other_rover_m")
        if isinstance(min_other, (int, float)):
            min_other_distances.append(float(min_other))
        time_key = f"{_safe_float(row.get('collision_time_s'), 0.0):.3f}"
        simultaneous_collision_times.setdefault(time_key, set()).add(participant_id)
    simultaneous = {
        time_s: sorted(participants)
        for time_s, participants in simultaneous_collision_times.items()
        if len(participants) > 1
    }
    return {
        "by_participant": by_participant,
        "by_nearest_gate": dict(sorted(by_nearest_gate.items())),
        "near_gate_center_within_2_5m": near_gate,
        "near_gate_fraction_2_5m": (near_gate / len(rows)) if rows else 0.0,
        "min_other_rover_distance_m": min(min_other_distances) if min_other_distances else None,
        "mean_min_other_rover_distance_m": (
            sum(min_other_distances) / len(min_other_distances) if min_other_distances else None
        ),
        "simultaneous_collision_times": simultaneous,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in CSV_FIELDS})


def _write_report(
    path: Path,
    analysis: Mapping[str, Any],
    event_path: Path,
    summary_path: Path,
    track_path: Path,
) -> None:
    aggregates = analysis.get("aggregates", {})
    rows = analysis.get("collisions", [])
    lines = [
        "# Multi-Rover Collision Analysis",
        "",
        f"- event_jsonl: {event_path}",
        f"- summary_json: {summary_path}",
        f"- track: {track_path}",
        f"- collision_count: {len(rows) if isinstance(rows, list) else 0}",
        f"- state_telemetry_available: {analysis.get('summary', {}).get('state_telemetry_available')}",
        "",
        "## Participant Summaries",
        _participants_table(analysis.get("participants", [])),
        "",
        "## Aggregates",
        f"- collisions_by_participant: {json.dumps(aggregates.get('by_participant', {}), sort_keys=True)}",
        f"- collisions_by_nearest_gate: {json.dumps(aggregates.get('by_nearest_gate', {}), sort_keys=True)}",
        f"- near_gate_fraction_within_2_5m: {_fmt(aggregates.get('near_gate_fraction_2_5m'))}",
        f"- min_other_rover_distance_m: {_fmt(aggregates.get('min_other_rover_distance_m'))}",
        f"- mean_min_other_rover_distance_m: {_fmt(aggregates.get('mean_min_other_rover_distance_m'))}",
        f"- simultaneous_collision_times: {json.dumps(aggregates.get('simultaneous_collision_times', {}), sort_keys=True)}",
        "",
        "## First 20 Collisions",
        _collisions_table(rows[:20] if isinstance(rows, list) else []),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _participants_table(rows: Any) -> str:
    if not isinstance(rows, list) or not rows:
        return "_No participant summaries._"
    columns = ["participant_id", "status", "completed_gates", "official_time_s", "collisions", "stuck_events", "out_of_bounds_events"]
    return _markdown_table(columns, rows)


def _collisions_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No collisions._"
    columns = [
        "participant_id",
        "collision_time_s",
        "completed_gates",
        "target_gate_id",
        "nearest_gate_id",
        "distance_to_nearest_gate_center_m",
        "min_distance_to_other_rover_m",
        "nearest_other_rover_id",
    ]
    return _markdown_table(columns, rows)


def _markdown_table(columns: list[str], rows: list[Mapping[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(_csv_value(row.get(column))) for column in columns) + " |")
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _vector3(value: Any) -> Vector3 | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None


def _distance(a: Vector3, b: Vector3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _time_key(time_s: float) -> float:
    return round(float(time_s), 6)


def _safe_float(value: Any, default: float) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(converted):
        return default
    return converted


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "none"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
