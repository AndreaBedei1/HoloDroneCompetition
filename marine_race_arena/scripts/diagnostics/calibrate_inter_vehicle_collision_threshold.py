"""Calibrate a geometric BlueROV2-vs-BlueROV2 collision threshold in HoloOcean."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from marine_race_arena.adapters import RaceAdapterError, select_adapter
from marine_race_arena.arena.arena_builder import ArenaBuilder
from marine_race_arena.config.loader import load_track_config
from marine_race_arena.config.schema import ParticipantConfig, TrackConfig, Vector3
from marine_race_arena.participants.participant import RaceParticipant


TRACK_PATH = Path("marine_race_arena/tracks/marine_race_horseshoe_bay.json")
OUTPUT_DIR = Path("results/benchmarks/inter_vehicle_collision_calibration")
AGENT_A = "bluerov2_01"
AGENT_B = "bluerov2_02"
SPAWN_TOLERANCE_M = 2.0
CSV_FIELDS = [
    "sample_id",
    "status",
    "offset_type",
    "longitudinal_offset_m",
    "lateral_offset_m",
    "vertical_offset_m",
    "relative_yaw_deg",
    "horizontal_distance_m",
    "vertical_distance_m",
    "distance_3d_m",
    "position_a",
    "position_b",
    "yaw_a_deg",
    "yaw_b_deg",
    "collision_a",
    "collision_b",
    "collision_any",
    "collision_sensor_keys_a",
    "collision_sensor_keys_b",
    "error",
]


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "calibration_samples.csv"
    summary_path = output_dir / "calibration_summary.json"
    report_path = output_dir / "calibration_report.md"

    config = load_track_config(
        args.track,
        benchmark_task="clean_gate",
        obstacles="none",
        current_profile="none",
        seed=args.seed,
    )
    samples = _generate_samples(config, quick=args.quick, max_samples=args.max_samples)
    completed_ids = _completed_sample_ids(csv_path) if args.resume else set()
    write_header = not csv_path.exists() or not args.resume
    if write_header:
        csv_path.write_text("", encoding="utf-8")

    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    try:
        with csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            for sample in samples:
                if int(sample["sample_id"]) in completed_ids:
                    continue
                row = _run_sample(config, sample, seed=args.seed, settle_ticks=args.settle_ticks, dt=args.dt)
                writer.writerow({field: _csv_value(row.get(field)) for field in CSV_FIELDS})
                handle.flush()
                rows.append(row)
                if args.progress and len(rows) % max(1, int(args.progress)) == 0:
                    print(f"Completed {len(rows)} new calibration samples; latest={row['sample_id']}.")
    except Exception as exc:
        summary = _summarize_rows(_load_rows(csv_path), wall_time_s=time.monotonic() - started)
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        report_path.write_text(_render_report(summary, csv_path), encoding="utf-8")
        print(f"Calibration failed: {exc}", file=sys.stderr)
        print(f"Partial samples: {csv_path}", file=sys.stderr)
        return 1

    all_rows = _load_rows(csv_path)
    summary = _summarize_rows(all_rows, wall_time_s=time.monotonic() - started)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(_render_report(summary, csv_path), encoding="utf-8")
    if int(summary.get("successful_samples") or 0) == 0:
        print(
            "Calibration failed: no valid two-agent HoloOcean placements were recorded. "
            f"See {report_path}.",
            file=sys.stderr,
        )
        return 1
    print(f"Calibration samples: {csv_path}")
    print(f"Calibration summary: {summary_path}")
    print(f"Calibration report: {report_path}")
    print(
        "Recommended thresholds: "
        f"xy={summary['recommendation']['inter_vehicle_collision_xy_threshold_m']} m, "
        f"z={summary['recommendation']['inter_vehicle_collision_z_threshold_m']} m, "
        f"release={summary['recommendation']['hysteresis_release_threshold_m']} m, "
        f"cooldown={summary['recommendation']['recommended_cooldown_s']} s"
    )
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default=str(TRACK_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.033)
    parser.add_argument("--settle-ticks", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Run a small smoke calibration.")
    parser.add_argument("--resume", action="store_true", help="Append only missing samples to an existing CSV.")
    parser.add_argument("--progress", type=int, default=25, help="Print progress every N new samples.")
    return parser


def _generate_samples(
    config: TrackConfig,
    *,
    quick: bool,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    if quick:
        specs = [
            (0.0, -0.75, 0.0, 0.0),
            (0.0, 0.75, 0.0, 90.0),
            (0.0, -1.25, 0.0, 0.0),
            (0.0, 1.25, 0.0, 90.0),
            (0.0, -1.75, 0.0, 180.0),
            (0.0, 1.75, 0.0, 45.0),
            (-0.75, 0.0, 0.0, 0.0),
            (1.25, 0.0, 0.0, 90.0),
            (-1.75, 0.0, 0.0, 180.0),
            (0.75, 0.75, 0.0, 45.0),
            (1.25, -1.25, 0.0, 135.0),
            (0.0, 1.25, 0.75, 0.0),
        ]
        return _samples_from_specs(config, specs[:max_samples] if max_samples else specs)
    longitudinal_values = [-1.75, -1.25, -0.75, -0.35, 0.0, 0.35, 0.75, 1.25, 1.75]
    lateral_values = [-1.75, -1.25, -0.75, -0.35, 0.0, 0.35, 0.75, 1.25, 1.75]
    vertical_values = [-0.75, -0.35, 0.0, 0.35, 0.75]
    yaw_values = [0.0, 45.0, 90.0, 135.0, 180.0]
    samples: list[dict[str, Any]] = []
    sample_id = 0
    for longitudinal in longitudinal_values:
        for lateral in lateral_values:
            for vertical in vertical_values:
                for yaw in yaw_values:
                    if math.sqrt(longitudinal**2 + lateral**2 + vertical**2) < 0.15:
                        continue
                    position_a, position_b, yaw_a, yaw_b = _relative_pose(
                        config,
                        longitudinal_offset_m=longitudinal,
                        lateral_offset_m=lateral,
                        vertical_offset_m=vertical,
                        relative_yaw_deg=yaw,
                    )
                    samples.append(
                        {
                            "sample_id": sample_id,
                            "offset_type": _offset_type(longitudinal, lateral, vertical),
                            "longitudinal_offset_m": longitudinal,
                            "lateral_offset_m": lateral,
                            "vertical_offset_m": vertical,
                            "relative_yaw_deg": yaw,
                            "position_a": position_a,
                            "position_b": position_b,
                            "yaw_a_deg": yaw_a,
                            "yaw_b_deg": yaw_b,
                        }
                    )
                    sample_id += 1
    samples = sorted(
        samples,
        key=lambda item: (
            _distance(item["position_a"], item["position_b"]),
            abs(float(item["relative_yaw_deg"])),
            int(item["sample_id"]),
        ),
    )
    if max_samples is not None:
        samples = samples[: max(0, int(max_samples))]
    return samples


def _samples_from_specs(
    config: TrackConfig,
    specs: Iterable[tuple[float, float, float, float]],
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for sample_id, (longitudinal, lateral, vertical, yaw) in enumerate(specs):
        position_a, position_b, yaw_a, yaw_b = _relative_pose(
            config,
            longitudinal_offset_m=longitudinal,
            lateral_offset_m=lateral,
            vertical_offset_m=vertical,
            relative_yaw_deg=yaw,
        )
        samples.append(
            {
                "sample_id": sample_id,
                "offset_type": _offset_type(longitudinal, lateral, vertical),
                "longitudinal_offset_m": longitudinal,
                "lateral_offset_m": lateral,
                "vertical_offset_m": vertical,
                "relative_yaw_deg": yaw,
                "position_a": position_a,
                "position_b": position_b,
                "yaw_a_deg": yaw_a,
                "yaw_b_deg": yaw_b,
            }
        )
    return samples


def _run_sample(
    base_config: TrackConfig,
    sample: Mapping[str, Any],
    *,
    seed: int,
    settle_ticks: int,
    dt: float,
) -> dict[str, Any]:
    config, participants = _sample_config(base_config, sample)
    arena = ArenaBuilder(config, seed=seed).build()
    adapter = select_adapter(
        "holoocean",
        config=config,
        arena=arena,
        allow_fallback=False,
        headless=True,
        record=False,
        seed=seed,
    )
    try:
        adapter.spawn_participants(participants)
        adapter.reset()
        for _ in range(max(1, int(settle_ticks))):
            for participant in participants.values():
                adapter.apply_command(participant.id, _zero_command(), participant.config.control_mode)
            adapter.step(dt)
        state_a = adapter.get_participant_state(AGENT_A)
        state_b = adapter.get_participant_state(AGENT_B)
        keys_a = _collision_sensor_keys(state_a.raw_sensors)
        keys_b = _collision_sensor_keys(state_b.raw_sensors)
        if not keys_a or not keys_b:
            raise RaceAdapterError(
                "CollisionSensor data is missing for calibration agents. "
                f"{AGENT_A} keys={sorted(state_a.raw_sensors.keys())}; "
                f"{AGENT_B} keys={sorted(state_b.raw_sensors.keys())}."
            )
        requested_a = _vector3(sample["position_a"])
        requested_b = _vector3(sample["position_b"])
        spawn_error_a = _distance(requested_a, state_a.position)
        spawn_error_b = _distance(requested_b, state_b.position)
        if spawn_error_a > SPAWN_TOLERANCE_M or spawn_error_b > SPAWN_TOLERANCE_M:
            return _row_from_sample(
                sample,
                status="SPAWN_MISMATCH",
                position_a=state_a.position,
                position_b=state_b.position,
                collision_a=adapter.get_collision_state(AGENT_A),
                collision_b=adapter.get_collision_state(AGENT_B),
                sensor_keys_a=keys_a,
                sensor_keys_b=keys_b,
                error=(
                    "HoloOcean did not spawn both calibration agents near the requested poses. "
                    f"{AGENT_A} requested={requested_a} actual={state_a.position} error={spawn_error_a:.3f}m; "
                    f"{AGENT_B} requested={requested_b} actual={state_b.position} error={spawn_error_b:.3f}m."
                ),
            )
        collision_a = adapter.get_collision_state(AGENT_A)
        collision_b = adapter.get_collision_state(AGENT_B)
        position_a = state_a.position
        position_b = state_b.position
        return _row_from_sample(
            sample,
            status="OK",
            position_a=position_a,
            position_b=position_b,
            collision_a=collision_a,
            collision_b=collision_b,
            sensor_keys_a=keys_a,
            sensor_keys_b=keys_b,
            error="",
        )
    finally:
        adapter.close()


def _sample_config(
    config: TrackConfig,
    sample: Mapping[str, Any],
) -> tuple[TrackConfig, dict[str, RaceParticipant]]:
    if not config.participants:
        raise ValueError("Calibration track must contain at least one participant template.")
    base = config.participants[0]
    position_a = _vector3(sample["position_a"])
    position_b = _vector3(sample["position_b"])
    yaw_a = float(sample["yaw_a_deg"])
    yaw_b = float(sample["yaw_b_deg"])
    participant_a = _participant_config(base, AGENT_A, position_a, yaw_a)
    participant_b = _participant_config(base, AGENT_B, position_b, yaw_b)
    sample_config = replace(
        config,
        participants=[participant_a, participant_b],
        currents=[],
        obstacles=[],
    )
    participants = {
        AGENT_A: RaceParticipant(participant_a, controller=None, position=position_a, rotation_rpy_deg=(0.0, 0.0, yaw_a)),
        AGENT_B: RaceParticipant(participant_b, controller=None, position=position_b, rotation_rpy_deg=(0.0, 0.0, yaw_b)),
    }
    return sample_config, participants


def _participant_config(
    base: ParticipantConfig,
    participant_id: str,
    position: Vector3,
    yaw_deg: float,
) -> ParticipantConfig:
    spawn = dict(base.spawn)
    spawn["position"] = list(position)
    spawn["rotation_rpy_deg"] = [0.0, 0.0, yaw_deg]
    spawn["start_delay_s"] = 0.0
    return replace(base, id=participant_id, spawn=spawn, start_delay_s=0.0)


def _relative_pose(
    config: TrackConfig,
    *,
    longitudinal_offset_m: float,
    lateral_offset_m: float,
    vertical_offset_m: float,
    relative_yaw_deg: float,
) -> tuple[Vector3, Vector3, float, float]:
    position_a = config.start.position
    yaw_a = float(config.start.rotation_rpy_deg[2])
    yaw_rad = math.radians(yaw_a)
    forward = (math.cos(yaw_rad), math.sin(yaw_rad))
    right = (-math.sin(yaw_rad), math.cos(yaw_rad))
    position_b = (
        position_a[0] + longitudinal_offset_m * forward[0] + lateral_offset_m * right[0],
        position_a[1] + longitudinal_offset_m * forward[1] + lateral_offset_m * right[1],
        position_a[2] + vertical_offset_m,
    )
    return (
        _clamp_to_bounds(config, position_a),
        _clamp_to_bounds(config, position_b),
        yaw_a,
        _wrap_degrees(yaw_a + relative_yaw_deg),
    )


def _row_from_sample(
    sample: Mapping[str, Any],
    *,
    status: str,
    position_a: Vector3,
    position_b: Vector3,
    collision_a: bool,
    collision_b: bool,
    sensor_keys_a: Iterable[str],
    sensor_keys_b: Iterable[str],
    error: str,
) -> dict[str, Any]:
    horizontal = _horizontal_distance(position_a, position_b)
    vertical = abs(position_a[2] - position_b[2])
    return {
        "sample_id": sample["sample_id"],
        "status": status,
        "offset_type": sample["offset_type"],
        "longitudinal_offset_m": sample["longitudinal_offset_m"],
        "lateral_offset_m": sample["lateral_offset_m"],
        "vertical_offset_m": sample["vertical_offset_m"],
        "relative_yaw_deg": sample["relative_yaw_deg"],
        "horizontal_distance_m": horizontal,
        "vertical_distance_m": vertical,
        "distance_3d_m": math.sqrt(horizontal**2 + vertical**2),
        "position_a": position_a,
        "position_b": position_b,
        "yaw_a_deg": sample["yaw_a_deg"],
        "yaw_b_deg": sample["yaw_b_deg"],
        "collision_a": bool(collision_a),
        "collision_b": bool(collision_b),
        "collision_any": bool(collision_a or collision_b),
        "collision_sensor_keys_a": sorted(str(key) for key in sensor_keys_a),
        "collision_sensor_keys_b": sorted(str(key) for key in sensor_keys_b),
        "error": error,
    }


def _summarize_rows(rows: list[dict[str, Any]], *, wall_time_s: float) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "OK"]
    spawn_mismatch_rows = [row for row in rows if row.get("status") == "SPAWN_MISMATCH"]
    contact_rows = [row for row in ok_rows if _bool_value(row.get("collision_any"))]
    clear_rows = [row for row in ok_rows if not _bool_value(row.get("collision_any"))]
    warnings: list[str] = []
    if not ok_rows:
        warnings.append("No successful calibration samples were recorded; HoloOcean may not be spawning two agents.")
    if spawn_mismatch_rows:
        warnings.append(
            f"{len(spawn_mismatch_rows)} samples had invalid HoloOcean spawn placement and were excluded."
        )
    if not contact_rows:
        warnings.append("No collision samples were observed; recommended thresholds use conservative defaults.")
    if contact_rows and max(_float(row["horizontal_distance_m"]) for row in contact_rows) > 2.0:
        warnings.append("Collision sensor reported contact at large horizontal distances; check for global contacts.")
    recommendation = _recommend_thresholds(contact_rows, clear_rows)
    by_yaw: dict[str, dict[str, int]] = {}
    for row in ok_rows:
        yaw = str(int(round(_float(row.get("relative_yaw_deg")))))
        bucket = by_yaw.setdefault(yaw, {"samples": 0, "contacts": 0})
        bucket["samples"] += 1
        if _bool_value(row.get("collision_any")):
            bucket["contacts"] += 1
    return {
        "sample_count": len(rows),
        "successful_samples": len(ok_rows),
        "spawn_mismatch_samples": len(spawn_mismatch_rows),
        "collision_samples": len(contact_rows),
        "clear_samples": len(clear_rows),
        "wall_time_s": wall_time_s,
        "collision_rate": len(contact_rows) / len(ok_rows) if ok_rows else None,
        "min_contact_horizontal_distance_m": _min_value(contact_rows, "horizontal_distance_m"),
        "max_contact_horizontal_distance_m": _max_value(contact_rows, "horizontal_distance_m"),
        "min_clear_horizontal_distance_m": _min_value(clear_rows, "horizontal_distance_m"),
        "min_contact_vertical_distance_m": _min_value(contact_rows, "vertical_distance_m"),
        "max_contact_vertical_distance_m": _max_value(contact_rows, "vertical_distance_m"),
        "contacts_by_relative_yaw_deg": by_yaw,
        "recommendation": recommendation,
        "warnings": warnings,
    }


def _recommend_thresholds(
    contact_rows: list[dict[str, Any]],
    clear_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    default_xy = 0.8
    default_z = 0.75
    if not contact_rows:
        xy = default_xy
        z = default_z
    else:
        near_z_contacts = [row for row in contact_rows if _float(row.get("vertical_distance_m")) <= 0.8]
        basis = near_z_contacts or contact_rows
        max_contact_xy = max(_float(row.get("horizontal_distance_m")) for row in basis)
        clear_above = [
            _float(row.get("horizontal_distance_m"))
            for row in clear_rows
            if _float(row.get("horizontal_distance_m")) > max_contact_xy
            and _float(row.get("vertical_distance_m")) <= 0.8
        ]
        if clear_above:
            xy = (max_contact_xy + min(clear_above)) / 2.0
        else:
            xy = max_contact_xy + 0.05
        xy = max(0.35, min(2.5, xy))
        z = max(_float(row.get("vertical_distance_m")) for row in contact_rows) + 0.05
        z = max(0.35, min(1.5, z))
    release = max(xy + 0.25, xy * 1.25)
    return {
        "inter_vehicle_collision_xy_threshold_m": round(xy, 3),
        "inter_vehicle_collision_z_threshold_m": round(z, 3),
        "hysteresis_release_threshold_m": round(release, 3),
        "recommended_cooldown_s": 1.0,
    }


def _render_report(summary: Mapping[str, Any], csv_path: Path) -> str:
    recommendation = summary.get("recommendation", {})
    warnings = summary.get("warnings", [])
    lines = [
        "# Inter-Vehicle Collision Calibration",
        "",
        "This calibration estimates geometric proximity thresholds for BlueROV2-vs-BlueROV2 contact.",
        "",
        "## Samples",
        f"- sample_count: {summary.get('sample_count')}",
        f"- successful_samples: {summary.get('successful_samples')}",
        f"- collision_samples: {summary.get('collision_samples')}",
        f"- clear_samples: {summary.get('clear_samples')}",
        f"- collision_rate: {summary.get('collision_rate')}",
        f"- samples_csv: {csv_path}",
        "",
        "## Recommendation",
        f"- inter_vehicle_collision_xy_threshold_m: {recommendation.get('inter_vehicle_collision_xy_threshold_m')}",
        f"- inter_vehicle_collision_z_threshold_m: {recommendation.get('inter_vehicle_collision_z_threshold_m')}",
        f"- hysteresis_release_threshold_m: {recommendation.get('hysteresis_release_threshold_m')}",
        f"- recommended_cooldown_s: {recommendation.get('recommended_cooldown_s')}",
        "",
        "## Diagnostics",
        f"- min_contact_horizontal_distance_m: {summary.get('min_contact_horizontal_distance_m')}",
        f"- max_contact_horizontal_distance_m: {summary.get('max_contact_horizontal_distance_m')}",
        f"- min_clear_horizontal_distance_m: {summary.get('min_clear_horizontal_distance_m')}",
        f"- max_contact_vertical_distance_m: {summary.get('max_contact_vertical_distance_m')}",
        f"- contacts_by_relative_yaw_deg: {json.dumps(summary.get('contacts_by_relative_yaw_deg', {}), sort_keys=True)}",
        f"- error: {summary.get('error', '')}",
        "",
        "## Warnings",
    ]
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _completed_sample_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    ids: set[int] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                ids.add(int(row.get("sample_id", "")))
            except ValueError:
                continue
    return ids


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _collision_sensor_keys(sensors: Mapping[str, Any]) -> list[str]:
    return [
        str(key)
        for key in sensors
        if "collision" in str(key).lower() or "contact" in str(key).lower()
    ]


def _offset_type(longitudinal: float, lateral: float, vertical: float) -> str:
    horizontal_long = abs(longitudinal) > 1e-9
    horizontal_lat = abs(lateral) > 1e-9
    if abs(vertical) > 1e-9 and not horizontal_long and not horizontal_lat:
        return "vertical"
    if horizontal_long and not horizontal_lat:
        return "longitudinal"
    if horizontal_lat and not horizontal_long:
        return "lateral"
    if horizontal_long and horizontal_lat:
        return "diagonal"
    return "overlap"


def _clamp_to_bounds(config: TrackConfig, position: Vector3) -> Vector3:
    bounds = config.world.bounds
    return (
        min(max(position[0], bounds.x_min), bounds.x_max),
        min(max(position[1], bounds.y_min), bounds.y_max),
        min(max(position[2], bounds.z_min), bounds.z_max),
    )


def _horizontal_distance(a: Vector3, b: Vector3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _distance(a: Vector3, b: Vector3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _vector3(value: Any) -> Vector3:
    return (float(value[0]), float(value[1]), float(value[2]))


def _wrap_degrees(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def _zero_command() -> dict[str, float]:
    return {"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}


def _csv_value(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return json.dumps(value)
    return value


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _min_value(rows: list[dict[str, Any]], field: str) -> float | None:
    return min((_float(row.get(field)) for row in rows), default=None)


def _max_value(rows: list[dict[str, Any]], field: str) -> float | None:
    return max((_float(row.get(field)) for row in rows), default=None)


if __name__ == "__main__":
    raise SystemExit(main())
