"""Run repeated Marine Race benchmark trials and aggregate the results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from marine_race_arena.arena.obstacle import (
    OBSTACLE_DENSITIES,
    OBSTACLE_MODES,
    OBSTACLE_PHYSICS_MODES,
    effective_obstacle_mode,
)
from marine_race_arena.config.benchmark_tasks import (
    BENCHMARK_TASK_CLEAN_GATE,
    BENCHMARK_TASK_CURRENT_GATE,
    BENCHMARK_TASK_OBSTACLE_GATE,
)
from marine_race_arena.config.loader import CURRENT_PROFILE_MODES, load_track_config
from marine_race_arena.config.validation import validate_track_config
from marine_race_arena.scripts import run_marine_race

BENCHMARK_TASKS = (
    BENCHMARK_TASK_CLEAN_GATE,
    BENCHMARK_TASK_OBSTACLE_GATE,
    BENCHMARK_TASK_CURRENT_GATE,
)
MANUAL_CONTROLLER_ALIASES = {"pygame", "pygame_keyboard", "keyboard", "manual", "manual_keyboard"}
DEBUG_CONTROLLER_ALIASES = {"oracle"}
DNF_STATUSES = {"DNF", "DSQ", "TIMEOUT"}
SUMMARY_CSV_FIELDS = [
    "number_of_runs",
    "completion_rate",
    "mean_official_time_s",
    "std_official_time_s",
    "mean_penalized_time_s",
    "std_penalized_time_s",
    "mean_completed_gates",
    "mean_collision_events",
    "mean_obstacle_collision_events",
    "mean_out_of_bounds_events",
    "mean_stuck_events",
    "total_dnf",
    "dnf_reasons",
    "manual_stop_count",
    "controller_error_count",
]


@dataclass
class BenchmarkRunResult:
    seed: int | None
    run_dir: Path
    return_code: int
    metadata: dict[str, Any] = field(default_factory=dict)
    summary_path: Path | None = None
    event_path: Path | None = None
    metadata_path: Path | None = None


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    controller_role = _controller_role(args.controller)
    if controller_role == "manual_demo":
        print(
            "Benchmark note: pygame/keyboard manual controllers are allowed for demos, "
            "but should not be treated as main scientific baselines."
        )
    elif controller_role == "debug_only":
        print("Benchmark note: oracle is debug-only and must not be reported as a competition-valid baseline.")

    run_results = []
    for run_index, seed in enumerate(args.seeds, start=1):
        run_dir = _unique_run_dir(output_dir / "runs" / f"run_{run_index:03d}_seed_{seed}")
        run_dir.mkdir(parents=True, exist_ok=False)
        metadata = _build_run_metadata(args, seed, controller_role)
        metadata["run_index"] = run_index
        metadata["run_dir"] = str(run_dir)
        metadata_path = run_dir / "benchmark_metadata.json"
        _write_json(metadata_path, metadata)

        race_args = _race_args(args, seed, run_dir)
        print(f"Starting benchmark run {run_index}/{len(args.seeds)} seed={seed} log_dir={run_dir}")
        return_code = _run_single_race(race_args, metadata)
        summary_path = _newest_file(run_dir, "*_summary.json")
        event_path = _newest_file(run_dir, "*.jsonl")
        metadata["return_code"] = return_code
        metadata["summary_path"] = str(summary_path) if summary_path is not None else None
        metadata["event_path"] = str(event_path) if event_path is not None else None
        _write_json(metadata_path, metadata)
        run_results.append(
            BenchmarkRunResult(
                seed=seed,
                run_dir=run_dir,
                return_code=return_code,
                metadata=metadata,
                summary_path=summary_path,
                event_path=event_path,
                metadata_path=metadata_path,
            )
        )

    aggregate, rows = aggregate_run_results(run_results)
    csv_path, json_path = write_aggregate_outputs(output_dir, aggregate, rows)
    print(f"Benchmark CSV: {csv_path}")
    print(f"Benchmark JSON: {json_path}")
    return 0 if all(result.return_code == 0 for result in run_results) else 1


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-task", required=True, choices=BENCHMARK_TASKS)
    parser.add_argument("--track", required=True, help="Path to the benchmark track JSON.")
    parser.add_argument("--controller", required=True, help="Built-in alias, module path, module:Class, or .py file.")
    parser.add_argument("--controller-class", default=None, help="Controller class for file/module controllers.")
    parser.add_argument("--adapter", choices=("fallback", "holoocean", "auto"), default="fallback")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="One or more deterministic run seeds.")
    parser.add_argument("--duration", type=float, default=None, help="Maximum race duration in seconds.")
    parser.add_argument("--dt", type=float, default=0.1, help="Race loop timestep.")
    parser.add_argument("--obstacles", choices=OBSTACLE_MODES, default=None)
    parser.add_argument("--obstacle-density", choices=OBSTACLE_DENSITIES, default=None)
    parser.add_argument("--obstacle-physics", choices=OBSTACLE_PHYSICS_MODES, default=None)
    parser.add_argument("--current-profile", choices=CURRENT_PROFILE_MODES, default=None)
    parser.add_argument("--output-dir", default="results/benchmarks")
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--print-beacon-targets", action="store_true")
    return parser


def _race_args(args: argparse.Namespace, seed: int, run_dir: Path) -> list[str]:
    race_args = [
        "--track",
        str(args.track),
        "--benchmark-task",
        args.benchmark_task,
        "--controller",
        args.controller,
        "--adapter",
        args.adapter,
        "--seed",
        str(seed),
        "--dt",
        str(args.dt),
        "--log-dir",
        str(run_dir),
    ]
    if args.controller_class:
        race_args.extend(["--controller-class", args.controller_class])
    if args.duration is not None:
        race_args.extend(["--duration", str(args.duration)])
    if args.obstacles is not None:
        race_args.extend(["--obstacles", args.obstacles])
    if args.obstacle_density is not None:
        race_args.extend(["--obstacle-density", args.obstacle_density])
    if args.obstacle_physics is not None:
        race_args.extend(["--obstacle-physics", args.obstacle_physics])
    if args.current_profile is not None:
        race_args.extend(["--current-profile", args.current_profile])
    if args.allow_fallback:
        race_args.append("--allow-fallback")
    if args.official:
        race_args.append("--official")
    if args.print_beacon_targets:
        race_args.append("--print-beacon-targets")
    return race_args


def _run_single_race(race_args: list[str], metadata: dict[str, Any]) -> int:
    try:
        return int(run_marine_race.main(race_args))
    except Exception as exc:  # pragma: no cover - defensive benchmark wrapper
        metadata["run_exception"] = f"{type(exc).__name__}: {exc}"
        print(f"Benchmark run failed before summary creation: {metadata['run_exception']}", file=sys.stderr)
        return 1


def _build_run_metadata(args: argparse.Namespace, seed: int, controller_role: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "task": args.benchmark_task,
        "track": str(args.track),
        "controller": args.controller,
        "controller_class": args.controller_class,
        "controller_role": controller_role,
        "manual_demo": controller_role == "manual_demo",
        "debug_only": controller_role == "debug_only",
        "adapter": args.adapter,
        "allow_fallback": bool(args.allow_fallback),
        "seed": seed,
        "obstacles_requested": args.obstacles,
        "obstacle_density_requested": args.obstacle_density,
        "obstacle_physics_requested": args.obstacle_physics,
        "current_profile_requested": args.current_profile,
        "duration_s": args.duration,
        "dt": args.dt,
        "official": bool(args.official),
        "print_beacon_targets": bool(args.print_beacon_targets),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        config = load_track_config(
            args.track,
            debug=True,
            benchmark_task=args.benchmark_task,
            obstacles=args.obstacles,
            obstacle_density=args.obstacle_density,
            obstacle_physics=args.obstacle_physics,
            current_profile=args.current_profile,
            seed=seed,
        )
    except Exception as exc:
        metadata["track_config_error"] = f"{type(exc).__name__}: {exc}"
        metadata["current_mode"] = "unknown"
        metadata["obstacle_mode"] = args.obstacles or "unknown"
        return metadata
    validation = validate_track_config(config)
    metadata["race_name"] = config.race.name
    metadata["benchmark_task"] = config.benchmark_task.mode
    metadata["obstacle_mode"] = effective_obstacle_mode(config)
    metadata["obstacle_density"] = config.obstacle_generation.density
    metadata["obstacle_physics"] = config.obstacle_generation.obstacle_physics
    metadata["current_mode"] = _current_mode(config.currents)
    metadata["current_profile"] = config.selected_current_profile or "track-default"
    metadata["current_count"] = len(config.currents)
    metadata["duration_s"] = args.duration if args.duration is not None else config.race.max_duration_s
    metadata["validation_errors"] = list(validation.errors)
    metadata["validation_warnings"] = list(validation.warnings)
    return metadata


def aggregate_run_results(run_results: Sequence[BenchmarkRunResult]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [_row_from_run_result(result) for result in run_results]
    aggregate = _aggregate_rows(rows)
    return aggregate, rows


def write_aggregate_outputs(
    output_dir: str | Path,
    aggregate: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "benchmark_summary.csv"
    json_path = output_path / "benchmark_summary.json"

    csv_row = dict(aggregate)
    csv_row["dnf_reasons"] = json.dumps(csv_row.get("dnf_reasons", {}), sort_keys=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_CSV_FIELDS)
        writer.writeheader()
        writer.writerow({field: _csv_value(csv_row.get(field)) for field in SUMMARY_CSV_FIELDS})

    _write_json(json_path, {"aggregate": dict(aggregate), "runs": list(rows)})
    return csv_path, json_path


def _row_from_run_result(result: BenchmarkRunResult) -> dict[str, Any]:
    summary = _read_json(result.summary_path) if result.summary_path is not None else {}
    participant = _primary_participant(summary)
    status = str(participant.get("status") or ("RUN_FAILED" if result.return_code else "UNKNOWN"))
    row = {
        "seed": result.seed,
        "run_dir": str(result.run_dir),
        "return_code": result.return_code,
        "summary_path": str(result.summary_path) if result.summary_path is not None else None,
        "event_path": str(result.event_path) if result.event_path is not None else None,
        "metadata_path": str(result.metadata_path) if result.metadata_path is not None else None,
        "participant_id": participant.get("participant_id"),
        "status": status,
        "official_time_s": _optional_float(participant.get("official_time_s")),
        "penalized_time_s": _optional_float(participant.get("penalized_time_s")),
        "completed_gates": _float_or_zero(participant.get("completed_gates")),
        "collision_events": _float_or_zero(participant.get("collisions")),
        "obstacle_collision_events": _float_or_zero(participant.get("obstacle_collisions")),
        "out_of_bounds_events": _float_or_zero(participant.get("out_of_bounds_events")),
        "stuck_events": _float_or_zero(participant.get("stuck_events")),
        "dnf_reason": _dnf_reason(status, result.event_path),
    }
    row.update(_json_safe_metadata(result.metadata))
    return row


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    number_of_runs = len(rows)
    completed = [row for row in rows if row.get("status") == "FINISHED"]
    dnf_reasons: dict[str, int] = {}
    for row in rows:
        reason = row.get("dnf_reason")
        if reason:
            dnf_reasons[str(reason)] = dnf_reasons.get(str(reason), 0) + 1

    return {
        "number_of_runs": number_of_runs,
        "completion_rate": (len(completed) / number_of_runs) if number_of_runs else 0.0,
        "mean_official_time_s": _mean(_numeric_values(row.get("official_time_s") for row in rows)),
        "std_official_time_s": _std(_numeric_values(row.get("official_time_s") for row in rows)),
        "mean_penalized_time_s": _mean(_numeric_values(row.get("penalized_time_s") for row in rows)),
        "std_penalized_time_s": _std(_numeric_values(row.get("penalized_time_s") for row in rows)),
        "mean_completed_gates": _mean(_numeric_values(row.get("completed_gates") for row in rows), default=0.0),
        "mean_collision_events": _mean(_numeric_values(row.get("collision_events") for row in rows), default=0.0),
        "mean_obstacle_collision_events": _mean(
            _numeric_values(row.get("obstacle_collision_events") for row in rows),
            default=0.0,
        ),
        "mean_out_of_bounds_events": _mean(
            _numeric_values(row.get("out_of_bounds_events") for row in rows),
            default=0.0,
        ),
        "mean_stuck_events": _mean(_numeric_values(row.get("stuck_events") for row in rows), default=0.0),
        "total_dnf": sum(1 for row in rows if row.get("status") in DNF_STATUSES),
        "dnf_reasons": dnf_reasons,
        "manual_stop_count": sum(1 for row in rows if row.get("status") == "MANUAL_STOP"),
        "controller_error_count": sum(1 for row in rows if row.get("status") == "CONTROLLER_ERROR"),
    }


def _primary_participant(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    participants = summary.get("participants")
    if not isinstance(participants, list) or not participants:
        return {}
    ranked = sorted(
        (participant for participant in participants if isinstance(participant, Mapping)),
        key=lambda participant: _float_or_zero(participant.get("rank"), default=math.inf),
    )
    return ranked[0] if ranked else {}


def _dnf_reason(status: str, event_path: Path | None) -> str | None:
    if status not in DNF_STATUSES:
        return None
    event_reason = _latest_dnf_event_reason(event_path)
    if event_reason:
        return event_reason
    return status.lower()


def _latest_dnf_event_reason(event_path: Path | None) -> str | None:
    if event_path is None or not event_path.exists():
        return None
    reason = None
    try:
        with event_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, Mapping) and event.get("event") == "dnf":
                    reason = str(event.get("reason") or "dnf")
    except OSError:
        return None
    return reason


def _controller_role(controller: str) -> str:
    normalized = Path(controller).stem.lower() if controller.endswith(".py") else controller.lower()
    if normalized in MANUAL_CONTROLLER_ALIASES:
        return "manual_demo"
    if normalized in DEBUG_CONTROLLER_ALIASES or "oracle" in normalized:
        return "debug_only"
    return "automatic"


def _current_mode(currents: Iterable[Any]) -> str:
    current_types = [str(getattr(current, "type", "")).strip() for current in currents]
    current_types = [current_type for current_type in current_types if current_type]
    if not current_types:
        return "none"
    return ",".join(sorted(set(current_types)))


def _unique_run_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for suffix in range(2, 1000):
        candidate = base.with_name(f"{base.name}_{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an unused run directory for {base}")


def _newest_file(directory: Path, pattern: str) -> Path | None:
    candidates = [path for path in directory.glob(pattern) if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _numeric_values(values: Iterable[Any]) -> list[float]:
    numeric = []
    for value in values:
        converted = _optional_float(value)
        if converted is not None:
            numeric.append(converted)
    return numeric


def _mean(values: Sequence[float], default: float | None = None) -> float | None:
    if not values:
        return default
    return sum(values) / len(values)


def _std(values: Sequence[float]) -> float | None:
    if not values:
        return None
    mean = _mean(values, default=0.0) or 0.0
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted):
        return None
    return converted


def _float_or_zero(value: Any, default: float = 0.0) -> float:
    converted = _optional_float(value)
    return default if converted is None else converted


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    return value


def _json_safe_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {f"metadata_{key}": value for key, value in metadata.items()}


if __name__ == "__main__":
    raise SystemExit(main())
