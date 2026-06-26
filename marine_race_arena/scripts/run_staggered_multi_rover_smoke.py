"""Run a clean staggered-start multi-rover smoke test."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from marine_race_arena.config.loader import load_track_config


TRACK_PATH = Path("marine_race_arena/tracks/marine_race_horseshoe_bay.json")
OUTPUT_DIR = Path("results/benchmarks/staggered_multi_rover_smoke")
DEFAULT_NUM_ROVERS = 2
DEFAULT_START_GAP_S = 90.0
DEFAULT_LATERAL_OFFSET_M = 3.0
DIAGNOSTIC_NUM_ROVERS = 3
DIAGNOSTIC_START_GAP_S = 20.0
DIAGNOSTIC_LATERAL_OFFSET_M = 1.5
TABLE_FIELDS = [
    "participant_id",
    "start_delay_s",
    "release_time_s",
    "status",
    "completed_gates",
    "expected_gates",
    "official_time_s",
    "green_to_finish_time_s",
    "penalized_time_s",
    "collisions",
    "obstacle_collisions",
    "out_of_bounds_events",
    "stuck_events",
    "final_rank",
]


@dataclass
class SmokeRun:
    adapter: str
    run_dir: Path
    return_code: int
    status: str
    wall_time_s: float
    summary_path: Path | None
    event_path: Path | None
    stdout_path: Path
    stderr_path: Path
    error: str | None = None


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    _apply_mode_defaults(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fallback = _run_adapter_smoke("fallback", args, output_dir / "fallback")
    fallback_rows = _rows_from_summary(fallback.summary_path)
    _write_table(output_dir / "fallback_multi_rover_smoke_table.csv", fallback_rows)

    holoocean = _run_adapter_smoke("holoocean", args, output_dir / "holoocean")
    holoocean_rows = _rows_from_summary(holoocean.summary_path)
    _write_table(output_dir / "multi_rover_smoke_table.csv", holoocean_rows)
    _copy_primary_outputs(output_dir, holoocean)

    _write_report(output_dir / "multi_rover_smoke_report.md", fallback, fallback_rows, holoocean, holoocean_rows, args)
    _print_brief_result(fallback, holoocean_rows, holoocean)

    return 0 if fallback.return_code == 0 and holoocean.return_code == 0 and bool(holoocean_rows) else 1


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default=str(TRACK_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--num-rovers", type=int, default=DEFAULT_NUM_ROVERS)
    parser.add_argument("--start-gap-s", type=float, default=DEFAULT_START_GAP_S)
    parser.add_argument("--staggered-lateral-offset-m", type=float, default=DEFAULT_LATERAL_OFFSET_M)
    parser.add_argument("--dt", type=float, default=0.033)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=float, default=560.0)
    parser.add_argument("--wall-timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--diagnostic-3-rover",
        action="store_true",
        help=(
            "Run the older 3-rover proximity diagnostic "
            f"({DIAGNOSTIC_START_GAP_S:g}s gap, {DIAGNOSTIC_LATERAL_OFFSET_M:g}m offset)."
        ),
    )
    return parser


def _apply_mode_defaults(args: argparse.Namespace) -> None:
    if not args.diagnostic_3_rover:
        args.mode_name = "stable_2_rover_demo"
        return
    args.mode_name = "diagnostic_3_rover_proximity"
    args.num_rovers = DIAGNOSTIC_NUM_ROVERS
    args.start_gap_s = DIAGNOSTIC_START_GAP_S
    args.staggered_lateral_offset_m = DIAGNOSTIC_LATERAL_OFFSET_M


def _run_adapter_smoke(adapter: str, args: argparse.Namespace, run_dir: Path) -> SmokeRun:
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / f"{adapter}_stdout.txt"
    stderr_path = run_dir / f"{adapter}_stderr.txt"
    command = [
        sys.executable,
        "-m",
        "marine_race_arena.scripts.run_marine_race",
        "--track",
        str(args.track),
        "--benchmark-task",
        "clean_gate",
        "--controller",
        "rule_gate_baseline",
        "--adapter",
        adapter,
        "--seed",
        str(args.seed),
        "--dt",
        str(args.dt),
        "--duration",
        str(args.duration),
        "--obstacles",
        "none",
        "--current-profile",
        "none",
        "--motion-compensation",
        "none",
        "--official",
        "--headless",
        "--staggered-start",
        "--num-rovers",
        str(args.num_rovers),
        "--start-gap-s",
        str(args.start_gap_s),
        "--staggered-lateral-offset-m",
        str(args.staggered_lateral_offset_m),
        "--log-participant-states",
        "--log-dir",
        str(run_dir),
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            check=False,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=max(1.0, float(args.wall_timeout_s)),
        )
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        status = "OK" if completed.returncode == 0 else "RUN_FAILED"
        return_code = int(completed.returncode)
        error = None if completed.returncode == 0 else _failure_hint(completed.stderr, completed.stdout)
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(exc.stdout or "", encoding="utf-8")
        stderr_path.write_text((exc.stderr or "") + "\nWALL_TIMEOUT\n", encoding="utf-8")
        status = "WALL_TIMEOUT"
        return_code = 124
        error = f"{adapter} smoke exceeded wall timeout {args.wall_timeout_s:.1f}s."
    wall_time_s = time.monotonic() - started
    return SmokeRun(
        adapter=adapter,
        run_dir=run_dir,
        return_code=return_code,
        status=status,
        wall_time_s=wall_time_s,
        summary_path=_newest_file(run_dir, "*_summary.json"),
        event_path=_newest_file(run_dir, "*.jsonl"),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        error=error,
    )


def _rows_from_summary(summary_path: Path | None) -> list[dict[str, Any]]:
    if summary_path is None or not summary_path.exists():
        return []
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    participants = summary.get("participants", [])
    if not isinstance(participants, list):
        return []
    expected_gates = _expected_gates(summary)
    rows = []
    for participant in participants:
        if not isinstance(participant, Mapping):
            continue
        rows.append(
            {
                "participant_id": participant.get("participant_id"),
                "start_delay_s": participant.get("start_delay_s", 0.0),
                "release_time_s": participant.get("release_time_s"),
                "status": participant.get("status"),
                "completed_gates": participant.get("completed_gates", 0),
                "expected_gates": expected_gates,
                "official_time_s": participant.get("official_time_s"),
                "green_to_finish_time_s": participant.get("green_to_finish_time_s"),
                "penalized_time_s": participant.get("penalized_time_s"),
                "collisions": participant.get("collisions", 0),
                "obstacle_collisions": participant.get("obstacle_collisions", 0),
                "out_of_bounds_events": participant.get("out_of_bounds_events", 0),
                "stuck_events": participant.get("stuck_events", 0),
                "final_rank": participant.get("rank"),
            }
        )
    return sorted(rows, key=lambda row: str(row.get("participant_id") or ""))


def _expected_gates(summary: Mapping[str, Any]) -> int | None:
    track_file = summary.get("track_file")
    if track_file:
        try:
            config = load_track_config(str(track_file), debug=True)
            return int(config.race.laps) * len(config.track.gate_sequence)
        except Exception:
            pass
    participants = summary.get("participants", [])
    if not isinstance(participants, list) or not participants:
        return None
    completed = [participant.get("completed_gates") for participant in participants if isinstance(participant, Mapping)]
    numeric = [int(value) for value in completed if isinstance(value, int)]
    return max(numeric) if numeric else None


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in TABLE_FIELDS})


def _copy_primary_outputs(output_dir: Path, run: SmokeRun) -> None:
    if run.summary_path is not None and run.summary_path.exists():
        shutil.copy2(run.summary_path, output_dir / "multi_rover_smoke_summary.json")
    if run.event_path is not None and run.event_path.exists():
        shutil.copy2(run.event_path, output_dir / "multi_rover_smoke_events.jsonl")


def _write_report(
    path: Path,
    fallback: SmokeRun,
    fallback_rows: list[dict[str, Any]],
    holoocean: SmokeRun,
    holoocean_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Staggered Multi-Rover Smoke",
        "",
        "Configuration:",
        f"- mode: {args.mode_name}",
        f"- track: {args.track}",
        "- controller: rule_gate_baseline",
        "- current_profile: none",
        "- obstacles: none",
        "- motion_compensation: none",
        f"- num_rovers: {args.num_rovers}",
        f"- start_gap_s: {args.start_gap_s}",
        f"- staggered_lateral_offset_m: {args.staggered_lateral_offset_m}",
        f"- dt: {args.dt}",
        f"- seed: {args.seed}",
        "",
        "This is staggered multi-participant evaluation. Rover-rover collision arbitration is not implemented yet. "
        "The stable default uses large temporal separation to avoid physical interaction while still exercising "
        "multi-agent spawning, release timing, independent referee state, timing, scoring, summaries, and ranking.",
        "",
        "## Fallback Smoke",
        _run_line(fallback),
        "",
        _table_markdown(fallback_rows),
        "",
        "## HoloOcean Smoke",
        _run_line(holoocean),
        "",
        _table_markdown(holoocean_rows),
        "",
        "## Interpretation",
        _interpretation(fallback, fallback_rows, holoocean, holoocean_rows, expected_count=args.num_rovers),
        "",
        "## Output Paths",
        f"- fallback summary: {_path_text(fallback.summary_path)}",
        f"- fallback events: {_path_text(fallback.event_path)}",
        f"- HoloOcean summary: {_path_text(holoocean.summary_path)}",
        f"- HoloOcean events: {_path_text(holoocean.event_path)}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_line(run: SmokeRun) -> str:
    line = (
        f"- adapter: {run.adapter}; status: {run.status}; return_code: {run.return_code}; "
        f"wall_time_s: {run.wall_time_s:.1f}"
    )
    if run.error:
        line += f"; error: {run.error}"
    return line


def _table_markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No participant summary rows were produced._"
    columns = [
        "participant_id",
        "start_delay_s",
        "status",
        "completed_gates",
        "expected_gates",
        "official_time_s",
        "collisions",
        "stuck_events",
        "final_rank",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(_csv_value(row.get(column))) for column in columns) + " |")
    return "\n".join(lines)


def _interpretation(
    fallback: SmokeRun,
    fallback_rows: list[dict[str, Any]],
    holoocean: SmokeRun,
    holoocean_rows: list[dict[str, Any]],
    expected_count: int,
) -> str:
    fallback_ok = fallback.return_code == 0 and len(fallback_rows) == expected_count
    fallback_finished = fallback_ok and all(row.get("status") == "FINISHED" for row in fallback_rows)
    holoocean_ok = holoocean.return_code == 0 and len(holoocean_rows) == expected_count
    holoocean_finished = holoocean_ok and all(row.get("status") == "FINISHED" for row in holoocean_rows)
    holoocean_clean = holoocean_finished and all(
        int(row.get("stuck_events") or 0) == 0
        and int(row.get("out_of_bounds_events") or 0) == 0
        and int(row.get("collisions") or 0) <= 2
        for row in holoocean_rows
    )
    if holoocean_clean:
        fallback_note = (
            "Fallback also finished all participants. "
            if fallback_finished
            else "Fallback produced the expected participant summaries but did not finish with this baseline. "
        )
        return (
            f"{fallback_note}"
            "HoloOcean produced a stable staggered multi-participant smoke result with zero stuck/OOB events "
            "and only near-zero contact counts. Each participant has separate state, timing, scoring, and "
            "ranking. This demonstration mode is ready to document."
        )
    if holoocean_finished:
        fallback_note = (
            "Fallback also finished all participants. "
            if fallback_finished
            else "Fallback produced the expected participant summaries but did not finish with this baseline. "
        )
        return (
            f"{fallback_note}"
            "All HoloOcean participants finished, but contacts or penalties remain. Treat this as a diagnostic "
            "multi-agent result rather than the stable clean demonstration."
        )
    if holoocean_ok:
        return (
            "HoloOcean produced the expected participant summaries, but not all participants finished. "
            "The staggered multi-agent plumbing works, but race behavior needs inspection before collision handling."
        )
    if fallback_ok:
        return (
            "Fallback multi-rover logic produced the expected participant summaries, but HoloOcean did not. "
            "Inspect the HoloOcean stdout/stderr and event paths for the failing stage."
        )
    return "Fallback did not produce a complete participant summary, so fix runner/referee logic first."


def _print_brief_result(fallback: SmokeRun, holoocean_rows: list[dict[str, Any]], holoocean: SmokeRun) -> None:
    print(f"Fallback smoke: {fallback.status} summary={_path_text(fallback.summary_path)}")
    print(f"HoloOcean smoke: {holoocean.status} summary={_path_text(holoocean.summary_path)}")
    if holoocean_rows:
        print("HoloOcean participants:")
        for row in holoocean_rows:
            print(
                "  "
                f"{row.get('participant_id')} delay={row.get('start_delay_s')} "
                f"status={row.get('status')} gates={row.get('completed_gates')}/{row.get('expected_gates')} "
                f"collisions={row.get('collisions')}"
            )


def _failure_hint(stderr: str, stdout: str) -> str:
    combined = (stderr.strip() or stdout.strip()).splitlines()
    if not combined:
        return "run failed without stderr/stdout"
    return combined[-1][:240]


def _newest_file(directory: Path, pattern: str) -> Path | None:
    candidates = [path for path in directory.glob(pattern) if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    return value


def _path_text(path: Path | None) -> str:
    return str(path) if path is not None else "none"


if __name__ == "__main__":
    raise SystemExit(main())
