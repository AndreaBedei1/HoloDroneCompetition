"""Benchmark independent evaluation cases at 4/6/8 HoloOcean workers."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Mapping

from marine_race_arena.learning.benchmark_transition_parallelism import _ResourceMonitor
from marine_race_arena.learning.holoocean_capacity import (
    MAX_ACTIVE_HOLOOCEAN_ENGINES,
    active_holodeck_processes,
)
from marine_race_arena.learning.longrun_checkpoint import atomic_write_json
from marine_race_arena.learning.provenance import now_utc
from marine_race_arena.learning.transition_evaluation import (
    evaluate_checkpoint_universal_transition_benchmark,
)


def select_evaluator_count(cases: list[Mapping[str, Any]]) -> int | None:
    stable = [row for row in cases if row.get("stable")]
    if not stable:
        return None
    selected = stable[0]
    for row in stable[1:]:
        if float(row["cases_per_second"]) >= 1.10 * float(selected["cases_per_second"]):
            selected = row
    return int(selected["workers"])


def run_case(
    checkpoint: Path,
    output: Path,
    *,
    algorithm: str,
    workers: int,
    transition_cases: int,
    seed: int,
) -> Dict[str, Any]:
    before = active_holodeck_processes()
    monitor = _ResourceMonitor()
    started = time.perf_counter()
    try:
        monitor.start()
        report = evaluate_checkpoint_universal_transition_benchmark(
            checkpoint,
            output_dir=output / f"workers_{workers}",
            seed=seed,
            difficulty="G1",
            transition_cases=transition_cases,
            full_cases_per_length=0,
            adapter="holoocean",
            frames_per_sec=False,
            max_steps=3600,
            parallel_workers=workers,
            algorithm=algorithm,
        )
        elapsed = time.perf_counter() - started
        return {
            "ok": True,
            "workers": workers,
            "cases": int(report["transition_cases"]),
            "wall_time_s": elapsed,
            "cases_per_second": int(report["transition_cases"]) / max(elapsed, 1e-9),
            "cases_per_second_per_worker": (
                int(report["transition_cases"]) / max(elapsed * workers, 1e-9)
            ),
            "resources": monitor.report(),
            "engines_before": before,
        }
    except Exception as exc:
        return {
            "ok": False,
            "workers": workers,
            "error": f"{type(exc).__name__}: {exc}",
            "resources": monitor.report(),
            "engines_before": before,
        }
    finally:
        monitor.close()
        time.sleep(1.0)


def finalize_case(row: Dict[str, Any]) -> Dict[str, Any]:
    after = active_holodeck_processes()
    before_pids = {int(value["pid"]) for value in row.get("engines_before", [])}
    before_owners = {int(value.get("parent_pid", -1)) for value in row.get("engines_before", [])}
    orphans = [
        int(value["pid"]) for value in after
        if int(value["pid"]) not in before_pids
        and int(value.get("parent_pid", -1)) not in before_owners
    ]
    row["engines_after"] = after
    row["orphan_engine_pids"] = orphans
    row["stable"] = bool(
        row.get("ok")
        and not orphans
        and len(after) <= MAX_ACTIVE_HOLOOCEAN_ENGINES
    )
    return row


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Dynamic evaluation capacity",
        "",
        "| Workers | Cases/s | Per-worker cases/s | Stable | Orphans |",
        "|---:|---:|---:|:---:|---:|",
    ]
    for row in report["cases"]:
        lines.append(
            f"| {row['workers']} | {row.get('cases_per_second', 0):.4f} | "
            f"{row.get('cases_per_second_per_worker', 0):.4f} | "
            f"{row.get('stable')} | {len(row.get('orphan_engine_pids', []))} |"
        )
    lines += ["", f"Selected evaluator workers: {report['selected_workers']}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--algorithm", choices=("ppo", "sac"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", default="4,6,8")
    parser.add_argument("--cases", type=int, default=48)
    parser.add_argument("--seed", type=int, default=99001)
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    rows = []
    consecutive_no_gain = 0
    best_rate = 0.0
    for workers in [int(value) for value in args.workers.split(",") if value.strip()]:
        row = finalize_case(run_case(
            Path(args.checkpoint), output,
            algorithm=args.algorithm,
            workers=workers,
            transition_cases=args.cases,
            seed=args.seed,
        ))
        rows.append(row)
        rate = float(row.get("cases_per_second", 0.0) or 0.0)
        if row.get("stable") and rate >= 1.10 * max(best_rate, 1e-12):
            best_rate = max(best_rate, rate)
            consecutive_no_gain = 0
        else:
            consecutive_no_gain += 1
        if consecutive_no_gain >= 2:
            break
    report = {
        "schema_version": "dynamic_evaluation_capacity_v1",
        "generated_utc": now_utc(),
        "global_engine_limit": MAX_ACTIVE_HOLOOCEAN_ENGINES,
        "cases": rows,
        "selected_workers": select_evaluator_count(rows),
    }
    atomic_write_json(output / "evaluation_capacity.json", report)
    markdown = output / "evaluation_capacity.md"
    temporary = markdown.with_suffix(".md.tmp")
    temporary.write_text(_markdown(report), encoding="utf-8")
    temporary.replace(markdown)
    return 0 if report["selected_workers"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
