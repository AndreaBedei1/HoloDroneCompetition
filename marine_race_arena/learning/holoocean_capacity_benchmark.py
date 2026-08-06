"""Measured HoloOcean capacity benchmark with strict resource headroom.

The previous ten-engine ceiling was an engineering safety limit, not a measured
hardware maximum.  This module raises the total active-engine count step by step
and keeps the layout that maximises *aggregate valid environment transitions per
second*, rejecting any layout that breaches the headroom or reliability limits.

Stopping rule: increasing stops after two consecutive engine counts fail to
improve aggregate valid throughput by at least ``MINIMUM_IMPROVEMENT``.  A larger
layout is never selected if it measures slower than a smaller one.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from marine_race_arena.learning.provenance import now_utc
from marine_race_arena.learning.rl_resource_governor import (
    DEFAULT_HEADROOM, HeadroomLimits, layout_rejected,
)

BENCHMARK_VERSION = "universal_transition_parallel_capacity_v2"
MINIMUM_IMPROVEMENT = 0.08
CONSECUTIVE_NON_IMPROVEMENTS = 2
DEFAULT_ENGINE_COUNTS = (8, 10, 12, 14, 16, 18, 20)


def _nvidia_smi() -> Dict[str, Any]:
    """GPU utilisation, VRAM and temperature; empty when unavailable."""

    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False, timeout=20,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {}
    rows = [line for line in out.splitlines() if line.strip()]
    if not rows:
        return {}
    used_total = []
    for line in rows:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            used_total.append(tuple(float(part) for part in parts))
        except ValueError:
            continue
    if not used_total:
        return {}
    gpu = max(row[0] for row in used_total)
    used = sum(row[1] for row in used_total)
    total = sum(row[2] for row in used_total)
    temperature = max(row[3] for row in used_total)
    return {
        "gpu_percent": gpu,
        "vram_used_mb": used,
        "vram_total_mb": total,
        "vram_percent": 100.0 * used / max(1.0, total),
        "gpu_temperature_c": temperature,
    }


def sample_resources() -> Dict[str, Any]:
    """One CPU/RAM/GPU sample; never fatal when a source is unavailable."""

    sample: Dict[str, Any] = {"sampled_utc": now_utc()}
    try:
        import psutil

        sample["cpu_percent"] = float(psutil.cpu_percent(interval=1.0))
        memory = psutil.virtual_memory()
        sample["memory_percent"] = float(memory.percent)
        sample["memory_used_gb"] = round(memory.used / 1024 ** 3, 2)
        sample["memory_total_gb"] = round(memory.total / 1024 ** 3, 2)
    except Exception:  # pragma: no cover - psutil is optional
        pass
    sample.update(_nvidia_smi())
    return sample


def aggregate_resources(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Sustained view: the mean of the samples, plus their peaks."""

    if not samples:
        return {}
    keys = {key for sample in samples for key in sample if key != "sampled_utc"}
    out: Dict[str, Any] = {}
    for key in sorted(keys):
        values = [
            float(sample[key]) for sample in samples
            if isinstance(sample.get(key), (int, float))
        ]
        if not values:
            continue
        out[key] = round(statistics.fmean(values), 3)
        out[f"peak_{key}"] = round(max(values), 3)
    out["samples"] = len(samples)
    return out


def headroom_report(
    resources: Mapping[str, Any], limits: HeadroomLimits = DEFAULT_HEADROOM
) -> Dict[str, Any]:
    """Remaining headroom per resource, as a fraction of the ceiling."""

    out: Dict[str, Any] = {}
    for key, ceiling in (
        ("cpu_percent", limits.max_cpu_percent),
        ("memory_percent", limits.max_memory_percent),
        ("gpu_percent", limits.max_gpu_percent),
        ("vram_percent", limits.max_vram_percent),
    ):
        value = resources.get(key)
        if isinstance(value, (int, float)):
            out[key] = {
                "observed": round(float(value), 2),
                "ceiling": float(ceiling),
                "headroom_fraction": round(
                    max(0.0, (float(ceiling) - float(value)) / max(1e-9, float(ceiling))), 4
                ),
            }
    if out:
        out["minimum_headroom_fraction"] = round(
            min(row["headroom_fraction"] for row in out.values()
                if isinstance(row, dict)), 4
        )
        out["meets_target_headroom"] = bool(
            out["minimum_headroom_fraction"] >= limits.target_headroom_fraction
        )
    return out


def select_layout(
    layouts: Sequence[Mapping[str, Any]],
    limits: HeadroomLimits = DEFAULT_HEADROOM,
) -> Dict[str, Any]:
    """Rank accepted layouts by aggregate valid throughput, then reliability."""

    accepted, rejected = [], []
    for layout in layouts:
        reasons = layout_rejected(layout, limits)
        record = dict(layout)
        record["rejected_reasons"] = list(reasons)
        (rejected if reasons else accepted).append(record)
    ranked = sorted(
        accepted,
        key=lambda row: (
            float(row.get("aggregate_valid_transitions_per_second", 0.0) or 0.0),
            -float(row.get("checkpoint_latency_s", 0.0) or 0.0),
            float(row.get("headroom", {}).get("minimum_headroom_fraction", 0.0) or 0.0),
            -int(row.get("total_engines", 0) or 0),
        ),
        reverse=True,
    )
    return {
        "schema_version": BENCHMARK_VERSION,
        "accepted": ranked,
        "rejected": rejected,
        "selected": ranked[0] if ranked else None,
    }


def should_stop_increasing(
    throughput_by_engines: Mapping[int, float],
    *,
    minimum_improvement: float = MINIMUM_IMPROVEMENT,
    consecutive: int = CONSECUTIVE_NON_IMPROVEMENTS,
) -> bool:
    """Stop after ``consecutive`` engine counts fail to improve by the margin."""

    counts = sorted(throughput_by_engines)
    if len(counts) <= consecutive:
        return False
    best = 0.0
    streak = 0
    for count in counts:
        value = float(throughput_by_engines[count] or 0.0)
        if value >= best * (1.0 + float(minimum_improvement)):
            streak = 0
            best = max(best, value)
        else:
            streak += 1
            best = max(best, value)
        if streak >= consecutive:
            return True
    return False


def next_engine_counts(
    measured: Mapping[int, float],
    candidates: Sequence[int] = DEFAULT_ENGINE_COUNTS,
) -> list[int]:
    """Remaining counts to try, honouring the stopping rule and step order."""

    if should_stop_increasing(measured):
        return []
    return [count for count in sorted(candidates) if count not in measured]


def allocation_candidates(
    total: int,
    *,
    ppo_options: Sequence[int],
    sac_options: Sequence[int],
    minimum_per_algorithm: int = 2,
) -> list[tuple]:
    """Feasible (ppo, sac) worker splits for a total engine budget."""

    out = []
    for ppo in sorted(ppo_options):
        if ppo < minimum_per_algorithm:
            continue
        for sac in sorted(sac_options):
            if sac < minimum_per_algorithm or ppo + sac != int(total):
                continue
            out.append((int(ppo), int(sac)))
    return out


def write_report(output_dir: str | Path, report: Mapping[str, Any]) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    target = path / "capacity_benchmark.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    (path / "capacity_benchmark.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    return target


def _markdown(report: Mapping[str, Any]) -> str:
    selection = report.get("selection") or {}
    lines = [
        "# HoloOcean parallel capacity (v2)",
        "",
        f"Generated: {report.get('generated_utc')}",
        "",
        "Layouts are ranked by aggregate valid environment transitions per",
        "second. A larger engine count is never selected if it measures slower,",
        "and any layout breaching the headroom or reliability limits is rejected",
        "outright.",
        "",
        "| Total engines | PPO | SAC | PPO tx/s | SAC tx/s | Aggregate tx/s | tx/s per worker | CPU% | RAM% | GPU% | VRAM% | Accepted |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    rows = list(selection.get("accepted") or []) + list(selection.get("rejected") or [])
    for row in sorted(rows, key=lambda r: (r.get("total_engines", 0), r.get("ppo_workers", 0))):
        resources = row.get("resources") or {}
        accepted = not row.get("rejected_reasons")
        lines.append(
            f"| {row.get('total_engines')} | {row.get('ppo_workers')} | "
            f"{row.get('sac_workers')} | "
            f"{row.get('ppo_valid_transitions_per_second', 0):.2f} | "
            f"{row.get('sac_valid_transitions_per_second', 0):.2f} | "
            f"{row.get('aggregate_valid_transitions_per_second', 0):.2f} | "
            f"{row.get('transitions_per_second_per_worker', 0):.2f} | "
            f"{resources.get('cpu_percent', float('nan')):.1f} | "
            f"{resources.get('memory_percent', float('nan')):.1f} | "
            f"{resources.get('gpu_percent', float('nan')):.1f} | "
            f"{resources.get('vram_percent', float('nan')):.1f} | "
            f"{'yes' if accepted else 'no'} |"
        )
    for row in selection.get("rejected") or []:
        lines.append("")
        lines.append(
            f"Rejected {row.get('total_engines')} engines "
            f"({row.get('ppo_workers')} PPO + {row.get('sac_workers')} SAC): "
            + ", ".join(row.get("rejected_reasons") or [])
        )
    selected = selection.get("selected")
    lines += ["", "## Selected layout", ""]
    if selected:
        lines += [
            f"- total engines: **{selected.get('total_engines')}**",
            f"- PPO rollout workers: **{selected.get('ppo_workers')}**",
            f"- SAC rollout workers: **{selected.get('sac_workers')}**",
            f"- aggregate valid throughput: **{selected.get('aggregate_valid_transitions_per_second', 0):.2f}** transitions/s",
            f"- minimum headroom: **{(selected.get('headroom') or {}).get('minimum_headroom_fraction', 0):.1%}**",
        ]
    else:
        lines.append("No layout satisfied the headroom and reliability limits.")
    stopping = report.get("stopping") or {}
    lines += [
        "", "## Stopping rule", "",
        f"- minimum improvement to keep increasing: {MINIMUM_IMPROVEMENT:.0%}",
        f"- consecutive non-improvements before stopping: {CONSECUTIVE_NON_IMPROVEMENTS}",
        f"- stopped early: {stopping.get('stopped_early')}",
        f"- engine counts measured: {stopping.get('measured_engine_counts')}",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="measured layouts JSON")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    layouts = json.loads(Path(args.input).read_text(encoding="utf-8"))
    selection = select_layout(layouts)
    measured = {}
    for row in layouts:
        total = int(row.get("total_engines", 0) or 0)
        value = float(row.get("aggregate_valid_transitions_per_second", 0.0) or 0.0)
        measured[total] = max(measured.get(total, 0.0), value)
    report = {
        "schema_version": BENCHMARK_VERSION,
        "generated_utc": now_utc(),
        "selection": selection,
        "stopping": {
            "minimum_improvement": MINIMUM_IMPROVEMENT,
            "consecutive_non_improvements": CONSECUTIVE_NON_IMPROVEMENTS,
            "measured_engine_counts": sorted(measured),
            "aggregate_by_engine_count": measured,
            "stopped_early": should_stop_increasing(measured),
        },
        "headroom_limits": DEFAULT_HEADROOM.as_dict(),
    }
    path = write_report(args.output_dir, report)
    print(json.dumps({
        "selected": selection["selected"], "report": str(path),
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
