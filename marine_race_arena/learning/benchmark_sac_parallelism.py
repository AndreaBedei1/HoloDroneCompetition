"""Incremental SAC capacity benchmark under the global ten-engine cap."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np

from marine_race_arena.learning.benchmark_transition_parallelism import _ResourceMonitor
from marine_race_arena.learning.holoocean_capacity import (
    MAX_ACTIVE_HOLOOCEAN_ENGINES,
    active_holodeck_processes,
    assert_unique_holoocean_uuids,
)
from marine_race_arena.learning.longrun_checkpoint import atomic_write_json
from marine_race_arena.learning.provenance import now_utc
from marine_race_arena.learning.sac_transition_policy import (
    SACTransitionAgent,
    transfer_ppo_mean_actor,
)
from marine_race_arena.learning.train_sac_transition import (
    _load_config,
    _make_vec_env,
    _resolve_source_checkpoint,
)


def _progress_rates(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if value.get("environment_transitions_per_second") is not None:
            rows.append(value)
    return rows


def _mean_last_rates(rows: Iterable[Mapping[str, Any]], count: int = 3) -> float | None:
    selected = list(rows)[-int(count):]
    if not selected:
        return None
    return statistics.fmean(float(row["environment_transitions_per_second"]) for row in selected)


def _system_memory_percent() -> float | None:
    try:
        import psutil

        return float(psutil.virtual_memory().percent)
    except ImportError:  # pragma: no cover
        return None


def _gpu_total_memory_mib() -> float | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False, timeout=3,
        )
        values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        return sum(values) if values else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def ppo_is_in_normal_rollout(run_dir: str | Path) -> bool:
    try:
        status = json.loads((Path(run_dir) / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return str(status.get("state", "")).lower() == "running"


def wait_for_ppo_normal_rollout(
    run_dir: str | Path, *, timeout_seconds: float
) -> None:
    started = time.monotonic()
    while not ppo_is_in_normal_rollout(run_dir):
        if time.monotonic() - started >= float(timeout_seconds):
            raise TimeoutError("PPO did not return to normal rollout before timeout")
        time.sleep(60.0)


def run_case(
    config: Mapping[str, Any],
    output: Path,
    *,
    workers: int,
    transitions: int,
    ppo_progress: Path,
    ppo_samples: int = 1,
) -> Dict[str, Any]:
    case_config = json.loads(json.dumps(config))
    case_config["n_envs"] = int(workers)
    run_dir = output / f"workers_{int(workers)}"
    before_engines = active_holodeck_processes()
    ppo_before_rows = _progress_rates(ppo_progress)
    ppo_before = _mean_last_rates(ppo_before_rows)
    before_count = len(ppo_before_rows)
    monitor = _ResourceMonitor()
    env = None
    started = time.perf_counter()
    try:
        source = _resolve_source_checkpoint(config["initialization"]["source_checkpoint"])
        agent = SACTransitionAgent(
            hidden_sizes=config["sac"]["hidden_sizes"],
            initial_std=config["sac"]["initial_std"],
        )
        transfer_ppo_mean_actor(agent.actor, source, validation_samples=256)
        env = _make_vec_env(case_config, run_dir, "G1")
        env.env_method("enable_sensor_health", True)
        observations = np.asarray(env.reset(), dtype=np.float32)
        identities = list(env.env_method("worker_identity"))
        assert_unique_holoocean_uuids(identities)
        launch_time = time.perf_counter() - started
        monitor.start()
        rollout_started = time.perf_counter()
        collector_steps = int(config["sac"].get("collector_steps", 128))
        minimum_transitions = max(
            int(transitions), int(workers) * collector_steps * 4
        )
        minimum_vector_steps = int(
            np.ceil(minimum_transitions / int(workers))
        )
        latencies = []
        vector_steps = 0
        while True:
            action = agent.act(observations, deterministic=False)
            step_started = time.perf_counter()
            observations, _, _, _ = env.step(action)
            latencies.append(time.perf_counter() - step_started)
            vector_steps += 1
            if vector_steps < minimum_vector_steps:
                continue
            if vector_steps % 32:
                continue
            concurrent_samples = max(0, len(_progress_rates(ppo_progress)) - before_count)
            if concurrent_samples >= max(1, int(ppo_samples)):
                break
            if time.perf_counter() - rollout_started >= 900.0:
                raise RuntimeError(
                    "PPO emitted no complete concurrent rollout sample in 15 minutes"
                )
        rollout_time = time.perf_counter() - rollout_started
        monitor.close()
        completed = vector_steps * int(workers)
        sensor = list(env.env_method("sensor_health"))
        resources = monitor.report()
        resources["system_ram_percent"] = _system_memory_percent()
        resources["gpu_total_memory_mib"] = _gpu_total_memory_mib()
        active_during = active_holodeck_processes()
        checkpoint_tmp = run_dir / "capacity_checkpoint.pt.tmp"
        checkpoint = run_dir / "capacity_checkpoint.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_started = time.perf_counter()
        import torch

        torch.save(agent.checkpoint_state(), checkpoint_tmp)
        os.replace(checkpoint_tmp, checkpoint)
        checkpoint_latency = time.perf_counter() - checkpoint_started
        checkpoint_integrity = isinstance(
            torch.load(checkpoint, map_location="cpu", weights_only=False), dict
        )
        stale = sum(
            sum(int(value) for value in row.get("repeated_fingerprints", {}).values())
            for row in sensor
        )
        cadence = {
            str(index): dict(row.get("observed_presence_rates") or {})
            for index, row in enumerate(sensor)
        }
        mean_latency = statistics.fmean(latencies)
        jitter = statistics.pstdev(latencies) if len(latencies) > 1 else 0.0
        return {
            "ok": True,
            "workers": int(workers),
            "total_active_engines": len(active_during),
            "maximum_active_engines": MAX_ACTIVE_HOLOOCEAN_ENGINES,
            "launch_wall_time_s": launch_time,
            "rollout_wall_time_s": rollout_time,
            "environment_transitions": completed,
            "sac_transitions_per_second": completed / max(rollout_time, 1e-9),
            "sac_transitions_per_second_per_worker": completed / max(rollout_time * workers, 1e-9),
            "mean_vector_step_latency_s": mean_latency,
            "p95_vector_step_latency_s": float(np.percentile(latencies, 95)),
            "rollout_jitter_s": jitter,
            "rollout_jitter_coefficient": jitter / max(mean_latency, 1e-9),
            "worker_identities": identities,
            "sensor_health": sensor,
            "sensor_cadence": cadence,
            "stale_observations": stale,
            "sensor_contract_valid": all(
                bool(row.get("sensor_contract_valid")) for row in sensor
            ),
            "engine_launch_failures": 0,
            "worker_crashes": 0,
            "checkpoint_latency_s": checkpoint_latency,
            "checkpoint_integrity_valid": checkpoint_integrity,
            "resources": resources,
            "engines_before": before_engines,
            "engines_during": active_during,
            "ppo_throughput_before": ppo_before,
            "ppo_progress_rows_before": before_count,
            "required_ppo_progress_samples": int(ppo_samples),
        }
    except Exception as exc:
        monitor.close()
        return {
            "ok": False,
            "workers": int(workers),
            "error": f"{type(exc).__name__}: {exc}",
            "engine_launch_failures": 1 if env is None else 0,
            "worker_crashes": 0 if env is None else 1,
            "engines_before": before_engines,
            "resources": monitor.report(),
            "ppo_throughput_before": ppo_before,
            "ppo_progress_rows_before": before_count,
        }
    finally:
        if env is not None:
            env.close()
        # Let process teardown become visible before orphan accounting.
        time.sleep(1.0)


def finalize_case(row: Dict[str, Any], ppo_progress: Path) -> Dict[str, Any]:
    after = active_holodeck_processes()
    progress = _progress_rates(ppo_progress)
    new_rows = progress[int(row.get("ppo_progress_rows_before", 0)):]
    ppo_during = _mean_last_rates(new_rows) if new_rows else None
    before = row.get("ppo_throughput_before")
    degradation = (
        None if before is None or ppo_during is None
        else max(0.0, (float(before) - float(ppo_during)) / max(float(before), 1e-9))
    )
    before_pids = {int(value["pid"]) for value in row.get("engines_before", [])}
    before_owners = {
        int(value["parent_pid"])
        for value in row.get("engines_before", [])
        if value.get("parent_pid") is not None
    }
    # PPO workers legitimately recycle their Holodeck process between episodes.
    # A new engine PID with the same parent is still PPO-owned, not an orphan
    # left by the benchmark case that just closed.
    orphan_pids = [
        int(value["pid"])
        for value in after
        if int(value["pid"]) not in before_pids
        and int(value.get("parent_pid", -1)) not in before_owners
    ]
    row.update({
        "engines_after": after,
        "orphan_engine_pids": orphan_pids,
        "ppo_throughput_during": ppo_during,
        "ppo_throughput_degradation_fraction": degradation,
        "aggregate_transitions_per_second": (
            float(row.get("sac_transitions_per_second", 0.0) or 0.0)
            + float(ppo_during or 0.0)
        ),
    })
    row["aggregate_transitions_per_second_per_worker"] = (
        float(row["aggregate_transitions_per_second"])
        / max(1, int(row.get("workers", 0)) + 2)
    )
    resources = row.get("resources") or {}
    ram = resources.get("system_ram_percent")
    gpu_memory = (resources.get("gpu_memory_mib") or {}).get("peak")
    total_vram = resources.get("gpu_total_memory_mib")
    vram_percent = (
        None if gpu_memory is None or not total_vram
        else 100.0 * float(gpu_memory) / float(total_vram)
    )
    row["stable"] = bool(
        row.get("ok")
        and row.get("sensor_contract_valid")
        and not orphan_pids
        and int(row.get("total_active_engines", 99)) <= MAX_ACTIVE_HOLOOCEAN_ENGINES
        and (ram is None or float(ram) < 90.0)
        and (vram_percent is None or vram_percent < 90.0)
        and bool(row.get("checkpoint_integrity_valid", True))
        and int(row.get("engine_launch_failures", 0)) == 0
        and int(row.get("worker_crashes", 0)) == 0
    )
    row["vram_peak_mib"] = gpu_memory
    row["vram_percent"] = vram_percent
    return row


def _select(cases: list[Mapping[str, Any]]) -> int | None:
    stable = [row for row in cases if row.get("stable")]
    if not stable:
        return None
    # Optimize aggregate useful throughput. PPO slowdown is reported, but is not
    # itself a rejection criterion.
    selected = stable[0]
    for row in stable[1:]:
        previous = float(selected["aggregate_transitions_per_second"])
        current = float(row["aggregate_transitions_per_second"])
        if current >= previous * 1.10:
            selected = row
    return int(selected["workers"])


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Concurrent PPO/SAC HoloOcean capacity benchmark",
        "",
        "| SAC workers | Total engines | PPO t/s | SAC t/s | Aggregate t/s | PPO degradation | Sensors | Orphans | Stable |",
        "|---:|---:|---:|---:|---:|---:|:---:|---:|:---:|",
    ]
    for row in report["cases"]:
        degradation = row.get("ppo_throughput_degradation_fraction")
        lines.append(
            f"| {row['workers']} | {row.get('total_active_engines', 'n/a')} | "
            f"{row.get('ppo_throughput_during') or 0:.3f} | "
            f"{row.get('sac_transitions_per_second', 0):.3f} | "
            f"{row.get('aggregate_transitions_per_second', 0):.3f} | "
            f"{'n/a' if degradation is None else f'{100*degradation:.1f}%'} | "
            f"{row.get('sensor_contract_valid')} | "
            f"{len(row.get('orphan_engine_pids', []))} | {row.get('stable')} |"
        )
    lines += ["", f"Selected SAC worker count: {report['selected_sac_workers']}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ppo-run", required=True)
    parser.add_argument("--workers", default="1,2,4,6,8")
    parser.add_argument("--transitions", type=int, default=1024)
    parser.add_argument("--ppo-samples", type=int, default=1)
    parser.add_argument("--ppo-wait-timeout", type=float, default=21600.0)
    args = parser.parse_args(argv)
    config = _load_config(args.config)
    output = Path(args.output_dir)
    wait_for_ppo_normal_rollout(
        args.ppo_run, timeout_seconds=args.ppo_wait_timeout
    )
    ppo_progress = Path(args.ppo_run) / "logs" / "progress.jsonl"
    cases = []
    consecutive_no_gain = 0
    best_aggregate = 0.0
    for workers in [int(value) for value in args.workers.split(",") if value.strip()]:
        wait_for_ppo_normal_rollout(
            args.ppo_run, timeout_seconds=args.ppo_wait_timeout
        )
        row = run_case(
            config, output, workers=workers,
            transitions=args.transitions, ppo_progress=ppo_progress,
            ppo_samples=args.ppo_samples,
        )
        row = finalize_case(row, ppo_progress)
        cases.append(row)
        print(json.dumps(row, indent=2), flush=True)
        aggregate = float(row.get("aggregate_transitions_per_second", 0.0) or 0.0)
        if row.get("stable") and aggregate >= 1.10 * max(best_aggregate, 1e-12):
            best_aggregate = max(best_aggregate, aggregate)
            consecutive_no_gain = 0
        else:
            consecutive_no_gain += 1
        if consecutive_no_gain >= 2:
            break
    report = {
        "schema_version": "concurrent_ppo_sac_capacity_v1",
        "generated_utc": now_utc(),
        "strict_global_engine_cap": MAX_ACTIVE_HOLOOCEAN_ENGINES,
        "ppo_run": args.ppo_run,
        "cases": cases,
        "selected_sac_workers": _select(cases),
    }
    atomic_write_json(output / "capacity_benchmark.json", report)
    markdown = output / "capacity_benchmark.md"
    temporary = markdown.with_suffix(".md.tmp")
    temporary.write_text(_markdown(report), encoding="utf-8")
    temporary.replace(markdown)
    return 0 if report["selected_sac_workers"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
