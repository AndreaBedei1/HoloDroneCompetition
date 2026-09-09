"""Real-HoloOcean one/two-worker and frame-throttling benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from marine_race_arena.learning.provenance import now_utc
from marine_race_arena.learning.train_ppo_transition import _make_vec_env


class _ResourceMonitor:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self.cpu: list[float] = []
        self.ram_gib: list[float] = []
        self.gpu_percent: list[float] = []
        self.gpu_memory_mib: list[float] = []
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def _sample(self) -> None:
        try:
            import psutil
        except ImportError:  # pragma: no cover - optional diagnostics dependency
            psutil = None
        if psutil is not None:
            psutil.cpu_percent(interval=None)
        while not self._stop.wait(0.5):
            if psutil is not None:
                self.cpu.append(float(psutil.cpu_percent(interval=None)))
                root = psutil.Process()
                processes = [root, *root.children(recursive=True)]
                rss = 0
                for process in processes:
                    try:
                        rss += process.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                self.ram_gib.append(rss / (1024.0 ** 3))
            try:
                probe = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=2,
                )
                for line in probe.stdout.splitlines():
                    fields = [part.strip() for part in line.split(",")]
                    if len(fields) >= 2:
                        self.gpu_percent.append(float(fields[0]))
                        self.gpu_memory_mib.append(float(fields[1]))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass

    @staticmethod
    def _summary(values: list[float]) -> Dict[str, float | None]:
        return {
            "mean": None if not values else float(statistics.fmean(values)),
            "peak": None if not values else float(max(values)),
        }

    def report(self) -> Dict[str, Any]:
        return {
            "cpu_percent": self._summary(self.cpu),
            "process_tree_ram_gib": self._summary(self.ram_gib),
            "gpu_utilization_percent": self._summary(self.gpu_percent),
            "gpu_memory_mib": self._summary(self.gpu_memory_mib),
        }


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _case_config(n_envs: int, frames_per_sec: bool) -> Dict[str, Any]:
    return {
        "n_envs": int(n_envs),
        "worker_seed_base": 73001,
        "curriculum": {"transition_focus_fraction": 0.8},
        "adapter": "holoocean",
        "allow_fallback": False,
        "holoocean_frames_per_sec": bool(frames_per_sec),
        "max_episode_steps": 600,
        "reward": {},
    }


def _run_case(
    output: Path,
    *,
    n_envs: int,
    frames_per_sec: bool,
    total_transitions: int,
    repeat: int,
) -> Dict[str, Any]:
    run_dir = output / (
        f"workers_{n_envs}_frames_{str(frames_per_sec).lower()}_repeat_{repeat}"
    )
    env = None
    monitor = _ResourceMonitor()
    launched = time.perf_counter()
    try:
        env = _make_vec_env(_case_config(n_envs, frames_per_sec), run_dir, "G1")
        env.env_method("enable_sensor_health", True)
        obs = env.reset()
        launch_time = time.perf_counter() - launched
        identities = env.env_method("worker_identity")
        monitor.start()
        rollout_started = time.perf_counter()
        digest = hashlib.sha256()
        vec_steps = int(math.ceil(total_transitions / n_envs))
        for step in range(vec_steps):
            phase = 0.07 * step
            action = np.asarray([
                0.35 + 0.15 * math.sin(phase),
                0.20 * math.sin(phase * 0.7),
                0.20 * math.cos(phase * 0.5),
                0.25 * math.sin(phase * 0.9),
            ], dtype=np.float32)
            actions = np.repeat(action[None, :], n_envs, axis=0)
            obs, _, _, _ = env.step(actions)
            digest.update(np.round(np.asarray(obs), 5).astype(np.float32).tobytes())
        rollout_time = time.perf_counter() - rollout_started
        monitor.close()
        completed = vec_steps * n_envs
        return {
            "ok": True,
            "n_envs": n_envs,
            "frames_per_sec": frames_per_sec,
            "repeat": repeat,
            "environment_transitions": completed,
            "engine_launch_wall_time_s": launch_time,
            "rollout_wall_time_s": rollout_time,
            "total_wall_time_s": time.perf_counter() - launched,
            "environment_transitions_per_second": completed / max(rollout_time, 1e-9),
            "worker_identities": identities,
            "sensor_health": env.env_method("sensor_health"),
            "observation_trace_sha256": digest.hexdigest(),
            "resources": monitor.report(),
            "engine_launch_failures": 0,
        }
    except Exception as exc:
        monitor.close()
        return {
            "ok": False,
            "n_envs": n_envs,
            "frames_per_sec": frames_per_sec,
            "repeat": repeat,
            "total_wall_time_s": time.perf_counter() - launched,
            "engine_launch_failures": 1,
            "error": f"{type(exc).__name__}: {exc}",
            "resources": monitor.report(),
        }
    finally:
        if env is not None:
            env.close()


def _summarize(cases: list[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, Any] = {}
    for n_envs in (1, 2):
        for frames in (False, True):
            selected = [
                row for row in cases
                if row["n_envs"] == n_envs and row["frames_per_sec"] is frames
            ]
            good = [row for row in selected if row.get("ok")]
            key = f"workers_{n_envs}_frames_{str(frames).lower()}"
            hashes = [row["observation_trace_sha256"] for row in good]
            sensor_contract_valid = all(
                worker.get("sensor_contract_valid", _legacy_sensor_contract_valid(worker))
                for row in good for worker in row["sensor_health"]
            )
            groups[key] = {
                "attempts": len(selected),
                "successful_attempts": len(good),
                "mean_transitions_per_second": (
                    None if not good else statistics.fmean(
                        row["environment_transitions_per_second"] for row in good
                    )
                ),
                "mean_rollout_wall_time_s": (
                    None if not good else statistics.fmean(
                        row["rollout_wall_time_s"] for row in good
                    )
                ),
                "engine_launch_failures": sum(
                    int(row.get("engine_launch_failures", 0)) for row in selected
                ),
                "exact_observation_trace_reproducibility": (
                    None if len(hashes) < 2 else len(set(hashes)) == 1
                ),
                "sensor_contract_valid": sensor_contract_valid,
                "raw_missing_sensor_emissions": sum(
                    sum(sum(worker["missing_frames"].values()) for worker in row["sensor_health"])
                    for row in good
                ),
            }
    one = groups["workers_1_frames_false"]["mean_transitions_per_second"]
    two = groups["workers_2_frames_false"]["mean_transitions_per_second"]
    speedup = None if not one or not two else two / one
    return {
        "groups": groups,
        "two_worker_unthrottled_speedup": speedup,
        "two_workers_meaningful": bool(speedup is not None and speedup >= 1.20),
        "deterministic_case_generation_reproducible": _sampler_reproducible(),
    }


def _legacy_sensor_contract_valid(worker: Dict[str, Any]) -> bool:
    frames = max(1, int(worker.get("frames_checked", 0)))
    missing = worker.get("missing_frames", {})
    repeated = worker.get("repeated_fingerprints", {})
    required_every_tick = ("FrontCamera", "IMUSensor", "DepthSensor")
    full_rate_ok = all(int(missing.get(name, frames)) == 0 for name in required_every_tick)
    dvl_presence = 1.0 - int(missing.get("DVLSensor", frames)) / frames
    return full_rate_ok and dvl_presence >= 0.42 and all(
        int(value) == 0 for value in repeated.values()
    )


def _sampler_reproducible() -> bool:
    from marine_race_arena.learning.transition_curriculum import TransitionGeometrySampler

    first = TransitionGeometrySampler(seed=99173, difficulty="G6")
    second = TransitionGeometrySampler(seed=99173, difficulty="G6")
    return [first.sample() for _ in range(128)] == [second.sample() for _ in range(128)]


def _markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Universal-transition HoloOcean parallel benchmark",
        "",
        f"Generated: {report['generated_utc']}",
        "",
        "| Workers | `frames_per_sec` | Successful | transitions/s | Wall time (s) | Launch failures | Sensor cadence/freshness valid | Exact trace match |",
        "|---:|:---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for key, row in report["summary"]["groups"].items():
        workers = key.split("_")[1]
        frames = key.rsplit("_", 1)[-1]
        rate = row["mean_transitions_per_second"]
        wall = row["mean_rollout_wall_time_s"]
        lines.append(
            f"| {workers} | {frames} | {row['successful_attempts']}/{row['attempts']} | "
            f"{'n/a' if rate is None else f'{rate:.3f}'} | "
            f"{'n/a' if wall is None else f'{wall:.3f}'} | "
            f"{row['engine_launch_failures']} | {row['sensor_contract_valid']} | "
            f"{row['exact_observation_trace_reproducibility']} |"
        )
    speedup = report["summary"]["two_worker_unthrottled_speedup"]
    lines += [
        "",
        f"Two-worker unthrottled speedup: {'n/a' if speedup is None else f'{speedup:.3f}x'}.",
        f"Two workers pass the >=1.20x meaningful-speedup rule: {report['summary']['two_workers_meaningful']}.",
        f"Deterministic unseen-case generation reproducible: {report['summary']['deterministic_case_generation_reproducible']}.",
        "",
        "DVL emits at its configured 15 Hz in a 30 Hz world; its alternating absence is expected. Exact noisy simulator traces may differ across engine launches even when deterministic case definitions match.",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--transitions", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--summarize-existing", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    existing_path = output / "parallel_benchmark.json"
    if args.summarize_existing:
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        existing["summary"] = _summarize(list(existing["cases"]))
        _atomic_text(existing_path, json.dumps(existing, indent=2))
        _atomic_text(output / "parallel_benchmark.md", _markdown(existing))
        return 0
    cases = []
    for frames in (False, True):
        for n_envs in (1, 2):
            for repeat in range(args.repeats):
                row = _run_case(
                    output,
                    n_envs=n_envs,
                    frames_per_sec=frames,
                    total_transitions=args.transitions,
                    repeat=repeat,
                )
                cases.append(row)
                print(json.dumps(row, indent=2), flush=True)
    report = {
        "schema_version": "universal_transition_parallel_benchmark_v1",
        "generated_utc": now_utc(),
        "total_transitions_per_case": args.transitions,
        "repeats": args.repeats,
        "cases": cases,
        "summary": _summarize(cases),
    }
    _atomic_text(output / "parallel_benchmark.json", json.dumps(report, indent=2))
    _atomic_text(output / "parallel_benchmark.md", _markdown(report))
    return 0 if all(row.get("ok") for row in cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
