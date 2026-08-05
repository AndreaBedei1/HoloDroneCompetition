"""Detached, persistent supervisor for independent PPO and SAC transition runs.

The supervisor never owns policy data.  It only validates atomic checkpoints,
starts or resumes each trainer, records process ownership, and refreshes the
equal-budget comparison.  A trainer's own evaluation/collapse rules remain the
authority for policy selection.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from marine_race_arena.learning.longrun_checkpoint import (
    atomic_append_jsonl,
    atomic_write_json,
    latest_valid_checkpoint,
)
from marine_race_arena.learning.provenance import now_utc
from marine_race_arena.learning.sac_transition_checkpoint import (
    latest_valid_sac_checkpoint,
)


STATES = {
    "INSPECT", "TRAINING", "EVALUATING", "RESUMING", "CAPACITY_BENCHMARK",
    "COMPARING", "PAUSED", "COMPLETED", "FAILED",
}
TRAINER_MODULES = {
    "ppo": "marine_race_arena.learning.train_ppo_transition",
    "sac": "marine_race_arena.learning.train_sac_transition",
}
MAX_RESTART_ATTEMPTS = 3


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _processes() -> Iterable[Any]:
    try:
        import psutil
    except ImportError:  # pragma: no cover - runtime dependency on Windows
        return []
    return psutil.process_iter(["pid", "name", "cmdline", "create_time"])


def _trainer_processes(module: str, worktree: str | Path | None = None) -> list[Any]:
    matches = []
    expected_cwd = None if worktree is None else os.path.normcase(os.path.abspath(worktree))
    for process in _processes():
        try:
            cmdline = list(process.info.get("cmdline") or [])
            # The real trainer is ``python -m ...``.  Conda's launcher and
            # conda-script Python wrapper may repeat the same arguments later
            # in their command line and must not count as trainers.
            if len(cmdline) < 3 or cmdline[1] != "-m":
                continue
            if cmdline[2] != module:
                continue
            if expected_cwd is not None:
                cwd = os.path.normcase(os.path.abspath(process.cwd()))
                if cwd != expected_cwd:
                    continue
            matches.append(process)
        except Exception:
            continue
    return matches


def _pid_alive(pid: int) -> bool:
    try:
        import psutil

        return psutil.Process(int(pid)).is_running()
    except Exception:
        return False


def _checkpoint_summary(algorithm: str, run_dir: Path) -> Optional[Dict[str, Any]]:
    expected = _read_json(run_dir / "run_manifest.json").get(
        "config_contract_sha256"
    )
    valid = (
        latest_valid_checkpoint(
            run_dir, expected_contract_hash=expected, safe_only=False
        )
        if algorithm == "ppo"
        else latest_valid_sac_checkpoint(
            run_dir, expected_contract_sha256=expected
        )
    )
    if valid is None:
        return None
    return {
        "timesteps": int(valid.timesteps),
        "model_path": str(valid.model_path),
        "manifest_path": str(valid.manifest_path),
        "manifest": dict(valid.manifest),
    }


def _owned_engines(trainer: Any) -> list[Dict[str, Any]]:
    result = []
    try:
        descendants = trainer.children(recursive=True)
    except Exception:
        descendants = []
    for process in descendants:
        try:
            name = process.name().lower()
            if name not in {"holodeck.exe", "holodeck"}:
                continue
            result.append({
                "pid": int(process.pid),
                "create_time": float(process.create_time()),
                "name": process.name(),
            })
        except Exception:
            continue
    return result


def cleanup_recorded_orphans(records: Iterable[Mapping[str, Any]]) -> list[int]:
    """Terminate only exact PID/create-time pairs previously owned by a trainer."""

    try:
        import psutil
    except ImportError:  # pragma: no cover
        return []
    terminated = []
    for record in records:
        try:
            process = psutil.Process(int(record["pid"]))
            if process.name().lower() not in {"holodeck.exe", "holodeck"}:
                continue
            if abs(float(process.create_time()) - float(record["create_time"])) > 0.01:
                continue
            process.terminate()
            try:
                process.wait(timeout=20)
            except psutil.TimeoutExpired:
                process.kill()
            terminated.append(int(process.pid))
        except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError, ValueError):
            continue
    return terminated


def _collapsed(status: Mapping[str, Any]) -> bool:
    selection = dict(status.get("checkpoint_selection_state") or {})
    message = str(status.get("message", "")).lower()
    return bool(selection.get("rollback")) or "collapse" in message


def _target_reached(status: Mapping[str, Any]) -> bool:
    current = int(status.get("total_environment_transitions", 0) or 0)
    target = int(status.get("target_environment_transitions", 0) or 0)
    return target > 0 and current >= target


def _launch_command(spec: Mapping[str, Any], *, resume: bool) -> list[str]:
    command = [
        sys.executable, "-m", TRAINER_MODULES[str(spec["algorithm"])], "train",
        "--config", str(spec["config"]), "--run-dir", str(spec["run_dir"]),
    ]
    if spec.get("n_envs") is not None:
        command += ["--n-envs", str(int(spec["n_envs"]))]
    if resume:
        command.append("--resume")
    return command


def _detached_flags() -> int:
    if os.name != "nt":
        return 0
    return (
        int(getattr(subprocess, "DETACHED_PROCESS", 0x00000008))
        | int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200))
        | int(getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))
    )


def _launch_trainer(spec: Mapping[str, Any], *, resume: bool) -> int:
    run_dir = Path(spec["run_dir"])
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    stdout = (run_dir / "logs" / "supervised_trainer.stdout.log").open("a", encoding="utf-8")
    stderr = (run_dir / "logs" / "supervised_trainer.stderr.log").open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            _launch_command(spec, resume=resume), cwd=str(spec["worktree"]),
            stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
            close_fds=True, creationflags=_detached_flags(),
            start_new_session=(os.name != "nt"),
        )
    finally:
        stdout.close()
        stderr.close()
    return int(process.pid)


def inspect_algorithm(
    spec: Mapping[str, Any], previous: Mapping[str, Any], retries: int,
) -> Dict[str, Any]:
    algorithm = str(spec["algorithm"])
    run_dir = Path(spec["run_dir"])
    status = _read_json(run_dir / "status.json")
    trainers = _trainer_processes(TRAINER_MODULES[algorithm], spec.get("worktree"))
    checkpoint = _checkpoint_summary(algorithm, run_dir)
    result: Dict[str, Any] = {
        "algorithm": algorithm,
        "run_dir": str(run_dir),
        "trainer_pids": [int(process.pid) for process in trainers],
        "trainer_count": len(trainers),
        "trainer_status": status,
        "checkpoint": checkpoint,
        "restart_attempts": int(retries),
        "owned_engines": [],
        "state": "INSPECT",
    }
    if len(trainers) > 1:
        result.update(state="FAILED", error="duplicate real trainer processes")
        return result
    if trainers:
        result["owned_engines"] = _owned_engines(trainers[0])
        result["state"] = "EVALUATING" if str(status.get("state", "")).lower() == "evaluating" else "TRAINING"
        previous_steps = int(
            (previous.get("trainer_status") or {}).get(
                "total_environment_transitions", -1
            ) or -1
        )
        current_steps = int(status.get("total_environment_transitions", 0) or 0)
        result["restart_attempts"] = 0 if current_steps > previous_steps else int(retries)
        return result

    result["removed_orphan_pids"] = cleanup_recorded_orphans(
        previous.get("owned_engines") or []
    )
    if (run_dir / "STOP_REQUESTED").exists() or _collapsed(status):
        result["state"] = "PAUSED"
        result["reason"] = "explicit_stop_or_trainer_collapse_rule"
        return result
    if _target_reached(status) or str(status.get("state", "")).lower() == "completed":
        result["state"] = "COMPLETED"
        return result
    if retries >= MAX_RESTART_ATTEMPTS:
        result.update(state="FAILED", error="automatic restart limit reached")
        return result
    if checkpoint is None and not bool(spec.get("allow_fresh_start", False)):
        result.update(state="FAILED", error="no valid atomic checkpoint")
        return result
    if checkpoint is None and any((run_dir / "checkpoints").glob("*.manifest.json")):
        result.update(state="FAILED", error="checkpoint history exists but none is valid")
        return result
    resume = checkpoint is not None
    result["launched_pid"] = _launch_trainer(spec, resume=resume)
    result["restart_attempts"] = retries + 1
    result["state"] = "RESUMING"
    result["resume"] = resume
    return result


def _evaluation_signature(run_dir: Path) -> list[list[Any]]:
    values = []
    for path in sorted((run_dir / "evaluations").glob("unseen_*/evaluation.json")):
        try:
            stat = path.stat()
            values.append([path.parent.name, int(stat.st_size), int(stat.st_mtime_ns)])
        except OSError:
            continue
    return values


def _refresh_comparison(config: Mapping[str, Any], signature: Any) -> bool:
    previous = config.get("comparison_signature")
    if signature == previous:
        return False
    command = [
        sys.executable, "-m", "marine_race_arena.learning.compare_ppo_sac_transition",
        "--ppo-run", str(config["algorithms"]["ppo"]["run_dir"]),
        "--sac-run", str(config["algorithms"]["sac"]["run_dir"]),
        "--output-dir", str(config["comparison_output"]),
    ]
    result = subprocess.run(
        command, cwd=str(config["supervisor_worktree"]), capture_output=True,
        text=True, check=False, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"comparison failed: {result.stderr[-1000:]}")
    return True


def _write_supervisor_state(state_dir: Path, value: Mapping[str, Any]) -> None:
    state = str(value.get("state"))
    if state not in STATES:
        raise ValueError(f"invalid supervisor state {state}")
    atomic_write_json(state_dir / "status.json", dict(value))
    atomic_append_jsonl(state_dir / "history.jsonl", dict(value))


def worker(state_dir: Path, *, once: bool = False) -> int:
    config_path = state_dir / "config.json"
    config = _read_json(config_path)
    if not config:
        raise FileNotFoundError(config_path)
    previous = _read_json(state_dir / "status.json")
    retries = dict(previous.get("restart_attempts") or {})
    state_dir.mkdir(parents=True, exist_ok=True)
    log_path = state_dir / "supervisor.log"
    while True:
        if (state_dir / "STOP_REQUESTED").exists():
            value = {
                "schema_version": "rl_autonomous_supervisor_v1",
                "updated_utc": now_utc(), "state": "PAUSED",
                "pid": os.getpid(), "message": "supervisor stop requested; trainers untouched",
                "algorithms": previous.get("algorithms", {}),
                "restart_attempts": retries,
            }
            _write_supervisor_state(state_dir, value)
            return 0
        rows: Dict[str, Any] = {}
        errors = []
        for name in ("ppo", "sac"):
            spec = config["algorithms"][name]
            before = dict((previous.get("algorithms") or {}).get(name) or {})
            row = inspect_algorithm(spec, before, int(retries.get(name, 0)))
            retries[name] = int(row.get("restart_attempts", retries.get(name, 0)))
            rows[name] = row
            if row["state"] == "FAILED":
                errors.append(f"{name}: {row.get('error', 'failed')}")
        state = "FAILED" if errors else (
            "EVALUATING" if any(row["state"] == "EVALUATING" for row in rows.values())
            else "RESUMING" if any(row["state"] == "RESUMING" for row in rows.values())
            else "TRAINING" if any(row["state"] == "TRAINING" for row in rows.values())
            else "COMPLETED" if all(row["state"] == "COMPLETED" for row in rows.values())
            else "PAUSED" if any(row["state"] == "PAUSED" for row in rows.values())
            else "INSPECT"
        )
        signature = {
            name: _evaluation_signature(Path(spec["run_dir"]))
            for name, spec in config["algorithms"].items()
        }
        comparison_updated = False
        comparison_error = None
        if not errors:
            try:
                comparison_updated = _refresh_comparison(
                    {**config, "comparison_signature": previous.get("comparison_signature")},
                    signature,
                )
            except Exception as exc:
                comparison_error = f"{type(exc).__name__}: {exc}"
        value = {
            "schema_version": "rl_autonomous_supervisor_v1",
            "updated_utc": now_utc(), "state": "FAILED" if errors else state,
            "pid": os.getpid(), "check_interval_seconds": int(config["interval_seconds"]),
            "algorithms": rows, "restart_attempts": retries,
            "comparison_signature": (
                previous.get("comparison_signature")
                if comparison_error else signature
            ),
            "comparison_updated": comparison_updated,
            "comparison_error": comparison_error,
            "errors": errors,
        }
        _write_supervisor_state(state_dir, value)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "utc": value["updated_utc"], "state": value["state"],
                "ppo": rows["ppo"]["state"], "sac": rows["sac"]["state"],
                "errors": errors,
            }, separators=(",", ":")) + "\n")
        previous = value
        if once or value["state"] in {"FAILED", "COMPLETED"}:
            return 1 if value["state"] == "FAILED" else 0
        time.sleep(int(config["interval_seconds"]))


def _algorithm_spec(args: argparse.Namespace, name: str) -> Dict[str, Any]:
    worktree = Path(getattr(args, f"{name}_worktree")).resolve()
    run_dir = Path(getattr(args, f"{name}_run"))
    config = Path(getattr(args, f"{name}_config"))
    if not run_dir.is_absolute():
        run_dir = worktree / run_dir
    if not config.is_absolute():
        config = worktree / config
    return {
        "algorithm": name,
        "worktree": str(worktree),
        "run_dir": str(run_dir.resolve()),
        "config": str(config.resolve()),
        "n_envs": getattr(args, f"{name}_workers"),
        "allow_fresh_start": bool(getattr(args, f"{name}_allow_fresh")),
    }


def start(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    existing = _read_json(state_dir / "status.json")
    if int(existing.get("pid", 0) or 0) and _pid_alive(int(existing["pid"])):
        raise RuntimeError(f"supervisor already active as PID {existing['pid']}")
    (state_dir / "STOP_REQUESTED").unlink(missing_ok=True)
    config = {
        "schema_version": "rl_autonomous_supervisor_config_v1",
        "created_utc": now_utc(),
        "interval_seconds": int(args.interval_seconds),
        "supervisor_worktree": str(Path(args.supervisor_worktree).resolve()),
        "comparison_output": str(Path(args.comparison_output).resolve()),
        "algorithms": {
            "ppo": _algorithm_spec(args, "ppo"),
            "sac": _algorithm_spec(args, "sac"),
        },
    }
    atomic_write_json(state_dir / "config.json", config)
    stdout = (state_dir / "supervisor.stdout.log").open("a", encoding="utf-8")
    stderr = (state_dir / "supervisor.stderr.log").open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", __name__, "worker", "--state-dir", str(state_dir)],
            cwd=config["supervisor_worktree"], stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr, close_fds=True,
            creationflags=_detached_flags(), start_new_session=(os.name != "nt"),
        )
    finally:
        stdout.close()
        stderr.close()
    initial = {
        "schema_version": "rl_autonomous_supervisor_v1", "updated_utc": now_utc(),
        "state": "INSPECT", "pid": int(process.pid), "algorithms": {},
        "restart_attempts": {"ppo": 0, "sac": 0},
        "message": "detached supervisor launched",
    }
    _write_supervisor_state(state_dir, initial)
    print(json.dumps(initial, indent=2))
    return 0


def status(state_dir: Path) -> int:
    value = _read_json(state_dir / "status.json")
    if not value:
        print(json.dumps({"state": "NOT_STARTED", "state_dir": str(state_dir)}, indent=2))
        return 1
    value["process_alive"] = _pid_alive(int(value.get("pid", 0) or 0))
    print(json.dumps(value, indent=2))
    return 0


def stop(state_dir: Path) -> int:
    if not (state_dir / "status.json").exists():
        raise FileNotFoundError(state_dir / "status.json")
    (state_dir / "STOP_REQUESTED").touch(exist_ok=True)
    print(json.dumps({
        "stop_requested": True, "scope": "supervisor_only",
        "trainers_untouched": True, "marker": str(state_dir / "STOP_REQUESTED"),
    }, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    start_parser = commands.add_parser("start")
    start_parser.add_argument("--state-dir", default="results/rl/autonomous_supervisor")
    start_parser.add_argument("--supervisor-worktree", required=True)
    start_parser.add_argument("--comparison-output", required=True)
    for name in ("ppo", "sac"):
        start_parser.add_argument(f"--{name}-worktree", required=True)
        start_parser.add_argument(f"--{name}-run", required=True)
        start_parser.add_argument(f"--{name}-config", required=True)
        start_parser.add_argument(f"--{name}-workers", type=int)
        start_parser.add_argument(f"--{name}-allow-fresh", action="store_true")
    start_parser.add_argument("--interval-seconds", type=int, default=300)
    worker_parser = commands.add_parser("worker")
    worker_parser.add_argument("--state-dir", required=True)
    worker_parser.add_argument("--once", action="store_true")
    status_parser = commands.add_parser("status")
    status_parser.add_argument("--state-dir", default="results/rl/autonomous_supervisor")
    stop_parser = commands.add_parser("stop")
    stop_parser.add_argument("--state-dir", default="results/rl/autonomous_supervisor")
    args = parser.parse_args(argv)
    if args.command == "start":
        return start(args)
    if args.command == "worker":
        return worker(Path(args.state_dir), once=args.once)
    if args.command == "status":
        return status(Path(args.state_dir))
    return stop(Path(args.state_dir))


if __name__ == "__main__":
    raise SystemExit(main())
