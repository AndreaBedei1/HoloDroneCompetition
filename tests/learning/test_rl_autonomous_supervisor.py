from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import json
import pytest

# The RL stack (gymnasium/torch/SB3) lives in requirements-rl.txt and is not
# installed in the benchmark environment; skip rather than fail collection.
pytest.importorskip("gymnasium")

from marine_race_arena.learning import rl_autonomous_supervisor as supervisor


def test_supervisor_command_never_uses_conda_wrapper(tmp_path):
    spec = {
        "algorithm": "sac", "config": tmp_path / "config.json",
        "run_dir": tmp_path / "run", "n_envs": 4,
    }
    command = supervisor._launch_command(spec, resume=True)
    assert command[:3] == [supervisor.sys.executable, "-m", supervisor.TRAINER_MODULES["sac"]]
    assert Path(command[0]).name.lower() in {"python", "python.exe"}
    assert command[-1] == "--resume"


def test_detached_worker_command_uses_importable_module_name(tmp_path):
    command = supervisor._worker_command(tmp_path.resolve())
    assert command == [
        supervisor.sys.executable,
        "-m",
        "marine_race_arena.learning.rl_autonomous_supervisor",
        "worker",
        "--state-dir",
        str(tmp_path.resolve()),
    ]
    assert "__main__" not in command


def test_runtime_output_inside_worktree_must_be_git_ignored(monkeypatch, tmp_path):
    monkeypatch.setattr(
        supervisor, "_git_relative_path_is_ignored", lambda *_args: False,
    )
    with pytest.raises(ValueError, match="must be git-ignored"):
        supervisor._validate_runtime_output_path(
            tmp_path / "runtime", [tmp_path],
        )


def test_runtime_output_accepts_ignored_or_external_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(
        supervisor, "_git_relative_path_is_ignored", lambda *_args: True,
    )
    supervisor._validate_runtime_output_path(tmp_path / "runtime", [tmp_path])
    supervisor._validate_runtime_output_path(
        tmp_path.parent / "external-runtime", [tmp_path],
    )


def test_trainer_matching_ignores_conda_wrappers(monkeypatch):
    class Process:
        def __init__(self, pid, command):
            self.pid = pid
            self.info = {"cmdline": command}

        def cwd(self):
            return "C:/repo"

    module = supervisor.TRAINER_MODULES["ppo"]
    monkeypatch.setattr(supervisor, "_processes", lambda: [
        Process(1, ["python.exe", "-m", module, "train"]),
        Process(2, ["python.exe", "conda-script.py", "run", "python", "-m", module]),
        Process(3, ["conda.exe", "run", "python", "-m", module]),
    ])
    assert [process.pid for process in supervisor._trainer_processes(module, "C:/repo")] == [1]


def test_algorithm_paths_resolve_relative_to_their_worktree(tmp_path):
    args = Namespace(
        ppo_worktree=str(tmp_path), ppo_run="results/run", ppo_config="configs/p.json",
        ppo_workers=2, ppo_allow_fresh=False,
    )
    spec = supervisor._algorithm_spec(args, "ppo")
    assert Path(spec["run_dir"]) == (tmp_path / "results/run").resolve()
    assert Path(spec["config"]) == (tmp_path / "configs/p.json").resolve()


def test_supervisor_declares_all_required_states():
    assert {
        "INSPECT", "TRAINING", "EVALUATING", "RESUMING",
        "CAPACITY_BENCHMARK", "COMPARING", "PAUSED", "COMPLETED", "FAILED",
    } <= supervisor.STATES


def test_worker_once_preserves_completed_runs_without_launch(monkeypatch, tmp_path):
    state_dir = tmp_path / "supervisor"
    algorithms = {}
    for name in ("ppo", "sac"):
        run = tmp_path / name
        run.mkdir()
        (run / "status.json").write_text(json.dumps({
            "state": "completed",
            "total_environment_transitions": 200_000,
            "target_environment_transitions": 200_000,
        }), encoding="utf-8")
        algorithms[name] = {
            "algorithm": name, "run_dir": str(run), "worktree": str(tmp_path),
            "config": str(tmp_path / f"{name}.json"), "n_envs": None,
            "allow_fresh_start": False,
        }
    state_dir.mkdir()
    (state_dir / "config.json").write_text(json.dumps({
        "interval_seconds": 300, "supervisor_worktree": str(tmp_path),
        "comparison_output": str(tmp_path / "comparison"),
        "algorithms": algorithms,
    }), encoding="utf-8")
    monkeypatch.setattr(supervisor, "_trainer_processes", lambda *args: [])
    monkeypatch.setattr(supervisor, "_checkpoint_summary", lambda *args: None)
    monkeypatch.setattr(supervisor, "_refresh_comparison", lambda *args: False)
    assert supervisor.worker(state_dir, once=True) == 0
    status = json.loads((state_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "COMPLETED"
    assert status["algorithms"]["ppo"]["state"] == "COMPLETED"
    assert status["algorithms"]["sac"]["state"] == "COMPLETED"
