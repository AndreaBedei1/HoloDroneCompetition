"""Resumable PPO workflow: metadata, checkpoints, resume, best-model (fallback only)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from marine_race_arena.learning.train_workflow import latest_checkpoint, run_ppo_training

TRACK = "marine_race_arena/tracks/training/stage1_single_gate.json"
ENV_KWARGS = dict(adapter="fallback", allow_fallback=True, max_steps=15, dt=0.1)
PPO_KWARGS = dict(n_steps=32, batch_size=16, n_epochs=2)


def _run(run_dir, total, resume=False):
    return run_ppo_training(
        TRACK,
        stage="stage1",
        algorithm="ppo",
        total_timesteps=total,
        train_seed=0,
        eval_seeds=[900, 901],
        run_dir=str(run_dir),
        hidden_sizes=(32, 32),
        checkpoint_freq=32,
        eval_freq=32,
        env_kwargs=ENV_KWARGS,
        ppo_kwargs=PPO_KWARGS,
        resume=resume,
    )


def test_run_produces_full_metadata_and_artifacts(tmp_path):
    run_dir, model = _run(tmp_path / "run", total=64)

    # Metadata files.
    for name in ("run_config.json", "environment.json", "seeds.json", "reward_config.json", "track.json", "track_sha256.txt", "reproduce.txt"):
        assert (run_dir / name).exists(), f"missing {name}"
    for sub in ("checkpoints", "best_model", "logs", "evaluation"):
        assert (run_dir / sub).is_dir()

    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["stage"] == "stage1"
    assert run_config["obs_encoding_version"] == "onboard_only_v1"
    assert run_config["obs_dim"] == 36

    env_json = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    assert env_json["packages"]["stable_baselines3"] is not None
    assert env_json["adapter_actual"] == "fallback"
    assert env_json["fallback_used"] is True
    assert env_json["wall_clock_s"] is not None
    assert env_json["final_num_timesteps"] >= 64

    seeds = json.loads((run_dir / "seeds.json").read_text(encoding="utf-8"))
    assert seeds["eval_seeds"] == [900, 901]

    # Checkpoints + evaluation + final model.
    assert latest_checkpoint(run_dir) is not None
    assert (run_dir / "evaluation" / "eval.csv").exists()
    assert (run_dir / "final_model.zip").exists()
    assert "completion_rate" in (run_dir / "evaluation" / "eval.csv").read_text(encoding="utf-8").splitlines()[0]


def test_never_overwrites_nonempty_run(tmp_path):
    run_dir = tmp_path / "run"
    _run(run_dir, total=32)
    with pytest.raises(FileExistsError):
        _run(run_dir, total=32, resume=False)


def test_resume_continues_from_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    _, model1 = _run(run_dir, total=32)
    steps1 = int(model1.num_timesteps)
    _, model2 = _run(run_dir, total=96, resume=True)
    steps2 = int(model2.num_timesteps)
    assert steps2 > steps1, "resume did not continue training from the checkpoint"


def test_resume_preserves_eval_history_and_best_metrics(tmp_path):
    run_dir = tmp_path / "run"
    _run(run_dir, total=64)
    eval_csv = run_dir / "evaluation" / "eval.csv"
    header = eval_csv.read_text(encoding="utf-8").splitlines()[0]
    assert "mean_time_finished" in header  # best-model tie-breaker column present
    rows_before = len(eval_csv.read_text(encoding="utf-8").strip().splitlines())
    assert (run_dir / "best_model" / "best_metrics.json").exists()
    _run(run_dir, total=160, resume=True)
    rows_after = len(eval_csv.read_text(encoding="utf-8").strip().splitlines())
    assert rows_after > rows_before, "resume erased evaluation history instead of appending"


def test_generated_reproduce_script_executes(tmp_path):
    """The reproduce.txt python body runs end-to-end and produces a fresh run."""
    run_dir = tmp_path / "run"
    _run(run_dir, total=64)
    reproduce = (run_dir / "reproduce.txt").read_text(encoding="utf-8")
    body = reproduce.split("python - <<'PY'", 1)[1].split("\nPY", 1)[0]
    fresh_root = tmp_path / "reproduced"
    env = dict(os.environ, MARINE_RACE_REPRODUCE_ROOT=str(fresh_root), PYTHONPATH=str(Path.cwd()))
    result = subprocess.run([sys.executable, "-c", body], capture_output=True, text=True, env=env, timeout=400)
    assert result.returncode == 0, result.stderr[-2000:]
    assert list(fresh_root.rglob("run_config.json")), "reproduce script created no run"


def test_incompatible_resume_is_rejected(tmp_path):
    from marine_race_arena.learning.train_workflow import run_ppo_training

    run_dir = tmp_path / "run"
    _run(run_dir, total=32)  # trained with hidden_sizes=(32, 32)
    with pytest.raises(ValueError, match="incompatible"):
        run_ppo_training(
            TRACK, stage="stage1", algorithm="ppo", total_timesteps=64, train_seed=0,
            eval_seeds=[900, 901], run_dir=str(run_dir), hidden_sizes=(64, 64),  # changed
            checkpoint_freq=32, eval_freq=32, env_kwargs=ENV_KWARGS, ppo_kwargs=PPO_KWARGS, resume=True,
        )
