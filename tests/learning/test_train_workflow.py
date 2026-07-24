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


def _bc_policy():
    from marine_race_arena.learning.bc_train import BCPolicy

    return BCPolicy(hidden_sizes=(32, 32))


def test_timestep_zero_eval_and_action_std_for_bcinit(tmp_path):
    run_dir = tmp_path / "run"
    run_ppo_training(
        TRACK, stage="stage1", algorithm="ppo_bcinit", total_timesteps=32, train_seed=0,
        eval_seeds=[900, 901], run_dir=str(run_dir), hidden_sizes=(32, 32),
        checkpoint_freq=32, eval_freq=32, env_kwargs=ENV_KWARGS, ppo_kwargs=PPO_KWARGS,
        bc_policy=_bc_policy(),
    )
    # Timestep-zero held-out evaluation, deterministic, with the required metrics.
    init = json.loads((run_dir / "evaluation" / "initial_eval.json").read_text(encoding="utf-8"))
    assert init["timesteps"] == 0 and init["bc_initialized"] is True and init["deterministic"] is True
    for k in ("completion_rate", "mean_gates", "mean_collisions", "mean_out_of_bounds", "mean_wrong_direction"):
        assert k in init
    assert init["model_initialization_source"].startswith("bc_transfer")
    assert "action_std" in init
    # eval.csv's first data row is the timestep-0 evaluation.
    csv_lines = (run_dir / "evaluation" / "eval.csv").read_text(encoding="utf-8").strip().splitlines()
    assert csv_lines[1].startswith("0,")
    # Per-axis action-std provenance persisted.
    astd = json.loads((run_dir / "action_std.json").read_text(encoding="utf-8"))
    assert set(astd["std_per_axis"]) == {"surge", "sway", "heave", "yaw"}
    rc = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert rc["bc_action_std_config"]["std_min"] == 0.05 and rc["action_std"] is not None
    # The timestep-0 policy is saved as the initial best.
    assert (run_dir / "best_model" / "best_metrics.json").exists()


def test_scratch_arm_keeps_default_action_std(tmp_path):
    run_dir = tmp_path / "run"
    _run(run_dir, total=32)  # no bc_policy -> scratch defaults to sb3_default std
    astd = json.loads((run_dir / "action_std.json").read_text(encoding="utf-8"))
    assert astd["source"] == "sb3_default" and astd["strategy"] == "sb3_default"
    init = json.loads((run_dir / "evaluation" / "initial_eval.json").read_text(encoding="utf-8"))
    assert init["model_initialization_source"] == "sb3_default"


def test_resume_does_not_duplicate_timestep_zero(tmp_path):
    run_dir = tmp_path / "run"
    _run(run_dir, total=32)
    _run(run_dir, total=96, resume=True)
    csv_text = (run_dir / "evaluation" / "eval.csv").read_text(encoding="utf-8").strip().splitlines()
    zero_rows = [ln for ln in csv_text[1:] if ln.split(",")[0] == "0"]
    assert len(zero_rows) == 1, "resume must not add a second timestep-zero row"


def test_stage2_run_produces_monitor_and_rich_eval(tmp_path):
    """A Stage-2 randomized fallback run wires the KL monitor, run_status, rich eval and
    reward-component diagnostic together."""
    from marine_race_arena.learning.curriculum import STAGE2_RANDOMIZATION

    run_dir = tmp_path / "run"
    env_kwargs = dict(adapter="fallback", allow_fallback=True, max_steps=15, dt=0.1,
                      start_randomization=STAGE2_RANDOMIZATION)
    run_ppo_training(
        TRACK, stage="stage2", algorithm="stage2_randomized_bcinit_controlled",
        total_timesteps=64, train_seed=9000, eval_seeds=[1410, 1411, 1412], run_dir=str(run_dir),
        hidden_sizes=(32, 32), checkpoint_freq=32, eval_freq=32, env_kwargs=env_kwargs,
        ppo_kwargs=dict(n_steps=32, batch_size=16, n_epochs=1, target_kl=0.01, clip_range=0.05),
        bc_policy=_bc_policy(), arm="bcinit_controlled", action_std_strategy="fixed", action_std_value=0.10,
        max_acceptable_kl=0.5, stage2=True,
    )
    # KL monitoring + run status.
    assert (run_dir / "training" / "ppo_update_metrics.csv").exists()
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status["run_status"] in ("COMPLETED", "ABORT_MAX_KL")
    assert "max_approx_kl" in status["kl_summary"]
    # Rich Stage-2 timestep-zero eval with interior/extreme split + per-seed rows.
    init = json.loads((run_dir / "evaluation" / "initial_eval.json").read_text(encoding="utf-8"))
    for k in ("completion_rate", "interior_completion", "extreme_completion", "oob_episodes",
              "mean_action_saturation", "per_seed"):
        assert k in init
    assert init["action_std"]["strategy"] == "fixed"
    # Reward-component diagnostic present.
    assert (run_dir / "evaluation" / "reward_components.json").exists()
    # Stage-2 best_metrics carries the aggregate (not the Stage-1 shape).
    best = json.loads((run_dir / "best_model" / "best_metrics.json").read_text(encoding="utf-8"))
    assert "completion_rate" in best and "extreme_completion" in best


def test_stage2_kl_safety_abort(tmp_path):
    """A tiny max_acceptable_kl trips the hard KL stop -> ABORT_MAX_KL, env still closed."""
    from marine_race_arena.learning.curriculum import STAGE2_RANDOMIZATION

    run_dir = tmp_path / "run"
    env_kwargs = dict(adapter="fallback", allow_fallback=True, max_steps=15, dt=0.1,
                      start_randomization=STAGE2_RANDOMIZATION)
    run_ppo_training(
        TRACK, stage="stage2", algorithm="stage2_randomized_scratch_controlled",
        total_timesteps=64, train_seed=9000, eval_seeds=[1410, 1411], run_dir=str(run_dir),
        hidden_sizes=(32, 32), checkpoint_freq=32, eval_freq=64, env_kwargs=env_kwargs,
        ppo_kwargs=dict(n_steps=32, batch_size=16, n_epochs=3, target_kl=0.01, clip_range=0.2),
        arm="scratch_controlled", action_std_strategy="fixed", action_std_value=0.10,
        max_acceptable_kl=1e-6, stage2=True,  # absurdly strict -> abort on the first update
    )
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status["run_status"] == "ABORT_MAX_KL"
    assert (run_dir / "final_model.zip").exists()  # latest safe checkpoint saved


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
