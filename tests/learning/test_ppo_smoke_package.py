"""Structure test for the compact PPO-smoke package builder (no HoloOcean/torch).

Uses fake run directories without an SB3 model, so the action-statistics step degrades
gracefully and no torch import is triggered.
"""

import json
from pathlib import Path

import pytest

from marine_race_arena.learning import build_ppo_smoke_package as bsp


def _fake_run(run_dir: Path, algorithm: str, bc_init: bool):
    (run_dir / "evaluation").mkdir(parents=True)
    (run_dir / "best_model").mkdir(parents=True)
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "run_config.json").write_text(json.dumps(
        {"algorithm": algorithm, "total_timesteps": 1000, "bc_initialized": bc_init}), encoding="utf-8")
    (run_dir / "evaluation" / "initial_eval.json").write_text(json.dumps(
        {"timesteps": 0, "completion_rate": 1.0 if bc_init else 0.0, "bc_initialized": bc_init,
         "mean_gates": 1.0 if bc_init else 0.0}), encoding="utf-8")
    (run_dir / "evaluation" / "eval.csv").write_text(
        "timesteps,completion_rate,mean_gates,mean_collisions,mean_time_finished\n"
        "0,{c},1.0,0.0,9.9\n1000,{c},1.0,0.0,9.8\n".format(c=1.0 if bc_init else 0.0), encoding="utf-8")
    (run_dir / "best_model" / "best_metrics.json").write_text(json.dumps(
        {"timesteps": 1000, "completion_rate": 1.0 if bc_init else 0.0}), encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps(
        {"adapter_actual": "holoocean", "fallback_used": False, "wall_clock_s": 500.0, "final_num_timesteps": 1000}),
        encoding="utf-8")
    (run_dir / "action_std.json").write_text(json.dumps(
        {"source": "bc_validation_residual" if bc_init else "sb3_default"}), encoding="utf-8")
    (run_dir / "logs" / "progress.csv").write_text(
        "train/approx_kl,train/entropy_loss\n0.004,-1.2\n", encoding="utf-8")
    (run_dir / "reproduce.txt").write_text("# reproduce\n", encoding="utf-8")


def test_build_smoke_package_structure(tmp_path, monkeypatch):
    bcinit = tmp_path / "ppo_bcinit" / "ts1"
    scratch = tmp_path / "ppo_scratch" / "ts1"
    _fake_run(bcinit, "ppo_bcinit", bc_init=True)
    _fake_run(scratch, "ppo_scratch", bc_init=False)
    pub = tmp_path / "public_ppo_smoke"
    monkeypatch.setattr(bsp, "PUB", pub)

    rc = bsp.main(["--bcinit", str(bcinit), "--scratch", str(scratch), "--resume", str(bcinit)])
    assert rc == 0

    # Required files.
    assert (pub / "README.md").exists()
    comp = json.loads((pub / "comparison.json").read_text(encoding="utf-8"))
    assert comp["development_seeds"] == [1200, 1201, 1202, 1203, 1204]
    assert comp["final_evaluation_requires_new_unseen_seeds"] is True
    assert comp["bcinit_1k"]["timestep_zero_completion"] == 1.0
    assert comp["scratch_1k"]["timestep_zero_completion"] == 0.0
    for arm in ("bcinit_1k", "scratch_1k"):
        for f in ("run_config.json", "initial_eval.json", "final_eval.json", "eval.csv",
                  "action_statistics.json", "environment.json", "model_hashes.json", "reproduce.txt"):
            assert (pub / arm / f).exists(), f"missing {arm}/{f}"
    ver = json.loads((pub / "resume_smoke" / "resume_verification.json").read_text(encoding="utf-8"))
    assert ver["no_duplicate_timestep_zero"] is True  # only one timesteps==0 row
    # README states the honest caveats.
    readme = (pub / "README.md").read_text(encoding="utf-8").lower()
    assert "not convergence" in readme and "development seed" in readme


def test_action_statistics_degrades_without_model(tmp_path):
    run_dir = tmp_path / "ppo_bcinit" / "ts1"
    _fake_run(run_dir, "ppo_bcinit", bc_init=True)
    stats = bsp._action_statistics(run_dir, {"source": "bc_validation_residual"})
    # No final_model.zip -> no sampled stats, but the training-log metrics are read.
    assert "sampled_action_saturation_frac" not in stats
    assert stats["training_log_metrics"].get("approx_kl") == 0.004
