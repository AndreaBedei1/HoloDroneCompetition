"""Tests for the Stage-1 PPO launcher (dry-run + output-dir safety; no HoloOcean)."""

import argparse
from pathlib import Path

import pytest

from marine_race_arena.learning import launch_stage1_ppo as launch


def _args(**over):
    base = dict(
        arm="bcinit", steps=1000, n_steps=None, batch_size=None, n_epochs=2, device="cpu",
        train_seed=0, eval_seeds=None, adapter="holoocean", allow_fallback=False,
        persistent_reset=False, bc_model=None, bc_report=None,
        checkpoint_freq=None, eval_freq=None, output_root="results/rl",
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_build_config_uses_conservative_defaults_and_dev_seeds():
    cfg = launch.build_config(_args())
    assert cfg["eval_seeds"] == [1200, 1201, 1202, 1203, 1204]
    ppo = cfg["ppo_kwargs"]
    assert ppo["learning_rate"] == 5e-5 and ppo["target_kl"] == 0.01 and ppo["clip_range"] == 0.1
    assert ppo["n_steps"] == 500 and ppo["batch_size"] == 100 and ppo["n_epochs"] == 2
    assert cfg["algorithm"] == "ppo_bcinit" and cfg["bc_initialized"] is True
    assert cfg["persistent_reset_used"] is False


def test_scratch_arm_needs_no_bc_model():
    cfg = launch.build_config(_args(arm="scratch"))
    assert cfg["algorithm"] == "ppo_scratch" and cfg["bc_model"] is None and cfg["bc_report"] is None


def test_dry_run_bcinit_verifies_hash_and_does_not_launch(tmp_path, capsys):
    rc = launch.main(["--arm", "bcinit", "--steps", "1000", "--dry-run",
                      "--output-root", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "model sha256 OK" in out
    # No run directory is created on a dry run.
    assert not any((tmp_path / "stage1").glob("**/run_config.json"))


def test_dry_run_scratch_passes_without_model(tmp_path, capsys):
    rc = launch.main(["--arm", "scratch", "--steps", "1000", "--dry-run", "--output-root", str(tmp_path)])
    assert rc == 0
    assert "DRY RUN" in capsys.readouterr().out


def test_refuses_nonempty_output_directory(tmp_path, monkeypatch, capsys):
    class _FixedDT:
        @staticmethod
        def now():
            class _N:
                def strftime(self, fmt):
                    return "FIXEDTS"
            return _N()

    monkeypatch.setattr(launch, "datetime", _FixedDT)
    run_dir = tmp_path / "stage1" / "ppo_scratch" / "FIXEDTS"
    run_dir.mkdir(parents=True)
    (run_dir / "something.txt").write_text("x", encoding="utf-8")
    rc = launch.main(["--arm", "scratch", "--steps", "1000", "--dry-run", "--output-root", str(tmp_path)])
    assert rc == 1
    assert "not empty" in capsys.readouterr().out


def test_resume_without_checkpoint_is_rejected(tmp_path, capsys):
    empty = tmp_path / "run"
    empty.mkdir()
    rc = launch.main(["--arm", "bcinit", "--steps", "1500", "--dry-run", "--resume-run", str(empty)])
    assert rc == 1
    assert "no checkpoint" in capsys.readouterr().out.lower()


def test_bcinit_hash_mismatch_is_rejected(tmp_path, capsys):
    # A fake model + mismatching hash file -> refuse to launch.
    model = tmp_path / "best_model.pt"
    model.write_bytes(b"not the real model")
    (tmp_path / "model_hash.json").write_text('{"sha256": "deadbeef"}', encoding="utf-8")
    rc = launch.main(["--arm", "bcinit", "--steps", "1000", "--dry-run",
                      "--bc-model", str(model), "--output-root", str(tmp_path)])
    assert rc == 1
    assert "mismatch" in capsys.readouterr().out.lower()
