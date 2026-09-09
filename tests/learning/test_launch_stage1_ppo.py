"""Tests for the PPO launcher (arms, calibration presets, conditions, seed safety)."""

import argparse
import json
from pathlib import Path

import pytest

from marine_race_arena.learning import launch_stage1_ppo as launch


def _args(**over):
    base = dict(
        arm="bcinit_controlled", condition="fixed", config="kl_safe_v1", steps=500,
        train_seed=None, eval_seeds=None, device="cpu", adapter="holoocean", allow_fallback=False,
        action_std=None, max_kl=None, learning_rate=None, n_steps=None, batch_size=None,
        n_epochs=None, clip_range=None, target_kl=None, checkpoint_freq=None, eval_freq=None,
        min_initial_completion=None, output_root="results/rl",
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_bcinit_controlled_kl_safe_v1_defaults():
    cfg = launch.build_config(_args())
    assert cfg["arm"] == "bcinit_controlled" and cfg["bc_initialized"] is True
    assert cfg["action_std_strategy"] == "fixed" and cfg["action_std_value"] == 0.10
    assert cfg["max_acceptable_kl"] == 0.02
    ppo = cfg["ppo_kwargs"]
    assert ppo["learning_rate"] == 1e-5 and ppo["clip_range"] == 0.05 and ppo["n_epochs"] == 1 and ppo["target_kl"] == 0.01
    assert cfg["eval_seeds"] == [1400, 1401, 1402, 1403, 1404]  # Stage-1 calibration seeds
    assert cfg["train_seed"] == 9000 and cfg["stage2"] is False


def test_scratch_controlled_shares_the_controlled_std():
    cfg = launch.build_config(_args(arm="scratch_controlled"))
    assert cfg["bc_initialized"] is False and cfg["bc_model"] is None
    assert cfg["action_std_strategy"] == "fixed" and cfg["action_std_value"] == 0.10  # SAME as bcinit_controlled


def test_scratch_default_keeps_sb3_default():
    cfg = launch.build_config(_args(arm="scratch_default"))
    assert cfg["bc_initialized"] is False and cfg["action_std_strategy"] == "sb3_default"


def test_arm_aliases():
    assert launch.build_config(_args(arm="bcinit"))["arm"] == "bcinit_controlled"
    assert launch.build_config(_args(arm="scratch"))["arm"] == "scratch_default"


def test_randomized_condition_selects_stage2_seeds_and_label():
    cfg = launch.build_config(_args(condition="randomized", steps=5000))
    assert cfg["stage2"] is True and cfg["stage"] == "stage2"
    assert cfg["algorithm"] == "stage2_randomized_bcinit_controlled"
    assert cfg["eval_seeds"] == list(range(1410, 1420))
    assert cfg["eval_freq"] == 1000 and cfg["checkpoint_freq"] == 1000


def test_action_std_scalar_and_per_axis_override():
    cfg = launch.build_config(_args(action_std="0.075"))
    assert cfg["action_std_strategy"] == "fixed" and cfg["action_std_value"] == 0.075
    cfg2 = launch.build_config(_args(action_std="0.1,0.05,0.05,0.08"))
    assert cfg2["action_std_value"] == [0.1, 0.05, 0.05, 0.08]


def test_config_preset_v2_lowers_learning_rate():
    cfg = launch.build_config(_args(config="kl_safe_v2"))
    assert cfg["ppo_kwargs"]["learning_rate"] == 5e-6
    cfg3 = launch.build_config(_args(config="kl_safe_v3"))
    assert cfg3["action_std_value"] == 0.075


def test_seed_safety_refuses_frozen_or_reserved_seeds(capsys):
    # Frozen Stage-2 seeds 1100-1149 must never be used for checkpoint selection.
    rc = launch.main(["--arm", "bcinit_controlled", "--condition", "randomized", "--steps", "5000",
                      "--eval-seeds", "1100-1104", "--dry-run"])
    assert rc == 1
    assert "held-out" in capsys.readouterr().out
    # Reserved final seeds too.
    rc2 = launch.main(["--arm", "bcinit_controlled", "--condition", "randomized", "--steps", "5000",
                       "--eval-seeds", "1500-1504", "--dry-run"])
    assert rc2 == 1


def test_dry_run_bcinit_verifies_hash(tmp_path, capsys):
    rc = launch.main(["--arm", "bcinit_controlled", "--condition", "randomized", "--steps", "5000",
                      "--dry-run", "--output-root", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "model sha256 OK" in out
    assert not any((tmp_path).glob("**/run_config.json"))


def test_dry_run_scratch_default_needs_no_model(tmp_path, capsys):
    rc = launch.main(["--arm", "scratch_default", "--condition", "randomized", "--steps", "5000",
                      "--dry-run", "--output-root", str(tmp_path)])
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
    run_dir = tmp_path / "stage2" / "stage2_randomized_scratch_default" / "FIXEDTS"
    run_dir.mkdir(parents=True)
    (run_dir / "x.txt").write_text("x", encoding="utf-8")
    rc = launch.main(["--arm", "scratch_default", "--condition", "randomized", "--steps", "5000",
                      "--dry-run", "--output-root", str(tmp_path)])
    assert rc == 1 and "not empty" in capsys.readouterr().out


def test_resume_without_checkpoint_is_rejected(tmp_path, capsys):
    empty = tmp_path / "run"
    empty.mkdir()
    rc = launch.main(["--arm", "bcinit_controlled", "--condition", "randomized", "--steps", "6000",
                      "--dry-run", "--resume-run", str(empty)])
    assert rc == 1 and "no checkpoint" in capsys.readouterr().out.lower()
