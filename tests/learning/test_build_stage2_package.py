"""Structure test for the Stage-2 diagnostic package builder (fake run dirs; no torch)."""

import json
from pathlib import Path

from marine_race_arena.learning import build_stage2_package as bsp


def _fake_arm(run_dir: Path, arm: str, *, completion, extreme, status="COMPLETED", best_ts=0):
    (run_dir / "evaluation").mkdir(parents=True)
    (run_dir / "best_model").mkdir(parents=True)
    (run_dir / "training").mkdir(parents=True)
    (run_dir / "run_config.json").write_text(json.dumps(
        {"arm": arm, "algorithm": f"stage2_randomized_{arm}", "ppo_kwargs": {"learning_rate": 1e-5, "n_epochs": 1, "clip_range": 0.05},
         "bc_action_std_config": {"strategy": "fixed", "value": 0.1}, "action_std": {"std_per_axis": {"surge": 0.1}}}), encoding="utf-8")
    (run_dir / "evaluation" / "initial_eval.json").write_text(json.dumps(
        {"timesteps": 0, "completion_rate": completion, "interior_completion": 1.0, "extreme_completion": extreme,
         "mean_action_saturation": 0.0, "per_seed": [{"seed": 1410}]}), encoding="utf-8")
    (run_dir / "evaluation" / "eval.csv").write_text(
        "timesteps,completion_rate,extreme_completion\n0,{c},{e}\n5000,{c},{e}\n".format(c=completion, e=extreme), encoding="utf-8")
    (run_dir / "best_model" / "best_metrics.json").write_text(json.dumps(
        {"timesteps": best_ts, "completion_rate": completion, "interior_completion": 1.0, "extreme_completion": extreme}), encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps(
        {"adapter_actual": "holoocean", "final_num_timesteps": 5000, "wall_clock_s": 2000.0}), encoding="utf-8")
    (run_dir / "run_status.json").write_text(json.dumps(
        {"run_status": status, "kl_summary": {"max_approx_kl": 0.015, "final_action_saturation": 0.0, "final_policy_std": 0.1}}), encoding="utf-8")
    (run_dir / "action_std.json").write_text(json.dumps({"strategy": "fixed"}), encoding="utf-8")
    (run_dir / "training" / "ppo_update_metrics.csv").write_text("num_timesteps,approx_kl\n5000,0.015\n", encoding="utf-8")
    (run_dir / "reproduce.txt").write_text("# reproduce\n", encoding="utf-8")


def test_build_stage2_package(tmp_path, monkeypatch):
    monkeypatch.setattr(bsp, "PUB", tmp_path / "pub")
    bc = tmp_path / "bc"
    sc = tmp_path / "sc"
    sd = tmp_path / "sd"
    _fake_arm(bc, "bcinit_controlled", completion=1.0, extreme=1.0)
    _fake_arm(sc, "scratch_controlled", completion=0.0, extreme=0.0)
    _fake_arm(sd, "scratch_default", completion=0.0, extreme=0.0)

    rc = bsp.main(["--bcinit", str(bc), "--scratch-controlled", str(sc), "--scratch-default", str(sd)])
    assert rc == 0
    pub = tmp_path / "pub"
    assert (pub / "README.md").exists()
    manifest = json.loads((pub / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert any("not final scientific results" in d for d in manifest["disclaimers"])
    assert (pub / "seed_registry_snapshot.json").exists()
    for arm in ("bcinit_controlled", "scratch_controlled", "scratch_default"):
        for f in ("run_config.json", "initial_eval.json", "best_eval.json", "eval_history.csv",
                  "final_eval.json", "action_statistics.json", "kl_metrics.csv", "model_hashes.json", "reproduce.txt"):
            assert (pub / arm / f).exists(), f"missing {arm}/{f}"
    ive = json.loads((pub / "comparison" / "interior_vs_extreme.json").read_text(encoding="utf-8"))
    assert ive["arms"]["bcinit_controlled"]["best_extreme"] == 1.0
    # per_seed detail is not published in the compact initial_eval
    assert "per_seed" not in json.loads((pub / "bcinit_controlled" / "initial_eval.json").read_text(encoding="utf-8"))


def test_calibration_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(bsp, "PUB", tmp_path / "pub")
    (tmp_path / "pub" / "calibration").mkdir(parents=True)
    cal = tmp_path / "cal"
    _fake_arm(cal, "bcinit_controlled", completion=1.0, extreme=1.0)
    out = bsp.build_calibration_section([cal])
    assert out["selected"] is not None and out["selected"]["passed_calibration"] is True
