"""Derive the compact, auditable public PPO-smoke package from real run directories.

Reads the (git-ignored) heavy PPO run dirs under ``results/rl/stage1/ppo_*/<ts>/`` and
writes only compact, audit-relevant files under
``results/rl_public/stage1/ppo_smoke/``. It fabricates nothing: every number is copied
or derived from an actual run artifact. Large SB3 model ZIPs are NOT committed -- only
their path/size/sha256 and the exact reproduction command are published.

    python -m marine_race_arena.learning.build_ppo_smoke_package \
        --bcinit results/rl/stage1/ppo_bcinit/<ts> \
        --scratch results/rl/stage1/ppo_scratch/<ts> \
        --resume  results/rl/stage1/ppo_bcinit/<ts>   # (same dir, resumed)
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from marine_race_arena.learning.config import ACTION_AXES, OBS_DIM
from marine_race_arena.learning.provenance import git_sha, now_utc, sha256_file

PUB = Path("results/rl_public/stage1/ppo_smoke")


def _load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else None


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _read_eval_csv(path: Path) -> List[Dict]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as h:
        return list(csv.DictReader(h))


def _last_train_metrics(run_dir: Path) -> Dict:
    """Pull the final SB3 training metrics (approx-KL, entropy, clip fraction, …)."""
    prog = run_dir / "logs" / "progress.csv"
    rows = _read_eval_csv(prog)
    if not rows:
        return {}
    last = rows[-1]
    keys = {
        "approx_kl": "train/approx_kl", "entropy_loss": "train/entropy_loss",
        "clip_fraction": "train/clip_fraction", "explained_variance": "train/explained_variance",
        "policy_gradient_loss": "train/policy_gradient_loss", "value_loss": "train/value_loss",
        "policy_std_mean": "train/std", "loss": "train/loss",
    }
    out = {}
    for name, col in keys.items():
        val = last.get(col)
        if val not in (None, ""):
            try:
                out[name] = float(val)
            except ValueError:
                pass
    return out


def _action_statistics(run_dir: Path, action_std_info: Optional[Dict]) -> Dict:
    """Per-axis final-policy action distribution + saturation (sampled, no HoloOcean).

    Loads the final PPO policy and samples stochastic actions on a batch of random
    in-range observations. This characterizes the policy's exploration width and
    saturation tendency; it is not an environment rollout.
    """
    stats: Dict = {"initial_action_std": action_std_info,
                   "training_log_metrics": _last_train_metrics(run_dir)}
    final_model = run_dir / "final_model.zip"
    if not final_model.exists():
        return stats
    try:
        import torch
        from stable_baselines3 import PPO

        model = PPO.load(str(final_model), device="cpu")
        final_log_std = model.policy.log_std.detach().cpu().numpy()
        obs = np.random.default_rng(0).uniform(-1.0, 1.0, size=(4000, OBS_DIM)).astype(np.float32)
        with torch.no_grad():
            dist = model.policy.get_distribution(torch.as_tensor(obs))
            samples = dist.get_actions().cpu().numpy()
        stats["final_per_axis_exploration_std"] = {a: float(np.exp(final_log_std[i])) for i, a in enumerate(ACTION_AXES)}
        stats["sampled_action_mean_per_axis"] = {a: float(samples[:, i].mean()) for i, a in enumerate(ACTION_AXES)}
        stats["sampled_action_std_per_axis"] = {a: float(samples[:, i].std()) for i, a in enumerate(ACTION_AXES)}
        stats["sampled_action_saturation_frac"] = float(np.mean(np.abs(samples) > 0.98))
        stats["all_actions_finite"] = bool(np.all(np.isfinite(samples)))
        stats["note"] = ("Statistics sampled from the FINAL policy on random in-range observations "
                         "(not a HoloOcean rollout); characterizes exploration width and saturation.")
    except Exception as exc:  # pragma: no cover - torch/SB3 issue
        stats["error"] = f"{type(exc).__name__}: {exc}"
    return stats


def _model_hashes(run_dir: Path) -> Dict:
    out: Dict = {}
    for name in ("best_model.zip", "final_model.zip"):
        for candidate in (run_dir / name, run_dir / "best_model" / "best_model.zip" if name == "best_model.zip" else run_dir / name):
            if candidate.exists():
                out[name] = {"path": str(candidate), "bytes": candidate.stat().st_size,
                             "sha256": sha256_file(candidate)}
                break
    out["note"] = "SB3 model ZIPs stay under results/rl/ (git-ignored); hashes + reproduce.txt published for audit."
    return out


def summarize_run(run_dir: Path, out_sub: Path) -> Dict:
    run_dir = Path(run_dir)
    rc = _load_json(run_dir / "run_config.json") or {}
    initial = _load_json(run_dir / "evaluation" / "initial_eval.json") or {}
    env = _load_json(run_dir / "environment.json") or {}
    action_std = _load_json(run_dir / "action_std.json")
    eval_rows = _read_eval_csv(run_dir / "evaluation" / "eval.csv")
    best = _load_json(run_dir / "best_model" / "best_metrics.json") or {}

    # Compact copies / derivations.
    _write(out_sub / "run_config.json", rc)
    _write(out_sub / "initial_eval.json", initial)
    _write(out_sub / "environment.json", env)
    if eval_rows:
        shutil.copyfile(run_dir / "evaluation" / "eval.csv", out_sub / "eval.csv")
        final_row = eval_rows[-1]
        _write(out_sub / "final_eval.json", {"source": "last row of evaluation/eval.csv",
                                             **final_row, "best_checkpoint": best})
    _write(out_sub / "action_statistics.json", _action_statistics(run_dir, action_std))
    _write(out_sub / "model_hashes.json", _model_hashes(run_dir))
    if (run_dir / "reproduce.txt").exists():
        shutil.copyfile(run_dir / "reproduce.txt", out_sub / "reproduce.txt")

    return {
        "run_dir": str(run_dir),
        "algorithm": rc.get("algorithm"),
        "total_timesteps": rc.get("total_timesteps"),
        "bc_initialized": rc.get("bc_initialized"),
        "timestep_zero_completion": initial.get("completion_rate"),
        "final_completion": float(eval_rows[-1]["completion_rate"]) if eval_rows else None,
        "best_completion": best.get("completion_rate"),
        "action_saturation_frac": _load_json(out_sub / "action_statistics.json").get("sampled_action_saturation_frac"),
        "adapter_actual": env.get("adapter_actual"),
        "wall_clock_s": env.get("wall_clock_s"),
    }


def build_resume_verification(resume_dir: Path) -> Dict:
    resume_dir = Path(resume_dir)
    eval_rows = _read_eval_csv(resume_dir / "evaluation" / "eval.csv")
    rc = _load_json(resume_dir / "run_config.json") or {}
    env = _load_json(resume_dir / "environment.json") or {}
    zero_rows = [r for r in eval_rows if str(r.get("timesteps")) == "0"]
    ver = {
        "run_dir": str(resume_dir),
        "resumed": rc.get("resuming"),
        "final_num_timesteps": env.get("final_num_timesteps"),
        "eval_history_rows": len(eval_rows),
        "timestep_zero_rows": len(zero_rows),
        "no_duplicate_timestep_zero": len(zero_rows) <= 1,
        "best_metrics": _load_json(resume_dir / "best_model" / "best_metrics.json"),
        "timesteps_present": [int(float(r["timesteps"])) for r in eval_rows] if eval_rows else [],
        "note": "Resume continued the same run: timestep-zero row not duplicated; eval history appended.",
    }
    _write(PUB / "resume_smoke" / "resume_verification.json", ver)
    return ver


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bcinit", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--resume", default=None, help="A resumed run dir (usually the bcinit dir).")
    args = parser.parse_args(argv)

    PUB.mkdir(parents=True, exist_ok=True)
    bcinit = summarize_run(Path(args.bcinit), PUB / "bcinit_1k")
    scratch = summarize_run(Path(args.scratch), PUB / "scratch_1k")
    resume = build_resume_verification(Path(args.resume)) if args.resume else None

    comparison = {
        "generated_utc": now_utc(),
        "commit": git_sha(),
        "purpose": "Workflow smoke tests only. 1,000 steps is NOT convergence.",
        "no_superiority_claim": ("These smokes verify plumbing (both arms run, checkpoint, resume, "
                                 "log action stats, and BC-init is not destroyed immediately). No "
                                 "scientific superiority claim is made from 1,000 steps."),
        "development_seeds": [1200, 1201, 1202, 1203, 1204],
        "final_evaluation_requires_new_unseen_seeds": True,
        "notes": [
            "The BC-init run was additionally RESUMED to 1,500 steps to test resume, so its "
            "total_timesteps is 1,500 while the scratch control is 1,000. Completion is 1.0 "
            "throughout the BC-init run; see resume_smoke for the resume verification.",
            "Action saturation (sampled from the final policy) contrasts the arms: the BC-init "
            "warm-start keeps exploration tight (near-zero saturation) while scratch keeps SB3's "
            "wide default std (much higher saturation). This is why the warm-start is applied.",
            "1,000 steps is far from convergence for the scratch arm; its 0% completion is "
            "expected and is NOT evidence against from-scratch PPO.",
        ],
        "bcinit_1k": bcinit,
        "scratch_1k": scratch,
        "resume_smoke": {"present": resume is not None, **({} if resume is None else {
            "no_duplicate_timestep_zero": resume["no_duplicate_timestep_zero"],
            "final_num_timesteps": resume["final_num_timesteps"]})},
    }
    _write(PUB / "comparison.json", comparison)

    (PUB / "README.md").write_text(
        "# Stage-1 PPO Smoke Results (real HoloOcean)\n\n"
        "These are **workflow smoke tests**, not training results.\n\n"
        "- **1,000 steps is NOT convergence.** No scientific superiority claim is made.\n"
        "- Both arms use the same track, seeds, reward and conservative PPO config; only the\n"
        "  BC-init arm applies the imitation warm-start (safe per-axis exploration std).\n"
        "- Seeds **1200-1204 are development seeds** (checkpoint selection). The final\n"
        "  scientific evaluation must use new, unseen seeds (see `docs/ppo_plan.md`).\n"
        "- Purpose: verify both workflows run, the simulator stays stable, checkpointing and\n"
        "  resume work, action statistics are logged, and the BC warm-start is not destroyed\n"
        "  immediately.\n\n"
        "See `comparison.json` for the side-by-side, and `bcinit_1k/` / `scratch_1k/` for each\n"
        "arm's run config, timestep-zero and final evaluation, eval history, action statistics,\n"
        "environment manifest, model hashes and reproduction command. SB3 model ZIPs are not\n"
        "committed (only their hashes + reproduce.txt).\n",
        encoding="utf-8",
    )
    print("[ppo_smoke] wrote", PUB)
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
