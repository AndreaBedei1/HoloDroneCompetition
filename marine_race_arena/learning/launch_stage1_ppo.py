"""User-facing launcher for the Stage-1/Stage-2 PPO experiments (calibration + fair arms).

One command starts a safe, reproducible real-HoloOcean PPO run without hand-writing
Python. It supports the fixed Stage-1 KL calibration and the randomized Stage-2
diagnostic, three fair arms, KL-safe presets, and a hard KL safety stop.

Arms:
  * ``bcinit_controlled``  — BC weights + controlled exploration std (the warm-start).
  * ``scratch_controlled`` — random weights + the SAME controlled std (fair comparison).
  * ``scratch_default``    — random weights + SB3's default std (exploration diagnostic).
  (``bcinit`` / ``scratch`` remain as aliases for the controlled / default arms.)

Safe defaults: real ``holoocean`` adapter, fallback disabled, fresh reset, CPU, the
committed public BC model + report (hash-verified), development seeds from the seed
registry, the ``kl_safe_v1`` KL-safe configuration, and a ``max_acceptable_kl`` hard stop.

Examples (marine_race_rl env):
  python -m marine_race_arena.learning.launch_stage1_ppo --arm bcinit_controlled --condition fixed --steps 500
  python -m marine_race_arena.learning.launch_stage1_ppo --arm bcinit_controlled --condition randomized --steps 5000
  python -m marine_race_arena.learning.launch_stage1_ppo --arm scratch_controlled --condition randomized --steps 5000 --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from marine_race_arena.learning.seed_registry import (
    DO_NOT_TRAIN_ON,
    PPO_TRAINING_ENV_SEED,
    STAGE1_KL_CALIBRATION_SEEDS,
    STAGE2_PPO_DEV_SEEDS,
)

TRACK = "marine_race_arena/tracks/training/stage1_single_gate.json"
PUBLIC_BC_MODEL = "results/rl_public/stage1/bc/model/best_model.pt"
PUBLIC_BC_REPORT = "results/rl_public/stage1/bc/bc_report.json"
PUBLIC_BC_MODEL_HASH = "results/rl_public/stage1/bc/model_hash.json"

# KL-safe configuration presets (Section 2). Every value is overridable on the CLI.
CONFIG_PRESETS: Dict[str, Dict] = {
    "kl_safe_v1": dict(learning_rate=1e-5, n_steps=500, batch_size=100, n_epochs=1, gamma=0.99,
                       gae_lambda=0.95, ent_coef=0.0, target_kl=0.01, clip_range=0.05,
                       action_std=0.10, max_acceptable_kl=0.02),
    "kl_safe_v2": dict(learning_rate=5e-6, n_steps=500, batch_size=100, n_epochs=1, gamma=0.99,
                       gae_lambda=0.95, ent_coef=0.0, target_kl=0.01, clip_range=0.05,
                       action_std=0.10, max_acceptable_kl=0.02),
    "kl_safe_v3": dict(learning_rate=1e-5, n_steps=500, batch_size=100, n_epochs=1, gamma=0.99,
                       gae_lambda=0.95, ent_coef=0.0, target_kl=0.01, clip_range=0.05,
                       action_std=0.075, max_acceptable_kl=0.02),
    # Legacy 1k-smoke config (residual std) -- kept so the earlier 0.05-std result reproduces.
    "smoke": dict(learning_rate=5e-5, n_steps=500, batch_size=100, n_epochs=2, gamma=0.99,
                  gae_lambda=0.95, ent_coef=0.0, target_kl=0.01, clip_range=0.1,
                  action_std=None, max_acceptable_kl=None),
}

ARM_ALIASES = {"bcinit": "bcinit_controlled", "scratch": "scratch_default"}
ARMS = {
    "bcinit_controlled": {"bc_init": True, "exploration": "controlled"},
    "scratch_controlled": {"bc_init": False, "exploration": "controlled"},
    "scratch_default": {"bc_init": False, "exploration": "default"},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _active_branch() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def _dirty_worktree() -> Optional[bool]:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=15)
        dirty = [ln for ln in out.stdout.splitlines() if ln.strip() and not ln.strip().endswith(("/",))
                 and "results/rl/" not in ln]
        return bool(dirty)
    except Exception:
        return None


def _parse_seeds(spec: str) -> List[int]:
    seeds: List[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def _parse_action_std(spec):
    if spec is None:
        return None
    if isinstance(spec, (int, float)):
        return float(spec)
    parts = [p for p in str(spec).split(",") if p != ""]
    return float(parts[0]) if len(parts) == 1 else [float(p) for p in parts]


def _latest_checkpoint_steps(run_dir: Path) -> int:
    from marine_race_arena.learning.train_workflow import _checkpoint_steps, latest_checkpoint

    ckpt = latest_checkpoint(run_dir)
    return _checkpoint_steps(ckpt) if ckpt is not None else 0


def build_config(args) -> Dict:
    arm = ARM_ALIASES.get(args.arm, args.arm)
    if arm not in ARMS:
        raise ValueError(f"unknown arm {args.arm!r}")
    spec = ARMS[arm]
    preset = dict(CONFIG_PRESETS[args.config])
    # CLI overrides.
    for key in ("learning_rate", "n_steps", "batch_size", "n_epochs", "clip_range", "target_kl"):
        val = getattr(args, key, None)
        if val is not None:
            preset[key] = val
    if args.max_kl is not None:
        preset["max_acceptable_kl"] = args.max_kl
    action_std = _parse_action_std(args.action_std) if args.action_std is not None else preset.get("action_std")

    randomized = args.condition == "randomized"
    steps = int(args.steps)
    ppo_kwargs = dict(learning_rate=preset["learning_rate"], n_steps=min(int(preset["n_steps"]), steps),
                      batch_size=int(preset["batch_size"]), n_epochs=int(preset["n_epochs"]),
                      gamma=preset["gamma"], gae_lambda=preset["gae_lambda"], ent_coef=preset["ent_coef"],
                      target_kl=preset["target_kl"], clip_range=preset["clip_range"], device=args.device)

    # Resolve the exploration-std strategy for this arm.
    if spec["exploration"] == "default":
        std_strategy, std_value = "sb3_default", None
    elif action_std is not None:
        std_strategy, std_value = "fixed", action_std
    else:  # bcinit_controlled with no explicit std -> residual-derived
        std_strategy, std_value = "residual", None

    eval_seeds = (_parse_seeds(args.eval_seeds) if args.eval_seeds
                  else (list(STAGE2_PPO_DEV_SEEDS) if randomized else list(STAGE1_KL_CALIBRATION_SEEDS)))

    return {
        "arm": arm,
        "condition": args.condition,
        "stage": "stage2" if randomized else "stage1",
        "stage2": randomized,
        "algorithm": f"{'stage2_randomized_' if randomized else ''}{arm}",
        "bc_initialized": spec["bc_init"],
        "config_preset": args.config,
        "track": TRACK,
        "steps": steps,
        "train_seed": int(args.train_seed) if args.train_seed is not None else PPO_TRAINING_ENV_SEED,
        "eval_seeds": eval_seeds,
        "adapter": args.adapter,
        "allow_fallback": bool(args.allow_fallback),
        "device": args.device,
        "bc_model": PUBLIC_BC_MODEL if spec["bc_init"] else None,
        "bc_report": PUBLIC_BC_REPORT if spec["bc_init"] else None,
        "action_std_strategy": std_strategy,
        "action_std_value": std_value,
        "ppo_kwargs": ppo_kwargs,
        "max_acceptable_kl": preset.get("max_acceptable_kl"),
        # Controlled arms should keep exploration tight; scratch_default legitimately uses SB3's
        # wide default std, so no saturation cap is applied to it.
        "max_action_saturation": (0.10 if spec["exploration"] == "controlled" else None),
        "checkpoint_freq": int(args.checkpoint_freq) if args.checkpoint_freq else (1000 if randomized else max(1, steps // 2)),
        "eval_freq": int(args.eval_freq) if args.eval_freq else (1000 if randomized else steps),
        "min_initial_completion": args.min_initial_completion,
        "output_root": args.output_root,
    }


def _check_seed_safety(cfg: Dict) -> Optional[str]:
    forbidden = set()
    for seeds in DO_NOT_TRAIN_ON.values():
        forbidden.update(seeds)
    used = set(cfg["eval_seeds"]) | {cfg["train_seed"]}
    clash = sorted(used & forbidden)
    return f"seeds {clash} are held-out / frozen and must not be used for training or checkpoint selection" if clash else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", default="bcinit_controlled",
                        choices=list(ARMS) + list(ARM_ALIASES))
    parser.add_argument("--condition", choices=["fixed", "randomized"], default="fixed")
    parser.add_argument("--config", choices=list(CONFIG_PRESETS), default="kl_safe_v1")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--eval-seeds", default=None)
    parser.add_argument("--output-root", default="results/rl")
    parser.add_argument("--resume-run", default=None)
    parser.add_argument("--adapter", default="holoocean")
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action-std", default=None, help="Override exploration std (scalar, or comma per-axis).")
    parser.add_argument("--max-kl", type=float, default=None, help="Hard KL safety stop (overrides the preset).")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--clip-range", type=float, default=None)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--min-initial-completion", type=float, default=None)
    args = parser.parse_args(argv)

    cfg = build_config(args)

    branch = _active_branch()
    if branch and branch != "feature/rl-controller":
        print(f"[launch] WARNING: active git branch is '{branch}', expected 'feature/rl-controller'.")
    if not Path(TRACK).exists():
        print(f"[launch] ERROR: track not found: {TRACK}")
        return 1
    seed_issue = _check_seed_safety(cfg)
    if seed_issue:
        print(f"[launch] ERROR: {seed_issue}")
        return 1
    print(f"[launch] track sha256: {_sha256(Path(TRACK))}")
    if cfg["bc_initialized"]:
        model = Path(cfg["bc_model"])
        expected = json.loads(Path(PUBLIC_BC_MODEL_HASH).read_text(encoding="utf-8")).get("sha256")
        if not model.exists():
            print(f"[launch] ERROR: BC model not found: {model}")
            return 1
        actual = _sha256(model)
        if expected and actual != expected:
            print(f"[launch] ERROR: BC model hash mismatch: {actual} != {expected}")
            return 1
        print(f"[launch] BC model sha256 OK: {actual}")

    resuming = bool(args.resume_run)
    if resuming:
        run_dir = Path(args.resume_run)
        if not run_dir.exists() or _latest_checkpoint_steps(run_dir) == 0:
            print(f"[launch] ERROR: --resume-run {run_dir} has no checkpoint to resume from.")
            return 1
        prior = _latest_checkpoint_steps(run_dir)
        if args.eval_freq is None:
            cfg["eval_freq"] = max(1, cfg["steps"] - prior)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(cfg["output_root"]) / cfg["stage"] / cfg["algorithm"] / ts
        if run_dir.exists() and any(run_dir.iterdir()):
            print(f"[launch] ERROR: output directory {run_dir} is not empty; refusing to overwrite.")
            return 1

    cfg["run_dir"] = str(run_dir)
    cfg["resuming"] = resuming
    cfg["dirty_worktree"] = _dirty_worktree()
    cfg["total_timesteps"] = 0 if args.eval_only else cfg["steps"]

    print("[launch] effective configuration:")
    print(json.dumps(cfg, indent=2))
    print(f"[launch] output directory: {run_dir}")

    if args.dry_run:
        print("[launch] DRY RUN: all checks passed; not launching HoloOcean.")
        return 0

    from marine_race_arena.learning.curriculum import STAGE2_RANDOMIZATION
    from marine_race_arena.learning.train_workflow import run_ppo_training

    env_kwargs = dict(adapter=cfg["adapter"], allow_fallback=cfg["allow_fallback"], max_steps=250, dt=0.1)
    if cfg["stage2"]:
        env_kwargs["start_randomization"] = STAGE2_RANDOMIZATION
    try:
        run_path, model = run_ppo_training(
            TRACK, stage=cfg["stage"], algorithm=cfg["algorithm"],
            total_timesteps=cfg["total_timesteps"], train_seed=cfg["train_seed"],
            eval_seeds=cfg["eval_seeds"], run_dir=str(run_dir), resume=resuming,
            bc_model_path=cfg["bc_model"], bc_report_path=cfg["bc_report"],
            arm=cfg["arm"], action_std_strategy=cfg["action_std_strategy"], action_std_value=cfg["action_std_value"],
            max_acceptable_kl=cfg["max_acceptable_kl"], max_action_saturation=cfg["max_action_saturation"],
            stage2=cfg["stage2"], env_kwargs=env_kwargs, hidden_sizes=(256, 256),
            checkpoint_freq=cfg["checkpoint_freq"], eval_freq=cfg["eval_freq"],
            ppo_kwargs=cfg["ppo_kwargs"], initial_eval=True, min_initial_completion=cfg["min_initial_completion"],
        )
    except Exception as exc:  # pragma: no cover
        print(f"[launch] FAILED: {type(exc).__name__}: {exc}")
        return 1

    status_file = run_path / "run_status.json"
    status = json.loads(status_file.read_text(encoding="utf-8")) if status_file.exists() else {}
    print(f"[launch] DONE. run_status={status.get('run_status')} results in {run_path}")
    return 0 if status.get("run_status") in (None, "COMPLETED") else 3


if __name__ == "__main__":
    raise SystemExit(main())
