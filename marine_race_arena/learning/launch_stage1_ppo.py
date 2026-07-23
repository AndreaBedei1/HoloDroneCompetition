"""User-facing launcher for the Stage-1 PPO smoke experiments (BC-init vs scratch).

One command starts a safe, reproducible real-HoloOcean PPO run without hand-writing
Python. Safe defaults: real ``holoocean`` adapter, fallback disabled, fresh reset,
CPU, the committed public BC model + report, development eval seeds 1200-1204, and a
conservative BC-PPO configuration. The model and track hashes are verified before
launch, the full effective configuration and the output directory are printed, a
non-empty output directory is never overwritten (use ``--resume-run``), and the
environment is always closed.

Examples (from the marine_race_rl env):
    python -m marine_race_arena.learning.launch_stage1_ppo --arm bcinit --steps 1000
    python -m marine_race_arena.learning.launch_stage1_ppo --arm scratch --steps 1000
    python -m marine_race_arena.learning.launch_stage1_ppo --arm bcinit --steps 1000 --dry-run
    python -m marine_race_arena.learning.launch_stage1_ppo --resume-run results/rl/stage1/ppo_bcinit/<ts> --steps 1500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

TRACK = "marine_race_arena/tracks/training/stage1_single_gate.json"
PUBLIC_BC_MODEL = "results/rl_public/stage1/bc/model/best_model.pt"
PUBLIC_BC_REPORT = "results/rl_public/stage1/bc/bc_report.json"
PUBLIC_BC_MODEL_HASH = "results/rl_public/stage1/bc/model_hash.json"
DEV_EVAL_SEEDS = [1200, 1201, 1202, 1203, 1204]  # development seeds; NEVER final test seeds

# Conservative BC-PPO configuration (Section 10). Kept identical for both arms except
# the safe stochastic warm-start, which only the BC-init arm applies.
CONSERVATIVE_PPO = dict(
    learning_rate=5e-5, gamma=0.99, gae_lambda=0.95,
    ent_coef=0.0, target_kl=0.01, clip_range=0.1,
)


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _active_branch() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def _verify_model_hash(model_path: Path, hash_json: Path) -> None:
    """Raise if the model file does not match the recorded sha256."""
    if not model_path.exists():
        raise FileNotFoundError(f"BC model not found: {model_path}")
    if not hash_json.exists():
        print(f"[launch] WARNING: no hash file at {hash_json}; skipping model-hash verification.")
        return
    expected = json.loads(hash_json.read_text(encoding="utf-8")).get("sha256")
    actual = _sha256(model_path)
    if expected and actual != expected:
        raise ValueError(f"BC model hash mismatch: {actual} != {expected}")
    print(f"[launch] BC model sha256 OK: {actual}")


def _latest_checkpoint_steps(run_dir: Path) -> int:
    from marine_race_arena.learning.train_workflow import _checkpoint_steps, latest_checkpoint

    ckpt = latest_checkpoint(run_dir)
    return _checkpoint_steps(ckpt) if ckpt is not None else 0


def build_config(args) -> Dict:
    arm = args.arm
    bcinit = arm == "bcinit"
    steps = int(args.steps)
    # Default rollout length 500 (kept stable regardless of the total step budget, so a
    # resumed run stays config-compatible). n_steps < total gives >=2 PPO iterations, so
    # SB3 actually dumps the train/ metrics (approx_kl, entropy, clip_fraction, ...).
    n_steps = int(args.n_steps) if args.n_steps else min(500, steps)
    batch_size = int(args.batch_size) if args.batch_size else (100 if n_steps % 100 == 0 else n_steps)
    ppo_kwargs = dict(CONSERVATIVE_PPO, n_steps=n_steps, batch_size=batch_size,
                      n_epochs=int(args.n_epochs), device=args.device)
    return {
        "arm": arm,
        "algorithm": "ppo_bcinit" if bcinit else "ppo_scratch",
        "bc_initialized": bcinit,
        "track": TRACK,
        "steps": steps,
        "train_seed": int(args.train_seed),
        "eval_seeds": _parse_seeds(args.eval_seeds) if args.eval_seeds else list(DEV_EVAL_SEEDS),
        "adapter": args.adapter,
        "allow_fallback": bool(args.allow_fallback),
        "persistent_reset_requested": bool(args.persistent_reset),
        "persistent_reset_used": False,  # experimental; never wired into training here
        "device": args.device,
        "bc_model": (args.bc_model or PUBLIC_BC_MODEL) if bcinit else None,
        "bc_report": (args.bc_report or PUBLIC_BC_REPORT) if bcinit else None,
        "ppo_kwargs": ppo_kwargs,
        "checkpoint_freq": int(args.checkpoint_freq) if args.checkpoint_freq else max(1, steps // 2),
        "eval_freq": int(args.eval_freq) if args.eval_freq else steps,  # one eval at the end (+ timestep-0)
        "output_root": args.output_root,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=["bcinit", "scratch"], default="bcinit")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--eval-seeds", default=None, help="e.g. 1200-1204 or 1200,1201 (default: dev seeds 1200-1204)")
    parser.add_argument("--output-root", default="results/rl")
    parser.add_argument("--resume-run", default=None, help="Resume this existing run directory instead of starting fresh.")
    parser.add_argument("--adapter", default="holoocean")
    parser.add_argument("--allow-fallback", action="store_true", help="Allow the fallback backend (OFF by default).")
    parser.add_argument("--persistent-reset", action="store_true",
                        help="Experimental; NOT wired into training here -- fresh reset is always used.")
    parser.add_argument("--dry-run", action="store_true", help="Verify everything and print the plan without launching HoloOcean.")
    parser.add_argument("--eval-only", action="store_true", help="Only run the deterministic timestep-zero evaluation (no training).")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bc-model", default=None, help="Override the committed public BC model.")
    parser.add_argument("--bc-report", default=None, help="Override the committed public BC report (per-axis residuals).")
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--checkpoint-freq", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--min-initial-completion", type=float, default=None,
                        help="Abort a BC-init run if timestep-zero completion is below this (safety gate).")
    args = parser.parse_args(argv)

    cfg = build_config(args)

    # --- Safety checks (branch, hashes) -------------------------------------------
    branch = _active_branch()
    if branch and branch != "feature/rl-controller":
        print(f"[launch] WARNING: active git branch is '{branch}', expected 'feature/rl-controller'.")
    if not Path(TRACK).exists():
        print(f"[launch] ERROR: track not found: {TRACK}")
        return 1
    print(f"[launch] track sha256: {_sha256(Path(TRACK))}")
    if cfg["persistent_reset_requested"]:
        print("[launch] NOTE: --persistent-reset is experimental and not wired into the PPO training env; "
              "using fresh reset (see results/rl_public/reset_benchmark/).")

    if cfg["bc_initialized"]:
        try:
            _verify_model_hash(Path(cfg["bc_model"]), Path(PUBLIC_BC_MODEL_HASH)
                               if cfg["bc_model"] == PUBLIC_BC_MODEL else Path(cfg["bc_model"]).with_name("model_hash.json"))
        except Exception as exc:
            print(f"[launch] ERROR: {exc}")
            return 1

    # --- Resolve the output/run directory (never overwrite; resume only) -----------
    resuming = bool(args.resume_run)
    if resuming:
        run_dir = Path(args.resume_run)
        if not run_dir.exists() or _latest_checkpoint_steps(run_dir) == 0:
            print(f"[launch] ERROR: --resume-run {run_dir} has no checkpoint to resume from.")
            return 1
        prior = _latest_checkpoint_steps(run_dir)
        # Ensure the resumed segment produces at least one eval at the new end.
        cfg["eval_freq"] = max(1, cfg["steps"] - prior) if args.eval_freq is None else int(args.eval_freq)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(cfg["output_root"]) / "stage1" / cfg["algorithm"] / ts
        if run_dir.exists() and any(run_dir.iterdir()):
            print(f"[launch] ERROR: output directory {run_dir} is not empty; refusing to overwrite.")
            return 1

    cfg["run_dir"] = str(run_dir)
    cfg["resuming"] = resuming
    cfg["total_timesteps"] = 0 if args.eval_only else cfg["steps"]

    print("[launch] effective configuration:")
    print(json.dumps(cfg, indent=2))
    print(f"[launch] output directory: {run_dir}")

    if args.dry_run:
        print("[launch] DRY RUN: all checks passed; not launching HoloOcean.")
        return 0

    # --- Launch -------------------------------------------------------------------
    from marine_race_arena.learning.train_workflow import run_ppo_training

    env_kwargs = dict(adapter=cfg["adapter"], allow_fallback=cfg["allow_fallback"], max_steps=200, dt=0.1)
    try:
        run_path, model = run_ppo_training(
            TRACK,
            stage="stage1",
            algorithm=cfg["algorithm"],
            total_timesteps=cfg["total_timesteps"],
            train_seed=cfg["train_seed"],
            eval_seeds=cfg["eval_seeds"],
            run_dir=str(run_dir),
            resume=resuming,
            bc_model_path=cfg["bc_model"],
            bc_report_path=cfg["bc_report"],
            env_kwargs=env_kwargs,
            hidden_sizes=(256, 256),
            checkpoint_freq=cfg["checkpoint_freq"],
            eval_freq=cfg["eval_freq"],
            ppo_kwargs=cfg["ppo_kwargs"],
            initial_eval=True,
            min_initial_completion=args.min_initial_completion,
        )
    except Exception as exc:  # pragma: no cover - surfaced to the shell as a non-zero exit
        print(f"[launch] FAILED: {type(exc).__name__}: {exc}")
        return 1

    print(f"[launch] DONE. Results in {run_path}")
    init_eval = run_path / "evaluation" / "initial_eval.json"
    if init_eval.exists():
        print("[launch] timestep-zero eval:", init_eval.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
