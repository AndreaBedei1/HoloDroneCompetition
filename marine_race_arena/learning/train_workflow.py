"""Resumable PPO training workflow with full, reproducible run metadata.

Wraps the PPO builder / BC->PPO transfer (:mod:`rl_train`) with the machinery a
long HoloOcean run needs: timestamped run directories that are never overwritten,
periodic checkpoints, automatic resume from the latest checkpoint, a held-out
evaluation callback that saves the best model by *completion rate* (not training
reward), CSV logging, and a complete provenance snapshot (config, seeds, reward
config, track copy + hash, git SHA, package/HoloOcean versions, adapter/fallback
status, observation-encoding version, reproduction command). The environment is
always closed in a ``finally`` block.

Stable-Baselines3, Gymnasium and PyTorch are RL-only dependencies.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from marine_race_arena.learning.config import ACTION_DIM, OBS_DIM, OBS_ENCODING_VERSION
from marine_race_arena.learning.gym_env import MarineRaceGymEnv
from marine_race_arena.learning.reward import RewardConfig, TrainingReward
from marine_race_arena.learning.rl_train import build_ppo, transfer_bc_to_ppo


# --------------------------------------------------------------------------- utils
def _git_sha() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None
    except Exception:  # pragma: no cover - git absent
        return None


def _package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in ("numpy", "torch", "gymnasium", "stable_baselines3", "holoocean"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = None
    return versions


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _checkpoint_steps(path: Path) -> int:
    for token in path.stem.split("_"):
        if token.isdigit():
            return int(token)
    return 0


def latest_checkpoint(run_dir) -> Optional[Path]:
    """The highest-step PPO checkpoint in ``run_dir/checkpoints`` (or None)."""
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return None
    zips = sorted(ckpt_dir.glob("ppo_*_steps.zip"), key=_checkpoint_steps)
    return zips[-1] if zips else None


# --------------------------------------------------------------- eval callback
def _make_completion_eval_callback():
    from stable_baselines3.common.callbacks import BaseCallback

    class CompletionEvalCallback(BaseCallback):
        """Evaluate held-out completion rate; save the best model on improvement."""

        def __init__(self, track, eval_seeds, eval_freq, best_dir, eval_csv, env_kwargs, reward_config, verbose=0):
            super().__init__(verbose)
            self.track = track
            self.eval_seeds = list(eval_seeds)
            self.eval_freq = int(eval_freq)
            self.best_dir = Path(best_dir)
            self.eval_csv = Path(eval_csv)
            self.env_kwargs = dict(env_kwargs or {})
            self.reward_config = reward_config
            self.best_completion = -1.0
            self._rows: List[Dict[str, float]] = []

        def _init_callback(self):
            self.best_dir.mkdir(parents=True, exist_ok=True)
            self.eval_csv.parent.mkdir(parents=True, exist_ok=True)

        def _evaluate(self) -> Tuple[float, float, float]:
            completions, gates, collisions = 0, [], []
            for seed in self.eval_seeds:
                env = MarineRaceGymEnv(
                    self.track, seed=int(seed), reward_fn=TrainingReward(self.reward_config), **self.env_kwargs
                )
                try:
                    obs, _ = env.reset(seed=int(seed))
                    done = False
                    while not done:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, _, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
                    progress = env.episode.referee_progress()
                    state = env.episode.context.referee.states[env.episode.participant_id]
                    if progress["status"] == "FINISHED":
                        completions += 1
                    gates.append(progress["valid_gate_crossings"])
                    collisions.append(int(state.collision_events))
                finally:
                    env.close()
            n = max(1, len(self.eval_seeds))
            return (
                completions / n,
                float(np.mean(gates)) if gates else 0.0,
                float(np.mean(collisions)) if collisions else 0.0,
            )

        def _on_step(self) -> bool:
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                rate, mean_gates, mean_collisions = self._evaluate()
                row = {
                    "timesteps": int(self.num_timesteps),
                    "completion_rate": rate,
                    "mean_gates": mean_gates,
                    "mean_collisions": mean_collisions,
                }
                self._rows.append(row)
                self._write_csv()
                self.logger.record("eval/completion_rate", rate)
                self.logger.record("eval/mean_gates", mean_gates)
                if rate > self.best_completion:
                    self.best_completion = rate
                    self.model.save(str(self.best_dir / "best_model"))
                    (self.best_dir / "best_completion.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            return True

        def _write_csv(self):
            if not self._rows:
                return
            with self.eval_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self._rows[0].keys()))
                writer.writeheader()
                writer.writerows(self._rows)

    return CompletionEvalCallback


# --------------------------------------------------------------- main workflow
def run_ppo_training(
    track: str,
    *,
    stage: str = "stageX",
    algorithm: str = "ppo",
    total_timesteps: int = 50000,
    train_seed: int = 0,
    eval_seeds: Sequence[int] = (900, 901, 902, 903, 904),
    output_root: str = "results/rl",
    timestamp: Optional[str] = None,
    run_dir: Optional[str] = None,
    bc_policy=None,
    reward_config: Optional[RewardConfig] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    hidden_sizes: Sequence[int] = (256, 256),
    checkpoint_freq: int = 5000,
    eval_freq: int = 5000,
    resume: bool = False,
    ppo_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Any]:
    """Train (or resume) PPO on ``track`` with full metadata and checkpoints.

    Returns ``(run_dir, model)``. A non-empty existing run directory is never
    overwritten unless ``resume=True`` and a checkpoint is present.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.logger import configure

    reward_config = reward_config or RewardConfig()
    env_kwargs = dict(env_kwargs or {})
    ppo_kwargs = dict(ppo_kwargs or {})

    # Resolve the run directory (timestamped, never overwritten unless resuming).
    if run_dir is not None:
        run_path = Path(run_dir)
    else:
        ts = timestamp or _timestamp()
        run_path = Path(output_root) / stage / algorithm / ts
    resuming = bool(resume) and latest_checkpoint(run_path) is not None
    if run_path.exists() and run_path.is_dir() and any(run_path.iterdir()) and not resuming:
        raise FileExistsError(
            f"run directory {run_path} exists and is not empty; use a new timestamp or resume=True"
        )
    for sub in ("checkpoints", "best_model", "logs", "evaluation"):
        (run_path / sub).mkdir(parents=True, exist_ok=True)

    # Track provenance.
    shutil.copyfile(track, run_path / "track.json")
    (run_path / "track_sha256.txt").write_text(_sha256(track), encoding="utf-8")

    adapter_requested = env_kwargs.get("adapter", "fallback")
    allow_fallback = env_kwargs.get("allow_fallback", True)

    env = MarineRaceGymEnv(track, seed=train_seed, reward_fn=TrainingReward(reward_config), **env_kwargs)

    start_time = time.time()
    if resuming:
        checkpoint = latest_checkpoint(run_path)
        model = PPO.load(str(checkpoint), env=env, device="cpu")
        remaining = max(0, total_timesteps - int(model.num_timesteps))
        reset_num_timesteps = False
    else:
        model = build_ppo(env, hidden_sizes=hidden_sizes, seed=train_seed, **ppo_kwargs)
        if bc_policy is not None:
            transfer_bc_to_ppo(bc_policy, model)
        remaining = int(total_timesteps)
        reset_num_timesteps = True

    model.set_logger(configure(str(run_path / "logs"), ["csv", "stdout"]))

    eval_cb_cls = _make_completion_eval_callback()
    callbacks = [
        CheckpointCallback(save_freq=checkpoint_freq, save_path=str(run_path / "checkpoints"), name_prefix="ppo"),
        eval_cb_cls(track, eval_seeds, eval_freq, run_path / "best_model", run_path / "evaluation" / "eval.csv", env_kwargs, reward_config),
    ]

    _write_metadata(
        run_path,
        stage=stage,
        algorithm=algorithm,
        track=track,
        total_timesteps=total_timesteps,
        train_seed=train_seed,
        eval_seeds=list(eval_seeds),
        hidden_sizes=list(hidden_sizes),
        reward_config=reward_config,
        adapter_requested=adapter_requested,
        allow_fallback=allow_fallback,
        bc_initialized=bc_policy is not None,
        resuming=resuming,
        checkpoint_freq=checkpoint_freq,
        eval_freq=eval_freq,
        ppo_kwargs=ppo_kwargs,
        env_kwargs=env_kwargs,
    )

    adapter_actual = adapter_requested
    try:
        if remaining > 0:
            model.learn(
                total_timesteps=remaining,
                callback=callbacks,
                reset_num_timesteps=reset_num_timesteps,
                progress_bar=False,
            )
        model.save(str(run_path / "final_model"))
        try:
            if env.episode._ctx is not None:  # noqa: SLF001 - record the adapter actually used
                adapter_actual = env.episode._ctx.adapter.name
        except Exception:  # pragma: no cover
            pass
    finally:
        env.close()

    wall_clock_s = time.time() - start_time
    _finalize_environment_json(run_path, adapter_actual=adapter_actual, wall_clock_s=wall_clock_s, num_timesteps=int(model.num_timesteps))
    return run_path, model


def _timestamp() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_metadata(run_path: Path, **info) -> None:
    reward_config = info["reward_config"]
    run_config = {
        "stage": info["stage"],
        "algorithm": info["algorithm"],
        "track": info["track"],
        "total_timesteps": info["total_timesteps"],
        "hidden_sizes": info["hidden_sizes"],
        "train_seed": info["train_seed"],
        "checkpoint_freq": info["checkpoint_freq"],
        "eval_freq": info["eval_freq"],
        "bc_initialized": info["bc_initialized"],
        "resuming": info["resuming"],
        "ppo_kwargs": info["ppo_kwargs"],
        "env_kwargs": info["env_kwargs"],
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "obs_encoding_version": OBS_ENCODING_VERSION,
    }
    (run_path / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (run_path / "seeds.json").write_text(
        json.dumps({"train_seed": info["train_seed"], "eval_seeds": info["eval_seeds"]}, indent=2), encoding="utf-8"
    )
    (run_path / "reward_config.json").write_text(json.dumps(asdict(reward_config), indent=2), encoding="utf-8")

    environment = {
        "packages": _package_versions(),
        "git_sha": _git_sha(),
        "adapter_requested": info["adapter_requested"],
        "allow_fallback": info["allow_fallback"],
        "obs_encoding_version": OBS_ENCODING_VERSION,
        "adapter_actual": None,
        "fallback_used": None,
        "wall_clock_s": None,
        "final_num_timesteps": None,
    }
    (run_path / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    reproduce = (
        "# Reproduce this PPO run\n"
        f"# git checkout {_git_sha() or '<commit>'}\n"
        "# conda activate marine_race_rl\n"
        "python - <<'PY'\n"
        "from marine_race_arena.learning.train_workflow import run_ppo_training\n"
        f"run_ppo_training({info['track']!r}, stage={info['stage']!r}, algorithm={info['algorithm']!r},\n"
        f"    total_timesteps={info['total_timesteps']}, train_seed={info['train_seed']},\n"
        f"    eval_seeds={info['eval_seeds']!r}, env_kwargs={info['env_kwargs']!r},\n"
        f"    hidden_sizes={info['hidden_sizes']!r}, checkpoint_freq={info['checkpoint_freq']}, eval_freq={info['eval_freq']})\n"
        "PY\n"
        "# To resume: pass run_dir=<this directory> and resume=True.\n"
    )
    (run_path / "reproduce.txt").write_text(reproduce, encoding="utf-8")


def _finalize_environment_json(run_path: Path, *, adapter_actual: str, wall_clock_s: float, num_timesteps: int) -> None:
    path = run_path / "environment.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover
        data = {}
    data["adapter_actual"] = adapter_actual
    data["fallback_used"] = adapter_actual == "fallback"
    data["wall_clock_s"] = round(float(wall_clock_s), 3)
    data["final_num_timesteps"] = int(num_timesteps)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
