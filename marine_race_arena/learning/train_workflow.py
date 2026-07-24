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

from marine_race_arena.learning.bc_ppo_init import initialize_action_std
from marine_race_arena.learning.config import ACTION_CONTRACT_VERSION, ACTION_DIM, OBS_DIM, OBS_ENCODING_VERSION
from marine_race_arena.learning.gym_env import MarineRaceGymEnv
from marine_race_arena.learning.ppo_monitor import (
    PPOUpdateRecorder,
    RunStatus,
    WarmStartAbort,
    make_kl_monitor_callback,
)
from marine_race_arena.learning.reward import RewardConfig, TrainingReward
from marine_race_arena.learning.rl_train import build_ppo, transfer_bc_to_ppo
from marine_race_arena.learning.stage2_eval import (
    aggregate_stage2,
    evaluate_stage2,
    log_reward_components,
    stage2_best_metric_key,
    stage2_is_better,
)


def evaluate_completion(model, track, eval_seeds, *, env_kwargs, reward_config) -> Dict[str, Any]:
    """Deterministic held-out completion metrics for a policy (shared by the eval
    callback and the timestep-zero evaluation). Never uses privileged state for control."""
    completions, gates, collisions, finished_times, oob, wrongdir = 0, [], [], [], [], []
    for seed in eval_seeds:
        env = MarineRaceGymEnv(track, seed=int(seed), reward_fn=TrainingReward(reward_config), **dict(env_kwargs or {}))
        try:
            obs, _ = env.reset(seed=int(seed))
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            progress = env.episode.referee_progress()
            state = env.episode.context.referee.states[env.episode.participant_id]
            if progress["status"] == "FINISHED":
                completions += 1
                finished_times.append(env.episode.step_count * env.episode.dt)
            gates.append(progress["valid_gate_crossings"])
            collisions.append(int(state.collision_events))
            oob.append(int(state.out_of_bounds_events))
            wrongdir.append(int(state.wrong_direction_crossings))
        finally:
            env.close()
    n = max(1, len(list(eval_seeds)))
    return {
        "completion_rate": completions / n,
        "mean_gates": float(np.mean(gates)) if gates else 0.0,
        "mean_collisions": float(np.mean(collisions)) if collisions else 0.0,
        "mean_out_of_bounds": float(np.mean(oob)) if oob else 0.0,
        "mean_wrong_direction": float(np.mean(wrongdir)) if wrongdir else 0.0,
        "mean_time_finished": (float(np.mean(finished_times)) if finished_times else None),
    }


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


def best_metric_key(row: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Best-model ordering (higher is better) with documented lexicographic tie-breaks:
    (1) completion rate, (2) mean gates, (3) fewer collisions, (4) lower finished time."""
    time_finished = row.get("mean_time_finished")
    neg_time = -float(time_finished) if time_finished is not None else 0.0
    return (
        float(row["completion_rate"]),
        float(row["mean_gates"]),
        -float(row["mean_collisions"]),
        neg_time,
    )


def strictly_better(new_row: Dict[str, Any], best_row: Optional[Dict[str, Any]]) -> bool:
    """A new evaluation replaces the best model only if strictly better by the key."""
    if best_row is None:
        return True
    return best_metric_key(new_row) > best_metric_key(best_row)


_EVAL_CSV_FIELDS = ["timesteps", "completion_rate", "mean_gates", "mean_collisions", "mean_time_finished"]


def _load_bc_report(bc_report_path: Optional[str], bc_model_path: Optional[str]):
    """Load the BC report (for per-axis residuals); fall back to one next to the model."""
    candidates: List[Path] = []
    if bc_report_path:
        candidates.append(Path(bc_report_path))
    if bc_model_path:
        candidates.append(Path(bc_model_path).with_name("bc_report.json"))
    for p in candidates:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8")), str(p)
            except Exception:  # pragma: no cover - malformed report
                return None, str(p)
    return None, (str(candidates[0]) if candidates else None)


def _load_action_std(run_path: Path) -> Dict[str, Any]:
    p = Path(run_path) / "action_std.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover
            return {}
    return {}


def _coerce_eval_row(raw: Dict[str, str]) -> Dict[str, Any]:
    """Coerce a CSV eval row back to numbers (int timesteps, floats elsewhere, None for blanks)."""
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        if v in (None, "", "None"):
            out[k] = None
        elif k == "timesteps":
            out[k] = int(float(v))
        else:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = v
    return out


def _append_eval_row(path: Path, row: Dict[str, Any]) -> None:
    exists = Path(path).exists()
    with Path(path).open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_EVAL_CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in _EVAL_CSV_FIELDS})


def _write_single_eval_csv(path: Path, row: Dict[str, Any]) -> None:
    """Create eval.csv seeded with one row (all scalar columns; nested values dropped)."""
    fields = [k for k, v in row.items() if not isinstance(v, (dict, list))]
    if "timesteps" in fields:
        fields = ["timesteps"] + [f for f in fields if f != "timesteps"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({k: row.get(k) for k in fields})


def _run_timestep_zero_eval(model, track, eval_seeds, *, env_kwargs, reward_config, run_path,
                            bc_initialized, action_std_info, min_initial_completion, stage2=False) -> Dict[str, Any]:
    """Deterministic held-out evaluation BEFORE ``model.learn()`` modifies the policy.

    Writes ``evaluation/initial_eval.json`` and a timestep-0 row to ``evaluation/eval.csv``,
    saves the timestep-0 policy as the initial best, and (Stage-2) logs the reward
    components by outcome. Aborts before training if a BC warm-start looks broken.
    """
    if stage2:
        full = evaluate_stage2(model, track, eval_seeds, env_kwargs=env_kwargs, reward_config=reward_config)
        per_seed = full.get("rows", [])
        metrics = {k: v for k, v in full.items() if k != "rows"}
    else:
        metrics = evaluate_completion(model, track, eval_seeds, env_kwargs=env_kwargs, reward_config=reward_config)
        per_seed = None
    initial = {
        "timesteps": 0,
        "phase": "timestep_zero",
        "deterministic": True,
        "bc_initialized": bool(bc_initialized),
        "model_initialization_source": (
            "bc_transfer+" + str(action_std_info.get("source")) if bc_initialized else str(action_std_info.get("source"))),
        "eval_seeds": list(eval_seeds),
        "action_std": action_std_info,
        **metrics,
    }
    if per_seed is not None:
        initial["per_seed"] = per_seed
    eval_dir = run_path / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "initial_eval.json").write_text(json.dumps(initial, indent=2), encoding="utf-8")
    row = {"timesteps": 0, **metrics}
    _write_single_eval_csv(eval_dir / "eval.csv", row)
    best_dir = run_path / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(best_dir / "best_model"))
    (best_dir / "best_metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    # Reward-component diagnostic (does not modify the reward).
    try:
        rc = log_reward_components(model, track, eval_seeds, env_kwargs=env_kwargs, reward_config=reward_config)
        (eval_dir / "reward_components.json").write_text(json.dumps(rc, indent=2), encoding="utf-8")
    except Exception:  # pragma: no cover - diagnostic must never break a run
        pass
    if bc_initialized and min_initial_completion is not None and metrics["completion_rate"] < float(min_initial_completion):
        from marine_race_arena.learning.ppo_monitor import WarmStartAbort

        raise WarmStartAbort(
            f"BC-initialized timestep-zero completion {metrics['completion_rate']:.3f} < required "
            f"{float(min_initial_completion):.3f}; the warm-start looks broken -- stopping before training.")
    return initial


# --------------------------------------------------------------- eval callback
def _make_completion_eval_callback():
    from stable_baselines3.common.callbacks import BaseCallback

    class CompletionEvalCallback(BaseCallback):
        """Evaluate held-out completion rate; save the best model on improvement."""

        def __init__(self, track, eval_seeds, eval_freq, best_dir, eval_csv, env_kwargs, reward_config,
                     stage2=False, verbose=0):
            super().__init__(verbose)
            self.track = track
            self.eval_seeds = list(eval_seeds)
            self.eval_freq = int(eval_freq)
            self.best_dir = Path(best_dir)
            self.eval_csv = Path(eval_csv)
            self.env_kwargs = dict(env_kwargs or {})
            self.reward_config = reward_config
            self.stage2 = bool(stage2)
            self._rows: List[Dict[str, float]] = []
            self._best_key = None
            self._best_row: Optional[Dict[str, float]] = None

        @property
        def _metric_key(self):
            return stage2_best_metric_key if self.stage2 else best_metric_key

        def _init_callback(self):
            self.best_dir.mkdir(parents=True, exist_ok=True)
            self.eval_csv.parent.mkdir(parents=True, exist_ok=True)
            # Resume: recover prior evaluation history and best metric, so a resumed
            # run never erases history nor overwrites a previously better model.
            if self.eval_csv.exists():
                try:
                    with self.eval_csv.open(newline="", encoding="utf-8") as handle:
                        for r in csv.DictReader(handle):
                            self._rows.append(_coerce_eval_row(r))
                except Exception:  # pragma: no cover - tolerate a partial/old CSV
                    self._rows = []
            best_path = self.best_dir / "best_metrics.json"
            if best_path.exists():
                try:
                    self._best_row = json.loads(best_path.read_text(encoding="utf-8"))
                    self._best_key = self._metric_key(self._best_row)
                except Exception:  # pragma: no cover
                    self._best_row = None
            if self._best_row is None and self._rows:
                self._best_row = max(self._rows, key=self._metric_key)
                self._best_key = self._metric_key(self._best_row)

        def _evaluate(self) -> Dict[str, float]:
            if self.stage2:
                agg = evaluate_stage2(self.model, self.track, self.eval_seeds,
                                      env_kwargs=self.env_kwargs, reward_config=self.reward_config)
                agg.pop("rows", None)  # keep the periodic history compact
                return agg
            return evaluate_completion(self.model, self.track, self.eval_seeds,
                                       env_kwargs=self.env_kwargs, reward_config=self.reward_config)

        def _is_better(self, new_row, best_row) -> bool:
            return stage2_is_better(new_row, best_row) if self.stage2 else strictly_better(new_row, best_row)

        def _on_step(self) -> bool:
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                metrics = self._evaluate()
                row = {"timesteps": int(self.num_timesteps), **metrics}
                self._rows.append(row)
                self._write_csv()
                self.logger.record("eval/completion_rate", metrics.get("completion_rate", 0.0))
                self.logger.record("eval/mean_gates", metrics.get("mean_gates", 0.0))
                if self._is_better(row, self._best_row):
                    self._best_row = row
                    self.model.save(str(self.best_dir / "best_model"))
                    (self.best_dir / "best_metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            return True

        def _write_csv(self):
            if not self._rows:
                return
            # Flat CSV: timesteps first, then every scalar key seen (nested dicts/lists go
            # to best_metrics.json only). Supports both Stage-1 and Stage-2 aggregates.
            fields = ["timesteps"]
            for r in self._rows:
                for k, v in r.items():
                    if k not in fields and not isinstance(v, (dict, list)):
                        fields.append(k)
            with self.eval_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for row in self._rows:
                    writer.writerow({k: row.get(k) for k in fields})

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
    bc_model_path: Optional[str] = None,
    bc_report_path: Optional[str] = None,
    arm: str = "bcinit",
    action_std_strategy: Optional[str] = None,
    action_std_value=None,
    bc_action_std_min: float = 0.05,
    bc_action_std_max: float = 0.15,
    bc_log_std_fallback: float = -2.5,
    max_acceptable_kl: Optional[float] = None,
    max_action_saturation: Optional[float] = None,
    stage2: bool = False,
    initial_eval: bool = True,
    min_initial_completion: Optional[float] = None,
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
    # Exploration-std strategy: BC-init defaults to residual-derived; scratch defaults to
    # SB3's own init. An explicit strategy (e.g. a fixed 0.10 for the controlled arms) wins.
    effective_std_strategy = action_std_strategy or ("residual" if (bc_policy is not None or bc_model_path) else "sb3_default")
    target_kl = ppo_kwargs.get("target_kl")
    std_config = {"strategy": effective_std_strategy, "value": action_std_value,
                  "std_min": bc_action_std_min, "std_max": bc_action_std_max, "log_std_fallback": bc_log_std_fallback}

    # BC initialization can be supplied as a policy object or a path (for reproducibility).
    bc_model_sha256 = None
    if bc_model_path is not None:
        bc_model_sha256 = _sha256(bc_model_path)
        if bc_policy is None:
            from marine_race_arena.learning.bc_train import load_policy

            bc_policy = load_policy(bc_model_path)

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
    # On resume, refuse to continue against an incompatible configuration.
    if resuming:
        _validate_resume_compatibility(
            run_path,
            current={
                "track_sha256": _sha256(track),
                "hidden_sizes": list(hidden_sizes),
                "bc_initialized": bc_policy is not None,
                "adapter_requested": env_kwargs.get("adapter", "fallback"),
                "current_profile": env_kwargs.get("current_profile"),
                "randomized": env_kwargs.get("start_randomization") is not None,
                "obs_encoding_version": OBS_ENCODING_VERSION,
                "action_dim": ACTION_DIM,
                "reward_config": asdict(reward_config),
                "ppo_kwargs": ppo_kwargs,
                "bc_action_std_config": std_config,
            },
        )

    for sub in ("checkpoints", "best_model", "logs", "evaluation"):
        (run_path / sub).mkdir(parents=True, exist_ok=True)

    # Track provenance.
    shutil.copyfile(track, run_path / "track.json")
    (run_path / "track_sha256.txt").write_text(_sha256(track), encoding="utf-8")

    adapter_requested = env_kwargs.get("adapter", "fallback")
    allow_fallback = env_kwargs.get("allow_fallback", True)

    # Stage-2 training varies the randomization seed per episode (train seed stream); eval
    # keeps explicit seeds, so env_kwargs (shared with eval) must NOT carry the stream.
    train_env_kwargs = dict(env_kwargs)
    if stage2 and env_kwargs.get("start_randomization") is not None:
        train_env_kwargs["episode_seed_stream"] = int(train_seed)
    env = MarineRaceGymEnv(track, seed=train_seed, reward_fn=TrainingReward(reward_config), **train_env_kwargs)

    start_time = time.time()
    action_std_info: Dict[str, Any]
    if resuming:
        checkpoint = latest_checkpoint(run_path)
        model = PPO.load(str(checkpoint), env=env, device="cpu")
        remaining = max(0, total_timesteps - int(model.num_timesteps))
        reset_num_timesteps = False
        # The learned log_std is restored from the checkpoint; keep the recorded
        # action-std provenance (do not re-initialize on resume).
        action_std_info = _load_action_std(run_path)
    else:
        model = build_ppo(env, hidden_sizes=hidden_sizes, seed=train_seed, **ppo_kwargs)
        bc_report_used = None
        if bc_policy is not None:
            transfer_bc_to_ppo(bc_policy, model)
        # Install the resolved exploration std (residual/fixed for the controlled arms;
        # sb3_default leaves SB3's own init). Fixed applies to scratch_controlled too.
        bc_report = None
        if effective_std_strategy == "residual":
            bc_report, bc_report_used = _load_bc_report(bc_report_path, bc_model_path)
        action_std_info = initialize_action_std(
            model, effective_std_strategy, bc_report=bc_report, value=action_std_value,
            action_dim=ACTION_DIM, std_min=bc_action_std_min, std_max=bc_action_std_max,
            log_std_fallback=bc_log_std_fallback,
        )
        action_std_info["arm"] = arm
        action_std_info["bc_initialized"] = bc_policy is not None
        action_std_info["bc_report_path"] = bc_report_used
        action_std_info["bc_report_sha256"] = (_sha256(bc_report_used)
                                               if bc_report_used and Path(bc_report_used).exists() else None)
        action_std_info["bc_model_path"] = bc_model_path
        action_std_info["bc_model_sha256"] = bc_model_sha256
        remaining = int(total_timesteps)
        reset_num_timesteps = True
        (run_path / "action_std.json").write_text(json.dumps(action_std_info, indent=2), encoding="utf-8")

    model.set_logger(configure(str(run_path / "logs"), ["csv", "stdout"]))

    # --- Timestep-zero held-out evaluation (fresh start only; never duplicated on resume) ---
    eval_cb_cls = _make_completion_eval_callback()
    callbacks = [
        CheckpointCallback(save_freq=checkpoint_freq, save_path=str(run_path / "checkpoints"), name_prefix="ppo"),
        eval_cb_cls(track, eval_seeds, eval_freq, run_path / "best_model", run_path / "evaluation" / "eval.csv",
                    env_kwargs, reward_config, stage2=stage2),
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
        bc_model_path=bc_model_path,
        bc_model_sha256=bc_model_sha256,
        action_std=action_std_info,
        bc_action_std_config=std_config,
        bc_report_path=bc_report_path,
        arm=arm,
        stage2=stage2,
        max_acceptable_kl=max_acceptable_kl,
        output_root=output_root,
        run_dir=str(run_path),
    )

    # Structured KL monitoring + hard safety stop (an SB3 callback; records each update
    # at the next rollout start and stops learn() cleanly if the hard KL is exceeded).
    recorder = PPOUpdateRecorder(model, run_path / "training" / "ppo_update_metrics.csv",
                                 target_kl=target_kl, max_acceptable_kl=max_acceptable_kl,
                                 max_action_saturation=max_action_saturation)
    callbacks.append(make_kl_monitor_callback(recorder))
    run_status = RunStatus.COMPLETED
    adapter_actual = adapter_requested
    try:
        if initial_eval and not resuming:
            _run_timestep_zero_eval(
                model, track, list(eval_seeds), env_kwargs=env_kwargs, reward_config=reward_config,
                run_path=run_path, bc_initialized=bc_policy is not None, action_std_info=action_std_info,
                min_initial_completion=min_initial_completion, stage2=stage2,
            )
        if remaining > 0:
            model.learn(total_timesteps=remaining, callback=callbacks,
                        reset_num_timesteps=reset_num_timesteps, progress_bar=False)
        recorder.record()  # capture the final update (not seen by the next rollout start)
        run_status = recorder.status
        model.save(str(run_path / "final_model"))
    except WarmStartAbort:
        run_status = "ABORT_INITIAL_EVAL"
        raise
    except Exception:  # pragma: no cover - engine / other failure
        run_status = RunStatus.SIMULATOR_FAILURE
        raise
    finally:
        try:
            if env.episode._ctx is not None:  # noqa: SLF001 - record the adapter actually used
                adapter_actual = env.episode._ctx.adapter.name
        except Exception:  # pragma: no cover
            pass
        env.close()
        if adapter_actual == "fallback" and adapter_requested != "fallback" and run_status == RunStatus.COMPLETED:
            run_status = RunStatus.SIMULATOR_FAILURE
        wall_clock_s = time.time() - start_time
        _finalize_environment_json(run_path, adapter_actual=adapter_actual, wall_clock_s=wall_clock_s,
                                   num_timesteps=int(model.num_timesteps), run_status=run_status,
                                   monitor_summary=recorder.summary())
    return run_path, model


def _timestamp() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _serializable_env_kwargs(env_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Make env_kwargs JSON-serializable (a StartRandomization becomes its dict)."""
    from dataclasses import asdict as _asdict, is_dataclass

    out: Dict[str, Any] = {}
    for key, value in (env_kwargs or {}).items():
        out[key] = _asdict(value) if is_dataclass(value) else value
    return out


def _validate_resume_compatibility(run_path: Path, *, current: Dict[str, Any]) -> None:
    """Raise if resuming against an incompatible configuration."""
    mismatches: List[str] = []
    track_hash_path = run_path / "track_sha256.txt"
    if track_hash_path.exists():
        prev = track_hash_path.read_text(encoding="utf-8").strip()
        if prev != current["track_sha256"]:
            mismatches.append(f"track_sha256 ({prev[:12]} != {current['track_sha256'][:12]})")
    run_config_path = run_path / "run_config.json"
    if run_config_path.exists():
        rc = json.loads(run_config_path.read_text(encoding="utf-8"))
        prev_env = rc.get("env_kwargs", {}) or {}
        prev = {
            "obs_encoding_version": rc.get("obs_encoding_version"),
            "action_dim": rc.get("action_dim"),
            "hidden_sizes": rc.get("hidden_sizes"),
            "bc_initialized": rc.get("bc_initialized"),
            "adapter": prev_env.get("adapter", "fallback"),
            "current_profile": prev_env.get("current_profile"),
            "randomized": bool(prev_env.get("start_randomization")),
        }
        cur = {
            "obs_encoding_version": current["obs_encoding_version"],
            "action_dim": current["action_dim"],
            "hidden_sizes": current["hidden_sizes"],
            "bc_initialized": current["bc_initialized"],
            "adapter": current["adapter_requested"],
            "current_profile": current["current_profile"],
            "randomized": current["randomized"],
        }
        for key in cur:
            if prev.get(key) != cur[key]:
                mismatches.append(f"{key} ({prev.get(key)} != {cur[key]})")
        prev_ppo = rc.get("ppo_kwargs", {}) or {}
        for key in ("n_steps", "batch_size", "n_epochs"):
            if prev_ppo.get(key) != current["ppo_kwargs"].get(key):
                mismatches.append(f"ppo.{key} ({prev_ppo.get(key)} != {current['ppo_kwargs'].get(key)})")
        prev_std = rc.get("bc_action_std_config")
        cur_std = current.get("bc_action_std_config")
        if prev_std is not None and cur_std is not None and prev_std != cur_std:
            mismatches.append(f"bc_action_std_config ({prev_std} != {cur_std})")
    reward_path = run_path / "reward_config.json"
    if reward_path.exists():
        if json.loads(reward_path.read_text(encoding="utf-8")) != current["reward_config"]:
            mismatches.append("reward_config")
    if mismatches:
        raise ValueError("cannot resume: incompatible configuration -> " + "; ".join(mismatches))


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
        "env_kwargs": _serializable_env_kwargs(info["env_kwargs"]),
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "obs_encoding_version": OBS_ENCODING_VERSION,
        "action_contract_version": ACTION_CONTRACT_VERSION,
        "arm": info.get("arm"),
        "stage2": bool(info.get("stage2")),
        "max_acceptable_kl": info.get("max_acceptable_kl"),
        "bc_action_std_config": info.get("bc_action_std_config"),
        "bc_report_path": info.get("bc_report_path"),
        "action_std": info.get("action_std"),
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

    (run_path / "reproduce.txt").write_text(build_reproduce_script(info, _git_sha()), encoding="utf-8")


def build_reproduce_script(info: Dict[str, Any], commit: Optional[str]) -> str:
    """Build a complete, executable reproduction script (no ellipsis placeholders).

    Machine-specific paths are exposed as clearly documented variables; everything
    else (seeds, reward config, PPO hyperparameters, randomization, BC model + hash)
    is captured literally so the run can be reproduced from scratch or resumed.
    """
    env_kwargs = _serializable_env_kwargs(info["env_kwargs"])
    reward_dict = asdict(info["reward_config"])
    bc_path = info.get("bc_model_path")
    bc_sha = info.get("bc_model_sha256")
    randomized = bool(env_kwargs.get("start_randomization"))

    lines: List[str] = []
    lines.append("# Reproduce this PPO run. Machine-specific paths are variables below.")
    lines.append(f"# commit: {commit or '<unknown>'}")
    lines.append("# git checkout " + (commit or "<commit>"))
    lines.append("# conda activate marine_race_rl")
    lines.append(f"# adapter={info['adapter_requested']}  fallback_allowed={info['allow_fallback']}  "
                 f"obs_encoding={OBS_ENCODING_VERSION}  bc_initialized={info['bc_initialized']}")
    if bc_path:
        lines.append(f"# BC model: {bc_path}  (sha256 {bc_sha})")
    lines.append("python - <<'PY'")
    lines.append("import os")
    lines.append("from marine_race_arena.learning.train_workflow import run_ppo_training")
    lines.append("from marine_race_arena.learning.reward import RewardConfig")
    if randomized:
        lines.append("from marine_race_arena.learning.randomization import StartRandomization")
    lines.append("")
    lines.append("# --- machine-specific variables (edit these) ---")
    lines.append(f"OUTPUT_ROOT = os.environ.get('MARINE_RACE_REPRODUCE_ROOT', {info['output_root']!r})")
    if bc_path:
        lines.append(f"BC_MODEL_PATH = {bc_path!r}  # sha256 {bc_sha}")
    lines.append("")
    lines.append(f"reward_config = RewardConfig(**{reward_dict!r})")
    env_literal = dict(env_kwargs)
    if randomized:
        rspec = env_literal.pop("start_randomization")
        lines.append(f"env_kwargs = {env_literal!r}")
        lines.append(f"env_kwargs['start_randomization'] = StartRandomization(**{rspec!r})")
    else:
        lines.append(f"env_kwargs = {env_literal!r}")
    lines.append("")
    lines.append("run_ppo_training(")
    lines.append(f"    {info['track']!r},")
    lines.append(f"    stage={info['stage']!r}, algorithm={info['algorithm']!r},")
    lines.append(f"    total_timesteps={info['total_timesteps']}, train_seed={info['train_seed']},")
    lines.append(f"    eval_seeds={list(info['eval_seeds'])!r},")
    lines.append("    output_root=OUTPUT_ROOT,")
    lines.append(f"    hidden_sizes={info['hidden_sizes']!r},")
    lines.append(f"    checkpoint_freq={info['checkpoint_freq']}, eval_freq={info['eval_freq']},")
    lines.append("    reward_config=reward_config, env_kwargs=env_kwargs,")
    lines.append(f"    ppo_kwargs={info['ppo_kwargs']!r},")
    lines.append(f"    arm={info.get('arm')!r}, stage2={bool(info.get('stage2'))!r}, "
                 f"max_acceptable_kl={info.get('max_acceptable_kl')!r},")
    std_cfg = info.get("bc_action_std_config") or {}
    if std_cfg:
        lines.append(f"    action_std_strategy={std_cfg.get('strategy')!r}, "
                     f"action_std_value={std_cfg.get('value')!r},")
        lines.append(f"    bc_action_std_min={std_cfg.get('std_min')!r}, "
                     f"bc_action_std_max={std_cfg.get('std_max')!r}, "
                     f"bc_log_std_fallback={std_cfg.get('log_std_fallback')!r},")
    if bc_path:
        lines.append("    bc_model_path=BC_MODEL_PATH,")
        bc_report = info.get("bc_report_path")
        if bc_report:
            lines.append(f"    bc_report_path={bc_report!r},")
    lines.append(")")
    lines.append("PY")
    lines.append("")
    lines.append("# To RESUME this exact run instead of starting fresh, add to the call above:")
    lines.append(f"#     run_dir={info['run_dir']!r}, resume=True")
    return "\n".join(lines) + "\n"


def _finalize_environment_json(run_path: Path, *, adapter_actual: str, wall_clock_s: float, num_timesteps: int,
                               run_status: str = "COMPLETED", monitor_summary: Optional[Dict[str, Any]] = None) -> None:
    path = run_path / "environment.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover
        data = {}
    data["adapter_actual"] = adapter_actual
    data["fallback_used"] = adapter_actual == "fallback"
    data["wall_clock_s"] = round(float(wall_clock_s), 3)
    data["final_num_timesteps"] = int(num_timesteps)
    data["run_status"] = run_status
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    # A dedicated, machine-readable run-status record (KL summary + status enum).
    (run_path / "run_status.json").write_text(json.dumps({
        "run_status": run_status,
        "adapter_actual": adapter_actual,
        "final_num_timesteps": int(num_timesteps),
        "wall_clock_s": round(float(wall_clock_s), 3),
        "kl_summary": monitor_summary or {},
    }, indent=2), encoding="utf-8")
