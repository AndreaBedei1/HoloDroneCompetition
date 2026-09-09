"""Reliability-first 27-D PPO smoke, starting from the working parent.

This runner intentionally loads ``parent_warmstart_27d.zip`` directly.  The
rejected BC checkpoint is never read.  It trains only on official circuits and
three exact fragments, then performs a matched closed-loop HoloOcean check.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from marine_race_arena.config.benchmark_tasks import BENCHMARK_TASK_CLEAN_GATE
from marine_race_arena.learning.gen2 import track_fragments as tf
from marine_race_arena.learning.gen2.fog_contract import verify_fog_sources
from marine_race_arena.learning.gen2.monitor_snapshot import Gen2MonitorSnapshotWrapper
from marine_race_arena.learning.gen2.ppo_reliability import ReliabilityReward, assert_reward_hierarchy
from marine_race_arena.learning.gen2.recurrent_policy import policy_parameter_count
from marine_race_arena.learning.gen2.train_ppo_27d import TrainingSource27d, materialize_sources_27d
from marine_race_arena.learning.gen2.transfer_27d import sha256_file
from marine_race_arena.learning.config_local_transition_27d import OBS_ENCODING_VERSION_LOCAL_TRANSITION_27D
from marine_race_arena.learning.gym_env import MarineRaceGymEnv


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def _make_env(source_payload: Mapping[str, Any], seed: int, monitor_path: str, snapshot_dir: str):
    from stable_baselines3.common.monitor import Monitor

    source = TrainingSource27d(**dict(source_payload))
    env = MarineRaceGymEnv(
        source.path, seed=int(seed), adapter="holoocean", allow_fallback=False,
        max_steps=int(source.max_steps), official=True, current_profile="none",
        benchmark_task=BENCHMARK_TASK_CLEAN_GATE,
        reward_fn=ReliabilityReward(source.gate_count),
        observation_encoding_version=OBS_ENCODING_VERSION_LOCAL_TRANSITION_27D,
        initial_body_velocity=source.initial_body_velocity,
    )
    # Only worker zero is wrapped; the monitor is still headless and adds no
    # environment steps.
    if str(monitor_path).endswith("worker_00"):
        env = Gen2MonitorSnapshotWrapper(env, output_dir=snapshot_dir, snapshot_interval_episodes=50, run_name="ppo_27d_smoke")
    return Monitor(env, filename=monitor_path)


def _make_vec_env(sources: Sequence[TrainingSource27d], run_dir: Path, seed: int):
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from marine_race_arena.learning.holoocean_capacity import capacity_snapshot

    capacity = capacity_snapshot()
    if int(capacity.get("available", 0)) < len(sources):
        raise RuntimeError(f"insufficient HoloOcean capacity: {capacity}")
    monitor_dir = run_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = run_dir / "monitor_snapshots"
    factories = []
    for index, source in enumerate(sources):
        factories.append(partial(
            _make_env, asdict(source), int(seed) + index * 10003,
            str(monitor_dir / f"worker_{index:02d}"), str(snapshot_dir),
        ))
    return SubprocVecEnv(factories, start_method="spawn"), capacity


class SmokeCallback:
    """Small callback object implemented as a SB3 BaseCallback subclass below."""


def _callback_class():
    from stable_baselines3.common.callbacks import BaseCallback

    class Callback(BaseCallback):
        def __init__(self, run_dir: Path, thresholds: Sequence[int], verbose: int = 1):
            super().__init__(verbose=verbose)
            self.run_dir = run_dir
            self.thresholds = list(sorted(int(x) for x in thresholds))
            self.saved = set()
            self.last_latest = 0
            self.latest_path = run_dir / "latest.zip"
            self.best_path = run_dir / "best.zip"
            self.records: List[Dict[str, Any]] = []
            self.rewards: List[float] = []
            self.components: Dict[str, float] = {}
            self.episodes: List[Dict[str, Any]] = []
            self.action_total = 0
            self.action_saturated = 0
            self.vision = self.orientation = self.obs_steps = 0
            self.best_key = (-1.0, -1.0, 0.0, 0.0)

        def _on_step(self) -> bool:
            rewards = self.locals.get("rewards")
            if rewards is not None:
                self.rewards.extend(float(x) for x in np.asarray(rewards).reshape(-1))
            actions = self.locals.get("actions")
            if actions is not None:
                arr = np.asarray(actions)
                self.action_total += int(arr.size)
                self.action_saturated += int(np.sum(np.abs(arr) >= 0.995))
            obs = self.locals.get("new_obs")
            if obs is not None:
                arr = np.asarray(obs)
                if arr.ndim == 2 and arr.shape[1] == 27:
                    self.obs_steps += int(arr.shape[0])
                    self.vision += int(np.sum((np.abs(arr[:, 5]) > 1e-4) | (arr[:, 7] > 1e-4)))
                    self.orientation += int(np.sum(arr[:, 24] > 0.5))
            infos = self.locals.get("infos") or []
            dones = self.locals.get("dones")
            for done, info in zip(dones if dones is not None else [], infos):
                for name, value in (info.get("reward_components") or {}).items():
                    self.components[name] = self.components.get(name, 0.0) + float(value)
                if done:
                    self.episodes.append({
                        "timestep": int(self.num_timesteps),
                        "completed": bool(info.get("completed", False)),
                        "gates": int(info.get("gates_completed", info.get("gate_crossings", 0))),
                        "status": str(info.get("status", "")),
                        "collision_events": int(info.get("collision_events", 0)) + int(info.get("obstacle_collision_events", 0)),
                        "out_of_bounds_events": int(info.get("out_of_bounds_events", 0)),
                        "wrong_direction_crossings": int(info.get("wrong_direction_crossings", 0)),
                        "missed_gate_attempts": int(info.get("missed_gate_attempts", 0)),
                    })
            self._maybe_checkpoint()
            return True

        def _on_rollout_end(self) -> None:
            self._maybe_checkpoint()

        def _metrics(self) -> Dict[str, Any]:
            logs = getattr(self.logger, "name_to_value", {}) if self.logger else {}
            completed = [x for x in self.episodes]
            completion = sum(bool(x["completed"]) for x in completed) / max(1, len(completed))
            gates = sum(x["gates"] for x in completed) / max(1, len(completed))
            safety = sum(x["collision_events"] + x["out_of_bounds_events"] + x["wrong_direction_crossings"] + x["missed_gate_attempts"] for x in completed)
            return {
                "timestep": int(self.num_timesteps),
                "approx_kl": float(logs.get("train/approx_kl", 0.0)),
                "policy_loss": float(logs.get("train/policy_gradient_loss", 0.0)),
                "value_loss": float(logs.get("train/value_loss", 0.0)),
                "entropy": float(logs.get("train/entropy_loss", 0.0)),
                "clip_fraction": float(logs.get("train/clip_fraction", 0.0)),
                "learning_rate": float(logs.get("train/learning_rate", 0.0)),
                "mean_reward_last500": float(np.mean(self.rewards[-500:])) if self.rewards else 0.0,
                "completion_rate": float(completion), "mean_gates": float(gates), "safety_events": int(safety),
                "vision_availability": self.vision / max(1, self.obs_steps),
                "orientation_availability": self.orientation / max(1, self.obs_steps),
                "action_saturation": self.action_saturated / max(1, self.action_total),
                "reward_components": dict(self.components),
                "episodes": len(completed),
            }

        def _maybe_checkpoint(self) -> None:
            step = int(self.num_timesteps)
            for threshold in self.thresholds:
                if step >= threshold and threshold not in self.saved:
                    self.model.save(str(self.run_dir / f"checkpoint_{threshold:05d}.zip"))
                    self.saved.add(threshold)
            metrics = self._metrics()
            should_snapshot = step - self.last_latest >= 256 or bool(self.saved and step in self.thresholds)
            if should_snapshot:
                self.model.save(str(self.latest_path))
                self.last_latest = step
                key = (metrics["completion_rate"], metrics["mean_gates"], -metrics["safety_events"], -metrics["timestep"] / 1e9)
                if key > self.best_key:
                    self.best_key = key
                    self.model.save(str(self.best_path))
            if not self.records or step - int(self.records[-1]["timestep"]) >= 256:
                self.records.append(metrics)
                _atomic_json(self.run_dir / "training_metrics.json", {"records": self.records, "episodes": self.episodes})
                if self.verbose:
                    print(json.dumps(metrics), flush=True)

        def _on_training_end(self) -> None:
            self._maybe_checkpoint()
            _atomic_json(self.run_dir / "training_metrics.json", {"records": self.records, "episodes": self.episodes})

    return Callback


def _matched_eval(parent: Path, candidate: Path, out: Path, seed: int) -> Dict[str, Any]:
    from sb3_contrib import RecurrentPPO
    from marine_race_arena.learning.gen2.evaluation import run_policy_episode
    from marine_race_arena.learning.gen2.recurrent_policy import Gen2RecurrentController

    rows = []
    for label, path in (("parent_warmstart_27d", parent), ("ppo_candidate_5k", candidate)):
        model = RecurrentPPO.load(str(path), device="cpu")
        controller = Gen2RecurrentController(model, deterministic=True)
        for index, track in enumerate(tf.OFFICIAL_TRACKS):
            result = run_policy_episode(controller, tf.track_path(track), seed=int(seed + index), adapter="holoocean", allow_fallback=False, max_steps=7000)
            rows.append({"policy": label, "track": track, **result.as_row()})
            print(json.dumps(rows[-1]), flush=True)
    report = {"rows": rows, "actual_adapter": "holoocean", "fallback_used": False, "fog": verify_fog_sources([tf.track_path(t) for t in tf.OFFICIAL_TRACKS])}
    _atomic_json(out / "matched_evaluation.json", report)
    return report


def run(*, parent: str | Path, out: str | Path, steps: int = 5120, seed: int = 31001) -> Dict[str, Any]:
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.utils import get_schedule_fn
    from marine_race_arena.learning.train_ppo_transition import close_vec_env_safely

    parent_path = Path(parent)
    run_dir = Path(out)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    parent_sha = sha256_file(parent_path)
    model_probe = RecurrentPPO.load(str(parent_path), device="cpu")
    if tuple(model_probe.observation_space.shape) != (27,):
        raise RuntimeError("PPO start checkpoint is not the approved 27-D parent warm-start")
    del model_probe
    reward_check = assert_reward_hierarchy()
    if not reward_check["all_hold"]:
        raise RuntimeError(reward_check)
    sources, source_audit = materialize_sources_27d(run_dir)
    source_audit["synthetic_included"] = False
    source_audit["fog"] = verify_fog_sources([Path(x.path) for x in sources])
    _atomic_json(run_dir / "run_config.json", {
        "parent": str(parent_path), "parent_sha256": parent_sha, "bc_policy_used": False,
        "observation_contract": OBS_ENCODING_VERSION_LOCAL_TRANSITION_27D, "observation_dim": 27,
        "total_timesteps_requested": int(steps), "sources": [asdict(x) for x in sources],
        "source_audit": source_audit, "reward_hierarchy": reward_check,
        "ppo": {"learning_rate": 1e-5, "clip_range": 0.06, "target_kl": 0.0075, "n_epochs": 1, "n_steps": 256, "batch_size": 128, "ent_coef": 0.0, "max_grad_norm": 0.5, "efficiency_reward": False, "reliability_reward": True},
    })
    env = None
    close_report: Dict[str, Any] = {}
    try:
        env, capacity = _make_vec_env(sources, run_dir, seed)
        model = RecurrentPPO.load(str(parent_path), env=env, device="cpu")
        model.learning_rate = 1e-5
        model.lr_schedule = get_schedule_fn(1e-5)
        for group in model.policy.optimizer.param_groups:
            group["lr"] = 1e-5
        model.clip_range = get_schedule_fn(0.06)
        model.target_kl = 0.0075
        model.n_epochs = 1
        model.max_grad_norm = 0.5
        model.ent_coef = 0.0
        if not all(p.requires_grad for p in model.policy.parameters()):
            raise RuntimeError("some PPO parameters are frozen")
        _atomic_json(run_dir / "start_audit.json", {"parent_sha256": parent_sha, "parameter_count": policy_parameter_count(model), "all_parameters_trainable": True, "capacity": capacity})
        Callback = _callback_class()
        callback = Callback(run_dir, thresholds=(1024, 2560, 5120), verbose=1)
        model.learn(total_timesteps=int(steps), callback=callback, reset_num_timesteps=True, progress_bar=False)
        model.save(str(run_dir / "candidate_5k.zip"))
        callback._maybe_checkpoint()
        _atomic_json(run_dir / "training_end.json", {"actual_timesteps": int(model.num_timesteps), "parameter_count": policy_parameter_count(model), "checkpoints": [str(x) for x in sorted(run_dir.glob("*.zip"))]})
    finally:
        if env is not None:
            close_report = close_vec_env_safely(env, timeout=120.0)
            _atomic_json(run_dir / "environment_close.json", close_report)
    if sha256_file(parent_path) != parent_sha:
        raise RuntimeError("parent checkpoint changed")
    evaluation = _matched_eval(parent_path, run_dir / "candidate_5k.zip", run_dir, seed + 1000)
    report = {"parent": str(parent_path), "parent_sha256": parent_sha, "candidate": str(run_dir / "candidate_5k.zip"), "candidate_sha256": sha256_file(run_dir / "candidate_5k.zip"), "training": json.loads((run_dir / "training_metrics.json").read_text()), "evaluation": evaluation, "environment_close": close_report, "ppo_started": True, "continued_beyond_5k": False}
    _atomic_json(run_dir / "smoke_report.json", report)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=5120)
    parser.add_argument("--seed", type=int, default=31001)
    args = parser.parse_args(argv)
    print(json.dumps(run(parent=args.parent, out=args.out, steps=args.steps, seed=args.seed), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
