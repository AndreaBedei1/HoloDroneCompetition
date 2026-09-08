"""Long, staged reliability-first PPO campaign for the approved 27-D parent.

The campaign always starts from the parent warm-start, keeps the parent
immutable, trains only onboard observations, and evaluates at recoverable
milestones.  It deliberately treats isolated safety events as diagnostics;
rollback is reserved for sustained completion/progress/KL collapse.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from marine_race_arena.config.benchmark_tasks import BENCHMARK_TASK_CLEAN_GATE
from marine_race_arena.learning.config_local_transition_27d import OBS_ENCODING_VERSION_LOCAL_TRANSITION_27D
from marine_race_arena.learning.gen2 import track_fragments as tf
from marine_race_arena.learning.gen2.fog_contract import verify_fog_sources
from marine_race_arena.learning.gen2.monitor_snapshot import Gen2MonitorSnapshotWrapper
from marine_race_arena.learning.gen2.ppo_reliability import ReliabilityReward, assert_reward_hierarchy
from marine_race_arena.learning.gen2.recurrent_policy import policy_parameter_count
from marine_race_arena.learning.gen2.train_ppo_27d import TrainingSource27d, materialize_sources_27d
from marine_race_arena.learning.gen2.transfer_27d import sha256_file
from marine_race_arena.learning.gym_env import MarineRaceGymEnv


COLLISION_PENALTY = -35.0
MILESTONES = (10_000, 25_000, 50_000, 75_000, 100_000, 150_000, 200_000)


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def _sources(run_dir: Path) -> tuple[List[TrainingSource27d], Dict[str, Any]]:
    base, audit = materialize_sources_27d(run_dir)
    # Full circuits are deliberately duplicated: three full tracks remain the
    # majority of samples even though their episodes are longer than fragments.
    full = [s for s in base if s.kind == "full_circuit"]
    fragments = [s for s in base if s.kind == "exact_fragment"]
    sources = full + full + fragments
    audit = dict(audit)
    audit.update({"synthetic_included": False, "source_balance": "6 full + 3 exact fragments"})
    audit["fog"] = verify_fog_sources([Path(s.path) for s in sources])
    return sources, audit


def _make_env(payload: Mapping[str, Any], seed: int, monitor_path: str, snapshot_dir: str):
    from stable_baselines3.common.monitor import Monitor

    source = TrainingSource27d(**dict(payload))
    env = MarineRaceGymEnv(
        source.path, seed=int(seed), adapter="holoocean", allow_fallback=False,
        max_steps=int(source.max_steps), official=True, current_profile="none",
        benchmark_task=BENCHMARK_TASK_CLEAN_GATE,
        reward_fn=ReliabilityReward(source.gate_count, collision_penalty=COLLISION_PENALTY),
        observation_encoding_version=OBS_ENCODING_VERSION_LOCAL_TRANSITION_27D,
        initial_body_velocity=source.initial_body_velocity,
    )
    if str(monitor_path).endswith("worker_00"):
        env = Gen2MonitorSnapshotWrapper(
            env, output_dir=snapshot_dir, snapshot_interval_episodes=50,
            run_name="ppo_27d_campaign",
        )
    return Monitor(env, filename=monitor_path)


def _make_vec(sources: Sequence[TrainingSource27d], run_dir: Path, seed: int):
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from marine_race_arena.learning.holoocean_capacity import capacity_snapshot

    capacity = capacity_snapshot()
    if int(capacity.get("available", 0)) < len(sources):
        raise RuntimeError(f"insufficient HoloOcean capacity: {capacity}")
    monitor_dir = run_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = run_dir / "monitor_snapshots"
    factories = [partial(
        _make_env, asdict(source), int(seed) + index * 10003,
        str(monitor_dir / f"worker_{index:02d}"), str(snapshot_dir),
    ) for index, source in enumerate(sources)]
    return SubprocVecEnv(factories, start_method="spawn"), capacity


def _callback_class():
    from stable_baselines3.common.callbacks import BaseCallback

    class Callback(BaseCallback):
        def __init__(self, run_dir: Path, thresholds: Sequence[int], verbose: int = 1):
            super().__init__(verbose=verbose)
            self.run_dir = run_dir
            self.thresholds = tuple(sorted(int(x) for x in thresholds))
            self.saved: set[int] = set()
            self.records: List[Dict[str, Any]] = []
            self.episodes: List[Dict[str, Any]] = []
            self.rewards: List[float] = []
            self.components: Dict[str, float] = {}
            self.vision = self.orientation = self.obs_steps = 0
            self.action_total = self.action_saturated = 0

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
                for key, value in (info.get("reward_components") or {}).items():
                    self.components[key] = self.components.get(key, 0.0) + float(value)
                if done:
                    self.episodes.append({
                        "timestep": int(self.num_timesteps),
                        "completed": bool(info.get("completed", False)),
                        "gates": int(info.get("gates_completed", 0)),
                        "status": str(info.get("status", "")),
                        "collision_events": int(info.get("collision_events", 0)) + int(info.get("obstacle_collision_events", 0)),
                        "out_of_bounds_events": int(info.get("out_of_bounds_events", 0)),
                        "wrong_direction_crossings": int(info.get("wrong_direction_crossings", 0)),
                        "missed_gate_attempts": int(info.get("missed_gate_attempts", 0)),
                    })
            self._checkpoint()
            return True

        def _metrics(self) -> Dict[str, Any]:
            logs = getattr(self.logger, "name_to_value", {}) if self.logger else {}
            n = max(1, len(self.episodes))
            return {
                "timestep": int(self.num_timesteps),
                "approx_kl": float(logs.get("train/approx_kl", 0.0)),
                "policy_loss": float(logs.get("train/policy_gradient_loss", 0.0)),
                "value_loss": float(logs.get("train/value_loss", 0.0)),
                "entropy": float(logs.get("train/entropy_loss", 0.0)),
                "clip_fraction": float(logs.get("train/clip_fraction", 0.0)),
                "learning_rate": float(logs.get("train/learning_rate", 0.0)),
                "mean_reward_last500": float(np.mean(self.rewards[-500:])) if self.rewards else 0.0,
                "completion_rate": float(sum(bool(x["completed"]) for x in self.episodes) / n),
                "mean_gates": float(sum(x["gates"] for x in self.episodes) / n),
                "safety_events": int(sum(x["collision_events"] + x["out_of_bounds_events"] + x["wrong_direction_crossings"] + x["missed_gate_attempts"] for x in self.episodes)),
                "vision_availability": self.vision / max(1, self.obs_steps),
                "orientation_availability": self.orientation / max(1, self.obs_steps),
                "action_saturation": self.action_saturated / max(1, self.action_total),
                "reward_components": dict(self.components),
                "episodes": len(self.episodes),
            }

        def _checkpoint(self) -> None:
            step = int(self.num_timesteps)
            for target in self.thresholds:
                if step >= target and target not in self.saved:
                    self.model.save(str(self.run_dir / f"checkpoint_{target:06d}.zip"))
                    self.saved.add(target)
            if not self.records or step - int(self.records[-1]["timestep"]) >= 512:
                m = self._metrics()
                self.records.append(m)
                _write(self.run_dir / "training_metrics.json", {"records": self.records, "episodes": self.episodes})
                if self.verbose:
                    print(json.dumps(m), flush=True)

        def _on_rollout_end(self) -> None:
            self._checkpoint()

        def _on_training_end(self) -> None:
            self._checkpoint()
            _write(self.run_dir / "training_metrics.json", {"records": self.records, "episodes": self.episodes})

    return Callback


def _audit_processes(path: Path, label: str) -> Dict[str, Any]:
    import subprocess
    rows: List[Dict[str, Any]] = []
    for p in __import__("subprocess").check_output(["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^(python|pythonw|Holodeck)(\\.exe)?$'} | ConvertTo-Json -Depth 5"], text=True, encoding="utf-8", errors="replace").splitlines():
        if p.strip():
            try:
                obj = json.loads(p)
                rows.extend(obj if isinstance(obj, list) else [obj])
            except json.JSONDecodeError:
                pass
    report = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "label": label, "processes": rows}
    _write(path, report)
    return report


def _matched_eval(parent: Path, candidate: Path, out: Path, seed: int) -> Dict[str, Any]:
    from sb3_contrib import RecurrentPPO
    from marine_race_arena.learning.gen2.evaluation import run_policy_episode
    from marine_race_arena.learning.gen2.recurrent_policy import Gen2RecurrentController

    rows = []
    for label, path in (("parent_warmstart_27d", parent), ("campaign_candidate", candidate)):
        model = RecurrentPPO.load(str(path), device="cpu")
        controller = Gen2RecurrentController(model, deterministic=True)
        for index, track in enumerate(tf.OFFICIAL_TRACKS):
            result = run_policy_episode(controller, tf.track_path(track), seed=int(seed + index), adapter="holoocean", allow_fallback=False, max_steps=7000)
            row = {"policy": label, "track": track, **result.as_row()}
            rows.append(row)
            print(json.dumps(row), flush=True)
        del controller, model
    report = {"rows": rows, "actual_adapter": "holoocean", "fallback_used": False, "fog": verify_fog_sources([tf.track_path(t) for t in tf.OFFICIAL_TRACKS])}
    _write(out, report)
    return report


def _clear_regression(report: Mapping[str, Any]) -> tuple[bool, str]:
    candidate = [x for x in report["rows"] if x["policy"] == "campaign_candidate"]
    if len(candidate) != 3:
        return True, "incomplete_evaluation"
    completed = sum(bool(x["succeeded"]) for x in candidate)
    gates = sum(int(x["gates_completed"]) for x in candidate)
    total = sum(int(x["gate_count"]) for x in candidate)
    max_kl = max(float(x) for x in report.get("training_kl", [0.0]))
    if completed < 2:
        return True, "completion_below_2_of_3"
    if total and gates / total < 0.85:
        return True, "gate_progress_below_85_percent"
    if max_kl > 0.02:
        return True, "kl_divergence"
    return False, "isolated_safety_events_allowed"


def run(parent: str | Path, out: str | Path, *, seed: int = 41001, max_steps: int = 200_000) -> Dict[str, Any]:
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.utils import get_schedule_fn
    from marine_race_arena.learning.train_ppo_transition import close_vec_env_safely

    parent_path, run_dir = Path(parent), Path(out)
    run_dir.mkdir(parents=True, exist_ok=True)
    parent_sha = sha256_file(parent_path)
    sources, source_audit = _sources(run_dir)
    if not source_audit["fog"]["all_valid"]:
        raise RuntimeError(source_audit["fog"])
    reward_check = assert_reward_hierarchy()
    config = {
        "parent": str(parent_path), "parent_sha256": parent_sha, "bc_policy_used": False,
        "candidate_5k_used": False, "observation_contract": OBS_ENCODING_VERSION_LOCAL_TRANSITION_27D,
        "observation_dim": 27, "max_timesteps": int(max_steps), "seed": int(seed),
        "sources": [asdict(s) for s in sources], "source_audit": source_audit,
        "reward": {"reliability_only": True, "efficiency_disabled": True, "collision_penalty": COLLISION_PENALTY, "entry_based": True},
        "ppo": {"learning_rate": 3e-6, "clip_range": 0.04, "target_kl": 0.004, "n_epochs": 1, "batch_size": 128, "n_steps": 256, "ent_coef": 0.0, "max_grad_norm": 0.5},
        "milestones": [x for x in MILESTONES if x <= max_steps], "actual_adapter": "holoocean", "fallback_used": False,
    }
    _write(run_dir / "campaign_config.json", config)
    _audit_processes(run_dir / "process_audit_start.json", "before_campaign")
    model = None
    env = None
    stages: List[Dict[str, Any]] = []
    current = 0
    try:
        env, capacity = _make_vec(sources, run_dir, seed)
        model = RecurrentPPO.load(str(parent_path), env=env, device="cpu")
        model.learning_rate = 3e-6
        model.lr_schedule = get_schedule_fn(3e-6)
        for group in model.policy.optimizer.param_groups:
            group["lr"] = 3e-6
        model.clip_range = get_schedule_fn(0.04)
        model.target_kl = 0.004
        model.n_epochs = 1
        model.max_grad_norm = 0.5
        model.ent_coef = 0.0
        if not all(p.requires_grad for p in model.policy.parameters()):
            raise RuntimeError("frozen parameter detected")
        _write(run_dir / "start_audit.json", {"parent_sha256": parent_sha, "parameter_count": policy_parameter_count(model), "all_parameters_trainable": True, "capacity": capacity})
        Callback = _callback_class()
        for target in [x for x in MILESTONES if x <= max_steps]:
            chunk = target - current
            if chunk <= 0:
                continue
            callback = Callback(run_dir, thresholds=[target], verbose=1)
            model.learn(total_timesteps=chunk, callback=callback, reset_num_timesteps=False, progress_bar=False)
            current = int(model.num_timesteps)
            candidate = run_dir / f"candidate_{target // 1000}k.zip"
            model.save(str(candidate))
            shutil.copy2(candidate, run_dir / "latest.zip")
            # The parent must remain byte-identical throughout the campaign.
            if sha256_file(parent_path) != parent_sha:
                raise RuntimeError("parent checkpoint changed")
            # Close all workers before real HoloOcean evaluation.
            close_vec_env_safely(env)
            env = None
            eval_report = _matched_eval(parent_path, candidate, run_dir / f"evaluation_{target // 1000}k.json", seed + target)
            # Attach current training KL diagnostics for the decision record.
            recent = callback.records[-4:]
            eval_report["training_kl"] = [float(x.get("approx_kl", 0.0)) for x in recent]
            _write(run_dir / f"evaluation_{target // 1000}k.json", eval_report)
            candidate_rows = [x for x in eval_report["rows"] if x["policy"] == "campaign_candidate"]
            stage = {"target": target, "actual_timesteps": current, "checkpoint": str(candidate), "sha256": sha256_file(candidate), "evaluation": eval_report, "training_tail": recent}
            bad, reason = _clear_regression(eval_report)
            stage["rollback_triggered"] = bad
            stage["decision"] = reason
            stages.append(stage)
            _write(run_dir / "training_decision_log.json", {"stages": stages, "incumbent": str(parent_path), "incumbent_sha256": parent_sha})
            _audit_processes(run_dir / f"process_audit_{target // 1000}k.json", f"after_{target}")
            if bad:
                break
            if target >= max_steps:
                break
            # Recreate workers and continue from the in-memory model.  This
            # keeps optimizer/LSTM weights while ensuring every phase cleans up.
            env, capacity = _make_vec(sources, run_dir, seed + target)
            model.set_env(env)
        _write(run_dir / "ppo_training_summary.json", {"stages": stages, "actual_timesteps": current, "parent_sha256": parent_sha, "final_candidate": stages[-1]["checkpoint"] if stages else None, "continued_beyond_5k": True})
        return {"stages": stages, "actual_timesteps": current}
    finally:
        if env is not None:
            try:
                close_vec_env_safely(env)
            except Exception:
                try:
                    env.close()
                except Exception:
                    pass
        _write(run_dir / "environment_close.json", {"closed": True, "method": "finally"})
        _audit_processes(run_dir / "process_audit_end.json", "after_campaign")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=41001)
    ap.add_argument("--max-steps", type=int, default=200_000)
    args = ap.parse_args()
    print(json.dumps(run(args.parent, args.out, seed=args.seed, max_steps=args.max_steps), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
