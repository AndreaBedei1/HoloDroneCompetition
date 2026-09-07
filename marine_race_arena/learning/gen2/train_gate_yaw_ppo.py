"""Conservative Gen-2 gate-yaw PPO smoke run.

The command is deliberately gated by a successful no-learning HoloOcean
preflight.  It expands the immutable 35-D parent to 38 inputs, proves exact
recurrent parity, trains only on the three official circuits and exact
contiguous fragments, evaluates parent and child on matched seeds, then exits.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from marine_race_arena.config.benchmark_tasks import BENCHMARK_TASK_CLEAN_GATE
from marine_race_arena.learning.config_local_transition_gate_yaw import (
    OBS_ENCODING_VERSION_LOCAL_TRANSITION_GATE_YAW,
)
from marine_race_arena.learning.gen2 import track_fragments as tf
from marine_race_arena.learning.gen2.fog_contract import verify_fog_sources
from marine_race_arena.learning.gen2.gate_yaw_transfer import (
    sha256_file,
    transfer_parent_to_gate_yaw,
    verify_expanded_recurrent_parity,
    verify_transfer_tensors,
)
from marine_race_arena.learning.gen2.ppo_reliability import (
    PPOReliabilityConfig,
    ReliabilityReward,
    assert_reward_hierarchy,
)
from marine_race_arena.learning.gen2.recurrent_policy import write_policy_manifest
from marine_race_arena.learning.gym_env import MarineRaceGymEnv


SMOKE_MAX_TIMESTEPS = 10_000
DEFAULT_SMOKE_TIMESTEPS = 6_144
SAFE_ENV_COUNT = 6


@dataclass(frozen=True)
class TrainingSource:
    name: str
    track: str
    kind: str
    path: str
    gate_count: int
    max_steps: int
    initial_body_velocity: Optional[tuple[float, float, float]] = None


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_suffix(path.suffix + ".partial")
    partial_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    partial_path.replace(path)


def _load_preflight(path: str | Path) -> Dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if not report.get("pass"):
        raise RuntimeError(f"preflight did not pass: {path}")
    if int(report.get("learning_updates", -1)) != 0:
        raise RuntimeError("preflight must contain zero learning updates")
    names = {row.get("track") for row in report.get("tracks", []) if row.get("pass")}
    if names != set(tf.OFFICIAL_TRACKS):
        raise RuntimeError(f"preflight does not cover all official tracks: {sorted(names)}")
    return report


def _pick_fragment(track: str, target_gate_1based: int) -> tf.TrackFragment:
    target = int(target_gate_1based) - 1
    candidates = [
        fragment
        for fragment in tf.enumerate_fragments(track, lengths=(3, 4))
        if fragment.start_index < target <= fragment.end_index
    ]
    if not candidates:
        raise RuntimeError(f"no exact fragment straddles {track} gate {target_gate_1based}")
    return min(
        candidates,
        key=lambda fragment: (
            fragment.length,
            abs(fragment.end_index - target),
            fragment.start_index,
        ),
    )


def materialize_training_sources(run_dir: str | Path) -> tuple[List[TrainingSource], Dict[str, Any]]:
    """Three full circuits plus one measured weak transition per circuit."""

    run_dir = Path(run_dir)
    sources: List[TrainingSource] = []
    source_paths: List[Path] = []
    geometry_reports: List[Dict[str, Any]] = []
    for track in tf.OFFICIAL_TRACKS:
        data = tf.load_track(track)
        path = tf.track_path(track)
        gate_count = len(data["track"]["gate_sequence"])
        source_paths.append(path)
        sources.append(
            TrainingSource(
                name=f"full_{track}",
                track=track,
                kind="full_circuit",
                path=str(path),
                gate_count=gate_count,
                max_steps=max(2000, gate_count * 900),
            )
        )

    # Last fixed-pipeline evaluation: H9, V10 and M4 were representative
    # observed failures.  These are selection metadata only, never features.
    weak_gate = {
        "horseshoe_bay": 9,
        "vertical_serpent": 10,
        "mixed_endurance": 4,
    }
    fragment_dir = run_dir / "training_tracks"
    for track in tf.OFFICIAL_TRACKS:
        fragment = _pick_fragment(track, weak_gate[track])
        path = tf.materialize_fragment(fragment, fragment_dir / f"{fragment.name}.json")
        geometry = tf.verify_geometry_preserved(fragment, path)
        if not geometry["identical"] or geometry["dangling_links"] or not geometry["bounds_inherited"]:
            raise RuntimeError(f"fragment geometry check failed: {geometry}")
        source_paths.append(path)
        geometry_reports.append(geometry)
        sources.append(
            TrainingSource(
                name=fragment.name,
                track=track,
                kind="exact_fragment",
                path=str(path),
                gate_count=fragment.length,
                max_steps=max(600, fragment.length * 900),
                initial_body_velocity=tf.inbound_body_velocity(fragment),
            )
        )

    if len(sources) != SAFE_ENV_COUNT:
        raise AssertionError("the fixed smoke mixture must have exactly six environments")
    fog = verify_fog_sources(source_paths)
    return sources, {"fog": fog, "geometry": geometry_reports}


def _make_source_env(source_payload: Mapping[str, Any], seed: int, monitor_path: str):
    from stable_baselines3.common.monitor import Monitor

    source = TrainingSource(**dict(source_payload))
    reward = ReliabilityReward(source.gate_count)
    env = MarineRaceGymEnv(
        source.path,
        seed=int(seed),
        adapter="holoocean",
        allow_fallback=False,
        max_steps=int(source.max_steps),
        official=True,
        current_profile="none",
        benchmark_task=BENCHMARK_TASK_CLEAN_GATE,
        reward_fn=reward,
        observation_encoding_version=OBS_ENCODING_VERSION_LOCAL_TRANSITION_GATE_YAW,
        initial_body_velocity=source.initial_body_velocity,
    )
    return Monitor(env, filename=monitor_path)


def _make_vec_env(sources: Sequence[TrainingSource], run_dir: Path, seed: int):
    from stable_baselines3.common.vec_env import SubprocVecEnv

    try:
        from marine_race_arena.learning.holoocean_capacity import capacity_snapshot

        capacity = capacity_snapshot()
        if int(capacity.get("available", 0)) < len(sources):
            raise RuntimeError(
                f"need {len(sources)} HoloOcean slots for the fixed safe mixture; "
                f"capacity={capacity}"
            )
    except ImportError:
        capacity = {"available": "unknown"}
    monitor_dir = run_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    factories = [
        partial(
            _make_source_env,
            asdict(source),
            int(seed) + index * 10_003,
            str(monitor_dir / f"worker_{index:02d}"),
        )
        for index, source in enumerate(sources)
    ]
    return SubprocVecEnv(factories, start_method="spawn"), capacity


def _close_vec_env(env, timeout_s: float = 60.0) -> Dict[str, Any]:
    from marine_race_arena.learning.train_ppo_transition import close_vec_env_safely

    return close_vec_env_safely(env, timeout=timeout_s)


def _evaluate_one(job: Mapping[str, Any]) -> Dict[str, Any]:
    from marine_race_arena.learning.gen2.track_eval import (
        evaluate_circuit,
        summarize_circuit,
    )

    checkpoint = str(job["checkpoint"])
    track = str(job["track"])
    rows = evaluate_circuit(
        checkpoint,
        track,
        trials=int(job.get("trials", 1)),
        adapter="holoocean",
        allow_fallback=False,
    )
    return {
        "policy": str(job["policy"]),
        "track": track,
        "summary": summarize_circuit(rows),
        "rows": [row.as_dict() for row in rows],
    }


def evaluate_matched(
    parent: str | Path,
    candidate: str | Path,
    *,
    trials: int,
    workers: int,
) -> Dict[str, Any]:
    jobs = [
        {"policy": policy, "checkpoint": str(checkpoint), "track": track, "trials": trials}
        for policy, checkpoint in (("parent", parent), ("candidate", candidate))
        for track in tf.OFFICIAL_TRACKS
    ]
    effective = max(1, min(int(workers), len(jobs)))
    if effective == 1:
        results = [_evaluate_one(job) for job in jobs]
    else:
        with mp.get_context("spawn").Pool(processes=effective) as pool:
            results = pool.map(_evaluate_one, jobs)
    by_policy: Dict[str, Dict[str, Any]] = {"parent": {}, "candidate": {}}
    rows: Dict[str, List[Dict[str, Any]]] = {"parent": [], "candidate": []}
    for result in results:
        by_policy[result["policy"]][result["track"]] = result["summary"]
        rows[result["policy"]].extend(result["rows"])
    comparison = {}
    for track in tf.OFFICIAL_TRACKS:
        old = by_policy["parent"][track]
        new = by_policy["candidate"][track]
        comparison[track] = {
            "completion": f"{old['completed']}/{old['trials']} -> {new['completed']}/{new['trials']}",
            "mean_gates": f"{old['mean_gates_completed']} -> {new['mean_gates_completed']}",
            "collision_episodes": f"{old['collision_episodes']} -> {new['collision_episodes']}",
            "out_of_bounds_episodes": f"{old['out_of_bounds_episodes']} -> {new['out_of_bounds_episodes']}",
            "wrong_direction_episodes": f"{old['wrong_direction_episodes']} -> {new['wrong_direction_episodes']}",
        }
    return {
        "matched_seed_protocol": True,
        "trials_per_track": int(trials),
        "workers": effective,
        "summaries": by_policy,
        "rows": rows,
        "comparison": comparison,
        "statistical_significance_claimed": False,
    }


def _new_column_stats(model) -> Dict[str, Any]:
    state = model.policy.state_dict()
    tensors = {
        key: value[:, 35:].detach().cpu().numpy()
        for key, value in state.items()
        if key.endswith("encoder.0.weight")
    }
    combined = np.concatenate([value.reshape(-1) for value in tensors.values()])
    return {
        "keys": list(tensors),
        "nonzero": int(np.count_nonzero(combined)),
        "values": int(combined.size),
        "max_abs": float(np.max(np.abs(combined))),
        "l2": float(np.linalg.norm(combined)),
    }


def run(
    *,
    parent_path: str | Path,
    preflight_path: str | Path,
    run_dir: str | Path,
    total_timesteps: int = DEFAULT_SMOKE_TIMESTEPS,
    seed: int = 8400,
    eval_trials: int = 1,
    eval_workers: int = SAFE_ENV_COUNT,
    authorize_long: bool = False,
) -> Dict[str, Any]:
    total_timesteps = int(total_timesteps)
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if total_timesteps > SMOKE_MAX_TIMESTEPS and not authorize_long:
        raise RuntimeError(
            f"refusing {total_timesteps} steps without --authorize-long; "
            f"smoke ceiling is {SMOKE_MAX_TIMESTEPS}"
        )
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    parent_path = Path(parent_path)
    parent_hash_before = sha256_file(parent_path)
    preflight = _load_preflight(preflight_path)
    reward_check = assert_reward_hierarchy()
    if not reward_check["all_hold"]:
        raise RuntimeError(f"reward hierarchy failed: {reward_check}")

    sources, source_audit = materialize_training_sources(run_dir)
    for source in source_audit["fog"]["sources"]:
        print(f"[smoke] source fog {source['track_path']}: {source['water_fog']}", flush=True)
    print(f"[smoke] fixed concurrency={SAFE_ENV_COUNT}; steps={total_timesteps}", flush=True)

    config = PPOReliabilityConfig(
        total_timesteps=total_timesteps,
        n_steps=256,
        batch_size=128,
        n_epochs=2,
        learning_rate=3e-5,
        clip_range=0.08,
        target_kl=0.01,
        checkpoint_every=1536,
        seed=int(seed),
        full_circuit_fraction=0.5,
    )
    run_config = {
        "schema_version": "gen2_gate_yaw_ppo_smoke_v1",
        "parent": str(parent_path),
        "parent_sha256": parent_hash_before,
        "preflight": str(preflight_path),
        "observation_contract": OBS_ENCODING_VERSION_LOCAL_TRANSITION_GATE_YAW,
        "observation_dim": 38,
        "new_features": [
            "gate_orientation_present",
            "gate_yaw_sin",
            "gate_yaw_cos",
        ],
        "ppo": config.as_dict(),
        "sources": [asdict(source) for source in sources],
        "source_audit": source_audit,
        "reward_hierarchy": reward_check,
        "privileged_reward_isolation": (
            "reward may score referee/simulator events; only the separately "
            "encoded 38-D onboard vector enters policy/recurrent state"
        ),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _atomic_json(run_dir / "run_config.json", run_config)

    env = None
    close_report: Dict[str, Any] = {}
    try:
        env, capacity = _make_vec_env(sources, run_dir, int(seed))
        model = transfer_parent_to_gate_yaw(
            parent_path,
            env=env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            target_kl=config.target_kl,
            seed=config.seed,
            device=config.device,
        )
        tensor_report = verify_transfer_tensors(parent_path, model)
        parity_report = verify_expanded_recurrent_parity(parent_path, model)
        if not tensor_report["exact"] or not parity_report["bit_exact"]:
            raise RuntimeError(
                f"warm-start preflight failed: tensors={tensor_report}, parity={parity_report}"
            )
        initial_path = run_dir / "initialized_38d_policy.zip"
        model.save(initial_path)
        initial_state = {
            key: value.detach().cpu().clone()
            for key, value in model.policy.state_dict().items()
        }
        _atomic_json(
            run_dir / "transfer_report.json",
            {
                "tensors": tensor_report,
                "recurrent_parity": parity_report,
                "capacity_at_start": capacity,
                "initialized_checkpoint": str(initial_path),
            },
        )

        from stable_baselines3.common.callbacks import CheckpointCallback

        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_freq_calls = max(1, config.checkpoint_every // len(sources))
        callback = CheckpointCallback(
            save_freq=save_freq_calls,
            save_path=str(checkpoint_dir),
            name_prefix="gate_yaw_smoke",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=True,
            progress_bar=False,
        )
        candidate_path = run_dir / "smoke_candidate.zip"
        model.save(candidate_path)
        final_state = model.policy.state_dict()
        changed_tensors = sum(
            not np.array_equal(
                initial_state[key].numpy(), final_state[key].detach().cpu().numpy()
            )
            for key in initial_state
        )
        update_report = {
            "requested_timesteps": total_timesteps,
            "actual_timesteps": int(model.num_timesteps),
            "ppo_updates": int(getattr(model, "_n_updates", 0)),
            "changed_policy_tensors": int(changed_tensors),
            "new_input_weights": _new_column_stats(model),
            "candidate": str(candidate_path),
            "candidate_sha256": sha256_file(candidate_path),
            "checkpoints": [str(path) for path in sorted(checkpoint_dir.glob("*.zip"))],
        }
        if update_report["ppo_updates"] <= 0 or update_report["changed_policy_tensors"] <= 0:
            raise RuntimeError(f"PPO did not update the policy: {update_report}")
        if update_report["new_input_weights"]["nonzero"] <= 0:
            raise RuntimeError(f"new yaw columns did not learn: {update_report}")
        _atomic_json(run_dir / "training_update_report.json", update_report)
        write_policy_manifest(
            model,
            run_dir / "smoke_candidate.manifest.json",
            parent=str(parent_path),
            parent_sha256=parent_hash_before,
            transfer=tensor_report,
            parity_before_optimization=parity_report,
            smoke=update_report,
        )
    finally:
        if env is not None:
            close_report = _close_vec_env(env)
            _atomic_json(run_dir / "environment_close.json", close_report)

    if sha256_file(parent_path) != parent_hash_before:
        raise RuntimeError("immutable parent checkpoint changed during the smoke run")

    evaluation = evaluate_matched(
        parent_path,
        candidate_path,
        trials=int(eval_trials),
        workers=int(eval_workers),
    )
    evaluation["fog"] = verify_fog_sources(
        [tf.track_path(track) for track in tf.OFFICIAL_TRACKS]
    )
    _atomic_json(run_dir / "matched_evaluation.json", evaluation)
    final_report = {
        "schema_version": "gen2_gate_yaw_ppo_smoke_result_v1",
        "run_dir": str(run_dir),
        "parent_immutable": True,
        "parent_sha256": parent_hash_before,
        "preflight_passed": bool(preflight["pass"]),
        "transfer": tensor_report,
        "parity": parity_report,
        "training": update_report,
        "environment_close": close_report,
        "evaluation": evaluation,
        "long_campaign_started": False,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _atomic_json(run_dir / "smoke_report.json", final_report)
    return final_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate the short 38-D PPO smoke")
    parser.add_argument("--parent", default="results/rl/gen2/best/best_completion_policy.zip")
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=DEFAULT_SMOKE_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=8400)
    parser.add_argument("--eval-trials", type=int, default=1)
    parser.add_argument("--eval-workers", type=int, default=SAFE_ENV_COUNT)
    parser.add_argument(
        "--authorize-long",
        action="store_true",
        help="required for more than 10k steps; use only after explicit human authorization",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(
        parent_path=args.parent,
        preflight_path=args.preflight,
        run_dir=args.out,
        total_timesteps=args.steps,
        seed=args.seed,
        eval_trials=args.eval_trials,
        eval_workers=args.eval_workers,
        authorize_long=args.authorize_long,
    )
    print(
        json.dumps(
            {
                "report": str(Path(args.out) / "smoke_report.json"),
                "actual_timesteps": report["training"]["actual_timesteps"],
                "long_campaign_started": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
