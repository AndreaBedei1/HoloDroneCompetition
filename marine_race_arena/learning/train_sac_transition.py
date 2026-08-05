"""Atomic independent multi-step SAC training for universal gate transitions."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from marine_race_arena.learning.config import ACTION_CONTRACT_VERSION
from marine_race_arena.learning.config_local_transition import (
    OBS_DIM_LOCAL_TRANSITION,
    OBS_ENCODING_VERSION_LOCAL_TRANSITION,
)
from marine_race_arena.learning.holoocean_capacity import (
    assert_unique_holoocean_uuids,
    reserve_holoocean_engines,
)
from marine_race_arena.learning.longrun_checkpoint import (
    atomic_append_jsonl,
    atomic_write_json,
    canonical_hash,
    capture_rng_state,
    restore_rng_state,
    sha256_file,
)
from marine_race_arena.learning.provenance import git_sha, now_utc
from marine_race_arena.learning.sac_replay_buffer import (
    DEFAULT_BATCH_COMPOSITION,
    MultiWorkerNStepAccumulator,
    StratifiedReplayBuffer,
    classify_replay_event,
    is_legal_bootstrap_truncation,
)
from marine_race_arena.learning.sac_transition_checkpoint import (
    atomic_save_sac_checkpoint,
    latest_valid_sac_checkpoint,
    load_sac_checkpoint,
)
from marine_race_arena.learning.sac_transition_policy import (
    SACTransitionAgent,
    transfer_ppo_mean_actor,
)
from marine_race_arena.learning.train_ppo_transition import (
    _dedicated_benchmark_due,
    _evaluation_summary,
    _next_evaluation_transition,
    apply_evaluation_selection,
)
from marine_race_arena.learning.transition_curriculum import (
    TransitionCurriculumController,
)
from marine_race_arena.learning.transition_env import UniversalTransitionEnv
from marine_race_arena.learning.transition_evaluation import (
    evaluate_checkpoint_universal_transition_benchmark,
    transition_checkpoint_rank_key,
)
from marine_race_arena.learning.transition_selection import (
    evaluate_competence_gate,
    thresholds_from_mapping,
)


SAC_BRANCH = "feature/rl-multistep-sac-transition"


def _load_config(path: str | Path) -> Dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "run_name", "output_root", "observation_version", "action_version",
        "initialization", "seed", "worker_seed_base", "new_environment_steps",
        "n_envs", "adapter", "allow_fallback", "holoocean_frames_per_sec",
        "max_episode_steps", "sac", "curriculum", "evaluation",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"SAC transition config missing {missing}")
    if config["observation_version"] != OBS_ENCODING_VERSION_LOCAL_TRANSITION:
        raise ValueError("SAC observation contract changed")
    if config["action_version"] != ACTION_CONTRACT_VERSION:
        raise ValueError("SAC action contract changed")
    if config["adapter"] == "holoocean" and bool(config["allow_fallback"]):
        raise ValueError("real SAC training cannot enable fallback")
    sac = config["sac"]
    defaults = {
        "n_step": 3,
        "gamma": 0.995,
        "tau": 0.005,
        "batch_size": 512,
        "replay_capacity": 1_000_000,
        "learning_starts": 10_000,
        "automatic_entropy": True,
        "target_entropy": -4.0,
    }
    for key, expected in defaults.items():
        if key not in sac:
            raise ValueError(f"SAC config missing {key}")
        if key == "automatic_entropy":
            if bool(sac[key]) is not expected:
                raise ValueError("automatic SAC entropy tuning is required")
        elif not np.isclose(float(sac[key]), float(expected)):
            raise ValueError(f"initial SAC engineering default changed: {key}")
    if int(config["n_envs"]) < 1:
        raise ValueError("SAC n_envs must be positive")
    thresholds_from_mapping(config.get("competence_gate"))
    return config


def _resolve_source_checkpoint(declared: str | Path) -> Path:
    source = Path(declared)
    if source.exists():
        return source.resolve()
    override = os.environ.get("HOLODRONE_PPO_WORKTREE")
    candidates = []
    if override:
        candidates.append(Path(override) / source)
    # The requested isolated worktree is a sibling of the live PPO worktree.
    if Path.cwd().name.endswith("-sac"):
        candidates.append(Path.cwd().with_name(Path.cwd().name[:-4]) / source)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(declared)


def _contract(config: Mapping[str, Any]) -> str:
    stable = dict(config)
    stable.pop("new_environment_steps", None)
    return canonical_hash({"sac_transition_config": stable, "training_code_sha": git_sha()})


def _run_contract(config: Mapping[str, Any], run_dir: Path) -> str:
    manifest = run_dir / "run_manifest.json"
    if manifest.exists():
        value = json.loads(manifest.read_text(encoding="utf-8"))
        stored = value.get("config_contract_sha256")
        if isinstance(stored, str) and len(stored) == 64:
            return stored
    return _contract(config)


def _preflight(config: Mapping[str, Any], *, allow_dirty: bool) -> Dict[str, Any]:
    branch = subprocess.run(
        ["git", "branch", "--show-current"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if branch != SAC_BRANCH:
        raise ValueError(f"wrong SAC branch {branch!r}")
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], check=True,
        capture_output=True, text=True,
    ).stdout.strip())
    if dirty and not allow_dirty:
        raise ValueError("commit and push SAC implementation before a non-smoke run")
    initialization = dict(config["initialization"])
    if initialization.get("mode") != "ppo_actor_mean_warm_start":
        raise ValueError("SAC must start from the common competent PPO actor")
    source = _resolve_source_checkpoint(initialization["source_checkpoint"])
    digest = sha256_file(source)
    if digest != initialization.get("source_sha256"):
        raise ValueError("SAC PPO actor source hash mismatch")
    return {
        "checked_utc": now_utc(),
        "branch": branch,
        "git_sha": git_sha(),
        "worktree_dirty": dirty,
        "adapter": config["adapter"],
        "fallback_disabled": not bool(config["allow_fallback"]),
        "observation_version": OBS_ENCODING_VERSION_LOCAL_TRANSITION,
        "observation_dim": OBS_DIM_LOCAL_TRANSITION,
        "action_version": ACTION_CONTRACT_VERSION,
        "n_envs": int(config["n_envs"]),
        "initialization_mode": initialization["mode"],
        "declared_source_checkpoint": initialization["source_checkpoint"],
        "resolved_source_checkpoint": str(source),
        "source_sha256": digest,
    }


def _make_worker(
    *, run_dir: str, worker_id: int, sampler_seed: int, difficulty: str,
    transition_focus_fraction: float, adapter: str, allow_fallback: bool,
    frames_per_sec: bool | int, max_episode_steps: int,
    reward_config: Mapping[str, Any],
):
    return UniversalTransitionEnv(
        run_dir=run_dir,
        worker_id=worker_id,
        sampler_seed=sampler_seed,
        difficulty=difficulty,
        transition_focus_fraction=transition_focus_fraction,
        adapter=adapter,
        allow_fallback=allow_fallback,
        frames_per_sec=frames_per_sec,
        max_episode_steps=max_episode_steps,
        reward_config=reward_config,
    )


def _make_vec_env(config: Mapping[str, Any], run_dir: Path, difficulty: str):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    n_envs = int(config["n_envs"])
    factories = [
        partial(
            _make_worker,
            run_dir=str(run_dir),
            worker_id=index,
            sampler_seed=int(config["worker_seed_base"]) + index * 10_003,
            difficulty=difficulty,
            transition_focus_fraction=float(config["curriculum"]["transition_focus_fraction"]),
            adapter=str(config["adapter"]),
            allow_fallback=bool(config["allow_fallback"]),
            frames_per_sec=config["holoocean_frames_per_sec"],
            max_episode_steps=int(config["max_episode_steps"]),
            reward_config=dict(config.get("reward") or {}),
        )
        for index in range(n_envs)
    ]
    reservation = (
        reserve_holoocean_engines(n_envs, owner=f"SAC rollout {run_dir}")
        if config["adapter"] == "holoocean"
        else _null_reservation()
    )
    with reservation:
        env = DummyVecEnv(factories) if n_envs == 1 else SubprocVecEnv(
            factories, start_method="spawn"
        )
    return env


class _null_reservation:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _worker_states(env: Any) -> list[Mapping[str, Any]]:
    return list(env.env_method("worker_state"))


def _worker_identities(env: Any) -> list[Mapping[str, Any]]:
    return list(env.env_method("worker_identity"))


def _validate_initialized_worker_identities(
    env: Any, config: Mapping[str, Any]
) -> None:
    """Validate engine ownership only after reset has launched HoloOcean.

    UniversalTransitionEnv intentionally creates its adapter lazily on the
    first reset.  Querying identities in ``_make_vec_env`` therefore observes
    a legitimate empty UUID and rejects every real worker before it can start.
    """

    if config["adapter"] == "holoocean":
        assert_unique_holoocean_uuids(_worker_identities(env))


def _restore_workers(env: Any, states: Sequence[Mapping[str, Any]]) -> None:
    if len(states) != int(env.num_envs):
        raise ValueError("SAC worker count changed on resume")
    for index, state in enumerate(states):
        env.env_method("load_worker_state", state, indices=index)


def _preserve_baseline(
    run_dir: Path,
    config: Mapping[str, Any],
    agent: SACTransitionAgent,
) -> Dict[str, Any]:
    import torch

    directory = run_dir / "baseline"
    record_path = directory / "initialization_baseline.json"
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        for path_key, hash_key in (
            ("copied_source_checkpoint", "source_ppo_sha256"),
            ("initial_sac_actor", "initial_sac_actor_sha256"),
        ):
            path = Path(record[path_key])
            if not path.exists() or sha256_file(path) != record[hash_key]:
                raise OSError(f"immutable SAC baseline failed integrity check: {path}")
        return record
    directory.mkdir(parents=True, exist_ok=True)
    source = _resolve_source_checkpoint(config["initialization"]["source_checkpoint"])
    source_target = directory / "source_ppo_actor_checkpoint.zip"
    source_tmp = directory / ".source_ppo_actor_checkpoint.partial.zip"
    shutil.copyfile(source, source_tmp)
    if sha256_file(source_tmp) != config["initialization"]["source_sha256"]:
        source_tmp.unlink(missing_ok=True)
        raise OSError("immutable SAC source copy hash mismatch")
    os.replace(source_tmp, source_target)
    actor_target = directory / "initial_sac_actor.pt"
    actor_tmp = directory / ".initial_sac_actor.partial.pt"
    torch.save(agent.actor.state_dict(), actor_tmp)
    os.replace(actor_tmp, actor_target)
    record = {
        "schema_version": "sac_transition_initialization_baseline_v1",
        "created_utc": now_utc(),
        "source_ppo_checkpoint": str(source),
        "source_ppo_sha256": sha256_file(source_target),
        "copied_source_checkpoint": str(source_target),
        "initial_sac_actor": str(actor_target),
        "initial_sac_actor_sha256": sha256_file(actor_target),
        "baseline_checkpoint": str(actor_target),
        "immutable": True,
    }
    atomic_write_json(record_path, record)
    return record


def _initial_selection() -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    aliases = {
        "last": None,
        "latest_competent": None,
        "latest_safe_competent": None,
        "best_universal_transition": None,
        "best_long_sequence": None,
    }
    selection = {
        "best_metrics": None,
        "best_timestep": None,
        "last_evaluation_metrics": None,
        "last_competence_verdict": None,
        "best_long_sequence_score": None,
        "consecutive_baseline_collapses": 0,
        "dedicated": {"last_timesteps": 0, "best_success_rate": None, "history": []},
        "rollback": None,
    }
    evaluation = {"history": [], "last": None, "best": None}
    return aliases, selection, evaluation


def _save(
    *, agent: SACTransitionAgent, replay: StratifiedReplayBuffer,
    accumulator: MultiWorkerNStepAccumulator, run_dir: Path, contract: str,
    total_transitions: int, curriculum: TransitionCurriculumController, env: Any,
    evaluation_state: Mapping[str, Any], aliases: Dict[str, Any],
    selection: Mapping[str, Any], config: Mapping[str, Any], reason: str,
    status: str = "unverified", stop_reason: Optional[str] = None,
):
    path = run_dir / "checkpoints" / f"sac_{int(total_transitions)}_steps.pt"
    aliases["last"] = str(path)
    return atomic_save_sac_checkpoint(
        agent, replay, run_dir,
        total_environment_transitions=int(total_transitions),
        config_contract_sha256=contract,
        curriculum_state=curriculum.state_dict(_worker_states(env)),
        n_step_state=accumulator.state_dict(),
        evaluation_state=evaluation_state,
        aliases=aliases,
        selection_state=selection,
        initialization=config["initialization"],
        worker_identities=_worker_identities(env),
        reason=reason,
        status=status,
        stop_reason=stop_reason,
    )


def _status(
    run_dir: Path, *, state: str, pid: int, total_transitions: int,
    target: int, agent: SACTransitionAgent, replay: StratifiedReplayBuffer,
    accumulator: MultiWorkerNStepAccumulator,
    curriculum: TransitionCurriculumController, env: Any,
    aliases: Mapping[str, Any], selection: Mapping[str, Any],
    config: Mapping[str, Any], last_losses: Optional[Mapping[str, float]],
    message: str,
) -> Dict[str, Any]:
    value = {
        "updated_utc": now_utc(),
        "state": state,
        "pid": int(pid),
        "total_environment_transitions": int(total_transitions),
        "target_environment_transitions": int(target),
        "gradient_updates": int(agent.gradient_updates),
        "entropy_coefficient": float(agent.alpha),
        "last_losses": None if last_losses is None else dict(last_losses),
        "replay_size": int(replay.size),
        "replay_capacity": int(replay.capacity),
        "replay_sampling": replay.sampling_report(),
        "pending_n_step_transitions": [len(queue) for queue in accumulator.queues],
        "observation_version": OBS_ENCODING_VERSION_LOCAL_TRANSITION,
        "observation_dim": OBS_DIM_LOCAL_TRANSITION,
        "action_version": ACTION_CONTRACT_VERSION,
        "architecture": "multistep_sac_twin_q",
        "n_step": int(config["sac"]["n_step"]),
        "n_envs": int(config["n_envs"]),
        "difficulty": curriculum.difficulty,
        "difficulty_entry_transitions": curriculum.state.difficulty_entry_transitions,
        "evaluation_history_count": len(curriculum.state.evaluation_history),
        "checkpoint_aliases": dict(aliases),
        "checkpoint_selection_state": dict(selection),
        "worker_identities": _worker_identities(env),
        "initialization": dict(config["initialization"]),
        "all_actions_policy_generated": True,
        "hybrid_or_expert_actions_active": False,
        "ppo_replay_imported": False,
        "message": message,
    }
    atomic_write_json(run_dir / "status.json", value)
    return value


def _run_dedicated(
    checkpoint: Path, run_dir: Path, *, config: Mapping[str, Any],
    difficulty: str, timesteps: int, reason: str,
    selection: Dict[str, Any],
) -> Dict[str, Any]:
    evaluation = config["evaluation"]
    output = run_dir / "evaluations" / f"dedicated_{timesteps:09d}"
    report = evaluate_checkpoint_universal_transition_benchmark(
        checkpoint,
        output_dir=output,
        seed=int(evaluation.get("dedicated_seed", evaluation["seed"])),
        difficulty=difficulty,
        transition_cases=int(evaluation["dedicated_transition_cases"]),
        full_cases_per_length=int(evaluation.get("dedicated_full_cases_per_length", 2)),
        adapter=str(config["adapter"]),
        frames_per_sec=config["holoocean_frames_per_sec"],
        max_steps=int(config["max_episode_steps"]),
        parallel_workers=int(evaluation.get("dedicated_parallel_workers", 2)),
        algorithm="sac",
    )
    metrics = report["metrics"]
    success = float(metrics.get("universal_transition_success_rate", 0.0) or 0.0)
    dedicated = selection["dedicated"]
    dedicated["last_timesteps"] = timesteps
    if dedicated.get("best_success_rate") is None or success > float(dedicated["best_success_rate"]):
        dedicated["best_success_rate"] = success
    dedicated["history"].append({
        "timesteps": timesteps, "reason": reason, "output_dir": str(output),
        "universal_transition_success_rate": success,
    })
    return {
        "reason": reason, "output_dir": str(output),
        "transition_cases": int(report.get("transition_cases", 0)),
        "metrics": metrics,
    }


def _add_interrupted_queues(
    accumulator: MultiWorkerNStepAccumulator,
    replay: StratifiedReplayBuffer,
) -> int:
    rows = accumulator.flush_all(bootstrap_allowed=False)
    replay.extend(rows)
    return len(rows)


def run_training(args: argparse.Namespace) -> int:
    import torch

    config = _load_config(args.config)
    if args.steps is not None:
        config["new_environment_steps"] = int(args.steps)
    if args.n_envs is not None:
        config["n_envs"] = int(args.n_envs)
    run_dir = Path(args.run_dir) if args.run_dir else Path(config["output_root"]) / config["run_name"]
    torch.set_num_threads(max(1, int(config.get("torch_threads", 1))))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    preflight = _preflight(config, allow_dirty=bool(args.allow_dirty_smoke))
    atomic_write_json(run_dir / "preflight.json", preflight)
    contract = _run_contract(config, run_dir)
    aliases, selection, evaluation_state = _initial_selection()
    curriculum = TransitionCurriculumController(
        initial_difficulty=config["curriculum"]["initial_difficulty"],
        maximum_difficulty=config["curriculum"]["maximum_difficulty"],
    )
    env = None
    total_transitions = 0
    agent: SACTransitionAgent
    replay: StratifiedReplayBuffer
    accumulator = MultiWorkerNStepAccumulator(
        n_step=int(config["sac"]["n_step"]),
        gamma=float(config["sac"]["gamma"]),
        n_workers=int(config["n_envs"]),
    )
    initialization_report = None
    env = _make_vec_env(config, run_dir, curriculum.difficulty)
    try:
        if args.resume:
            checkpoint = latest_valid_sac_checkpoint(
                run_dir, expected_contract_sha256=contract
            )
            if checkpoint is None:
                raise RuntimeError("no valid SAC checkpoint to resume")
            agent, replay, state = load_sac_checkpoint(checkpoint)
            total_transitions = int(state["total_environment_transitions"])
            worker_states = curriculum.load_state_dict(state["curriculum"])
            _restore_workers(env, worker_states)
            accumulator.load_state_dict(state["n_step"])
            # Simulator state is deliberately not serialized.  On restart the
            # saved partial queues become shortened terminal returns rather than
            # being joined to a different freshly reset track.
            flushed = _add_interrupted_queues(accumulator, replay)
            aliases.update(state.get("checkpoint_aliases") or {})
            selection.update(state.get("checkpoint_selection_state") or {})
            evaluation_state.update(state.get("evaluation") or {})
            restore_rng_state(state["rng"])
            atomic_append_jsonl(run_dir / "logs" / "resume.jsonl", {
                "utc": now_utc(), "checkpoint": str(checkpoint.model_path),
                "timesteps": total_transitions,
                "interrupted_n_step_entries_flushed": flushed,
            })
        else:
            if any((run_dir / "checkpoints").glob("*.manifest.json")):
                raise ValueError("SAC run contains checkpoints; use --resume")
            sac = config["sac"]
            agent = SACTransitionAgent(
                hidden_sizes=sac["hidden_sizes"],
                actor_learning_rate=sac["actor_learning_rate"],
                critic_learning_rate=sac["critic_learning_rate"],
                entropy_learning_rate=sac["entropy_learning_rate"],
                tau=sac["tau"],
                target_entropy=sac["target_entropy"],
                initial_alpha=sac["initial_alpha"],
                initial_std=sac["initial_std"],
            )
            source = _resolve_source_checkpoint(config["initialization"]["source_checkpoint"])
            initialization_report = transfer_ppo_mean_actor(
                agent.actor, source,
                validation_samples=int(config["initialization"].get("parity_samples", 2048)),
                tolerance=float(config["initialization"].get("parity_tolerance", 2e-6)),
            )
            initialization_report.update({
                "mode": "ppo_actor_mean_warm_start",
                "source_sha256": sha256_file(source),
                "critics_initialized_from_scratch": True,
                "target_critics_initialized_from_scratch": True,
                "entropy_initialized_independently": True,
                "replay_initial_size": 0,
            })
            replay = StratifiedReplayBuffer(
                capacity=int(sac["replay_capacity"]),
                seed=int(config["seed"]) + 71,
                composition=sac.get("replay_composition", DEFAULT_BATCH_COMPOSITION),
            )

        baseline = _preserve_baseline(run_dir, config, agent)
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.exists():
            atomic_write_json(manifest_path, {
                "schema_version": "universal_transition_sac_run_v1",
                "created_utc": now_utc(),
                "git_sha": git_sha(),
                "config_contract_sha256": contract,
                "config": config,
                "initialization_report": initialization_report,
                "immutable_initialization_baseline": baseline,
                "algorithmic_independence": {
                    "ppo_critic_copied": False,
                    "ppo_optimizer_copied": False,
                    "ppo_replay_imported": False,
                    "ppo_demonstration_loss": False,
                    "hybrid_actions": False,
                    "joint_losses": False,
                },
                "replay_event_categories": list(config["sac"].get(
                    "event_categories", []
                )),
                "replay_composition": dict(replay.composition),
            })
        atomic_write_json(run_dir / "reward_config.json", dict(config.get("reward") or {}))

        observations = np.asarray(env.reset(), dtype=np.float32)
        _validate_initialized_worker_identities(env, config)
        env.env_method("set_total_environment_transitions", total_transitions)
        target = int(config["new_environment_steps"])
        target -= target % int(config["n_envs"])
        evaluation_frequency = int(config["evaluation"]["frequency"])
        early_schedule = tuple(map(int, config["evaluation"].get("early_schedule") or ()))
        next_evaluation = _next_evaluation_transition(
            evaluation_state, evaluation_frequency, early_schedule, target
        )
        checkpoint_frequency = int(config["evaluation"]["checkpoint_frequency"])
        next_checkpoint = ((total_transitions // checkpoint_frequency) + 1) * checkpoint_frequency
        thresholds = thresholds_from_mapping(config.get("competence_gate"))
        rollback_limit = int(config["evaluation"].get("rollback_consecutive_evaluations", 2))
        update_credit = 0.0
        last_losses = None
        stop = {"requested": False}

        def request_stop(signum=None, frame=None):
            stop["requested"] = True

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        stop_file = run_dir / "STOP_REQUESTED"
        _status(
            run_dir, state="running", pid=os.getpid(), total_transitions=total_transitions,
            target=target, agent=agent, replay=replay, accumulator=accumulator,
            curriculum=curriculum, env=env, aliases=aliases, selection=selection,
            config=config, last_losses=last_losses, message="multi-step SAC initialized",
        )

        collector_steps = int(config["sac"].get("collector_steps", 128))
        learning_starts = int(config["sac"]["learning_starts"])
        batch_size = int(config["sac"]["batch_size"])
        updates_per_transition = float(config["sac"].get("updates_per_transition", 0.25))
        while total_transitions < target and not stop["requested"] and not stop_file.exists():
            block_start = time.perf_counter()
            block_transitions = 0
            update_time = replay_time = 0.0
            block_updates = 0
            for _ in range(collector_steps):
                if total_transitions >= target or stop["requested"] or stop_file.exists():
                    break
                actions = agent.act(observations, deterministic=False)
                next_observations, rewards, dones, infos = env.step(actions)
                next_observations = np.asarray(next_observations, dtype=np.float32)
                for worker_id in range(int(config["n_envs"])):
                    info = dict(infos[worker_id])
                    done = bool(dones[worker_id])
                    truncated = bool(done and info.get("TimeLimit.truncated", False))
                    terminated = bool(done and not truncated)
                    terminal_observation = np.asarray(
                        info.get("terminal_observation", next_observations[worker_id]),
                        dtype=np.float32,
                    )
                    category = classify_replay_event(
                        info, observations[worker_id], terminal_observation
                    )
                    emitted = accumulator.append(
                        worker_id,
                        observation=observations[worker_id],
                        action=actions[worker_id],
                        reward=float(rewards[worker_id]),
                        next_observation=terminal_observation,
                        terminated=terminated,
                        truncated=truncated,
                        bootstrap_allowed=(
                            not terminated and (
                                not truncated or is_legal_bootstrap_truncation(info)
                            )
                        ),
                        event_category=category,
                    )
                    replay.extend(emitted)
                observations = next_observations
                total_transitions += int(config["n_envs"])
                block_transitions += int(config["n_envs"])
                curriculum.set_total_environment_transitions(total_transitions)
                env.env_method("set_total_environment_transitions", total_transitions)
                if total_transitions >= learning_starts and replay.size >= batch_size:
                    update_credit += int(config["n_envs"]) * updates_per_transition
                    if agent.gradient_updates == 0:
                        # The 10k diagnostic must prove the learner can update,
                        # even when a fractional UTD ratio has accumulated less
                        # than one full update at the exact threshold.
                        update_credit = max(update_credit, 1.0)
                    while update_credit >= 1.0:
                        sampled = time.perf_counter()
                        batch = replay.sample(batch_size)
                        replay_time += time.perf_counter() - sampled
                        updated = time.perf_counter()
                        last_losses = agent.update(batch)
                        update_time += time.perf_counter() - updated
                        update_credit -= 1.0
                        block_updates += 1

            elapsed = time.perf_counter() - block_start
            atomic_append_jsonl(run_dir / "logs" / "progress.jsonl", {
                "utc": now_utc(),
                "total_environment_transitions": total_transitions,
                "difficulty": curriculum.difficulty,
                "environment_transitions_per_second": block_transitions / max(elapsed, 1e-9),
                "collector_wall_time_s": max(0.0, elapsed - update_time - replay_time),
                "learner_wall_time_s": update_time,
                "replay_sampling_wall_time_s": replay_time,
                "gradient_updates": int(agent.gradient_updates),
                "block_gradient_updates": block_updates,
                "learner_updates_per_second": block_updates / max(update_time, 1e-9),
                "replay_size": replay.size,
                "entropy_coefficient": agent.alpha,
                "losses": last_losses,
            })

            if total_transitions >= next_evaluation:
                _add_interrupted_queues(accumulator, replay)
                checkpoint_started = time.perf_counter()
                boundary = _save(
                    agent=agent, replay=replay, accumulator=accumulator,
                    run_dir=run_dir, contract=contract,
                    total_transitions=total_transitions, curriculum=curriculum,
                    env=env, evaluation_state=evaluation_state, aliases=aliases,
                    selection=selection, config=config,
                    reason="pre_evaluation_atomic_boundary",
                )
                checkpoint_time = time.perf_counter() - checkpoint_started
                _status(
                    run_dir, state="evaluating", pid=os.getpid(), total_transitions=total_transitions,
                    target=target, agent=agent, replay=replay, accumulator=accumulator,
                    curriculum=curriculum, env=env, aliases=aliases, selection=selection,
                    config=config, last_losses=last_losses,
                    message="SAC rollout engines will close before evaluation",
                )
                worker_states = _worker_states(env)
                learner_rng = capture_rng_state()
                env.close()
                env = None
                evaluation_started = time.perf_counter()
                output = run_dir / "evaluations" / f"unseen_{total_transitions:09d}"
                report = evaluate_checkpoint_universal_transition_benchmark(
                    boundary.model_path,
                    output_dir=output,
                    seed=int(config["evaluation"]["seed"]),
                    difficulty=curriculum.difficulty,
                    transition_cases=int(config["evaluation"]["transition_cases"]),
                    full_cases_per_length=int(config["evaluation"]["full_cases_per_length"]),
                    adapter=str(config["adapter"]),
                    frames_per_sec=config["holoocean_frames_per_sec"],
                    max_steps=int(config["max_episode_steps"]),
                    parallel_workers=int(config["evaluation"].get("intermediate_parallel_workers", 2)),
                    algorithm="sac",
                )
                report["timesteps"] = total_transitions
                report["nominal_timestep"] = next_evaluation
                report["evaluation_level"] = "intermediate"
                competent = evaluate_competence_gate(report["metrics"], thresholds).passed
                candidate_best = selection["best_metrics"] is None or (
                    transition_checkpoint_rank_key(report["metrics"], thresholds)
                    > transition_checkpoint_rank_key(selection["best_metrics"], thresholds)
                )
                dedicated_reason = _dedicated_benchmark_due(
                    metrics=report["metrics"], competent=competent,
                    candidate_best=candidate_best,
                    dedicated_state=selection["dedicated"], timesteps=total_transitions,
                    target=target, config=config,
                )
                if dedicated_reason:
                    report["dedicated"] = _run_dedicated(
                        boundary.model_path, run_dir, config=config,
                        difficulty=curriculum.difficulty, timesteps=total_transitions,
                        reason=dedicated_reason, selection=selection,
                    )
                curriculum.observe_evaluation(
                    report["metrics"], total_transitions, thresholds=thresholds
                )
                evaluation_time = time.perf_counter() - evaluation_started
                env = _make_vec_env(config, run_dir, curriculum.difficulty)
                _restore_workers(env, worker_states)
                env.env_method("set_difficulty", curriculum.difficulty)
                env.env_method("set_efficiency_unlocked", curriculum.state.efficiency_reward_active)
                env.env_method("set_total_environment_transitions", total_transitions)
                observations = np.asarray(env.reset(), dtype=np.float32)
                _validate_initialized_worker_identities(env, config)
                restore_rng_state(learner_rng)
                outcome = apply_evaluation_selection(
                    report["metrics"], checkpoint_path=str(boundary.model_path),
                    timesteps=total_transitions, aliases=aliases, selection=selection,
                    thresholds=thresholds, rollback_limit=rollback_limit,
                    baseline_record=baseline,
                )
                report["competence"] = outcome["competence"]
                atomic_write_json(output / "evaluation.json", report)
                summary = _evaluation_summary(report, str(output))
                summary["nominal_timestep"] = next_evaluation
                evaluation_state["last"] = summary
                evaluation_state["history"].append(summary)
                if outcome["candidate_better"]:
                    evaluation_state["best"] = summary
                _save(
                    agent=agent, replay=replay, accumulator=accumulator,
                    run_dir=run_dir, contract=contract,
                    total_transitions=total_transitions, curriculum=curriculum,
                    env=env, evaluation_state=evaluation_state, aliases=aliases,
                    selection=selection, config=config,
                    reason="unseen_transition_evaluation",
                    status="safe" if outcome["safe"] else "unsafe",
                )
                atomic_append_jsonl(run_dir / "logs" / "timing.jsonl", {
                    "utc": now_utc(), "timesteps": total_transitions,
                    "nominal_timestep": next_evaluation,
                    "evaluation_wall_time_s": evaluation_time,
                    "checkpoint_wall_time_s": checkpoint_time,
                })
                next_evaluation = _next_evaluation_transition(
                    evaluation_state, evaluation_frequency, early_schedule, target
                )
                next_checkpoint = max(next_checkpoint, total_transitions + checkpoint_frequency)
                if outcome["collapsed"]:
                    stop["requested"] = True

            if total_transitions >= next_checkpoint and env is not None:
                started = time.perf_counter()
                _save(
                    agent=agent, replay=replay, accumulator=accumulator,
                    run_dir=run_dir, contract=contract,
                    total_transitions=total_transitions, curriculum=curriculum,
                    env=env, evaluation_state=evaluation_state, aliases=aliases,
                    selection=selection, config=config, reason="periodic",
                )
                atomic_append_jsonl(run_dir / "logs" / "timing.jsonl", {
                    "utc": now_utc(), "timesteps": total_transitions,
                    "checkpoint_wall_time_s": time.perf_counter() - started,
                })
                next_checkpoint += checkpoint_frequency
            if env is not None:
                _status(
                    run_dir, state="running", pid=os.getpid(), total_transitions=total_transitions,
                    target=target, agent=agent, replay=replay, accumulator=accumulator,
                    curriculum=curriculum, env=env, aliases=aliases, selection=selection,
                    config=config, last_losses=last_losses,
                    message="independent multi-step SAC actively advancing",
                )

        reason = (
            "baseline_collapse_rollback" if selection.get("rollback")
            else "graceful_stop" if stop["requested"] or stop_file.exists()
            else "target_reached"
        )
        _add_interrupted_queues(accumulator, replay)
        _save(
            agent=agent, replay=replay, accumulator=accumulator,
            run_dir=run_dir, contract=contract, total_transitions=total_transitions,
            curriculum=curriculum, env=env, evaluation_state=evaluation_state,
            aliases=aliases, selection=selection, config=config, reason=reason,
            stop_reason=reason,
        )
        _status(
            run_dir, state="completed" if reason == "target_reached" else "stopped",
            pid=os.getpid(), total_transitions=total_transitions, target=target,
            agent=agent, replay=replay, accumulator=accumulator,
            curriculum=curriculum, env=env, aliases=aliases, selection=selection,
            config=config, last_losses=last_losses, message=reason,
        )
        stop_file.unlink(missing_ok=True)
        return 0
    except Exception as exc:
        try:
            atomic_write_json(run_dir / "failure.json", {
                "utc": now_utc(), "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
        except Exception:
            pass
        raise
    finally:
        if env is not None:
            env.close()


def status_command(run_dir: str | Path) -> int:
    path = Path(run_dir) / "status.json"
    if not path.exists():
        print(json.dumps({"state": "not_started", "run_dir": str(run_dir)}, indent=2))
        return 1
    value = json.loads(path.read_text(encoding="utf-8"))
    pid = int(value.get("pid", 0) or 0)
    alive = False
    if pid:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, check=False,
        )
        alive = str(pid) in result.stdout and "No tasks" not in result.stdout
    value["process_alive"] = alive
    print(json.dumps(value, indent=2))
    return 0


def stop_command(run_dir: str | Path) -> int:
    run = Path(run_dir)
    if not (run / "status.json").exists():
        raise FileNotFoundError(run / "status.json")
    stop = run / "STOP_REQUESTED"
    stop.touch(exist_ok=True)
    print(json.dumps({
        "stop_requested": True, "stop_file": str(stop),
        "mechanism": "atomic SAC checkpoint after current collector/evaluation boundary",
    }, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--run-dir")
    train.add_argument("--steps", type=int)
    train.add_argument("--n-envs", type=int)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--allow-dirty-smoke", action="store_true")
    status = commands.add_parser("status")
    status.add_argument("run_dir")
    stop = commands.add_parser("stop")
    stop.add_argument("run_dir")
    args = parser.parse_args(argv)
    if args.command == "train":
        return run_training(args)
    if args.command == "status":
        return status_command(args.run_dir)
    return stop_command(args.run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
