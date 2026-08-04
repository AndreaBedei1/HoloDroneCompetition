"""Atomic feed-forward PPO training for universal local gate transitions."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from marine_race_arena.learning.config import ACTION_CONTRACT_VERSION
from marine_race_arena.learning.config_local_transition import (
    OBS_DIM_LOCAL_TRANSITION,
    OBS_ENCODING_VERSION_LOCAL_TRANSITION,
)
from marine_race_arena.learning.longrun_checkpoint import (
    atomic_save_checkpoint,
    atomic_write_json,
    canonical_hash,
    capture_rng_state,
    latest_valid_checkpoint,
    load_checkpoint_state,
    restore_model_training_state,
    restore_rng_state,
    sha256_file,
)
from marine_race_arena.learning.provenance import git_sha, now_utc
from marine_race_arena.learning.train_multigate_longrun import (
    AbsoluteLearningRateSchedule,
)
from marine_race_arena.learning.transition_curriculum import (
    TransitionCurriculumController,
)
from marine_race_arena.learning.transition_env import UniversalTransitionEnv
from marine_race_arena.learning.transition_evaluation import (
    evaluate_checkpoint_universal_transition_benchmark,
    transition_checkpoint_rank_key,
)
from marine_race_arena.learning.transition_policy import (
    build_local_transition_ppo,
    initialize_local_transition_policy,
)
from marine_race_arena.learning.transition_selection import (
    baseline_collapse_criteria,
    evaluate_competence_gate,
    thresholds_from_mapping,
)

INITIALIZATION_MODES = (
    "scratch",
    "selective_warm_start",
    "warm_start_transition_checkpoint",
)
WARM_START_MODES = ("selective_warm_start", "warm_start_transition_checkpoint")


def _load_config(path: str | Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "run_name", "output_root", "observation_version", "action_version",
        "initialization", "seed", "new_environment_steps", "n_envs",
        "adapter", "allow_fallback", "holoocean_frames_per_sec", "ppo",
        "curriculum", "evaluation",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"transition config missing {missing}")
    if value["observation_version"] != OBS_ENCODING_VERSION_LOCAL_TRANSITION:
        raise ValueError("local transition observation contract changed")
    if value["action_version"] != ACTION_CONTRACT_VERSION:
        raise ValueError("action contract changed")
    if int(value["n_envs"]) < 1:
        raise ValueError("n_envs must be positive")
    rollout = int(value["n_envs"]) * int(value["ppo"]["n_steps"])
    if rollout != 2048:
        raise ValueError(f"total rollout size must remain 2048, got {rollout}")
    if value["adapter"] == "holoocean" and value["allow_fallback"]:
        raise ValueError("HoloOcean transition training cannot allow fallback")
    mode = value["initialization"].get("mode")
    if mode not in INITIALIZATION_MODES:
        raise ValueError("unknown initialization mode")
    # Validated eagerly so a malformed gate cannot silently fall back to the
    # defaults on a million-transition run.
    thresholds_from_mapping(value.get("competence_gate"))
    schedule = value["evaluation"].get("early_schedule") or []
    if any(int(point) <= 0 for point in schedule):
        raise ValueError("early evaluation schedule must be positive")
    return value


def _run_dir(config: Mapping[str, Any], override: Optional[str]) -> Path:
    return Path(override) if override else Path(config["output_root"]) / str(config["run_name"])


def _contract(config: Mapping[str, Any]) -> str:
    stable = dict(config)
    stable.pop("new_environment_steps", None)
    return canonical_hash({"transition_config": stable, "training_code_sha": git_sha()})


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
    if branch != "feature/rl-universal-gate-transition":
        raise ValueError(f"wrong branch {branch!r}")
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], check=True,
        capture_output=True, text=True,
    ).stdout.strip())
    if dirty and not allow_dirty:
        raise ValueError("commit implementation before a non-smoke run")
    initialization = config["initialization"]
    source = initialization.get("source_checkpoint")
    actual_sha = None
    if initialization["mode"] in WARM_START_MODES:
        if not source or not Path(source).exists():
            raise FileNotFoundError(source)
        actual_sha = sha256_file(source)
        if actual_sha != initialization.get("source_sha256"):
            raise ValueError("warm-start checkpoint hash mismatch")
    return {
        "checked_utc": now_utc(),
        "branch": branch,
        "git_sha": git_sha(),
        "worktree_dirty": dirty,
        "adapter": config["adapter"],
        "fallback_disabled": not bool(config["allow_fallback"]),
        "observation_version": OBS_ENCODING_VERSION_LOCAL_TRANSITION,
        "observation_dim": OBS_DIM_LOCAL_TRANSITION,
        "n_envs": int(config["n_envs"]),
        "rollout_size": int(config["n_envs"]) * int(config["ppo"]["n_steps"]),
        "initialization_mode": initialization["mode"],
        "initialization_checkpoint": source,
        "initialization_sha256": actual_sha,
    }


def _make_worker(
    *,
    run_dir: str,
    worker_id: int,
    sampler_seed: int,
    difficulty: str,
    transition_focus_fraction: float,
    adapter: str,
    allow_fallback: bool,
    frames_per_sec: bool | int,
    max_episode_steps: int,
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
    base_seed = int(config["worker_seed_base"])
    factories = [
        partial(
            _make_worker,
            run_dir=str(run_dir),
            worker_id=index,
            sampler_seed=base_seed + index * 10_003,
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
    if n_envs == 1:
        return DummyVecEnv(factories)
    # Explicit spawn is required on Windows and prevents HoloOcean handles from
    # being inherited across simulation processes.
    return SubprocVecEnv(factories, start_method="spawn")


def _worker_states(env: Any) -> list[Mapping[str, Any]]:
    return list(env.env_method("worker_state"))


def _worker_identities(env: Any) -> list[Mapping[str, Any]]:
    return list(env.env_method("worker_identity"))


def _restore_workers(env: Any, worker_states: list[Mapping[str, Any]]) -> None:
    if len(worker_states) != int(env.num_envs):
        raise ValueError("worker count changed on resume")
    for index, state in enumerate(worker_states):
        env.env_method("load_worker_state", state, indices=index)


def _recreate_training_env(
    model: Any,
    config: Mapping[str, Any],
    run_dir: Path,
    curriculum: TransitionCurriculumController,
    worker_states: list[Mapping[str, Any]],
) -> Any:
    """Reopen isolated rollout workers after an out-of-process evaluation.

    The caller closes the old vector environment before launching evaluation,
    so training and evaluation engines never coexist.  Sampler state is loaded
    before reset; the centrally selected curriculum settings are then applied
    to every worker before its next geometry is sampled.
    """

    env = _make_vec_env(config, run_dir, curriculum.difficulty)
    try:
        _restore_workers(env, worker_states)
        env.env_method("set_difficulty", curriculum.difficulty)
        env.env_method(
            "set_efficiency_unlocked",
            curriculum.state.efficiency_reward_active,
        )
        env.env_method(
            "set_total_environment_transitions",
            int(model.num_timesteps),
        )
        env.reset()
        model.set_env(env)
        model._last_obs = None
        return env
    except BaseException:
        env.close()
        raise


def _next_evaluation_transition(
    evaluation_state: Mapping[str, Any],
    frequency: int,
    early_schedule: Sequence[int] = (),
    target: Optional[int] = None,
) -> int:
    """Return the first scheduled evaluation not present in checkpoint state.

    Early checkpoints are evaluated on an explicit dense schedule so a collapse
    of the warm-start behaviour is detected within tens of thousands of
    transitions rather than after the regular interval.  The training target is
    always evaluated once, so the final model is validated even when it does not
    land on a schedule boundary.
    """

    completed = {
        int(report.get("timesteps", 0) or 0)
        for report in list(evaluation_state.get("history") or [])
        if isinstance(report, Mapping)
    }
    last_completed = max(completed, default=0)
    for point in sorted({int(value) for value in early_schedule}):
        if point > last_completed:
            return point
    regular = ((last_completed // int(frequency)) + 1) * int(frequency)
    if target is not None and regular > int(target) and int(target) not in completed:
        return int(target)
    return regular


def _preserve_initialization_baseline(
    run_dir: Path, config: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    """Copy the warm-start source into an immutable per-run baseline.

    The baseline is the behaviour this run must not lose.  It is written once
    and never overwritten, so a rollback target always exists even if the source
    experiment directory is later moved.
    """

    initialization = config["initialization"]
    source = initialization.get("source_checkpoint")
    if initialization["mode"] not in WARM_START_MODES or not source:
        return None
    baseline_dir = run_dir / "baseline"
    record_path = baseline_dir / "initialization_baseline.json"
    if record_path.exists():
        return json.loads(record_path.read_text(encoding="utf-8"))
    baseline_dir.mkdir(parents=True, exist_ok=True)
    target = baseline_dir / "initialization_policy.zip"
    tmp = baseline_dir / ".initialization_policy.partial.zip"
    shutil.copyfile(source, tmp)
    digest = sha256_file(tmp)
    if digest != sha256_file(source):
        tmp.unlink(missing_ok=True)
        raise OSError("initialization baseline copy hash mismatch")
    os.replace(tmp, target)
    record = {
        "schema_version": "universal_transition_initialization_baseline_v1",
        "created_utc": now_utc(),
        "source_checkpoint": str(source),
        "source_sha256": digest,
        "baseline_checkpoint": str(target),
        "baseline_sha256": digest,
        "immutable": True,
    }
    atomic_write_json(record_path, record)
    return record


def _dedicated_benchmark_due(
    *,
    metrics: Mapping[str, Any],
    competent: bool,
    candidate_best: bool,
    dedicated_state: Mapping[str, Any],
    timesteps: int,
    target: int,
    config: Mapping[str, Any],
) -> Optional[str]:
    """Reason to spend the expensive ~1,000-case benchmark, or ``None``.

    The dense intermediate schedule is cheap; the dedicated benchmark runs only
    when a checkpoint has actually earned validation, or at final validation.
    """

    evaluation = config["evaluation"]
    if int(timesteps) >= int(target):
        return "final_model_validation"
    if not competent:
        return None
    last = int(dedicated_state.get("last_timesteps", 0) or 0)
    minimum_interval = int(
        evaluation.get("dedicated_min_interval_transitions", 0) or 0
    )
    if last and int(timesteps) - last < minimum_interval:
        return None
    if dedicated_state.get("best_success_rate") is None:
        return "first_competent_checkpoint"
    margin = float(evaluation.get("dedicated_material_improvement", 0.05) or 0.0)
    success = float(metrics.get("universal_transition_success_rate", 0.0) or 0.0)
    if success >= float(dedicated_state["best_success_rate"]) + margin:
        return "material_competence_improvement"
    if candidate_best:
        return "new_candidate_best_checkpoint"
    return None


def _training_work_pending(current: int, target: int, next_evaluation: int) -> bool:
    """Include a pending target-boundary evaluation without extra training."""

    return int(current) < int(target) or int(current) >= int(next_evaluation)


def _evaluation_summary(report: Mapping[str, Any], output_dir: str) -> Dict[str, Any]:
    """Checkpoint-sized view of an evaluation report.

    Keeps everything resume, safety and selection logic reads, and drops the
    per-episode rows, which remain available on disk.
    """

    dedicated = report.get("dedicated")
    return {
        "schema_version": report.get("schema_version"),
        "observation_version": report.get("observation_version"),
        "timesteps": int(report.get("timesteps", 0) or 0),
        "evaluation_level": report.get("evaluation_level", "intermediate"),
        "seed": report.get("seed"),
        "difficulty": report.get("difficulty"),
        "transition_cases": report.get("transition_cases"),
        "full_cases_per_length": report.get("full_cases_per_length"),
        "output_dir": output_dir,
        "metrics": dict(report.get("metrics") or {}),
        "competence": report.get("competence"),
        "dedicated": None if not isinstance(dedicated, Mapping) else {
            "reason": dedicated.get("reason"),
            "output_dir": dedicated.get("output_dir"),
            "transition_cases": dedicated.get("transition_cases"),
            "metrics": dict(dedicated.get("metrics") or {}),
        },
    }


def apply_evaluation_selection(
    metrics: Mapping[str, Any],
    *,
    checkpoint_path: str,
    timesteps: int,
    aliases: Dict[str, Any],
    selection: Dict[str, Any],
    thresholds: Any,
    rollback_limit: int,
    baseline_record: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Update checkpoint aliases and rollback state from one evaluation.

    Competence qualification precedes every decision: a policy that avoids
    safety events by not moving can never be marked competent or safe, become
    the best checkpoint, or promote geometric difficulty.
    """

    selection["last_evaluation_metrics"] = dict(metrics)
    verdict = evaluate_competence_gate(metrics, thresholds)
    selection["last_competence_verdict"] = verdict.as_dict()
    candidate_better = verdict.passed and (
        selection["best_metrics"] is None
        or transition_checkpoint_rank_key(metrics, thresholds)
        > transition_checkpoint_rank_key(selection["best_metrics"], thresholds)
    )
    safe = (
        verdict.passed
        and bool(metrics.get("safety_clean"))
        and float(metrics.get("universal_transition_success_rate", 0.0) or 0.0) >= 0.99
    )
    if verdict.passed:
        aliases["latest_competent"] = checkpoint_path
    if safe:
        aliases["latest_safe_competent"] = checkpoint_path
    if candidate_better:
        aliases["best_universal_transition"] = checkpoint_path
        selection["best_metrics"] = dict(metrics)
        selection["best_timestep"] = int(timesteps)
    long_sequence = metrics.get("long_sequence_completion_score")
    if verdict.passed and long_sequence is not None and (
        selection["best_long_sequence_score"] is None
        or float(long_sequence) > float(selection["best_long_sequence_score"])
    ):
        aliases["best_long_sequence"] = checkpoint_path
        selection["best_long_sequence_score"] = float(long_sequence)
    collapse = baseline_collapse_criteria(metrics)
    selection["consecutive_baseline_collapses"] = (
        int(selection["consecutive_baseline_collapses"]) + 1 if collapse else 0
    )
    collapsed = int(selection["consecutive_baseline_collapses"]) >= int(rollback_limit)
    if collapsed:
        selection["rollback"] = {
            "triggered_utc": now_utc(),
            "total_environment_transitions": int(timesteps),
            "consecutive_evaluations_below_baseline": int(
                selection["consecutive_baseline_collapses"]
            ),
            "failed_criteria": list(collapse),
            "rollback_checkpoint": (
                aliases["latest_competent"]
                or (baseline_record or {}).get("baseline_checkpoint")
            ),
            "initialization_baseline": dict(baseline_record or {}) or None,
            "action": "stop_before_spending_remaining_transitions",
        }
    return {
        "competence": verdict.as_dict(),
        "competent": verdict.passed,
        "safe": safe,
        "candidate_better": candidate_better,
        "collapsed": collapsed,
        "collapse_criteria": list(collapse),
    }


def _run_dedicated_benchmark(
    run_dir: Path,
    *,
    model: Any,
    config: Mapping[str, Any],
    difficulty: str,
    reason: str,
    selection: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate an earned checkpoint on the full unseen benchmark.

    Runs against the atomic checkpoint written immediately before evaluation, so
    the benchmark is resumable per case and independent of the live learner.
    """

    timesteps = int(model.num_timesteps)
    evaluation = config["evaluation"]
    checkpoint = run_dir / "checkpoints" / f"ppo_{timesteps}_steps.zip"
    output = run_dir / "evaluations" / f"dedicated_{timesteps:09d}"
    report = evaluate_checkpoint_universal_transition_benchmark(
        checkpoint,
        output_dir=output,
        seed=int(evaluation.get("dedicated_seed", evaluation["seed"])),
        difficulty=difficulty,
        transition_cases=int(evaluation["dedicated_transition_cases"]),
        full_cases_per_length=int(
            evaluation.get("dedicated_full_cases_per_length", 2)
        ),
        adapter=str(config["adapter"]),
        frames_per_sec=config["holoocean_frames_per_sec"],
        max_steps=int(config["max_episode_steps"]),
        parallel_workers=int(evaluation.get("dedicated_parallel_workers", 2)),
    )
    metrics = report["metrics"]
    success = float(metrics.get("universal_transition_success_rate", 0.0) or 0.0)
    state = selection["dedicated"]
    state["last_timesteps"] = timesteps
    previous = state.get("best_success_rate")
    if previous is None or success > float(previous):
        state["best_success_rate"] = success
    state["history"].append({
        "timesteps": timesteps,
        "reason": reason,
        "output_dir": str(output),
        "universal_transition_success_rate": success,
    })
    return {
        "reason": reason,
        "output_dir": str(output),
        "transition_cases": int(report.get("transition_cases", 0)),
        "metrics": metrics,
    }


def _save(
    model: Any,
    run_dir: Path,
    contract: str,
    curriculum: TransitionCurriculumController,
    env: Any,
    evaluation_state: Mapping[str, Any],
    aliases: Dict[str, Any],
    selection: Dict[str, Any],
    config: Mapping[str, Any],
    *,
    status: str,
    reason: str,
    stop_reason: Optional[str] = None,
):
    path = run_dir / "checkpoints" / f"ppo_{int(model.num_timesteps)}_steps.zip"
    aliases["last"] = str(path)
    curriculum_state = curriculum.state_dict(_worker_states(env))
    return atomic_save_checkpoint(
        model,
        run_dir,
        total_timesteps=int(model.num_timesteps),
        config_contract_hash=contract,
        curriculum_state=curriculum_state,
        evaluation_state=dict(evaluation_state),
        status=status,
        reason=reason,
        extra_state={
            "architecture": "feedforward_ppo",
            "checkpoint_aliases": dict(aliases),
            "checkpoint_selection_state": dict(selection),
            "initialization": dict(config["initialization"]),
            "worker_identities": _worker_identities(env),
            "n_envs": int(config["n_envs"]),
            "rollout_size": int(config["n_envs"]) * int(config["ppo"]["n_steps"]),
            "stop_reason": stop_reason,
        },
    )


def _status(
    run_dir: Path,
    *,
    state: str,
    model: Any,
    curriculum: TransitionCurriculumController,
    env: Any,
    aliases: Mapping[str, Any],
    selection: Mapping[str, Any],
    config: Mapping[str, Any],
    message: Optional[str] = None,
) -> Dict[str, Any]:
    value = {
        "updated_utc": now_utc(),
        "state": state,
        "pid": os.getpid(),
        "total_environment_transitions": int(model.num_timesteps),
        "target_environment_transitions": int(config["new_environment_steps"]),
        "observation_version": OBS_ENCODING_VERSION_LOCAL_TRANSITION,
        "observation_dim": OBS_DIM_LOCAL_TRANSITION,
        "architecture": "feedforward_ppo",
        "n_envs": int(config["n_envs"]),
        "n_steps_per_env": int(config["ppo"]["n_steps"]),
        "rollout_size": int(config["n_envs"]) * int(config["ppo"]["n_steps"]),
        "difficulty": curriculum.difficulty,
        "difficulty_entry_transitions": curriculum.state.difficulty_entry_transitions,
        "consecutive_qualifying_evaluations": curriculum.state.consecutive_qualifying_evaluations,
        "efficiency_reward_active": curriculum.state.efficiency_reward_active,
        "difficulty_history": curriculum.state.difficulty_history,
        "evaluation_history_count": len(curriculum.state.evaluation_history),
        "checkpoint_aliases": dict(aliases),
        "checkpoint_selection_state": dict(selection),
        "competence_gate": thresholds_from_mapping(
            config.get("competence_gate")
        ).as_dict(),
        "evaluation_schedule": {
            "early_schedule": list(config["evaluation"].get("early_schedule") or []),
            "frequency": int(config["evaluation"]["frequency"]),
            "intermediate_transition_cases": int(
                config["evaluation"]["transition_cases"]
            ),
            "intermediate_full_cases_per_length": int(
                config["evaluation"]["full_cases_per_length"]
            ),
            "dedicated_transition_cases": int(
                config["evaluation"].get("dedicated_transition_cases", 0) or 0
            ),
        },
        "worker_identities": _worker_identities(env),
        "initialization": dict(config["initialization"]),
        "all_actions_policy_generated": True,
        "hybrid_or_expert_actions_active": False,
        "message": message,
    }
    atomic_write_json(run_dir / "status.json", value)
    return value


def _build_new_model(config: Mapping[str, Any], env: Any):
    ppo = config["ppo"]
    schedule = AbsoluteLearningRateSchedule(
        ppo["learning_rate"],
        ppo["final_learning_rate"],
        ppo["learning_rate_schedule"],
    )
    model = build_local_transition_ppo(
        env,
        seed=int(config["seed"]),
        learning_rate=schedule,
        hidden_sizes=ppo["hidden_sizes"],
        n_steps=ppo["n_steps"],
        batch_size=ppo["batch_size"],
        n_epochs=ppo["n_epochs"],
        gamma=ppo["gamma"],
        gae_lambda=ppo["gae_lambda"],
        clip_range=ppo["clip_range"],
        target_kl=ppo["target_kl"],
        ent_coef=ppo["ent_coef"],
    )
    initialization = config["initialization"]
    report = initialize_local_transition_policy(
        model,
        mode=initialization["mode"],
        source_checkpoint=initialization.get("source_checkpoint"),
    )
    return model, report


def run_training(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = _run_dir(config, None)
    if args.steps is not None:
        config["new_environment_steps"] = int(args.steps)
    if args.n_envs is not None:
        config["n_envs"] = int(args.n_envs)
        config["ppo"]["n_steps"] = 2048 // int(args.n_envs)
    import torch
    torch.set_num_threads(max(1, int(config.get("torch_threads", 1))))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    preflight = _preflight(config, allow_dirty=bool(args.allow_dirty_smoke))
    atomic_write_json(run_dir / "preflight.json", preflight)
    contract = _run_contract(config, run_dir)
    curriculum = TransitionCurriculumController(
        initial_difficulty=config["curriculum"]["initial_difficulty"],
        maximum_difficulty=config["curriculum"]["maximum_difficulty"],
    )
    env: Optional[Any] = _make_vec_env(config, run_dir, curriculum.difficulty)
    aliases: Dict[str, Any] = {
        "last": None,
        "latest_competent": None,
        "latest_safe_competent": None,
        "best_universal_transition": None,
        "best_long_sequence": None,
    }
    selection: Dict[str, Any] = {
        "best_metrics": None,
        "best_timestep": None,
        "last_evaluation_metrics": None,
        "last_competence_verdict": None,
        "best_long_sequence_score": None,
        "consecutive_baseline_collapses": 0,
        "dedicated": {
            "last_timesteps": 0,
            "best_success_rate": None,
            "history": [],
        },
        "rollback": None,
    }
    evaluation_state: Dict[str, Any] = {"history": [], "last": None, "best": None}
    initialization_report = None
    if args.resume:
        checkpoint = latest_valid_checkpoint(
            run_dir, expected_contract_hash=contract, safe_only=False
        )
        if checkpoint is None:
            raise RuntimeError("no valid local-transition checkpoint")
        state = load_checkpoint_state(checkpoint)
        from stable_baselines3 import PPO
        model = PPO.load(str(checkpoint.model_path), env=env, device="cpu")
        if int(model.num_timesteps) != int(state["total_timesteps"]):
            raise ValueError("model and transition sidecar timesteps disagree")
        if tuple(model.observation_space.shape) != (OBS_DIM_LOCAL_TRANSITION,):
            raise ValueError("local-transition observation shape changed")
        worker_states = curriculum.load_state_dict(state["curriculum"])
        _restore_workers(env, worker_states)
        restore_model_training_state(model, state["model_training"])
        restore_rng_state(state["rng"])
        extra = state["extra"]
        aliases.update(extra.get("checkpoint_aliases") or {})
        selection.update(extra.get("checkpoint_selection_state") or {})
        evaluation_state.update(state.get("evaluation") or {})
        env.env_method("set_difficulty", curriculum.difficulty)
        env.env_method(
            "set_efficiency_unlocked", curriculum.state.efficiency_reward_active
        )
        model._last_obs = None
        env.reset()
    else:
        if any((run_dir / "checkpoints").glob("*.manifest.json")):
            raise ValueError("run contains checkpoints; use --resume")
        model, initialization_report = _build_new_model(config, env)

    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        atomic_write_json(manifest_path, {
            "schema_version": "universal_transition_run_v1",
            "created_utc": now_utc(),
            "git_sha": git_sha(),
            "config_contract_sha256": contract,
            "config": config,
            "initialization_report": initialization_report,
            "active_controller_scope": "feedforward_ppo_only",
            "historical_sequence_run_preserved": True,
        })
    atomic_write_json(run_dir / "reward_config.json", dict(config.get("reward") or {}))

    stop = {"requested": False}
    def request_stop(signum=None, frame=None):
        stop["requested"] = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    stop_file = run_dir / "STOP_REQUESTED"

    rollout_size = int(config["n_envs"]) * int(config["ppo"]["n_steps"])
    configured_target = int(config["new_environment_steps"])
    target = max(rollout_size, (configured_target // rollout_size) * rollout_size)
    schedule = getattr(model, "learning_rate", None)
    if hasattr(schedule, "set_training_horizon"):
        schedule.set_training_horizon(
            sb3_total_timesteps=target,
            absolute_total_timesteps=target,
        )
    evaluation_frequency = int(config["evaluation"]["frequency"])
    checkpoint_frequency = int(config["evaluation"]["checkpoint_frequency"])
    early_schedule = tuple(
        int(value) for value in (config["evaluation"].get("early_schedule") or ())
    )
    thresholds = thresholds_from_mapping(config.get("competence_gate"))
    rollback_limit = int(
        config["evaluation"].get("rollback_consecutive_evaluations", 2) or 2
    )
    baseline_record = _preserve_initialization_baseline(run_dir, config)
    next_evaluation = _next_evaluation_transition(
        evaluation_state, evaluation_frequency, early_schedule, target
    )
    next_checkpoint = ((int(model.num_timesteps) // checkpoint_frequency) + 1) * checkpoint_frequency
    _status(
        run_dir, state="running", model=model, curriculum=curriculum, env=env,
        aliases=aliases, selection=selection, config=config,
        message="universal transition PPO initialized",
    )
    try:
        while (
            _training_work_pending(
                int(model.num_timesteps), target, next_evaluation
            )
            and not stop["requested"]
            and not stop_file.exists()
        ):
            if int(model.num_timesteps) < target:
                schedule = getattr(model, "learning_rate", None)
                if hasattr(schedule, "set_training_horizon"):
                    schedule.set_training_horizon(
                        sb3_total_timesteps=int(model.num_timesteps) + rollout_size,
                        absolute_total_timesteps=target,
                    )
                started = time.perf_counter()
                model.learn(
                    total_timesteps=rollout_size,
                    reset_num_timesteps=False,
                    progress_bar=False,
                )
                elapsed = time.perf_counter() - started
                curriculum.set_total_environment_transitions(int(model.num_timesteps))
                env.env_method(
                    "set_total_environment_transitions", int(model.num_timesteps)
                )
                with (run_dir / "logs" / "progress.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(json.dumps({
                        "utc": now_utc(),
                        "total_environment_transitions": int(model.num_timesteps),
                        "difficulty": curriculum.difficulty,
                        "learning_rate": float(model.policy.optimizer.param_groups[0]["lr"]),
                        "rollout_wall_time_s": elapsed,
                        "environment_transitions_per_second": rollout_size / max(elapsed, 1e-9),
                    }) + "\n")

            if int(model.num_timesteps) >= next_evaluation:
                output = run_dir / "evaluations" / f"unseen_{int(model.num_timesteps):09d}"
                boundary = _save(
                    model, run_dir, contract, curriculum, env, evaluation_state,
                    aliases, selection, config, status="unverified",
                    reason="pre_evaluation_atomic_boundary",
                )
                _status(
                    run_dir, state="evaluating", model=model,
                    curriculum=curriculum, env=env, aliases=aliases,
                    selection=selection, config=config,
                    message="training workers will be released for evaluation",
                )
                worker_states = _worker_states(env)
                learner_rng = capture_rng_state()
                env.close()
                env = None
                evaluation_error: Optional[BaseException] = None
                report = None
                try:
                    # Evaluate the atomic checkpoint written above rather than
                    # the live model: identical weights, resumable per case, and
                    # the learner object is never touched by evaluator engines.
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
                        parallel_workers=int(
                            config["evaluation"].get(
                                "intermediate_parallel_workers", 1
                            )
                        ),
                    )
                    report["timesteps"] = int(model.num_timesteps)
                    report["evaluation_level"] = "intermediate"
                    dedicated_reason = _dedicated_benchmark_due(
                        metrics=report["metrics"],
                        competent=evaluate_competence_gate(
                            report["metrics"], thresholds
                        ).passed,
                        candidate_best=(
                            selection["best_metrics"] is None
                            or transition_checkpoint_rank_key(
                                report["metrics"], thresholds
                            )
                            > transition_checkpoint_rank_key(
                                selection["best_metrics"], thresholds
                            )
                        ),
                        dedicated_state=selection["dedicated"],
                        timesteps=int(model.num_timesteps),
                        target=target,
                        config=config,
                    )
                    if dedicated_reason is not None:
                        # The learner's engines are already closed here, so the
                        # dedicated benchmark never coexists with rollout
                        # workers.
                        report["dedicated"] = _run_dedicated_benchmark(
                            run_dir,
                            model=model,
                            config=config,
                            difficulty=curriculum.difficulty,
                            reason=dedicated_reason,
                            selection=selection,
                        )
                    curriculum.observe_evaluation(
                        report["metrics"],
                        int(model.num_timesteps),
                        thresholds=thresholds,
                    )
                except BaseException as exc:
                    evaluation_error = exc
                finally:
                    env = _recreate_training_env(
                        model, config, run_dir, curriculum, worker_states
                    )
                    # Neither evaluator engine launches nor replacement worker
                    # construction may perturb the learner's stochastic state.
                    restore_rng_state(learner_rng)
                if evaluation_error is not None:
                    raise evaluation_error
                assert report is not None
                metrics = report["metrics"]
                outcome = apply_evaluation_selection(
                    metrics,
                    checkpoint_path=str(
                        run_dir / "checkpoints"
                        / f"ppo_{int(model.num_timesteps)}_steps.zip"
                    ),
                    timesteps=int(model.num_timesteps),
                    aliases=aliases,
                    selection=selection,
                    thresholds=thresholds,
                    rollback_limit=rollback_limit,
                    baseline_record=baseline_record,
                )
                report["competence"] = outcome["competence"]
                atomic_write_json(output / "evaluation.json", report)
                # Only the summary is carried in checkpoint state; per-episode
                # rows stay in evaluations/*/evaluation.json so that state files
                # do not grow with every scheduled evaluation.
                summary = _evaluation_summary(report, str(output))
                evaluation_state["last"] = summary
                evaluation_state["history"].append(summary)
                if outcome["candidate_better"]:
                    evaluation_state["best"] = summary
                safe = outcome["safe"]
                collapsed = outcome["collapsed"]
                _save(
                    model, run_dir, contract, curriculum, env, evaluation_state,
                    aliases, selection, config,
                    status="safe" if safe else "unsafe",
                    reason="unseen_transition_evaluation",
                )
                next_evaluation = _next_evaluation_transition(
                    evaluation_state, evaluation_frequency, early_schedule, target
                )
                next_checkpoint = max(
                    next_checkpoint,
                    int(model.num_timesteps) + checkpoint_frequency,
                )
                if collapsed:
                    stop["requested"] = True

            if int(model.num_timesteps) >= next_checkpoint:
                _save(
                    model, run_dir, contract, curriculum, env, evaluation_state,
                    aliases, selection, config,
                    status="unverified", reason="periodic",
                )
                next_checkpoint += checkpoint_frequency
            _status(
                run_dir, state="running", model=model, curriculum=curriculum,
                env=env, aliases=aliases, selection=selection, config=config,
                message="universal transition PPO actively advancing",
            )

        if selection.get("rollback"):
            reason = "baseline_collapse_rollback"
        elif stop["requested"] or stop_file.exists():
            reason = "graceful_stop"
        else:
            reason = "target_reached"
        _save(
            model, run_dir, contract, curriculum, env, evaluation_state,
            aliases, selection, config, status="unverified", reason=reason,
            stop_reason=reason,
        )
        _status(
            run_dir,
            state="completed" if reason == "target_reached" else "stopped",
            model=model, curriculum=curriculum, env=env, aliases=aliases,
            selection=selection, config=config, message=reason,
        )
        if stop_file.exists():
            stop_file.unlink()
        return 0
    except Exception as exc:
        if env is not None:
            try:
                _save(
                    model, run_dir, contract, curriculum, env, evaluation_state,
                    aliases, selection, config, status="unverified",
                    reason="exception", stop_reason=repr(exc),
                )
                _status(
                    run_dir, state="failed", model=model, curriculum=curriculum,
                    env=env, aliases=aliases, selection=selection, config=config,
                    message=repr(exc),
                )
            except Exception:
                pass
        (run_dir / "failure.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
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
    if pid > 0:
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
    status = run / "status.json"
    if not status.exists():
        raise FileNotFoundError(status)
    stop = run / "STOP_REQUESTED"
    stop.touch(exist_ok=True)
    print(json.dumps({
        "stop_requested": True,
        "stop_file": str(stop),
        "mechanism": "atomic checkpoint after current rollout/evaluation",
    }, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--run-dir")
    train.add_argument("--steps", type=int)
    train.add_argument("--n-envs", type=int, choices=(1, 2, 4))
    train.add_argument("--resume", action="store_true")
    train.add_argument("--allow-dirty-smoke", action="store_true")
    status = sub.add_parser("status")
    status.add_argument("run_dir")
    stop = sub.add_parser("stop")
    stop.add_argument("run_dir")
    args = parser.parse_args(argv)
    if args.command == "train":
        return run_training(args)
    if args.command == "status":
        return status_command(args.run_dir)
    return stop_command(args.run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
