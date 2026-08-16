"""Atomic feed-forward PPO training for universal local gate transitions."""

from __future__ import annotations

import argparse
import copy
import json

import numpy as np
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
    atomic_append_jsonl,
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
from marine_race_arena.learning.ppo_lr_rewarm import (
    LearningRateRewarm,
    rewarm_policy_from_mapping,
)
from marine_race_arena.learning.provenance import git_sha, now_utc
from marine_race_arena.learning.rl_evaluation_scheduler import scheduled_evaluation
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
    "parallel_continuation",
)
WARM_START_MODES = ("selective_warm_start", "warm_start_transition_checkpoint")
# Adding rollout workers must not silently multiply the PPO batch.  The
# effective rollout stays within this tolerance of the successful 2,048-step
# configuration so the continuation remains comparable to the policy history it
# extends.
BASELINE_ROLLOUT_SIZE = 2048
ROLLOUT_SIZE_TOLERANCE = 0.25
BASELINE_BATCH_SIZE = 256
MINIBATCH_ALIGNMENT = 64


def supported_worker_shardings(
    baseline: int = BASELINE_ROLLOUT_SIZE,
    tolerance: float = ROLLOUT_SIZE_TOLERANCE,
    worker_counts: Sequence[int] = (1, 2, 4, 6, 8, 10, 12, 16),
    batch_size: int = BASELINE_BATCH_SIZE,
) -> Dict[int, int]:
    """Return ``{n_envs: n_steps_per_env}`` keeping the rollout near ``baseline``.

    The total rollout must stay within ``tolerance`` of the size the parent run
    learned with and must remain an exact multiple of the minibatch size, so a
    reshard changes only *who collects* the transitions, never the number of
    gradient steps PPO takes per rollout.  ``n_steps`` is a multiple of 64.
    """

    low = baseline * (1.0 - float(tolerance))
    high = baseline * (1.0 + float(tolerance))
    out: Dict[int, int] = {}
    for workers in worker_counts:
        best = None
        for steps in range(MINIBATCH_ALIGNMENT, baseline + 1, MINIBATCH_ALIGNMENT):
            rollout = int(workers) * steps
            if not low <= rollout <= high or rollout % int(batch_size):
                continue
            score = (abs(rollout - baseline), steps)
            if best is None or score < best[0]:
                best = (score, steps)
        if best is not None:
            out[int(workers)] = int(best[1])
    return out


def reshard_workers(config: Dict[str, Any], n_envs: int) -> Dict[str, Any]:
    """Apply a supported worker sharding, preserving the effective rollout."""

    shardings = supported_worker_shardings()
    if int(n_envs) not in shardings:
        raise ValueError(
            f"worker count {n_envs} has no sharding within "
            f"{ROLLOUT_SIZE_TOLERANCE:.0%} of {BASELINE_ROLLOUT_SIZE}; "
            f"supported: {sorted(shardings)}"
        )
    steps = shardings[int(n_envs)]
    config["n_envs"] = int(n_envs)
    config["ppo"]["n_steps"] = int(steps)
    return config


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
    low = BASELINE_ROLLOUT_SIZE * (1.0 - ROLLOUT_SIZE_TOLERANCE)
    high = BASELINE_ROLLOUT_SIZE * (1.0 + ROLLOUT_SIZE_TOLERANCE)
    if not low <= rollout <= high:
        raise ValueError(
            f"total rollout size must stay within {ROLLOUT_SIZE_TOLERANCE:.0%} of "
            f"{BASELINE_ROLLOUT_SIZE}, got {rollout}"
        )
    if int(value["ppo"]["n_steps"]) % MINIBATCH_ALIGNMENT:
        raise ValueError(
            f"n_steps must be a multiple of {MINIBATCH_ALIGNMENT} so every "
            "worker sharding yields the same minibatch count"
        )
    if rollout % int(value["ppo"]["batch_size"]):
        raise ValueError("rollout size must divide evenly into PPO minibatches")
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
    # The PPO trainer runs from either the PPO worktree or the SAC worktree,
    # whose branch also carries the parallel-continuation and governor code.
    if branch not in {
        "feature/rl-universal-gate-transition",
        "feature/rl-multistep-sac-transition",
    }:
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
    elif initialization["mode"] == "parallel_continuation":
        parent_path, parent_manifest = _parent_checkpoint(config)
        source = str(parent_path)
        actual_sha = parent_manifest["model_sha256"]
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
    sequence_curriculum: Mapping[str, Any] | None = None,
    hard_case_mixture: Mapping[str, Any] | None = None,
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
        sequence_curriculum=sequence_curriculum,
        hard_case_mixture=hard_case_mixture,
        dataset_split="train",
        algorithm="ppo",
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
            sequence_curriculum=dict(config.get("sequence_curriculum") or {}),
            hard_case_mixture=dict(config.get("hard_case_mixture") or {}),
        )
        for index in range(n_envs)
    ]
    if n_envs == 1:
        return DummyVecEnv(factories)
    # Explicit spawn is required on Windows and prevents HoloOcean handles from
    # being inherited across simulation processes.
    return SubprocVecEnv(factories, start_method="spawn")


def close_vec_env_safely(env: Any, *, timeout: float = 60.0) -> Dict[str, Any]:
    """Close rollout workers without masking a failure or blocking forever.

    When a worker dies -- a HoloOcean engine start can time out under load --
    SubprocVecEnv.close() waits on a pipe that will never answer.  In the
    production run that turned a recoverable engine timeout into a process that
    hung for hours holding the run directory.  Closing is therefore bounded: on
    timeout the worker processes are terminated directly so the trainer can exit
    cleanly and be resumed from its last atomic checkpoint.
    """

    import threading

    if env is None:
        return {"closed": True, "method": "no_env"}
    outcome: Dict[str, Any] = {"closed": False, "method": "graceful"}

    def _close():
        try:
            env.close()
            outcome["closed"] = True
        except BaseException as exc:  # a dead worker must not mask the cause
            outcome["error"] = repr(exc)

    thread = threading.Thread(target=_close, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive() or not outcome["closed"]:
        outcome["method"] = "forced"
        for process in list(getattr(env, "processes", []) or []):
            try:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10)
                if process.is_alive():
                    process.kill()
            except BaseException:
                continue
    return outcome


def _worker_states(env: Any) -> list[Mapping[str, Any]]:
    return list(env.env_method("worker_state"))


def _worker_identities(env: Any) -> list[Mapping[str, Any]]:
    return list(env.env_method("worker_identity"))


def _hard_case_composition(env: Any) -> Optional[Dict[str, Any]]:
    """The base/hard mixture the live workers actually realised.

    A configured mixture that never materialises -- a worker built without it,
    or a cap that keeps firing -- looks exactly like a healthy run in every
    other status field.  Surfaced while the run is live so the realised share
    can be compared against the requested one before the run is over, not
    reconstructed from the episode log afterwards.
    """

    reported = [state for state in env.env_method("hard_case_composition") if state]
    if not reported:
        return None
    episodes = sum(int(state["episodes"]) for state in reported)
    hard = sum(int(state["hard_episodes"]) for state in reported)
    families: Dict[str, int] = {}
    for state in reported:
        for name, count in dict(state.get("family_counts") or {}).items():
            families[str(name)] = families.get(str(name), 0) + int(count)
    return {
        "schema_version": reported[0].get("schema_version"),
        # Reported separately from the worker count so "one worker silently
        # started without the mixture" is visible in the status file itself.
        "workers_reporting": len(reported),
        "workers_total": int(getattr(env, "num_envs", len(reported)) or len(reported)),
        "episodes": episodes,
        "base_episodes": sum(int(state["base_episodes"]) for state in reported),
        "hard_episodes": hard,
        "requested_hard_fraction": reported[0].get("requested_hard_fraction"),
        "realised_hard_fraction": round(hard / episodes, 6) if episodes else 0.0,
        "max_hard_fraction": reported[0].get("max_hard_fraction"),
        "hard_fraction_within_cap": all(
            bool(state.get("hard_fraction_within_cap")) for state in reported
        ),
        "capped_draws": sum(int(state.get("capped_draws", 0)) for state in reported),
        "family_counts": dict(sorted(families.items())),
        "family_share_of_hard": {
            name: round(count / hard, 6) for name, count in sorted(families.items())
        } if hard else {},
        "per_worker": reported,
    }


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
    if initialization["mode"] == "parallel_continuation":
        # The verified parent is the behaviour the continuation must not lose.
        source = str(_parent_checkpoint(config)[0])
    elif initialization["mode"] not in WARM_START_MODES or not source:
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
    with scheduled_evaluation(
        evaluation,
        fixed_workers=int(evaluation.get("dedicated_parallel_workers", 2)),
        owner=f"ppo:{run_dir.name}:dedicated:{timesteps}",
        audit_path=run_dir / "logs" / "evaluation_allocation.jsonl",
    ) as allocation:
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
            parallel_workers=int(allocation["selected_workers"]),
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
        "hard_case_mixture": {
            "requested": dict(config.get("hard_case_mixture") or {}),
            "realised": _hard_case_composition(env),
        },
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


def _parent_checkpoint(config: Mapping[str, Any]):
    """Resolve and hash-verify the parent checkpoint of a continuation."""

    initialization = config["initialization"]
    parent_run = Path(initialization["parent_run_dir"])
    declared = initialization.get("parent_checkpoint")
    if not declared:
        raise ValueError("parallel_continuation requires parent_checkpoint")
    path = Path(declared)
    if not path.is_absolute() and not path.exists():
        path = parent_run / "checkpoints" / Path(declared).name
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = path.parent / f"{path.stem}.manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    digest = sha256_file(path)
    if digest != value["model_sha256"]:
        raise ValueError("parent checkpoint hash does not match its manifest")
    declared_sha = initialization.get("parent_sha256")
    if declared_sha and declared_sha != digest:
        raise ValueError("parent checkpoint hash does not match the config")
    if value["observation_version"] != OBS_ENCODING_VERSION_LOCAL_TRANSITION:
        raise ValueError("parent checkpoint uses a different observation contract")
    if int(value["policy_observation_dim"]) != OBS_DIM_LOCAL_TRANSITION:
        raise ValueError("parent checkpoint uses a different observation dimension")
    if value["action_version"] != ACTION_CONTRACT_VERSION:
        raise ValueError("parent checkpoint uses a different action contract")
    return path, value


def _policy_vector(model: Any):
    """Flat snapshot of the policy parameters, for measuring update magnitude."""

    import torch

    return torch.cat([p.detach().reshape(-1).clone() for p in model.policy.parameters()])


def _optimizer_activity(model: Any, before: Any) -> Dict[str, Any]:
    """What the last PPO update actually did.

    ``approx_kl`` and ``clip_fraction`` come from SB3's own logger; the
    parameter movement is measured directly because a near-zero KL with a
    near-zero clip fraction is exactly what a stalled optimizer looks like.
    """

    import torch

    logged = dict(getattr(getattr(model, "logger", None), "name_to_value", {}) or {})
    after = _policy_vector(model)
    with torch.no_grad():
        delta = float((after - before).norm())
        scale = float(before.norm())
    return {
        "approx_kl": float(logged.get("train/approx_kl", 0.0) or 0.0),
        "clip_fraction": float(logged.get("train/clip_fraction", 0.0) or 0.0),
        "entropy_loss": float(logged.get("train/entropy_loss", 0.0) or 0.0),
        "value_loss": float(logged.get("train/value_loss", 0.0) or 0.0),
        "policy_gradient_loss": float(logged.get("train/policy_gradient_loss", 0.0) or 0.0),
        "explained_variance": float(logged.get("train/explained_variance", 0.0) or 0.0),
        "policy_update_norm": delta,
        "relative_policy_update": delta / max(1e-12, scale),
    }


def _install_continuation_schedule(model: Any, config: Mapping[str, Any]) -> None:
    """Give the continuation its own learning-rate schedule.

    ``AbsoluteLearningRateSchedule`` ratchets: ``last_value = min(last_value,
    value)``, deliberately so that *extending* a run can never restart a decayed
    schedule.  The parent's schedule arrives exhausted
    (current_progress_remaining 0.0, last_value at its floor), so a continuation
    that inherits it can never reach its own configured rate no matter what the
    optimizer param groups say -- SB3 re-reads the schedule on every ``train()``
    call and would immediately pull the rate back down.

    A new continuation is not an extension: it gets a fresh schedule over its
    own horizon, which is the one path the ratchet's docstring leaves open.
    """

    ppo = config["ppo"]
    schedule = AbsoluteLearningRateSchedule(
        float(ppo["learning_rate"]),
        float(ppo["final_learning_rate"]),
        str(ppo["learning_rate_schedule"]),
    )
    model.learning_rate = schedule
    model.lr_schedule = schedule
    # The new run starts at the beginning of its own horizon.
    model._current_progress_remaining = 1.0


def _position_continuation_schedule(
    model: Any, *, current_timesteps: int, target_timesteps: int
) -> None:
    """Put a fresh continuation schedule at the run's real resume position.

    Verifying the freshly installed schedule at progress ``1.0`` is insufficient:
    the training loop gives it the absolute horizon immediately before
    ``learn()``, and SB3 then queries it at the current absolute transition.  The
    old resume repair passed its launch assertion but still fell back to the
    decayed value on the first real update.  Querying the schedule once at the
    resume position lets ``_apply_learning_rate`` express the intended controller
    rate relative to the correct base value.
    """

    schedule = getattr(model, "learning_rate", None)
    if not hasattr(schedule, "set_training_horizon"):
        return
    current = max(1, int(current_timesteps))
    target = int(target_timesteps)
    schedule.set_training_horizon(
        sb3_total_timesteps=current,
        absolute_total_timesteps=target,
    )
    # With sb3_total_timesteps=current, progress 0.0 denotes exactly the
    # checkpoint boundary.  AbsoluteLearningRateSchedule deliberately ratchets
    # this value, which is what the subsequent re-warm scaling must start from.
    schedule(0.0)


def _intended_resume_learning_rate(
    config: Mapping[str, Any], run_dir: Path
) -> float:
    """The rate a resumed run should come back at.

    Not simply the configured value: the re-warm controller may legitimately have
    moved the rate during the run, and throwing that decision away on every
    restart would make the effective rate depend on how often the machine was
    rebooted.  So the last re-warm decision wins when there is one, clamped to
    the policy's own bounds, and the configured rate is the fallback.
    """

    configured = float(config["ppo"]["learning_rate"])
    spec = config.get("lr_rewarm")
    path = Path(run_dir) / "logs" / "lr_rewarm.jsonl"
    if not spec or not path.is_file():
        return configured
    # Only records that actually CHANGED the rate count as decisions.  Every
    # rollout logs the rate it observed, so reading the last line would adopt
    # whatever the optimizer happened to be running -- including a rate a broken
    # resume had just imposed, which would launder the bug into the intended
    # value and make it permanent.
    last: Optional[float] = None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("changed"):
                continue
            value = record.get("learning_rate")
            if value is not None:
                last = float(value)
    except (OSError, ValueError, TypeError):
        return configured
    if last is None:
        return configured
    policy = rewarm_policy_from_mapping(spec)
    return min(
        max(last, float(policy.minimum_learning_rate)),
        float(policy.maximum_learning_rate),
    )


def assert_learning_rate_applied(
    model: Any, *, expected: float, restored: float, configured: float,
    tolerance: float = 1e-12,
) -> Dict[str, Any]:
    """Fail loudly unless every optimizer param group runs the intended rate.

    A run that reports one learning rate in its manifest while the optimizer
    uses another is indistinguishable from a healthy run until someone reads
    the raw parameter deltas weeks later.
    """

    groups = [float(g["lr"]) for g in model.policy.optimizer.param_groups]
    wrong = [value for value in groups if abs(value - float(expected)) > tolerance]
    if wrong:
        raise ValueError(
            "PPO continuation learning rate not applied: expected "
            f"{expected!r}, optimizer param groups {groups!r}"
        )
    schedule = getattr(model, "learning_rate", None)
    schedule_value = None
    if callable(schedule):
        # SB3 re-reads the schedule at the start of every train(); if it does
        # not yield the intended rate the param groups will simply be undone.
        schedule_value = float(schedule(float(model._current_progress_remaining)))
        if abs(schedule_value - float(expected)) > max(tolerance, 1e-9):
            raise ValueError(
                "PPO continuation schedule does not yield the intended rate: "
                f"expected {expected!r}, schedule returns {schedule_value!r}"
            )
    elif hasattr(schedule, "last_value"):
        schedule_value = float(schedule.last_value)
    return {
        "configured_initial_lr": float(configured),
        "restored_parent_lr": float(restored),
        "effective_optimizer_lr": groups[0],
        "optimizer_param_group_lrs": groups,
        "scheduler_reported_lr": schedule_value,
        "lr_rewarm_verified": True,
    }


def _apply_learning_rate(model: Any, learning_rate: float) -> None:
    """Set the optimizer rate and keep the schedule consistent with it."""

    intended = float(learning_rate)
    schedule = getattr(model, "learning_rate", None)
    if callable(schedule):
        progress = float(getattr(model, "_current_progress_remaining", 1.0))
        scheduled = float(schedule(progress))
        multiplier = getattr(schedule, "multiplier", None)
        if (
            multiplier is not None
            and hasattr(schedule, "last_value")
            and scheduled > 0.0
            and abs(float(multiplier)) > 1e-15
        ):
            # Preserve the schedule's base curve while making the controller's
            # explicit rate the value SB3 will see at this transition.  Calling
            # scale() is not sufficient here: its 1.5x single-step bound can
            # prevent a legitimate bounded re-warm decision from surviving the
            # next SB3 schedule query.
            nominal = scheduled / float(multiplier)
            schedule.multiplier = intended / nominal
            schedule.last_value = intended
    for group in model.policy.optimizer.param_groups:
        group["lr"] = intended


def policy_parameter_sha256(model: Any) -> str:
    """Content hash of a PPO policy's parameters (weights + log_std)."""

    import hashlib

    import torch

    digest = hashlib.sha256()
    state = model.policy.state_dict()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(
            torch.as_tensor(state[name]).detach().cpu().contiguous().numpy().tobytes()
        )
    return digest.hexdigest()


def assert_policy_matches_parent(model: Any, parent_path: "str | Path") -> Dict[str, Any]:
    """Fail loudly when a continuation is not actually continuing anything.

    A continuation that quietly starts from random weights is indistinguishable
    from a healthy run until the first unseen evaluation returns 0% -- which is
    exactly how ~51k transitions were spent training a fresh network that the
    reports still described as an 84%-success continuation.
    """

    from stable_baselines3 import PPO

    parent = PPO.load(str(parent_path), device="cpu")
    live_hash = policy_parameter_sha256(model)
    parent_hash = policy_parameter_sha256(parent)
    if live_hash != parent_hash:
        raise ValueError(
            "parallel continuation did not preserve the parent policy: "
            f"live {live_hash} != parent {parent_hash}"
        )
    return {
        "policy_parity_verified": True,
        "policy_sha256": live_hash,
        "parent_checkpoint": str(parent_path),
    }


def _continue_from_parent(config: Mapping[str, Any], env: Any, run_dir: Path):
    """Continue a verified parent run under a new worker sharding.

    Everything that defines *where the policy is in its learning trajectory* is
    inherited: actor and critic weights, optimizer moments, learning-rate
    schedule progress, total transition count, curriculum, evaluation history
    and checkpoint ranking.  Only the transient rollout collection state is
    rebuilt, because the worker count changed.
    """

    from stable_baselines3 import PPO

    path, manifest = _parent_checkpoint(config)
    parent_state = json.loads(
        (path.parent / f"{path.stem}.state.json").read_text(encoding="utf-8")
    )
    model = PPO.load(str(path), env=env, device="cpu")
    if tuple(model.observation_space.shape) != (OBS_DIM_LOCAL_TRANSITION,):
        raise ValueError("parent observation shape changed")
    # ``_setup_model`` rebuilds the rollout buffer for the new (n_envs, n_steps)
    # geometry -- but it also constructs a brand new randomly initialised policy
    # and optimizer, silently discarding everything ``PPO.load`` just restored.
    # That is how a run advertised as a continuation of an 84%-success parent
    # actually trained a fresh network: |theta| fell from 488.75 to the 34.13
    # initialisation scale and log_std reverted to SB3's 0.0 default, so the
    # deterministic policy emitted ~0 actions at every evaluation.  Snapshot the
    # weights and optimizer moments first, then put them back.
    policy_state = copy.deepcopy(model.policy.state_dict())
    optimizer_state = copy.deepcopy(model.policy.optimizer.state_dict())
    model.n_steps = int(config["ppo"]["n_steps"])
    model.n_envs = int(env.num_envs)
    model._setup_model()
    model.policy.load_state_dict(policy_state)
    model.policy.optimizer.load_state_dict(optimizer_state)
    model.set_env(env)
    model._last_obs = None
    restore_model_training_state(model, parent_state["model_training"])
    # ``restore_model_training_state`` restores the parent's *learning rate* as
    # well as its optimizer moments and schedule progress: it assigns the stored
    # rate onto every optimizer param group and reloads the schedule's
    # last_value.  A continuation that asks for a different rate therefore had
    # its request silently discarded -- PPO v4 advertised 7e-6 and actually ran
    # at 3.489754098360656e-06, byte-identical to its parent.  Historical
    # moments stay useful; the historical *rate* must not outrank the new run's
    # explicit policy, so it is re-established here, after the restore.
    restored_learning_rate = float(model.policy.optimizer.param_groups[0]["lr"])
    continuation_learning_rate = float(config["ppo"]["learning_rate"])
    _install_continuation_schedule(model, config)
    _position_continuation_schedule(
        model,
        current_timesteps=int(model.num_timesteps),
        target_timesteps=int(config["new_environment_steps"]),
    )
    _apply_learning_rate(model, continuation_learning_rate)
    learning_rate_report = assert_learning_rate_applied(
        model,
        expected=continuation_learning_rate,
        restored=restored_learning_rate,
        configured=continuation_learning_rate,
    )
    if int(model.num_timesteps) != int(manifest["total_timesteps"]):
        raise ValueError("parent model and sidecar disagree on total timesteps")
    # Matching counters prove nothing about the weights.  Compare the live
    # policy against the parent on disk: this is the assertion that would have
    # stopped the lost run on its very first rollout.
    parity = assert_policy_matches_parent(model, path)
    report = {
        "mode": "parallel_continuation",
        "parent_run_dir": str(config["initialization"]["parent_run_dir"]),
        "parent_checkpoint": str(path),
        "parent_sha256": manifest["model_sha256"],
        "parent_total_environment_transitions": int(manifest["total_timesteps"]),
        "parent_status": manifest.get("status"),
        "parent_reason": manifest.get("reason"),
        "reshard_reason": config["initialization"].get(
            "reshard_reason", "increase measured valid transitions per second"
        ),
        "previous_n_envs": int(
            config["initialization"].get("parent_n_envs", 0) or 0
        ),
        "previous_n_steps": int(
            config["initialization"].get("parent_n_steps", 0) or 0
        ),
        "n_envs": int(config["n_envs"]),
        "n_steps": int(config["ppo"]["n_steps"]),
        "rollout_size": int(config["n_envs"]) * int(config["ppo"]["n_steps"]),
        "actor_and_critic_weights_preserved": True,
        "optimizer_state_preserved": True,
        "learning_rate_schedule_preserved": True,
        "total_environment_transitions_preserved": int(model.num_timesteps),
        "rollout_buffer_reset": True,
        "workers_recreated": True,
        "target_environment_transitions": int(config["new_environment_steps"]),
        **parity,
        **learning_rate_report,
    }
    return model, report, parent_state


def _inherit_parent_history(
    parent_state: Mapping[str, Any],
    *,
    curriculum: TransitionCurriculumController,
    aliases: Dict[str, Any],
    selection: Dict[str, Any],
    evaluation_state: Dict[str, Any],
) -> None:
    """Carry ranking, curriculum and evaluation history into the continuation."""

    curriculum.load_state_dict(parent_state["curriculum"])
    extra = dict(parent_state.get("extra") or {})
    for key, value in (extra.get("checkpoint_aliases") or {}).items():
        if key in aliases and value:
            aliases[key] = value
    parent_selection = dict(extra.get("checkpoint_selection_state") or {})
    for key in (
        "best_metrics", "best_timestep", "last_evaluation_metrics",
        "last_competence_verdict", "best_long_sequence_score",
        "consecutive_baseline_collapses", "dedicated",
    ):
        if parent_selection.get(key) is not None:
            selection[key] = parent_selection[key]
    parent_evaluation = dict(parent_state.get("evaluation") or {})
    if parent_evaluation.get("history"):
        evaluation_state["history"] = list(parent_evaluation["history"])
    for key in ("last", "best"):
        if parent_evaluation.get(key) is not None:
            evaluation_state[key] = parent_evaluation[key]


def run_training(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = _run_dir(config, None)
    if args.steps is not None:
        config["new_environment_steps"] = int(args.steps)
    if args.n_envs is not None:
        reshard_workers(config, int(args.n_envs))
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
        checkpoint_policy_sha256 = policy_parameter_sha256(model)
        worker_states = curriculum.load_state_dict(state["curriculum"])
        _restore_workers(env, worker_states)
        restored_learning_rate = float(
            (state["model_training"].get("optimizer_learning_rates") or [0.0])[0]
        )
        restore_model_training_state(model, state["model_training"])
        # Same defect as the continuation path, and it bit for real: restoring the
        # checkpoint reinstates its ratcheted AbsoluteLearningRateSchedule, and
        # because that schedule is absolute over the whole horizon it reports a
        # decayed rate the moment SB3 re-reads it.  A run paused at 4.5e-06 came
        # back at 3.2048e-06 -- a 29% cut nothing announced.  Resuming must
        # therefore restate the intended rate exactly as a fresh continuation
        # does, and prove it reached every optimizer param group.
        resume_learning_rate = _intended_resume_learning_rate(config, run_dir)
        _install_continuation_schedule(model, config)
        _position_continuation_schedule(
            model,
            current_timesteps=int(model.num_timesteps),
            target_timesteps=int(config["new_environment_steps"]),
        )
        _apply_learning_rate(model, resume_learning_rate)
        resume_lr_report = assert_learning_rate_applied(
            model,
            expected=resume_learning_rate,
            restored=restored_learning_rate,
            configured=float(config["ppo"]["learning_rate"]),
        )
        resumed_policy_sha256 = policy_parameter_sha256(model)
        if resumed_policy_sha256 != checkpoint_policy_sha256:
            raise ValueError(
                "PPO resume changed policy parameters before the first update"
            )
        atomic_write_json(run_dir / "logs" / "resume_verification.json", {
            "schema_version": "ppo_resume_verification_v1",
            "utc": now_utc(),
            "checkpoint": str(checkpoint.model_path),
            "checkpoint_model_sha256": checkpoint.manifest["model_sha256"],
            "total_environment_transitions": int(model.num_timesteps),
            "intended_learning_rate": float(resume_learning_rate),
            "policy_sha256_before_restore": checkpoint_policy_sha256,
            "policy_sha256_after_restore": resumed_policy_sha256,
            "policy_parity_verified": True,
            **resume_lr_report,
        })
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
    elif config["initialization"]["mode"] == "parallel_continuation":
        if any((run_dir / "checkpoints").glob("*.manifest.json")):
            raise ValueError("run contains checkpoints; use --resume")
        model, initialization_report, parent_state = _continue_from_parent(
            config, env, run_dir
        )
        _inherit_parent_history(
            parent_state, curriculum=curriculum, aliases=aliases,
            selection=selection, evaluation_state=evaluation_state,
        )
        # Worker sampler state is deliberately not restored: the reshard changes
        # the worker count, so each worker starts a fresh deterministic sampler
        # from its own seed while everything defining the learning trajectory is
        # inherited from the parent.
        env.env_method("set_difficulty", curriculum.difficulty)
        env.env_method(
            "set_efficiency_unlocked", curriculum.state.efficiency_reward_active
        )
        env.env_method(
            "set_total_environment_transitions", int(model.num_timesteps)
        )
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
    rewarm = (
        LearningRateRewarm(
            rewarm_policy_from_mapping(config.get("lr_rewarm")),
            learning_rate=float(model.policy.optimizer.param_groups[0]["lr"]),
        )
        if config.get("lr_rewarm") else None
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
                policy_before = _policy_vector(model)
                model.learn(
                    total_timesteps=rollout_size,
                    reset_num_timesteps=False,
                    progress_bar=False,
                )
                elapsed = time.perf_counter() - started
                update_optimizer_lrs = [
                    float(group["lr"])
                    for group in model.policy.optimizer.param_groups
                ]
                schedule = getattr(model, "learning_rate", None)
                update_schedule_lr = (
                    float(schedule(float(model._current_progress_remaining)))
                    if callable(schedule) else None
                )
                if update_schedule_lr is not None and any(
                    abs(value - update_schedule_lr) > 1e-12
                    for value in update_optimizer_lrs
                ):
                    raise RuntimeError(
                        "PPO optimizer and learning-rate schedule diverged after "
                        f"train(): groups={update_optimizer_lrs!r}, "
                        f"schedule={update_schedule_lr!r}"
                    )
                # One rollout completed: measure what the optimizer actually did
                # and let the bounded KL-aware controller adjust the step size.
                # The live v3 run moved 1.3e-5 (relative) per rollout with KL
                # near zero and clipping inactive -- competent, but not learning.
                activity = _optimizer_activity(model, policy_before)
                rewarm_record = (
                    rewarm.observe(activity, timesteps=int(model.num_timesteps))
                    if rewarm is not None else None
                )
                if rewarm_record is not None:
                    # Log EVERY rollout, not only rate changes.  Logging only
                    # changes made "controller never fired" indistinguishable
                    # from "controller never ran", which is exactly why v4's
                    # zero decisions could not be diagnosed from its logs.
                    atomic_append_jsonl(run_dir / "logs" / "lr_rewarm.jsonl", rewarm_record)
                    if rewarm_record["changed"]:
                        _apply_learning_rate(model, rewarm.learning_rate)
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
                        "learning_rate": update_optimizer_lrs[0],
                        "intended_learning_rate": (
                            float(rewarm_record["learning_rate_before"])
                            if rewarm_record is not None else update_optimizer_lrs[0]
                        ),
                        "optimizer_param_group_lrs": update_optimizer_lrs,
                        "scheduler_learning_rate": update_schedule_lr,
                        "next_learning_rate": float(
                            model.policy.optimizer.param_groups[0]["lr"]
                        ),
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
                close_vec_env_safely(env)
                env = None
                evaluation_error: Optional[BaseException] = None
                report = None
                try:
                    # Evaluate the atomic checkpoint written above rather than
                    # the live model: identical weights, resumable per case, and
                    # the learner object is never touched by evaluator engines.
                    with scheduled_evaluation(
                        config["evaluation"],
                        fixed_workers=int(config["evaluation"].get("intermediate_parallel_workers", 1)),
                        owner=f"ppo:{run_dir.name}:intermediate:{int(model.num_timesteps)}",
                        audit_path=run_dir / "logs" / "evaluation_allocation.jsonl",
                    ) as allocation:
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
                            parallel_workers=int(allocation["selected_workers"]),
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
            close_vec_env_safely(env)


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
    train.add_argument(
        "--n-envs", type=int,
        choices=tuple(sorted(supported_worker_shardings())),
    )
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
