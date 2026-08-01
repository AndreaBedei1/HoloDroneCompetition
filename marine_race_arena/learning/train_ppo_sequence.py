"""PPO-only S1--S5 sequence training with atomic graceful resume.

The action path contains only ``model.predict`` during evaluation and SB3 PPO
rollouts during training. Deterministic code is limited to onboard target
tracking, crossing/progression state, reward, termination, and scoring.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from marine_race_arena.learning.config import ACTION_CONTRACT_VERSION
from marine_race_arena.learning.config_sequence import (
    OBS_DIM_SEQUENCE,
    OBS_ENCODING_VERSION_SEQUENCE,
)
from marine_race_arena.learning.longrun_checkpoint import (
    atomic_save_checkpoint,
    atomic_write_json,
    canonical_hash,
    latest_valid_checkpoint,
    load_checkpoint_state,
    restore_model_training_state,
    restore_rng_state,
    sha256_file,
)
from marine_race_arena.learning.provenance import git_sha, now_utc
from marine_race_arena.learning.reward_sequence import (
    SequenceRewardConfig,
    SequenceTrainingReward,
)
from marine_race_arena.learning.seed_registry import (
    PPO_SEQUENCE_CURRICULUM_EVAL_SEEDS,
    PPO_SEQUENCE_OFFICIAL_HOLDOUT_SEEDS,
    PPO_SEQUENCE_TRAINING_SEEDS,
    PPO_SEQUENCE_UNSEEN_HOLDOUT_SEEDS,
)
from marine_race_arena.learning.sequence_curriculum import SequenceCurriculumSampler
from marine_race_arena.learning.sequence_env import SequenceCurriculumEnv
from marine_race_arena.learning.sequence_evaluation import (
    OFFICIAL_TRACKS,
    evaluate_sequence_suite,
    is_better_sequence_checkpoint,
)
from marine_race_arena.learning.sequence_policy import (
    build_sequence_ppo,
    initialize_from_ppo900462,
    load_sequence_model,
)
from marine_race_arena.learning.train_multigate_longrun import AbsoluteLearningRateSchedule


def _load_config(path: str | Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "run_name", "output_root", "initialization_checkpoint",
        "initialization_sha256", "architecture", "new_environment_steps",
        "seed", "ppo", "curriculum", "evaluation",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"sequence config missing {missing}")
    if value.get("observation_version") != OBS_ENCODING_VERSION_SEQUENCE:
        raise ValueError("sequence observation contract changed")
    if value.get("action_version") != ACTION_CONTRACT_VERSION:
        raise ValueError("sequence action contract changed")
    active = value.get("active_workflow", {})
    if active.get("controller_family") != "ppo_only" or active.get("expert_actions_or_losses") is not False:
        raise ValueError("active sequence workflow must remain PPO-only")
    return value


def _run_dir(config: Mapping[str, Any], override: Optional[str]) -> Path:
    return Path(override) if override else Path(config["output_root"]) / config["run_name"]


def _contract(config: Mapping[str, Any]) -> str:
    stable = dict(config)
    stable.pop("new_environment_steps", None)
    return canonical_hash({"sequence_config": stable, "training_code_sha": git_sha()})


def _run_contract(config: Mapping[str, Any], run_dir: Path) -> str:
    manifest = run_dir / "run_manifest.json"
    if manifest.exists():
        value = json.loads(manifest.read_text(encoding="utf-8"))
        stored = value.get("config_contract_sha256")
        if isinstance(stored, str) and len(stored) == 64:
            return stored
    return _contract(config)


def _check_preflight(config: Mapping[str, Any], *, allow_dirty: bool) -> Dict[str, Any]:
    source = Path(config["initialization_checkpoint"])
    if not source.exists():
        raise FileNotFoundError(source)
    actual = sha256_file(source)
    if actual != config["initialization_sha256"]:
        raise ValueError(f"initialization hash mismatch: {actual}")
    branch = subprocess.run(
        ["git", "branch", "--show-current"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if branch != "feature/rl-multigate-reliability-first":
        raise ValueError(f"wrong branch {branch!r}")
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], check=True,
        capture_output=True, text=True,
    ).stdout.strip())
    if dirty and not allow_dirty:
        raise ValueError("worktree is dirty; commit implementation before long training")
    if config.get("adapter") != "holoocean" or config.get("allow_fallback"):
        raise ValueError("long sequence training requires real HoloOcean without fallback")
    return {
        "checked_utc": now_utc(), "branch": branch, "git_sha": git_sha(),
        "worktree_dirty": dirty, "initialization_checkpoint": str(source),
        "initialization_sha256": actual, "adapter": config["adapter"],
        "ppo_only": True, "observation_dim": OBS_DIM_SEQUENCE,
    }


def _make_sampler(config: Mapping[str, Any]) -> SequenceCurriculumSampler:
    curriculum = config["curriculum"]
    return SequenceCurriculumSampler(
        seed=int(PPO_SEQUENCE_TRAINING_SEEDS[0] + int(config["seed"]) % 1000),
        initial_stage=curriculum["initial_stage"],
        maximum_stage=curriculum["maximum_stage"],
        replay_mixture=curriculum["replay_mixture"],
    )


def _make_env(config: Mapping[str, Any], run_dir: Path, sampler: SequenceCurriculumSampler, reward: SequenceRewardConfig):
    return SequenceCurriculumEnv(
        sampler, run_dir=run_dir, seed=int(PPO_SEQUENCE_TRAINING_SEEDS[0]),
        reward_factory=lambda: SequenceTrainingReward(reward),
        env_kwargs={
            "adapter": config["adapter"], "allow_fallback": False,
            "current_profile": "none", "max_steps": int(config["max_episode_steps"]),
        },
    )


def _new_model(config: Mapping[str, Any], env: Any, run_dir: Path) -> tuple[Any, Dict[str, Any]]:
    from stable_baselines3 import PPO

    ppo = config["ppo"]
    schedule = AbsoluteLearningRateSchedule(
        ppo["learning_rate"], ppo["final_learning_rate"], ppo["learning_rate_schedule"]
    )
    model = build_sequence_ppo(
        env, architecture=config["architecture"], seed=int(config["seed"]),
        learning_rate=schedule, hidden_sizes=ppo["hidden_sizes"],
        n_steps=ppo["n_steps"], batch_size=ppo["batch_size"],
        n_epochs=ppo["n_epochs"], gamma=ppo["gamma"], gae_lambda=ppo["gae_lambda"],
        clip_range=ppo["clip_range"], target_kl=ppo["target_kl"], ent_coef=ppo["ent_coef"],
    )
    source = PPO.load(config["initialization_checkpoint"], device="cpu")
    if tuple(source.observation_space.shape) != (59,):
        raise ValueError("initialization is not the selected v3 PPO policy")
    transfer = initialize_from_ppo900462(source, model, config["architecture"])
    model.num_timesteps = 0
    model.longrun_initialization_checkpoint = config["initialization_checkpoint"]
    model.longrun_initialization_sha256 = config["initialization_sha256"]
    return model, transfer


def _resume_model(config: Mapping[str, Any], env: Any, run_dir: Path, sampler: SequenceCurriculumSampler, contract: str):
    checkpoint = latest_valid_checkpoint(run_dir, expected_contract_hash=contract, safe_only=False)
    if checkpoint is None:
        raise RuntimeError("no valid sequence checkpoint is available")
    state = load_checkpoint_state(checkpoint)
    architecture = state["extra"].get("architecture")
    if architecture != config["architecture"]:
        raise ValueError("resume architecture mismatch")
    model = load_sequence_model(str(checkpoint.model_path), env=env, architecture=architecture)
    if int(model.num_timesteps) != int(state["total_timesteps"]):
        raise ValueError("model timestep and atomic sidecar disagree")
    if tuple(model.observation_space.shape) != (OBS_DIM_SEQUENCE,):
        raise ValueError("resume observation shape changed")
    sampler.load_state_dict(state["curriculum"])
    restore_model_training_state(model, state["model_training"])
    restore_rng_state(state["rng"])
    return model, state, checkpoint


def _state_payload(
    *,
    architecture: str,
    aliases: Mapping[str, Any],
    selection: Mapping[str, Any],
    initialization: Mapping[str, Any],
    stop_reason: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "architecture": architecture,
        "checkpoint_aliases": dict(aliases),
        "checkpoint_selection_state": dict(selection),
        "initialization": dict(initialization),
        "stop_reason": stop_reason,
    }


def _write_status(
    run_dir: Path,
    *,
    state: str,
    model: Any,
    sampler: SequenceCurriculumSampler,
    aliases: Mapping[str, Any],
    selection: Mapping[str, Any],
    config: Mapping[str, Any],
    message: Optional[str] = None,
) -> Dict[str, Any]:
    status = {
        "updated_utc": now_utc(), "state": state, "pid": os.getpid(),
        "new_environment_steps": int(model.num_timesteps),
        "target_new_environment_steps": int(config["new_environment_steps"]),
        "initialization_policy_steps": 900462,
        "initialization_checkpoint": config["initialization_checkpoint"],
        "initialization_sha256": config["initialization_sha256"],
        "architecture": config["architecture"],
        "observation_version": OBS_ENCODING_VERSION_SEQUENCE,
        "curriculum_stage": sampler.current_stage,
        "stage_entry_timesteps": sampler.state.stage_entry_timesteps,
        "consecutive_qualifying_evaluations": sampler.state.consecutive_qualifying_evaluations,
        "efficiency_reward_active": sampler.state.efficiency_reward_active,
        "replay_mixture": sampler.state.replay_mixture,
        "curriculum_stage_history": sampler.state.curriculum_stage_history,
        "evaluation_history_count": len(sampler.state.evaluation_history),
        "checkpoint_aliases": dict(aliases),
        "checkpoint_selection_state": dict(selection),
        "all_actions_policy_generated": True,
        "hybrid_or_bc_active": False,
        "message": message,
    }
    atomic_write_json(run_dir / "status.json", status)
    return status


def _save(
    model: Any, run_dir: Path, contract: str, sampler: SequenceCurriculumSampler,
    evaluation_state: Mapping[str, Any], aliases: Dict[str, Any], selection: Dict[str, Any],
    config: Mapping[str, Any], *, status: str, reason: str, stop_reason: Optional[str] = None,
):
    aliases["last"] = str(
        run_dir / "checkpoints" / f"ppo_{int(model.num_timesteps)}_steps.zip"
    )
    checkpoint = atomic_save_checkpoint(
        model, run_dir, total_timesteps=int(model.num_timesteps),
        config_contract_hash=contract, curriculum_state=sampler.state_dict(),
        evaluation_state=dict(evaluation_state), status=status, reason=reason,
        extra_state=_state_payload(
            architecture=config["architecture"], aliases=aliases, selection=selection,
            initialization={"checkpoint": config["initialization_checkpoint"],
                            "sha256": config["initialization_sha256"],
                            "policy_steps": 900462}, stop_reason=stop_reason,
        ),
    )
    return checkpoint


def run_training(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    if args.run_name:
        config["run_name"] = args.run_name
    if args.architecture:
        config["architecture"] = args.architecture
    if args.steps is not None:
        config["new_environment_steps"] = int(args.steps)
    if args.n_steps is not None:
        config["ppo"]["n_steps"] = int(args.n_steps)
    if args.batch_size is not None:
        config["ppo"]["batch_size"] = int(args.batch_size)
    import torch
    torch.set_num_threads(max(1, int(config.get("torch_threads", 1))))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    run_dir = _run_dir(config, args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    preflight = _check_preflight(config, allow_dirty=args.allow_dirty_smoke)
    atomic_write_json(run_dir / "preflight.json", preflight)
    contract = _run_contract(config, run_dir)
    sampler = _make_sampler(config)
    reward = SequenceRewardConfig(reward_phase="reliability")
    env = _make_env(config, run_dir, sampler, reward)
    aliases: Dict[str, Any] = {"last": None, "best_sequence": None, "latest_safe": None}
    selection: Dict[str, Any] = {"best_metrics": None, "best_timestep": None, "last_full_metrics": None}
    evaluation_state: Dict[str, Any] = {"history": [], "last": None, "best": None}
    transfer: Optional[Dict[str, Any]] = None
    if args.resume:
        model, resume_state, _ = _resume_model(config, env, run_dir, sampler, contract)
        extra = resume_state["extra"]
        aliases.update(extra.get("checkpoint_aliases") or {})
        selection.update(extra.get("checkpoint_selection_state") or {})
        evaluation_state.update(resume_state.get("evaluation") or {})
        reward.set_efficiency_unlocked(sampler.state.efficiency_reward_active)
    else:
        if any((run_dir / "checkpoints").glob("*.manifest.json")):
            raise ValueError("run directory already contains checkpoints; use --resume")
        model, transfer = _new_model(config, env, run_dir)

    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        atomic_write_json(manifest_path, {
            "schema_version": "ppo_sequence_run_v1", "created_utc": now_utc(),
            "git_sha": git_sha(), "config_contract_sha256": contract,
            "config": config, "transfer": transfer,
            "historical_controllers_archived_not_deleted": True,
            "active_controller_scope": "ppo_only",
        })
    atomic_write_json(run_dir / "reward_config.json", reward.__dict__)
    atomic_write_json(run_dir / "seed_roles.json", {
        "training": PPO_SEQUENCE_TRAINING_SEEDS,
        "curriculum_evaluation": PPO_SEQUENCE_CURRICULUM_EVAL_SEEDS,
        "unseen_holdout": PPO_SEQUENCE_UNSEEN_HOLDOUT_SEEDS,
        "official_holdout": PPO_SEQUENCE_OFFICIAL_HOLDOUT_SEEDS,
    })

    stop = {"requested": False}
    def request_stop(signum=None, frame=None):
        stop["requested"] = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    stop_file = run_dir / "STOP_REQUESTED"
    configured_target = int(config["new_environment_steps"])
    rollout_size = int(config["ppo"]["n_steps"])
    target = max(rollout_size, (configured_target // rollout_size) * rollout_size)
    schedule = getattr(model, "learning_rate", None)
    if hasattr(schedule, "set_training_horizon"):
        schedule.set_training_horizon(
            sb3_total_timesteps=target, absolute_total_timesteps=target
        )
    checkpoint_frequency = int(config["evaluation"]["checkpoint_frequency"])
    full_frequency = int(config["evaluation"]["full_frequency"])
    official_frequency = int(config["evaluation"]["official_frequency"])
    next_checkpoint = ((int(model.num_timesteps) // checkpoint_frequency) + 1) * checkpoint_frequency
    next_full = ((int(model.num_timesteps) // full_frequency) + 1) * full_frequency
    next_official = ((int(model.num_timesteps) // official_frequency) + 1) * official_frequency
    _write_status(run_dir, state="running", model=model, sampler=sampler,
                  aliases=aliases, selection=selection, config=config,
                  message="PPO-only sequence training initialized")
    try:
        while int(model.num_timesteps) < target and not stop["requested"] and not stop_file.exists():
            # SB3 recomputes ``progress_remaining`` against the horizon passed
            # to each learn() call. Map that local horizon back to the absolute
            # new-step target so a chunked/resumed run never restarts or
            # prematurely exhausts the learning-rate schedule.
            schedule = getattr(model, "learning_rate", None)
            if hasattr(schedule, "set_training_horizon"):
                schedule.set_training_horizon(
                    sb3_total_timesteps=int(model.num_timesteps) + rollout_size,
                    absolute_total_timesteps=target,
                )
            model.learn(total_timesteps=rollout_size, reset_num_timesteps=False, progress_bar=False)
            sampler.set_timesteps(int(model.num_timesteps))
            reward.set_efficiency_unlocked(sampler.state.efficiency_reward_active)
            with (run_dir / "logs" / "progress.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"utc": now_utc(), "timesteps": int(model.num_timesteps),
                                         "stage": sampler.current_stage,
                                         "learning_rate": float(model.policy.optimizer.param_groups[0]["lr"])}) + "\n")

            if int(model.num_timesteps) >= next_full:
                env.close(); model._last_obs = None
                count = int(config["evaluation"]["full_cases"])
                seeds = PPO_SEQUENCE_CURRICULUM_EVAL_SEEDS[:count]
                full_eval_dir = run_dir / "evaluations" / f"full_{int(model.num_timesteps):09d}"
                report = evaluate_sequence_suite(
                    model, architecture=config["architecture"], stage=sampler.current_stage,
                    seeds=seeds, output_dir=full_eval_dir,
                    adapter=config["adapter"], max_steps=int(config["max_episode_steps"]),
                )
                report["timesteps"] = int(model.num_timesteps)
                atomic_write_json(full_eval_dir / "evaluation.json", report)
                metrics = report["metrics"]
                sampler.observe_full_evaluation(metrics, int(model.num_timesteps))
                evaluation_state["last"] = report
                evaluation_state["history"].append(report)
                selection["last_full_metrics"] = metrics
                safe = bool(metrics["safety_clean"])
                candidate_better = is_better_sequence_checkpoint(metrics, selection.get("best_metrics"))
                checkpoint_path = str(
                    run_dir / "checkpoints" / f"ppo_{int(model.num_timesteps)}_steps.zip"
                )
                if safe:
                    aliases["latest_safe"] = checkpoint_path
                if safe and candidate_better:
                    aliases["best_sequence"] = checkpoint_path
                    selection["best_metrics"] = metrics
                    selection["best_timestep"] = int(model.num_timesteps)
                    evaluation_state["best"] = report
                checkpoint = _save(
                    model, run_dir, contract, sampler, evaluation_state, aliases, selection, config,
                    status="safe" if safe else "unsafe", reason="full_evaluation",
                )
                next_full += full_frequency
                next_checkpoint = max(next_checkpoint, int(model.num_timesteps) + checkpoint_frequency)

            if int(model.num_timesteps) >= next_official:
                env.close(); model._last_obs = None
                count = int(config["evaluation"]["official_cases"])
                report = evaluate_sequence_suite(
                    model, architecture=config["architecture"], stage=sampler.current_stage,
                    seeds=PPO_SEQUENCE_OFFICIAL_HOLDOUT_SEEDS[:count],
                    output_dir=run_dir / "evaluations" / f"official_{int(model.num_timesteps):09d}",
                    adapter=config["adapter"], max_steps=int(config["max_episode_steps"]), official=True,
                )
                evaluation_state["history"].append(report)
                selection["last_official_metrics"] = report["metrics"]
                next_official += official_frequency
                if (sampler.current_stage == "S5"
                        and sampler.state.consecutive_qualifying_evaluations >= 2
                        and (report["metrics"].get("official_completion_rate") or 0.0) >= 0.90
                        and report["metrics"]["safety_clean"]
                        and all(
                            any(
                                episode["track"] == track
                                and episode["full_sequence_completion"]
                                for episode in report["episodes"]
                            )
                            for track in OFFICIAL_TRACKS
                        )):
                    stop["requested"] = True

            if int(model.num_timesteps) >= next_checkpoint:
                _save(model, run_dir, contract, sampler, evaluation_state, aliases, selection,
                      config, status="unverified", reason="periodic")
                next_checkpoint += checkpoint_frequency
            _write_status(run_dir, state="running", model=model, sampler=sampler,
                          aliases=aliases, selection=selection, config=config,
                          message="training actively advancing")

        reason = "graceful_stop" if stop["requested"] or stop_file.exists() else "target_reached"
        checkpoint = _save(model, run_dir, contract, sampler, evaluation_state, aliases,
                           selection, config, status="unverified", reason=reason, stop_reason=reason)
        _write_status(run_dir, state="stopped" if reason == "graceful_stop" else "completed",
                      model=model, sampler=sampler, aliases=aliases, selection=selection,
                      config=config, message=reason)
        if stop_file.exists():
            stop_file.unlink()
        return 0
    except Exception as exc:
        try:
            _save(model, run_dir, contract, sampler, evaluation_state, aliases,
                  selection, config, status="unverified", reason="exception", stop_reason=repr(exc))
            _write_status(run_dir, state="failed", model=model, sampler=sampler,
                          aliases=aliases, selection=selection, config=config,
                          message=repr(exc))
        finally:
            (run_dir / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
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
        result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                                capture_output=True, text=True)
        alive = str(pid) in result.stdout and "No tasks" not in result.stdout
    value["process_alive"] = alive
    print(json.dumps(value, indent=2))
    aliases = value.get("checkpoint_aliases") or {}
    required = ("last", "best_sequence", "latest_safe")
    consistent = all(aliases.get(key) for key in required)
    return 0 if value.get("new_environment_steps", 0) > 0 and consistent else 2


def stop_command(run_dir: str | Path) -> int:
    target = Path(run_dir) / "STOP_REQUESTED"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch()
    print(f"graceful stop requested: {target}")
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    model = load_sequence_model(
        args.checkpoint, architecture=args.architecture
    )
    seeds = PPO_SEQUENCE_UNSEEN_HOLDOUT_SEEDS[: int(args.cases)]
    report = evaluate_sequence_suite(
        model, architecture=args.architecture, stage=args.stage,
        seeds=seeds, output_dir=args.out, adapter=args.adapter,
        max_steps=args.max_steps, official=args.official,
    )
    print(json.dumps(report["metrics"], indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", default="configs/rl/ppo_sequence_curriculum.json")
    train.add_argument("--run-name", default=None)
    train.add_argument("--run-dir", default=None)
    train.add_argument("--architecture", choices=("feedforward_ppo", "recurrent_ppo_lstm"), default=None)
    train.add_argument("--steps", type=int, default=None)
    train.add_argument("--n-steps", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--allow-dirty-smoke", action="store_true")
    status = sub.add_parser("status"); status.add_argument("run_dir")
    stop = sub.add_parser("stop"); stop.add_argument("run_dir")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--architecture", choices=("feedforward_ppo", "recurrent_ppo_lstm"), required=True)
    evaluate.add_argument("--stage", default="S1", choices=("S1", "S2", "S3", "S4", "S5"))
    evaluate.add_argument("--cases", type=int, default=8)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--adapter", choices=("holoocean", "fallback"), default="holoocean")
    evaluate.add_argument("--max-steps", type=int, default=3600)
    evaluate.add_argument("--official", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "status":
        return status_command(args.run_dir)
    if args.command == "stop":
        return stop_command(args.run_dir)
    if args.command == "evaluate":
        return evaluate_command(args)
    return run_training(args)


if __name__ == "__main__":
    raise SystemExit(main())
