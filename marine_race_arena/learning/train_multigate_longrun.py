"""Crash-safe, monitored, resumable multi-day multi-gate PPO training.

The default command targets one million *real HoloOcean environment steps*.  It
never enables fallback and never constructs a rule controller for runtime
actions.  Use ``--smoke`` only for bounded pipeline validation.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from marine_race_arena.learning.config import ACTION_CONTRACT_VERSION
from marine_race_arena.learning.config_v3 import (
    OBS_DIM_V3,
    OBS_ENCODING_VERSION_V3,
)
from marine_race_arena.learning.longrun_checkpoint import (
    atomic_append_jsonl,
    atomic_copy_checkpoint,
    atomic_save_checkpoint,
    atomic_write_json,
    canonical_hash,
    capture_model_training_state,
    capture_rng_state,
    latest_valid_checkpoint,
    load_checkpoint_state,
    restore_model_training_state,
    restore_rng_state,
    sha256_file,
)
from marine_race_arena.learning.longrun_config import (
    DEFAULT_R1_CHECKPOINT,
    LongRunConfig,
)
from marine_race_arena.learning.longrun_env import (
    CurriculumMarineRaceEnv,
    apply_observation_mode,
)
from marine_race_arena.learning.longrun_evaluation import evaluate_longrun_policy
from marine_race_arena.learning.longrun_monitor import (
    StatusStore,
    make_longrun_callback,
    resume_status_snapshot,
)
from marine_race_arena.learning.model_contract_v3 import (
    assert_clean_worktree,
    validate_v3_model,
)
from marine_race_arena.learning.parametric_curriculum import CurriculumSampler
from marine_race_arena.learning.provenance import git_sha, now_utc
from marine_race_arena.learning.reward_v3 import (
    MultiGateRewardConfig,
    MultiGateTrainingReward,
)
from marine_race_arena.learning.rl_train import build_ppo, transfer_bc_to_ppo
from marine_race_arena.learning.seed_registry import (
    MULTIGATE_LONGRUN_DEV_EVAL_SEEDS,
    MULTIGATE_LONGRUN_PPO_TRAINING_SEEDS,
    MULTIGATE_RELIABILITY_DEV_EVAL_SEEDS,
    MULTIGATE_RELIABILITY_PPO_TRAINING_SEEDS,
)


class AbsoluteLearningRateSchedule:
    """Picklable bounded schedule; SB3 supplies absolute progress on resume."""

    def __init__(self, start: float, end: float, kind: str):
        self.start = float(start)
        self.end = float(end)
        self.kind = str(kind)
        self.multiplier = 1.0
        self.last_value = float(start)
        self.absolute_total_timesteps: Optional[int] = None
        self.sb3_total_timesteps: Optional[int] = None

    def __call__(self, progress_remaining: float) -> float:
        p = float(np.clip(progress_remaining, 0.0, 1.0))
        absolute_total = getattr(self, "absolute_total_timesteps", None)
        sb3_total = getattr(self, "sb3_total_timesteps", None)
        if absolute_total and sb3_total:
            absolute_timesteps = (1.0 - p) * int(sb3_total)
            p = float(
                np.clip(
                    1.0 - absolute_timesteps / int(absolute_total),
                    0.0,
                    1.0,
                )
            )
        if self.kind == "constant":
            base = self.start
        elif self.kind == "cosine":
            base = self.end + 0.5 * (self.start - self.end) * (
                1.0 - np.cos(np.pi * p)
            )
        else:
            base = self.end + (self.start - self.end) * p
        value = float(np.clip(base * self.multiplier, self.end, self.start * 1.5))
        # Extending a run must not restart or increase a decayed schedule. A
        # deliberate bounded scale() call is the only path allowed to raise it.
        self.last_value = min(self.last_value, value)
        return self.last_value

    def scale(self, factor: float) -> None:
        self.multiplier = float(np.clip(self.multiplier * float(factor), 0.25, 1.5))
        self.last_value = float(
            np.clip(self.last_value * float(factor), self.end, self.start * 1.5)
        )

    def set_training_horizon(
        self, *, sb3_total_timesteps: int, absolute_total_timesteps: int
    ) -> None:
        if sb3_total_timesteps <= 0 or absolute_total_timesteps <= 0:
            raise ValueError("learning-rate horizons must be positive")
        self.sb3_total_timesteps = int(sb3_total_timesteps)
        self.absolute_total_timesteps = int(absolute_total_timesteps)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "absolute_learning_rate_schedule_v1",
            "start": self.start,
            "end": self.end,
            "kind": self.kind,
            "multiplier": self.multiplier,
            "last_value": self.last_value,
            "absolute_total_timesteps": self.absolute_total_timesteps,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("schema_version") not in {
            None,
            "absolute_learning_rate_schedule_v1",
        }:
            raise ValueError("unsupported learning-rate schedule state")
        if (
            float(state.get("start", self.start)) != self.start
            or float(state.get("end", self.end)) != self.end
            or str(state.get("kind", self.kind)) != self.kind
        ):
            raise ValueError("learning-rate schedule configuration changed")
        self.multiplier = float(state.get("multiplier", self.multiplier))
        self.last_value = float(state.get("last_value", self.last_value))
        absolute_total = state.get("absolute_total_timesteps")
        if absolute_total is not None:
            self.absolute_total_timesteps = int(absolute_total)


def _contract_dict(config: LongRunConfig) -> Dict[str, Any]:
    value = config.to_dict()
    # These fields may grow/change on a compatible continuation. Everything that
    # changes policy, observations, actions, curriculum semantics, or safety stays.
    for name in ("total_timesteps", "run_name", "output_root"):
        value.pop(name, None)
    return value


def config_contract_hash(config: LongRunConfig) -> str:
    return canonical_hash(
        {"config": _contract_dict(config), "training_code_sha": git_sha()}
    )


def run_contract_hash(config: LongRunConfig) -> str:
    """Keep a run's original contract stable across compatible bug-fix commits."""
    manifest_path = config.run_dir / "run_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            stored = str(manifest.get("config_contract_sha256", ""))
            if len(stored) == 64:
                return stored
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return config_contract_hash(config)


def _git_branch() -> str:
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _available_ram_gb() -> Optional[float]:
    if platform.system() != "Windows":
        return None

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    value = MemoryStatusEx()
    value.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        return None
    return value.ullAvailPhys / (1024**3)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if platform.system() == "Windows":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
        )
        return str(pid) in result.stdout and "No tasks" not in result.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def preflight(
    config: LongRunConfig,
    *,
    allow_smoke: bool = False,
    allow_dirty_smoke: bool = False,
) -> Dict[str, Any]:
    config.validate(allow_smoke=allow_smoke)
    if not (allow_smoke and allow_dirty_smoke):
        assert_clean_worktree()
    branch = _git_branch()
    expected_branch = (
        "feature/rl-multigate-reliability-first"
        if config.training_profile == "reliability_first"
        else "feature/rl-multigate-longrun"
    )
    if branch != expected_branch:
        raise RuntimeError(
            f"{config.training_profile} runs require {expected_branch}, found {branch!r}"
        )
    checkpoint = Path(config.initialization_checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    actual_sha = sha256_file(checkpoint)
    if actual_sha != config.initialization_sha256:
        raise ValueError(
            f"initialization hash mismatch: {actual_sha} != "
            f"{config.initialization_sha256}"
        )
    contract = validate_v3_model(checkpoint)
    if config.initialization_kind == "ppo_weights" and contract["kind"] != "ppo":
        raise ValueError("ppo_weights initialization requires an SB3 ZIP")
    if config.initialization_kind == "bc_v3" and contract["kind"] not in {
        "bc",
        "bc_v3_gated",
    }:
        raise ValueError("bc_v3 initialization requires a v3 BC checkpoint")
    config.run_dir.parent.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(config.run_dir.parent)
    free_gb = disk.free / (1024**3)
    if free_gb < config.reliability.minimum_free_disk_gb:
        raise RuntimeError(
            f"only {free_gb:.2f} GiB free; "
            f"{config.reliability.minimum_free_disk_gb:.2f} GiB required"
        )
    import torch

    try:
        import holoocean

        installed_packages = list(holoocean.installed_packages())
    except Exception:
        installed_packages = []
    if config.adapter == "holoocean" and "Ocean" not in installed_packages:
        raise RuntimeError("required HoloOcean Ocean package is not installed")
    training_seeds = (
        MULTIGATE_RELIABILITY_PPO_TRAINING_SEEDS
        if config.training_profile == "reliability_first"
        else MULTIGATE_LONGRUN_PPO_TRAINING_SEEDS
    )
    if config.seed not in training_seeds:
        raise ValueError("training seed is outside the allocated long-run range")
    tensorboard_available = False
    try:
        import tensorboard  # noqa: F401

        tensorboard_available = True
    except Exception:
        pass
    if not tensorboard_available:
        raise RuntimeError(
            "tensorboard is required for long-run logging; run the prepare script"
        )
    conflicts = []
    for pid_file in Path(config.output_root).glob("*/pid.txt"):
        if pid_file.parent == config.run_dir:
            continue
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except Exception:
            continue
        if _pid_alive(pid):
            conflicts.append({"run_dir": str(pid_file.parent), "pid": pid})
    if conflicts:
        raise RuntimeError(f"conflicting training process(es): {conflicts}")
    return {
        "checked_utc": now_utc(),
        "branch": branch,
        "git_sha": git_sha(),
        "worktree_clean": not (allow_smoke and allow_dirty_smoke),
        "initialization": contract,
        "free_disk_gb": round(free_gb, 3),
        "available_ram_gb": (
            round(_available_ram_gb(), 3)
            if _available_ram_gb() is not None
            else None
        ),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_devices": int(torch.cuda.device_count()),
        "holoocean_packages": installed_packages,
        "tensorboard_available": tensorboard_available,
        "conflicting_processes": conflicts,
    }


def _prepare_run_dirs(run_dir: Path) -> None:
    for relative in (
        "checkpoints",
        "best_models",
        "evaluations",
        "tensorboard",
        "logs",
        "crash_reports",
        "generated_tracks",
    ):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)


def _configure_logging(config: LongRunConfig) -> logging.Logger:
    logger = logging.getLogger(f"multigate_longrun.{config.run_name}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(
        config.run_dir / "logs" / "trainer.log",
        maxBytes=config.reliability.maximum_log_bytes,
        backupCount=config.reliability.log_backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(handler)
    return logger


def _set_sb3_logger(model: Any, run_dir: Path) -> None:
    from stable_baselines3.common.logger import (
        CSVOutputFormat,
        Logger,
        TensorBoardOutputFormat,
    )

    formats = [
        CSVOutputFormat(str(run_dir / "logs" / "progress.csv")),
        TensorBoardOutputFormat(str(run_dir / "tensorboard")),
    ]
    model.set_logger(Logger(str(run_dir / "logs"), formats))


def _set_learning_rate_horizon(
    model: Any, *, learn_target: int, schedule_target: int
) -> None:
    seen = set()
    for schedule in (
        getattr(model, "learning_rate", None),
        getattr(model, "lr_schedule", None),
        getattr(getattr(model, "lr_schedule", None), "value_schedule", None),
    ):
        if schedule is None or id(schedule) in seen:
            continue
        seen.add(id(schedule))
        if hasattr(schedule, "set_training_horizon"):
            schedule.set_training_horizon(
                sb3_total_timesteps=int(learn_target),
                absolute_total_timesteps=int(schedule_target),
            )


def _migrate_learning_rate_schedule(model: Any, config: LongRunConfig) -> None:
    """Replace cloud-pickled legacy schedule code while preserving its state."""
    from stable_baselines3.common.utils import FloatSchedule

    previous = getattr(model, "learning_rate", None)
    replacement = AbsoluteLearningRateSchedule(
        config.ppo.learning_rate,
        config.ppo.final_learning_rate,
        config.ppo.learning_rate_schedule,
    )
    if previous is not None:
        replacement.multiplier = float(
            getattr(previous, "multiplier", replacement.multiplier)
        )
        replacement.last_value = float(
            getattr(previous, "last_value", replacement.last_value)
        )
    model.learning_rate = replacement
    model.lr_schedule = FloatSchedule(replacement)


def _make_sampler(config: LongRunConfig) -> CurriculumSampler:
    curriculum = config.curriculum
    return CurriculumSampler(
        seed=config.seed,
        initial_stage=curriculum.initial_stage,
        maximum_stage=curriculum.maximum_stage,
        mixture=(
            curriculum.retention_fraction,
            curriculum.straight_fraction,
            curriculum.current_stage_fraction,
            curriculum.previous_stage_fraction,
            curriculum.failure_case_fraction,
        ),
        early_mixture=(
            curriculum.early_retention_fraction,
            curriculum.early_straight_fraction,
            curriculum.early_current_stage_fraction,
            curriculum.early_previous_stage_fraction,
            curriculum.early_failure_case_fraction,
        ),
        early_until_timesteps=curriculum.early_replay_until_timesteps,
        promotion_config=curriculum.promotion,
        sensor_noise=curriculum.sensor_noise,
    )


def _make_env(
    config: LongRunConfig,
    sampler: CurriculumSampler,
    reward_config: MultiGateRewardConfig,
) -> Any:
    env = CurriculumMarineRaceEnv(
        sampler,
        run_dir=config.run_dir,
        seed=config.seed,
        reward_factory=lambda: MultiGateTrainingReward(reward_config),
        env_kwargs={
            "adapter": config.adapter,
            "allow_fallback": config.allow_fallback,
            "current_profile": config.current_profile,
            "max_steps": config.max_episode_steps,
            "observation_encoding_version": config.observation_version,
        },
    )
    return apply_observation_mode(
        env, policy_mode=config.policy_mode, frame_stack=config.frame_stack
    )


def _transfer_ppo_initialization(source: Any, target: Any, frame_stack: int) -> None:
    """Copy R1 into a fresh optimizer, expanding only first layers for stacking."""
    import torch

    source_state = source.policy.state_dict()
    target_state = target.policy.state_dict()
    if frame_stack == 1:
        target.policy.load_state_dict(source_state, strict=True)
        return
    expanded: Dict[str, Any] = {}
    for name, destination in target_state.items():
        if name not in source_state:
            raise ValueError(f"initial policy is missing tensor {name}")
        origin = source_state[name]
        if origin.shape == destination.shape:
            expanded[name] = origin
        elif (
            origin.ndim == 2
            and destination.ndim == 2
            and origin.shape[0] == destination.shape[0]
            and origin.shape[1] == OBS_DIM_V3
            and destination.shape[1] == OBS_DIM_V3 * frame_stack
        ):
            value = torch.zeros_like(destination)
            # Initially reproduce R1 exactly by using only the newest frame.
            value[:, -OBS_DIM_V3:] = origin
            expanded[name] = value
        else:
            raise ValueError(
                f"cannot expand initialization tensor {name}: "
                f"{tuple(origin.shape)} -> {tuple(destination.shape)}"
            )
    target.policy.load_state_dict(expanded, strict=True)


def _new_model(
    config: LongRunConfig, env: Any, sampler: CurriculumSampler
) -> Any:
    schedule = AbsoluteLearningRateSchedule(
        config.ppo.learning_rate,
        config.ppo.final_learning_rate,
        config.ppo.learning_rate_schedule,
    )
    algorithm_class = None
    if config.training_profile == "reliability_first":
        from marine_race_arena.learning.reliability_first import (
            ReliabilityFirstPPO,
        )

        algorithm_class = ReliabilityFirstPPO
    model = build_ppo(
        env,
        hidden_sizes=config.ppo.hidden_sizes,
        seed=config.seed,
        learning_rate=schedule,
        n_steps=config.ppo.n_steps,
        batch_size=config.ppo.batch_size,
        n_epochs=config.ppo.n_epochs,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
        clip_range=config.ppo.clip_range,
        target_kl=config.ppo.target_kl,
        ent_coef=config.ppo.ent_coef,
        vf_coef=config.ppo.vf_coef,
        max_grad_norm=config.ppo.max_grad_norm,
        algorithm_class=algorithm_class,
    )
    if config.initialization_kind == "ppo_weights":
        from stable_baselines3 import PPO

        source = PPO.load(config.initialization_checkpoint, device="cpu")
        if tuple(source.observation_space.shape) != (OBS_DIM_V3,):
            raise ValueError("initial PPO observation shape is incompatible")
        _transfer_ppo_initialization(source, model, config.frame_stack)
    else:
        from marine_race_arena.learning.bc_v3_transfer import load_v3_policy

        transfer_bc_to_ppo(
            load_v3_policy(config.initialization_checkpoint), model
        )
    with __import__("torch").no_grad():
        std = np.asarray(
            model.policy.log_std.detach().cpu().exp().numpy(), dtype=float
        )
        if np.any(std < 0.05) or np.any(std > 0.20):
            model.policy.log_std.fill_(float(np.log(config.ppo.initial_action_std)))
    model.obs_encoding_version = OBS_ENCODING_VERSION_V3
    model.action_contract_version = ACTION_CONTRACT_VERSION
    model.longrun_config_contract = config_contract_hash(config)
    model.longrun_policy_mode = config.policy_mode
    model.longrun_frame_stack = config.frame_stack
    _attach_retention_regularizer(model, config, sampler)
    return model


def _attach_retention_regularizer(
    model: Any, config: LongRunConfig, sampler: CurriculumSampler
) -> None:
    if config.training_profile != "reliability_first":
        return
    from marine_race_arena.learning.reliability_first import (
        OfflineRetentionRegularizer,
    )

    regularizer = None
    if config.retention.enabled:
        regularizer = OfflineRetentionRegularizer(
            dataset_paths=config.retention.dataset_paths,
            initial_bc_checkpoint=config.initialization_checkpoint,
            seed=config.seed,
            batch_size=config.retention.batch_size,
            weight=config.retention.weight,
            maximum_weight=config.retention.maximum_weight,
            final_weight=config.retention.final_weight,
            initial_policy_kl_weight=config.retention.initial_policy_kl_weight,
            decay_until_timesteps=config.retention.decay_until_timesteps,
            active_through_stage=config.retention.active_through_stage,
            every_updates=config.retention.every_updates,
            initial_action_std=config.ppo.initial_action_std,
            maximum_gradient_norm=config.ppo.max_grad_norm,
        )
    model.attach_retention_regularizer(regularizer, lambda: sampler.current_stage)


def _jsonl_rows_through(path: Path, timesteps: int) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            if int(row.get("timesteps", row.get("num_timesteps", 0))) <= timesteps:
                rows.append(row)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return rows


def _restore_legacy_retention_state(
    model: Any, run_dir: Path, timesteps: int
) -> None:
    """Reconstruct the v1 retention state that was omitted from old sidecars."""
    regularizer = getattr(model, "_retention_regularizer", None)
    if regularizer is None:
        return
    regularizer.calls = int(getattr(model, "_n_updates", 0)) // max(
        1, int(getattr(model, "n_epochs", 1))
    )
    update_rows = _jsonl_rows_through(
        run_dir / "update_metrics.jsonl", timesteps
    )
    applied_batches = sum(
        float(row.get("bc_retention_loss") or 0.0) > 0.0
        for row in update_rows
    )
    batch_size = min(regularizer.batch_size, len(regularizer.observations))
    for _ in range(applied_batches):
        regularizer.rng.integers(
            0, len(regularizer.observations), size=batch_size
        )
    rollback_rows = _jsonl_rows_through(
        run_dir / "logs" / "rollback_history.jsonl", timesteps
    )
    for row in rollback_rows:
        retained = row.get("retention_weight_after")
        if retained is not None and regularizer.base_weight > 0:
            regularizer.weight_multiplier = max(
                regularizer.weight_multiplier,
                float(retained) / regularizer.base_weight,
            )
    if update_rows:
        latest = update_rows[-1]
        regularizer.last_metrics = {
            "bc_retention_loss": latest.get("bc_retention_loss"),
            "initial_policy_kl": latest.get("initial_policy_kl"),
            "retention_weight": latest.get("retention_weight"),
        }


def _upgrade_resume_state(
    config: LongRunConfig,
    checkpoint: Any,
    state: Dict[str, Any],
    model: Any,
) -> Dict[str, Any]:
    """Backfill fields absent from v1 checkpoints without changing the source."""
    upgraded = copy.deepcopy(state)
    current = int(upgraded["total_timesteps"])
    evaluation = upgraded.setdefault("evaluation", {})
    history = list(evaluation.get("history", []))
    evaluation.setdefault("last", history[-1] if history else None)
    extra = upgraded.setdefault("extra", {})
    rollback_history = list(
        extra.get(
            "rollback_history",
            _jsonl_rows_through(
                config.run_dir / "logs" / "rollback_history.jsonl", current
            ),
        )
    )
    extra["rollback_history"] = rollback_history
    if rollback_history:
        latest_rollback = rollback_history[-1]
        extra.setdefault("last_rollback_reason", latest_rollback.get("reasons"))
        extra.setdefault("last_rollback_source", latest_rollback.get("source"))
        extra["rollback_count"] = max(
            int(extra.get("rollback_count", 0)),
            int(latest_rollback.get("attempt", 0)),
        )
    else:
        extra.setdefault("last_rollback_reason", None)
        extra.setdefault("last_rollback_source", None)
        extra.setdefault("rollback_count", 0)
    extra.setdefault("reward_phase", "legacy")
    extra.setdefault("low_kl_count", 0)
    extra.setdefault("high_kl_count", 0)
    best = dict(evaluation.get("best", {}))
    aliases = dict(extra.get("checkpoint_aliases", {}))
    aliases.update(
        {
            alias: selected.get("checkpoint")
            for alias, selected in best.items()
            if selected.get("checkpoint")
        }
    )
    aliases["last"] = str(checkpoint.model_path)
    if checkpoint.manifest.get("status") == "safe":
        aliases["latest_safe"] = str(checkpoint.model_path)
    extra["checkpoint_aliases"] = aliases
    if not upgraded.get("model_training"):
        _restore_legacy_retention_state(model, config.run_dir, current)
        upgraded["model_training"] = capture_model_training_state(model)
    return upgraded


def _load_resume_model(
    config: LongRunConfig,
    env: Any,
    sampler: CurriculumSampler,
) -> tuple[Any, Dict[str, Any], Any]:
    if config.training_profile == "reliability_first":
        from marine_race_arena.learning.reliability_first import (
            ReliabilityFirstPPO as PPO,
        )
    else:
        from stable_baselines3 import PPO

    contract = run_contract_hash(config)
    checkpoint = latest_valid_checkpoint(
        config.run_dir, expected_contract_hash=contract
    )
    if checkpoint is None:
        raise RuntimeError("no valid compatible checkpoint is available to resume")
    prior_rng = capture_rng_state()
    try:
        state = load_checkpoint_state(checkpoint)
        if int(state["total_timesteps"]) != checkpoint.timesteps:
            raise ValueError("resume checkpoint manifest and sidecar disagree")
        candidate_sampler = _make_sampler(config)
        candidate_sampler.load_state_dict(state["curriculum"])
        model = PPO.load(str(checkpoint.model_path), env=env, device="cpu")
        _migrate_learning_rate_schedule(model, config)
        if int(model.num_timesteps) != checkpoint.timesteps:
            raise ValueError("resume model and sidecar timestep disagree")
        if getattr(model, "obs_encoding_version", None) != OBS_ENCODING_VERSION_V3:
            raise ValueError("resume checkpoint observation version is incompatible")
        if (
            getattr(model, "action_contract_version", None)
            != ACTION_CONTRACT_VERSION
        ):
            raise ValueError("resume checkpoint action version is incompatible")
        if (
            getattr(model, "longrun_config_contract", contract) != contract
        ):
            raise ValueError("resume checkpoint training contract is incompatible")
        if (
            getattr(model, "longrun_policy_mode", "feedforward")
            != config.policy_mode
        ):
            raise ValueError("resume checkpoint policy mode is incompatible")
        if int(getattr(model, "longrun_frame_stack", 1)) != config.frame_stack:
            raise ValueError("resume checkpoint frame stack is incompatible")
        expected_dim = OBS_DIM_V3 * config.frame_stack
        if tuple(model.observation_space.shape or ()) != (expected_dim,):
            raise ValueError("resume checkpoint observation shape is incompatible")
        _attach_retention_regularizer(model, config, sampler)
        state = _upgrade_resume_state(config, checkpoint, state, model)
        restore_model_training_state(model, state["model_training"])
        sampler.load_state_dict(state["curriculum"])
        # Model/environment construction can consume Python, NumPy, and Torch
        # entropy. Restore the checkpoint RNG only after every component is ready.
        restore_rng_state(state["rng"])
        return model, state, checkpoint
    except BaseException:
        restore_rng_state(prior_rng)
        raise


def _write_initial_manifest(
    config: LongRunConfig, preflight_result: Dict[str, Any]
) -> None:
    manifest = {
        "schema_version": "multigate_longrun_run_v1",
        "created_utc": now_utc(),
        "status": "LONG-RUN TRAINING PIPELINE RUN",
        "git_sha": git_sha(),
        "branch": _git_branch(),
        "config_contract_sha256": config_contract_hash(config),
        "initialization_checkpoint": config.initialization_checkpoint,
        "initialization_sha256": config.initialization_sha256,
        "initialization_mode": (
            "weights_only_with_new_longrun_optimizer"
            if config.initialization_kind == "ppo_weights"
            else "bc_v3_to_new_longrun_optimizer"
        ),
        "observation_version": config.observation_version,
        "policy_mode": config.policy_mode,
        "frame_stack": config.frame_stack,
        "policy_observation_dim": OBS_DIM_V3 * config.frame_stack,
        "action_version": config.action_version,
        "rule_action_weight": 0,
        "hybrid_blending": False,
        "rule_controller_instantiated": False,
        "training_profile": config.training_profile,
        "runtime_expert_queries": 0,
        "retention_source": "offline_recorded_datasets_only",
        "preflight": preflight_result,
    }
    atomic_write_json(config.run_dir / "run_manifest.json", manifest)


def _is_recoverable_simulator_error(exc: BaseException) -> bool:
    return isinstance(
        exc, (RuntimeError, OSError, EOFError, ConnectionError, TimeoutError)
    ) and not isinstance(exc, (ValueError, AssertionError))


def _write_crash_report(
    run_dir: Path, exc: BaseException, *, restart_index: int, timesteps: int
) -> Path:
    report = {
        "created_utc": now_utc(),
        "restart_index": int(restart_index),
        "total_timesteps": int(timesteps),
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    path = (
        run_dir
        / "crash_reports"
        / f"crash_{int(time.time())}_{restart_index:02d}.json"
    )
    atomic_write_json(path, report)
    return path


def run_longrun(
    config: LongRunConfig,
    *,
    resume: bool = False,
    additional_timesteps: Optional[int] = None,
    smoke: bool = False,
    allow_dirty_smoke: bool = False,
) -> Dict[str, Any]:
    run_dir = config.run_dir
    if resume:
        if not (run_dir / "config.json").exists():
            raise FileNotFoundError(run_dir / "config.json")
        original = LongRunConfig.load(run_dir / "config.json")
        if config_contract_hash(original) != config_contract_hash(config):
            raise ValueError("resume configuration is incompatible")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"run directory {run_dir} is not empty; choose a new name or resume"
        )
    contract_hash = run_contract_hash(config)

    _prepare_run_dirs(run_dir)
    preflight_result = preflight(
        config, allow_smoke=smoke, allow_dirty_smoke=allow_dirty_smoke
    )
    if not resume:
        config.save(run_dir / "config.json")
        _write_initial_manifest(config, preflight_result)
    atomic_write_json(run_dir / "preflight.json", preflight_result)
    (run_dir / "pid.txt").write_text(str(os.getpid()), encoding="utf-8")
    stop_request = run_dir / "stop.requested"
    if resume and stop_request.exists():
        stop_request.unlink()
    logger = _configure_logging(config)
    phase = config.reliability.reward_phase
    reward_config = MultiGateRewardConfig(
        reward_phase=(
            "reliability"
            if phase.enabled
            else "legacy"
        ),
        reliability_time_cost=phase.reliability_time_cost,
        efficiency_time_cost=phase.efficiency_time_cost,
        efficiency_detour_penalty=phase.efficiency_detour_penalty,
        efficiency_jerk_penalty=phase.efficiency_jerk_penalty,
        efficiency_energy_penalty=phase.efficiency_energy_penalty,
        per_step_efficiency_penalty_cap=phase.per_step_efficiency_penalty_cap,
    )
    if reward_config.timeout_penalty <= reward_config.maximum_episode_time_term(
        config.max_episode_steps
    ):
        raise ValueError(
            "failure penalty must exceed the maximum episode time term"
        )
    atomic_write_json(run_dir / "reward_config.json", reward_config.__dict__)
    sampler = _make_sampler(config)
    env = _make_env(config, sampler, reward_config)
    resume_state: Optional[Dict[str, Any]] = None
    resume_checkpoint = None
    if resume:
        model, resume_state, resume_checkpoint = _load_resume_model(
            config, env, sampler
        )
        reward_config.set_phase(
            str(
                resume_state.get("extra", {}).get(
                    "reward_phase", reward_config.reward_phase
                )
            )
        )
    else:
        model = _new_model(config, env, sampler)
    _set_sb3_logger(model, run_dir)
    current = int(model.num_timesteps)
    target = (
        current + int(additional_timesteps)
        if resume and additional_timesteps is not None
        else int(config.total_timesteps)
    )
    if target <= current:
        raise ValueError(
            f"target {target} must exceed checkpoint timesteps {current}"
        )
    initial_status: Dict[str, Any] = {
        "run_name": config.run_name,
        "target_total_timesteps": target,
        "curriculum_stage": sampler.current_stage,
        "training_profile": config.training_profile,
        "reward_phase": reward_config.reward_phase,
        "rollback_count": 0,
        "stage_entry_timesteps": sampler.state.stage_entry_timesteps,
        "steps_in_current_stage": current - sampler.state.stage_entry_timesteps,
        "next_promotion_eligibility_step": (
            sampler.state.stage_entry_timesteps
            + int(
                config.curriculum.promotion.minimum_stage_timesteps.get(
                    sampler.current_stage, 0
                )
            )
        ),
    }
    if resume_state is not None:
        initial_status.update(
            resume_status_snapshot(
                run_dir=run_dir,
                config=config,
                sampler=sampler,
                state=resume_state,
                last_checkpoint=(
                    str(resume_checkpoint.model_path)
                    if resume_checkpoint is not None
                    else None
                ),
            )
        )
    status_store = StatusStore(run_dir, initial_status)
    status_store.update(
        state="RUNNING",
        total_timesteps=current,
        target_total_timesteps=target,
        curriculum_stage=sampler.current_stage,
        stop_reason=None,
    )
    atomic_append_jsonl(
        run_dir / "logs" / "resume_history.jsonl",
        {
            "utc": now_utc(),
            "resume": bool(resume),
            "from_timesteps": current,
            "target_timesteps": target,
            "pid": os.getpid(),
        },
    )

    def evaluate(model_to_eval: Any, mode: str, timesteps: int) -> Dict[str, Any]:
        count = (
            config.evaluation.light_episodes
            if mode == "light"
            else config.evaluation.full_episodes
        )
        evaluation_seeds = (
            MULTIGATE_RELIABILITY_DEV_EVAL_SEEDS
            if config.training_profile == "reliability_first"
            else MULTIGATE_LONGRUN_DEV_EVAL_SEEDS
        )
        seeds = evaluation_seeds[: count + 4]

        def evaluation_progress(progress: Mapping[str, Any]) -> None:
            status_store.update(
                state="EVALUATING",
                evaluation_mode=progress["mode"],
                evaluation_cases_completed=int(progress["completed"]),
                evaluation_cases_total=int(progress["total"]),
            )

        status_store.update(
            state="EVALUATING",
            evaluation_mode=mode,
            evaluation_cases_completed=0,
            evaluation_cases_total=0,
        )
        try:
            return evaluate_longrun_policy(
                model_to_eval,
                stage=sampler.current_stage,
                mode=mode,
                seeds=seeds,
                output_dir=run_dir / "evaluations",
                env_kwargs={
                    "adapter": config.adapter,
                    "allow_fallback": config.allow_fallback,
                    "current_profile": config.current_profile,
                    "max_steps": config.max_episode_steps,
                    "observation_encoding_version": config.observation_version,
                },
                reward_config=reward_config,
                timesteps=timesteps,
                policy_mode=config.policy_mode,
                frame_stack=config.frame_stack,
                progress_callback=evaluation_progress,
            )
        finally:
            status_store.update(state="RUNNING")

    callback = make_longrun_callback(
        run_dir=run_dir,
        config=config,
        sampler=sampler,
        status_store=status_store,
        contract_hash=contract_hash,
        evaluate_fn=evaluate,
        reward_config=reward_config,
    )
    if resume_state is not None:
        callback.restore_pipeline_state(
            resume_state, resume_checkpoint=resume_checkpoint
        )
    else:
        initial = atomic_save_checkpoint(
            model,
            run_dir,
            total_timesteps=current,
            config_contract_hash=contract_hash,
            curriculum_state=sampler.state_dict(),
            evaluation_state={"history": [], "best": {}},
            status="safe",
            reason="initialized_from_frozen_checkpoint",
            extra_state={"automatic_changes": []},
        )
        for alias in ("initial_bc_v3", "latest_safe", "last"):
            atomic_copy_checkpoint(
                initial, run_dir / "best_models" / f"{alias}.zip"
            )
        status_store.update(last_checkpoint=str(initial.model_path))

    restart_count = 0
    final_exception: Optional[BaseException] = None
    try:
        while int(model.num_timesteps) < target:
            remaining = target - int(model.num_timesteps)
            try:
                logger.info(
                    "learn start current=%s remaining=%s stage=%s",
                    model.num_timesteps,
                    remaining,
                    sampler.current_stage,
                )
                _set_learning_rate_horizon(
                    model,
                    learn_target=target,
                    schedule_target=config.total_timesteps,
                )
                model.learn(
                    total_timesteps=remaining,
                    callback=callback,
                    reset_num_timesteps=False,
                    progress_bar=False,
                )
                callback.finalize()
                break
            except BaseException as exc:
                final_exception = exc
                report_path = _write_crash_report(
                    run_dir,
                    exc,
                    restart_index=restart_count,
                    timesteps=int(model.num_timesteps),
                )
                status_store.update(
                    detected_crash=str(report_path),
                    state="RECOVERING",
                    simulator_restarts=restart_count,
                )
                if (
                    not _is_recoverable_simulator_error(exc)
                    or restart_count >= config.reliability.max_simulator_restarts
                ):
                    raise
                restart_count += 1
                logger.exception(
                    "recoverable simulator failure; restart %s/%s",
                    restart_count,
                    config.reliability.max_simulator_restarts,
                )
                env.close()
                sampler = _make_sampler(config)
                env = _make_env(config, sampler, reward_config)
                (
                    model,
                    resume_state,
                    resume_checkpoint,
                ) = _load_resume_model(config, env, sampler)
                _set_sb3_logger(model, run_dir)
                callback = make_longrun_callback(
                    run_dir=run_dir,
                    config=config,
                    sampler=sampler,
                    status_store=status_store,
                    contract_hash=contract_hash,
                    evaluate_fn=evaluate,
                    reward_config=reward_config,
                )
                callback.restore_pipeline_state(
                    resume_state, resume_checkpoint=resume_checkpoint
                )
                status_store.update(
                    simulator_restarts=restart_count,
                    total_timesteps=int(model.num_timesteps),
                    state="RUNNING",
                )
                final_exception = None
    finally:
        env.close()
        state = status_store.data.get("state")
        if final_exception is not None:
            state = "CRASHED"
        elif state not in {"COMPLETED", "STOPPED"}:
            state = (
                "COMPLETED"
                if int(model.num_timesteps) >= target
                else "STOPPED"
            )
        status_store.update(
            state=state,
            process_alive=False,
            total_timesteps=int(model.num_timesteps),
            simulator_restarts=restart_count,
        )
        try:
            model.logger.close()
        except Exception:
            pass
        for handler in tuple(logger.handlers):
            try:
                handler.close()
            finally:
                logger.removeHandler(handler)
    return dict(status_store.data)


def _load_or_default_config(path: Optional[str]) -> LongRunConfig:
    return LongRunConfig.load(path) if path else LongRunConfig()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--additional-timesteps", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--curriculum-stage", default=None)
    parser.add_argument("--maximum-curriculum-stage", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-frequency", type=int, default=None)
    parser.add_argument("--checkpoint-frequency", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--adapter", choices=("holoocean", "fallback"), default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-dirty-smoke", action="store_true")
    args = parser.parse_args(argv)

    if args.resume and args.run_dir:
        config_path = Path(args.run_dir) / "config.json"
        config = LongRunConfig.load(config_path)
    else:
        config = _load_or_default_config(args.config)
    if args.run_name:
        config.run_name = args.run_name
    if args.run_dir:
        path = Path(args.run_dir)
        config.output_root = str(path.parent)
        config.run_name = path.name
    if args.total_timesteps is not None:
        config.total_timesteps = args.total_timesteps
    if args.curriculum_stage:
        config.curriculum.initial_stage = args.curriculum_stage
    if args.maximum_curriculum_stage:
        config.curriculum.maximum_stage = args.maximum_curriculum_stage
    if args.seed is not None:
        config.seed = args.seed
    if args.eval_frequency is not None:
        config.evaluation.light_frequency = args.eval_frequency
        config.evaluation.full_frequency = max(
            args.eval_frequency, 5 * args.eval_frequency
        )
    if args.checkpoint_frequency is not None:
        config.evaluation.checkpoint_frequency = args.checkpoint_frequency
    if args.n_steps is not None:
        config.ppo.n_steps = args.n_steps
    if args.batch_size is not None:
        config.ppo.batch_size = args.batch_size
    if args.adapter:
        config.adapter = args.adapter
        config.allow_fallback = args.adapter == "fallback"
    status = run_longrun(
        config,
        resume=args.resume,
        additional_timesteps=args.additional_timesteps,
        smoke=args.smoke,
        allow_dirty_smoke=args.allow_dirty_smoke,
    )
    print(json.dumps(status, indent=2))
    return 0 if status.get("state") in {"COMPLETED", "STOPPED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
