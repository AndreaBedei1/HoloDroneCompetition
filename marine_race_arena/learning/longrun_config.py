"""Validated configuration for multi-day observation-v3 PPO training.

The long-run configuration is deliberately separate from the 5k diagnostic
launcher.  A JSON snapshot is written into every run and is part of the resume
compatibility contract.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from marine_race_arena.learning.config import ACTION_CONTRACT_VERSION
from marine_race_arena.learning.config_v3 import OBS_ENCODING_VERSION_V3

LONGRUN_SCHEMA_VERSION = "multigate_longrun_config_v1"
DEFAULT_R1_CHECKPOINT = (
    "results/rl/multigate_v3/r1/ppo_multigate_v3/20260727_133031/"
    "best_model/best_model.zip"
)
DEFAULT_R1_SHA256 = (
    "de7e835132ee57fbe94ee3a2388f0f7554fd6bf41228c8e048000c61fd5b0b6b"
)
SUPPORTED_TIMESTEP_PRESETS = (100_000, 250_000, 500_000, 1_000_000, 2_000_000)
CURRICULUM_STAGES = tuple(f"C{i}" for i in range(8))


@dataclass
class PPOConfig:
    hidden_sizes: Tuple[int, int] = (256, 256)
    learning_rate: float = 3e-5
    final_learning_rate: float = 5e-6
    learning_rate_schedule: str = "linear"
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.10
    target_kl: float = 0.01
    high_kl: float = 0.02
    absolute_kl_stop: float = 0.03
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    initial_action_std: float = 0.12
    low_kl_threshold: float = 0.001
    low_kl_updates: int = 5
    max_automatic_changes: int = 1

    def validate(self) -> None:
        if len(self.hidden_sizes) != 2 or any(int(v) <= 0 for v in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain two positive layer widths")
        if not 0 < self.final_learning_rate <= self.learning_rate:
            raise ValueError("final_learning_rate must be in (0, learning_rate]")
        if self.learning_rate_schedule not in {"linear", "cosine", "constant"}:
            raise ValueError("learning_rate_schedule must be linear, cosine, or constant")
        if self.n_steps <= 0 or self.batch_size <= 0:
            raise ValueError("n_steps and batch_size must be positive")
        if self.n_steps % self.batch_size:
            raise ValueError("batch_size must divide n_steps for the single training env")
        if self.n_epochs <= 0:
            raise ValueError("n_epochs must be positive")
        if not 0 < self.gamma <= 1 or not 0 < self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be in (0, 1]")
        if not 0 < self.clip_range <= 0.3:
            raise ValueError("clip_range must be in (0, 0.3]")
        if not 0 < self.target_kl < self.high_kl < self.absolute_kl_stop:
            raise ValueError("KL thresholds must satisfy target < high < absolute stop")
        if not 0.05 <= self.initial_action_std <= 0.20:
            raise ValueError("initial_action_std must be between 0.05 and 0.20")
        if self.max_automatic_changes not in (0, 1):
            raise ValueError("bounded adaptation permits zero or one automatic change")


@dataclass
class CurriculumConfig:
    initial_stage: str = "C0"
    maximum_stage: str = "C4"
    auto_promote: bool = True
    allow_demotion: bool = True
    retention_fraction: float = 0.20
    straight_fraction: float = 0.20
    current_stage_fraction: float = 0.30
    previous_stage_fraction: float = 0.20
    failure_case_fraction: float = 0.10
    sensor_noise: bool = True

    def validate(self) -> None:
        if self.initial_stage not in CURRICULUM_STAGES:
            raise ValueError(f"unknown initial curriculum stage {self.initial_stage!r}")
        if self.maximum_stage not in CURRICULUM_STAGES:
            raise ValueError(f"unknown maximum curriculum stage {self.maximum_stage!r}")
        if CURRICULUM_STAGES.index(self.maximum_stage) < CURRICULUM_STAGES.index(
            self.initial_stage
        ):
            raise ValueError("maximum_stage cannot precede initial_stage")
        mixture = (
            self.retention_fraction,
            self.straight_fraction,
            self.current_stage_fraction,
            self.previous_stage_fraction,
            self.failure_case_fraction,
        )
        if any(v < 0 for v in mixture) or abs(sum(mixture) - 1.0) > 1e-9:
            raise ValueError("curriculum replay fractions must be nonnegative and sum to 1")


@dataclass
class EvaluationConfig:
    checkpoint_frequency: int = 10_000
    light_frequency: int = 10_000
    full_frequency: int = 50_000
    light_episodes: int = 5
    full_episodes: int = 20
    plateau_steps: int = 100_000
    plateau_min_evaluations: int = 3

    def validate(self) -> None:
        for name in ("checkpoint_frequency", "light_frequency", "full_frequency"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.full_frequency % self.light_frequency:
            raise ValueError("full_frequency must be a multiple of light_frequency")
        if self.light_episodes < 3 or self.full_episodes < 10:
            raise ValueError("light/full evaluation require at least 3/10 episodes")
        if self.plateau_steps < self.full_frequency:
            raise ValueError("plateau_steps must cover at least one full-evaluation interval")


@dataclass
class ReliabilityConfig:
    status_frequency_steps: int = 250
    status_frequency_seconds: int = 30
    stall_timeout_seconds: int = 900
    max_simulator_restarts: int = 5
    max_reset_failures: int = 3
    max_nan_events: int = 1
    minimum_free_disk_gb: float = 10.0
    maximum_log_bytes: int = 20 * 1024 * 1024
    log_backup_count: int = 5
    graceful_stop_timeout_seconds: int = 300

    def validate(self) -> None:
        if self.status_frequency_steps <= 0 or self.status_frequency_seconds <= 0:
            raise ValueError("status update frequencies must be positive")
        if self.stall_timeout_seconds <= self.status_frequency_seconds:
            raise ValueError("stall timeout must exceed the status update interval")
        if self.max_simulator_restarts < 0 or self.max_reset_failures <= 0:
            raise ValueError("restart/reset limits are invalid")
        if self.minimum_free_disk_gb < 1.0:
            raise ValueError("minimum_free_disk_gb must be at least 1")
        if self.maximum_log_bytes < 1024 * 1024 or self.log_backup_count < 1:
            raise ValueError("rotating log limits are too small")


@dataclass
class LongRunConfig:
    run_name: str = "r2_longrun_seed22001"
    total_timesteps: int = 1_000_000
    seed: int = 22001
    output_root: str = "results/rl/multigate_longrun"
    adapter: str = "holoocean"
    allow_fallback: bool = False
    current_profile: str = "none"
    max_episode_steps: int = 1800
    initialization_checkpoint: str = DEFAULT_R1_CHECKPOINT
    initialization_sha256: str = DEFAULT_R1_SHA256
    initialization_kind: str = "ppo_weights"
    observation_version: str = OBS_ENCODING_VERSION_V3
    action_version: str = ACTION_CONTRACT_VERSION
    policy_mode: str = "feedforward"
    frame_stack: int = 1
    ppo: PPOConfig = field(default_factory=PPOConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    schema_version: str = LONGRUN_SCHEMA_VERSION

    @property
    def run_dir(self) -> Path:
        return Path(self.output_root) / self.run_name

    def validate(self, *, allow_smoke: bool = False) -> None:
        if self.schema_version != LONGRUN_SCHEMA_VERSION:
            raise ValueError(f"unsupported config schema {self.schema_version!r}")
        if not self.run_name or any(c in self.run_name for c in '<>:"/\\|?*'):
            raise ValueError("run_name is empty or contains an invalid Windows path character")
        minimum = 1 if allow_smoke else min(SUPPORTED_TIMESTEP_PRESETS)
        if int(self.total_timesteps) < minimum:
            raise ValueError(f"total_timesteps must be at least {minimum}")
        if self.adapter not in {"holoocean", "fallback"}:
            raise ValueError("adapter must be holoocean or fallback")
        if self.adapter == "holoocean" and self.allow_fallback:
            raise ValueError("real long-run training must never allow fallback")
        if self.current_profile != "none":
            raise ValueError("the initial long run is current-free")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if self.observation_version != OBS_ENCODING_VERSION_V3:
            raise ValueError("long-run feed-forward policy requires observation-v3")
        if self.action_version != ACTION_CONTRACT_VERSION:
            raise ValueError("incompatible action contract")
        if self.policy_mode not in {"feedforward", "frame_stack"}:
            raise ValueError("policy_mode must be feedforward or frame_stack")
        if self.policy_mode == "feedforward" and self.frame_stack != 1:
            raise ValueError("feedforward policy_mode requires frame_stack=1")
        if self.policy_mode == "frame_stack" and self.frame_stack not in {3, 4}:
            raise ValueError("experimental frame stacking supports 3 or 4 frames")
        if self.policy_mode == "frame_stack" and self.initialization_kind != "ppo_weights":
            raise ValueError("frame stacking currently requires ppo_weights initialization")
        if self.initialization_kind not in {"ppo_weights", "bc_v3"}:
            raise ValueError("initialization_kind must be ppo_weights or bc_v3")
        if len(self.initialization_sha256) != 64:
            raise ValueError("initialization_sha256 must be a SHA-256 hex digest")
        self.ppo.validate()
        self.curriculum.validate()
        self.evaluation.validate()
        self.reliability.validate()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["ppo"]["hidden_sizes"] = list(self.ppo.hidden_sizes)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LongRunConfig":
        raw = dict(data)
        raw["ppo"] = PPOConfig(**raw.get("ppo", {}))
        raw["ppo"].hidden_sizes = tuple(raw["ppo"].hidden_sizes)
        raw["curriculum"] = CurriculumConfig(**raw.get("curriculum", {}))
        raw["evaluation"] = EvaluationConfig(**raw.get("evaluation", {}))
        raw["reliability"] = ReliabilityConfig(**raw.get("reliability", {}))
        return cls(**raw)

    @classmethod
    def load(cls, path: str | Path) -> "LongRunConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(target)


def apply_overrides(config: LongRunConfig, overrides: Iterable[Tuple[str, Any]]) -> None:
    """Apply dotted CLI overrides without silently accepting unknown fields."""
    for dotted, value in overrides:
        owner: Any = config
        parts = dotted.split(".")
        for part in parts[:-1]:
            if not hasattr(owner, part):
                raise ValueError(f"unknown configuration field {dotted!r}")
            owner = getattr(owner, part)
        if not hasattr(owner, parts[-1]):
            raise ValueError(f"unknown configuration field {dotted!r}")
        setattr(owner, parts[-1], value)
