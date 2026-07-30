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
class RetentionConfig:
    enabled: bool = False
    reference_checkpoint: Optional[str] = None
    weight: float = 0.10
    maximum_weight: float = 0.20
    initial_policy_kl_weight: float = 0.02
    batch_size: int = 256
    every_updates: int = 1
    decay_until_timesteps: int = 150_000
    final_weight: float = 0.0
    active_through_stage: str = "C2"
    dataset_paths: Tuple[str, ...] = ()

    def validate(self) -> None:
        if not 0.0 <= self.weight <= self.maximum_weight <= 0.20:
            raise ValueError("retention weights must satisfy 0 <= weight <= maximum <= 0.20")
        if not 0.0 <= self.final_weight <= self.weight:
            raise ValueError("retention final_weight must be in [0, weight]")
        if not 0.0 <= self.initial_policy_kl_weight <= 0.20:
            raise ValueError("initial-policy KL weight must be in [0, 0.20]")
        if self.batch_size <= 0 or self.every_updates <= 0:
            raise ValueError("retention batch_size/every_updates must be positive")
        if self.decay_until_timesteps <= 0:
            raise ValueError("retention decay_until_timesteps must be positive")
        if self.active_through_stage not in CURRICULUM_STAGES:
            raise ValueError("unknown retention active_through_stage")
        if self.enabled and not self.dataset_paths:
            raise ValueError("enabled retention requires offline dataset_paths")
        if self.reference_checkpoint is not None and not str(
            self.reference_checkpoint
        ).strip():
            raise ValueError("retention reference_checkpoint cannot be empty")


@dataclass
class PromotionConfig:
    required_consecutive_full_evaluations: int = 1
    minimum_stage_timesteps: Dict[str, int] = field(default_factory=dict)
    overall_completion_rate: float = 0.80
    single_gate_completion_rate: float = 0.0
    straight_completion_rate: float = 0.80
    left_completion_rate: float = 0.0
    right_completion_rate: float = 0.0
    maximum_collision_episodes: int = 2**31 - 1
    maximum_out_of_bounds_episodes: int = 2**31 - 1
    maximum_wrong_direction_episodes: int = 2**31 - 1
    maximum_previous_gate_returns: int = 2**31 - 1

    def validate(self) -> None:
        if self.required_consecutive_full_evaluations <= 0:
            raise ValueError("promotion requires at least one consecutive full evaluation")
        if any(
            stage not in CURRICULUM_STAGES or int(value) < 0
            for stage, value in self.minimum_stage_timesteps.items()
        ):
            raise ValueError("minimum_stage_timesteps contains an invalid stage/duration")
        rates = (
            self.overall_completion_rate,
            self.single_gate_completion_rate,
            self.straight_completion_rate,
            self.left_completion_rate,
            self.right_completion_rate,
        )
        if any(not 0.0 <= value <= 1.0 for value in rates):
            raise ValueError("promotion completion thresholds must be in [0, 1]")
        limits = (
            self.maximum_collision_episodes,
            self.maximum_out_of_bounds_episodes,
            self.maximum_wrong_direction_episodes,
            self.maximum_previous_gate_returns,
        )
        if any(int(value) < 0 for value in limits):
            raise ValueError("promotion safety/return limits must be non-negative")


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
    early_replay_until_timesteps: int = 0
    early_retention_fraction: float = 0.20
    early_straight_fraction: float = 0.20
    early_current_stage_fraction: float = 0.30
    early_previous_stage_fraction: float = 0.20
    early_failure_case_fraction: float = 0.10
    geometry_ramp_timesteps: int = 0
    sensor_noise: bool = True
    promotion: PromotionConfig = field(default_factory=PromotionConfig)

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
        early_mixture = (
            self.early_retention_fraction,
            self.early_straight_fraction,
            self.early_current_stage_fraction,
            self.early_previous_stage_fraction,
            self.early_failure_case_fraction,
        )
        if any(v < 0 for v in early_mixture) or abs(sum(early_mixture) - 1.0) > 1e-9:
            raise ValueError("early curriculum replay fractions must sum to 1")
        if self.early_replay_until_timesteps < 0:
            raise ValueError("early_replay_until_timesteps must be non-negative")
        if self.geometry_ramp_timesteps < 0:
            raise ValueError("geometry_ramp_timesteps must be non-negative")
        self.promotion.validate()


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
class RollbackConfig:
    enabled: bool = False
    maximum_attempts: int = 2
    completion_drop: float = 0.10
    single_gate_floor: float = 0.90
    straight_floor: float = 0.85
    directional_floor: float = 0.70
    maximum_safety_episodes: int = 0
    maximum_previous_gate_returns: int = 2
    learning_rate_scale: float = 0.5
    retention_weight_scale: float = 1.25

    def validate(self) -> None:
        if self.maximum_attempts < 0:
            raise ValueError("rollback maximum_attempts must be non-negative")
        if not 0.0 <= self.completion_drop <= 1.0:
            raise ValueError("rollback completion_drop must be in [0, 1]")
        for value in (
            self.single_gate_floor,
            self.straight_floor,
            self.directional_floor,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("rollback completion floors must be in [0, 1]")
        if self.maximum_safety_episodes < 0 or self.maximum_previous_gate_returns < 0:
            raise ValueError("rollback event limits must be non-negative")
        if not 0.0 < self.learning_rate_scale <= 1.0:
            raise ValueError("rollback learning_rate_scale must be in (0, 1]")
        if self.retention_weight_scale < 1.0:
            raise ValueError("rollback retention_weight_scale must be at least 1")


@dataclass
class RewardPhaseConfig:
    enabled: bool = False
    reliable_full_evaluations: int = 2
    reliability_time_cost: float = 0.0
    efficiency_time_cost: float = 0.002
    efficiency_detour_penalty: float = 0.004
    efficiency_jerk_penalty: float = 0.01
    efficiency_energy_penalty: float = 0.002
    per_step_efficiency_penalty_cap: float = 0.03
    action_change_penalty: float = 0.04

    def validate(self) -> None:
        if self.reliable_full_evaluations <= 0:
            raise ValueError("reward phase requires a positive reliable-evaluation streak")
        values = (
            self.reliability_time_cost,
            self.efficiency_time_cost,
            self.efficiency_detour_penalty,
            self.efficiency_jerk_penalty,
            self.efficiency_energy_penalty,
            self.per_step_efficiency_penalty_cap,
            self.action_change_penalty,
        )
        if any(value < 0 for value in values):
            raise ValueError("reward phase coefficients must be non-negative")


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
    rollback: RollbackConfig = field(default_factory=RollbackConfig)
    reward_phase: RewardPhaseConfig = field(default_factory=RewardPhaseConfig)

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
        self.rollback.validate()
        self.reward_phase.validate()


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
    training_profile: str = "longrun"
    ppo: PPOConfig = field(default_factory=PPOConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)
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
        if self.training_profile not in {"longrun", "reliability_first"}:
            raise ValueError("training_profile must be longrun or reliability_first")
        if self.initialization_kind not in {"ppo_weights", "bc_v3"}:
            raise ValueError("initialization_kind must be ppo_weights or bc_v3")
        if len(self.initialization_sha256) != 64:
            raise ValueError("initialization_sha256 must be a SHA-256 hex digest")
        self.ppo.validate()
        self.retention.validate()
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
        raw["retention"] = RetentionConfig(**raw.get("retention", {}))
        raw["retention"].dataset_paths = tuple(raw["retention"].dataset_paths)
        curriculum = dict(raw.get("curriculum", {}))
        curriculum["promotion"] = PromotionConfig(**curriculum.get("promotion", {}))
        raw["curriculum"] = CurriculumConfig(**curriculum)
        raw["evaluation"] = EvaluationConfig(**raw.get("evaluation", {}))
        reliability = dict(raw.get("reliability", {}))
        reliability["rollback"] = RollbackConfig(**reliability.get("rollback", {}))
        reliability["reward_phase"] = RewardPhaseConfig(
            **reliability.get("reward_phase", {})
        )
        raw["reliability"] = ReliabilityConfig(**reliability)
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
