"""Configuration loading and validation for marine race tracks."""

from marine_race_arena.config.benchmark_tasks import BENCHMARK_TASK_MODES, BenchmarkTaskConfig
from marine_race_arena.config.loader import (
    CURRENT_PROFILE_MODES,
    describe_current_profile,
    load_track_config,
    with_benchmark_task,
    with_current_profile,
    with_obstacle_options,
)
from marine_race_arena.config.schema import ObstacleGenerationConfig, TrackConfig
from marine_race_arena.config.validation import TrackValidationError, validate_track_config

__all__ = [
    "BENCHMARK_TASK_MODES",
    "BenchmarkTaskConfig",
    "CURRENT_PROFILE_MODES",
    "describe_current_profile",
    "ObstacleGenerationConfig",
    "TrackConfig",
    "TrackValidationError",
    "load_track_config",
    "validate_track_config",
    "with_benchmark_task",
    "with_current_profile",
    "with_obstacle_options",
]
