"""Configuration loading and validation for marine race tracks."""

from marine_race_arena.config.benchmark_tasks import BENCHMARK_TASK_MODES, BenchmarkTaskConfig
from marine_race_arena.config.loader import load_track_config, with_benchmark_task
from marine_race_arena.config.schema import TrackConfig
from marine_race_arena.config.validation import TrackValidationError, validate_track_config

__all__ = [
    "BENCHMARK_TASK_MODES",
    "BenchmarkTaskConfig",
    "TrackConfig",
    "TrackValidationError",
    "load_track_config",
    "validate_track_config",
    "with_benchmark_task",
]
