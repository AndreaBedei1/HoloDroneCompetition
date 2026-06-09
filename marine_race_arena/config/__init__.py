"""Configuration loading and validation for marine race tracks."""

from marine_race_arena.config.loader import load_track_config
from marine_race_arena.config.schema import TrackConfig
from marine_race_arena.config.validation import TrackValidationError, validate_track_config

__all__ = ["TrackConfig", "TrackValidationError", "load_track_config", "validate_track_config"]

