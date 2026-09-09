"""Onboard-only observation contract for PPO sequence progression.

The first 59 values are exactly ``onboard_multigate_rl_v3``. The appended
values describe only the local sensor-confirmed course tracker; no simulator
pose, gate geometry, future gate coordinates, or referee state is exposed.
"""

from __future__ import annotations

from dataclasses import dataclass

from marine_race_arena.learning.config_v3 import (
    FEATURE_BOUNDS_V3,
    FEATURE_NAMES_V3,
    MultiGateLearningContext,
)

OBS_ENCODING_VERSION_SEQUENCE = "onboard_ppo_sequence_v4"
MAX_SEQUENCE_GATES = 22
TIME_SINCE_CROSSING_SCALE_STEPS = 100.0

SEQUENCE_FEATURE_NAMES = (
    "current_gate_index_norm",
    "remaining_gate_count_norm",
    "sequence_progress_norm",
    "previous_gate_cleared",
    "new_target_acquired_since_crossing",
    "time_since_confirmed_crossing_norm",
)
FEATURE_NAMES_SEQUENCE = FEATURE_NAMES_V3 + SEQUENCE_FEATURE_NAMES
FEATURE_BOUNDS_SEQUENCE = FEATURE_BOUNDS_V3 + ((0.0, 1.0),) * len(
    SEQUENCE_FEATURE_NAMES
)
OBS_DIM_SEQUENCE = len(FEATURE_NAMES_SEQUENCE)


@dataclass
class SequenceLearningContext(MultiGateLearningContext):
    current_gate_index: int = 0
    remaining_gate_count: int = 1
    sequence_progress: float = 0.0
    previous_gate_cleared: bool = False
    new_target_acquired_since_crossing: bool = False
    time_since_confirmed_crossing_steps: int = 0
