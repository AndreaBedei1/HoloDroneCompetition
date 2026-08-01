"""Encoder for the PPO-only long-sequence observation contract."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from marine_race_arena.learning.config_sequence import (
    FEATURE_BOUNDS_SEQUENCE,
    MAX_SEQUENCE_GATES,
    OBS_DIM_SEQUENCE,
    TIME_SINCE_CROSSING_SCALE_STEPS,
    SequenceLearningContext,
)
from marine_race_arena.learning.observation_encoder_v3 import encode_observation_v3


def encode_observation_sequence(
    observation: Mapping[str, Any],
    context: Optional[SequenceLearningContext] = None,
) -> np.ndarray:
    context = context or SequenceLearningContext()
    base = encode_observation_v3(observation, context)
    total = max(1, int(context.total_beacons) * int(context.laps))
    current = max(0, min(total - 1, int(context.current_gate_index)))
    remaining = max(0, min(total, int(context.remaining_gate_count)))
    sequence = np.asarray(
        [
            current / max(1, total - 1),
            remaining / max(1, min(MAX_SEQUENCE_GATES, total)),
            float(np.clip(context.sequence_progress, 0.0, 1.0)),
            float(bool(context.previous_gate_cleared)),
            float(bool(context.new_target_acquired_since_crossing)),
            float(np.clip(
                context.time_since_confirmed_crossing_steps
                / TIME_SINCE_CROSSING_SCALE_STEPS,
                0.0,
                1.0,
            )),
        ],
        dtype=np.float32,
    )
    vector = np.concatenate((base, sequence)).astype(np.float32, copy=False)
    if vector.shape != (OBS_DIM_SEQUENCE,):
        raise ValueError(
            f"encoded sequence observation has shape {vector.shape}, "
            f"expected ({OBS_DIM_SEQUENCE},)"
        )
    low = np.asarray([item[0] for item in FEATURE_BOUNDS_SEQUENCE], dtype=np.float32)
    high = np.asarray([item[1] for item in FEATURE_BOUNDS_SEQUENCE], dtype=np.float32)
    return np.clip(np.nan_to_num(vector), low, high).astype(np.float32, copy=False)
