"""Onboard-only multi-gate observation v3.

The first 36 columns are byte-for-byte the frozen v1 encoding.  Temporal
features are appended from :class:`MultiGateLearningContext`; the encoder never
reads simulator pose, gate geometry, or referee state.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from marine_race_arena.learning.config import RANGE_SCALE_M
from marine_race_arena.learning.config_v3 import (
    FEATURE_BOUNDS_V3,
    FORWARD_DISPLACEMENT_SCALE_M,
    OBS_DIM_V3,
    RANGE_DELTA_SCALE_M,
    STEPS_SINCE_BEACON_CHANGE_SCALE,
    STEPS_SINCE_SEEN_SCALE,
    MultiGateLearningContext,
)
from marine_race_arena.learning.observation_encoder import encode_observation


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _flag(value: bool) -> float:
    return 1.0 if value else 0.0


def encode_observation_v3(
    observation: Mapping[str, Any],
    context: Optional[MultiGateLearningContext] = None,
) -> np.ndarray:
    """Encode an official observation into ``onboard_multigate_rl_v3``."""
    context = context or MultiGateLearningContext()
    base = encode_observation(observation, context)

    temporal = np.asarray(
        [
            _flag(context.camera_present),
            _flag(context.vision_recently_seen),
            _flag(context.vision_recently_lost),
            _clip(context.steps_since_gate_seen / STEPS_SINCE_SEEN_SCALE, 0.0, 1.0)
            if context.steps_since_gate_seen_present
            else 0.0,
            _flag(context.steps_since_gate_seen_present),
            _clip(context.last_seen_center_x, -1.0, 1.0) if context.last_seen_present else 0.0,
            _clip(context.last_seen_center_y, -1.0, 1.0) if context.last_seen_present else 0.0,
            _clip(context.last_seen_area_fraction, 0.0, 1.0) if context.last_seen_present else 0.0,
            _clip(context.last_seen_confidence, 0.0, 1.0) if context.last_seen_present else 0.0,
            _flag(context.last_seen_present),
            _flag(context.gate_was_recently_centered),
            _flag(context.gate_was_recently_large),
            _clip(
                context.forward_displacement_since_visual_loss_m
                / FORWARD_DISPLACEMENT_SCALE_M,
                0.0,
                1.0,
            )
            if context.forward_displacement_since_visual_loss_present
            else 0.0,
            _flag(context.forward_displacement_since_visual_loss_present),
            _clip(context.beacon_range_delta_m / RANGE_DELTA_SCALE_M, -1.0, 1.0)
            if context.beacon_range_delta_present
            else 0.0,
            _flag(context.beacon_range_delta_present),
            _clip(context.beacon_range_min_recent_m / RANGE_SCALE_M, 0.0, 1.0)
            if context.beacon_range_min_recent_present
            else 0.0,
            _flag(context.beacon_range_min_recent_present),
            _flag(context.beacon_now_receding),
            _flag(context.expected_beacon_changed),
            _clip(
                context.steps_since_beacon_change / STEPS_SINCE_BEACON_CHANGE_SCALE,
                0.0,
                1.0,
            ),
            _flag(context.previous_gate_in_rear_sector),
            _flag(context.previous_gate_bearing_present),
        ],
        dtype=np.float32,
    )
    vector = np.concatenate((base, temporal)).astype(np.float32, copy=False)
    vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)
    if vector.shape != (OBS_DIM_V3,):
        raise ValueError(f"encoded v3 observation has shape {vector.shape}, expected ({OBS_DIM_V3},)")

    lows = np.asarray([bound[0] for bound in FEATURE_BOUNDS_V3], dtype=np.float32)
    highs = np.asarray([bound[1] for bound in FEATURE_BOUNDS_V3], dtype=np.float32)
    return np.clip(vector, lows, highs).astype(np.float32, copy=False)
