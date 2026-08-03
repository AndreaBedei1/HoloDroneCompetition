"""Encoder for :data:`onboard_local_transition_v1`."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import numpy as np

from marine_race_arena.learning.config import FEATURE_NAMES, LearningContext
from marine_race_arena.learning.config_local_transition import (
    BEACON_BEARING_RATE_SCALE_DEG_S,
    BEACON_ELEVATION_RATE_SCALE_DEG_S,
    BEACON_RANGE_RATE_SCALE_M_S,
    FEATURE_BOUNDS_LOCAL_TRANSITION,
    OBS_DIM_LOCAL_TRANSITION,
    TARGET_CHANGE_TIME_SCALE_S,
    VISION_CENTER_RATE_SCALE_S,
    LocalTransitionLearningContext,
)
from marine_race_arena.learning.observation_encoder import encode_observation


_V1_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
_INSTANTANEOUS_SOURCE_NAMES = (
    "beacon_present", "beacon_bearing_sin", "beacon_bearing_cos",
    "beacon_elevation_norm", "beacon_range_norm", "beacon_signal_strength",
    "beacon_age_norm", "vision_present", "vision_center_x", "vision_center_y",
    "vision_area_fraction", "vision_confidence", "depth_norm", "depth_present",
    "depth_error_norm", "depth_ref_present", "dvl_surge_norm", "dvl_sway_norm",
    "dvl_heave_norm", "dvl_present", "imu_yaw_rate_norm", "imu_present",
    "prev_surge", "prev_sway", "prev_heave", "prev_yaw",
)


def _finite_clip(value: Any, scale: float) -> float:
    try:
        converted = float(value) / float(scale)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0
    if not math.isfinite(converted):
        return 0.0
    return float(np.clip(converted, -1.0, 1.0))


def encode_observation_local_transition(
    observation: Mapping[str, Any],
    context: Optional[LocalTransitionLearningContext] = None,
) -> np.ndarray:
    """Return the exact finite, clipped 35-value local policy observation."""

    context = context or LocalTransitionLearningContext()
    base_context = LearningContext(
        expected_beacon_id=context.expected_beacon_id,
        depth_reference_m=context.depth_reference_m,
        prev_action=context.prev_action,
    )
    base = encode_observation(observation, base_context)
    instantaneous = [float(base[_V1_INDEX[name]]) for name in _INSTANTANEOUS_SOURCE_NAMES]
    temporal = [
        _finite_clip(context.vision_center_x_rate_s, VISION_CENTER_RATE_SCALE_S)
        if context.vision_rate_valid else 0.0,
        _finite_clip(context.vision_center_y_rate_s, VISION_CENTER_RATE_SCALE_S)
        if context.vision_rate_valid else 0.0,
        float(bool(context.vision_rate_valid)),
        _finite_clip(
            context.beacon_bearing_rate_deg_s,
            BEACON_BEARING_RATE_SCALE_DEG_S,
        ) if context.beacon_rate_valid else 0.0,
        _finite_clip(
            context.beacon_elevation_rate_deg_s,
            BEACON_ELEVATION_RATE_SCALE_DEG_S,
        ) if context.beacon_rate_valid else 0.0,
        _finite_clip(context.beacon_range_rate_m_s, BEACON_RANGE_RATE_SCALE_M_S)
        if context.beacon_rate_valid else 0.0,
        float(bool(context.beacon_rate_valid)),
        float(bool(context.target_changed_recently)),
        float(np.clip(
            float(context.time_since_target_change_s) / TARGET_CHANGE_TIME_SCALE_S,
            0.0,
            1.0,
        )),
    ]
    vector = np.asarray(instantaneous + temporal, dtype=np.float32)
    vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)
    if vector.shape != (OBS_DIM_LOCAL_TRANSITION,):
        raise ValueError(
            f"local-transition observation has shape {vector.shape}, "
            f"expected ({OBS_DIM_LOCAL_TRANSITION},)"
        )
    low = np.asarray([bound[0] for bound in FEATURE_BOUNDS_LOCAL_TRANSITION], dtype=np.float32)
    high = np.asarray([bound[1] for bound in FEATURE_BOUNDS_LOCAL_TRANSITION], dtype=np.float32)
    return np.clip(vector, low, high).astype(np.float32, copy=False)

