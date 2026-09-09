"""Encoder for :data:`onboard_local_transition_gate_yaw_v2`."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import numpy as np

from marine_race_arena.learning.config_local_transition_gate_yaw import (
    FEATURE_BOUNDS_LOCAL_TRANSITION_GATE_YAW,
    OBS_DIM_LOCAL_TRANSITION_GATE_YAW,
    LocalTransitionGateYawLearningContext,
)
from marine_race_arena.learning.observation_encoder_local_transition import (
    encode_observation_local_transition,
)


def encode_observation_local_transition_gate_yaw(
    observation: Mapping[str, Any],
    context: Optional[LocalTransitionGateYawLearningContext] = None,
) -> np.ndarray:
    """Return the exact finite, clipped 38-value gate-yaw observation.

    If the camera orientation is unavailable or invalid, all three appended
    values are zero.  In particular, ``cos(0)`` is *not* emitted without its
    availability mask.
    """

    context = context or LocalTransitionGateYawLearningContext()
    prefix = encode_observation_local_transition(observation, context)
    present = bool(context.gate_orientation_present)
    try:
        yaw_deg = float(context.gate_yaw_deg)
    except (TypeError, ValueError):
        yaw_deg = float("nan")
    present = present and math.isfinite(yaw_deg)
    if present:
        yaw_rad = math.radians(yaw_deg)
        suffix = np.asarray(
            [1.0, math.sin(yaw_rad), math.cos(yaw_rad)], dtype=np.float32
        )
    else:
        suffix = np.zeros(3, dtype=np.float32)

    vector = np.concatenate((prefix, suffix)).astype(np.float32, copy=False)
    vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)
    if vector.shape != (OBS_DIM_LOCAL_TRANSITION_GATE_YAW,):
        raise ValueError(
            f"gate-yaw observation has shape {vector.shape}, expected "
            f"({OBS_DIM_LOCAL_TRANSITION_GATE_YAW},)"
        )
    low = np.asarray(
        [bound[0] for bound in FEATURE_BOUNDS_LOCAL_TRANSITION_GATE_YAW],
        dtype=np.float32,
    )
    high = np.asarray(
        [bound[1] for bound in FEATURE_BOUNDS_LOCAL_TRANSITION_GATE_YAW],
        dtype=np.float32,
    )
    return np.clip(vector, low, high).astype(np.float32, copy=False)

