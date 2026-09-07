"""Encoder for :data:`onboard_local_transition_27d_v1`.

Transforms official onboard observation and context into the exact 27-D vector:
 1 beacon_present
 2 beacon_bearing_sin
 3 beacon_bearing_cos
 4 beacon_elevation_norm
 5 beacon_range_norm
 6 vision_center_x
 7 vision_center_y
 8 vision_area_fraction
 9 depth_norm
10 dvl_surge_norm
11 dvl_sway_norm
12 dvl_heave_norm
13 dvl_present
14 imu_yaw_rate_norm
15 previous_surge
16 previous_sway
17 previous_heave
18 previous_yaw
19 vision_center_x_rate
20 vision_center_y_rate
21 beacon_bearing_rate
22 beacon_elevation_rate
23 beacon_range_rate
24 target_changed_recently
25 gate_orientation_present
26 gate_yaw_sin
27 gate_yaw_cos

Missing-data rules:
- Visual absence: center_x = 0, center_y = 0, area_fraction = 0
- Non-valid / uncalculable rates: rate = 0
- Orientation unavailable: gate_orientation_present = 0, gate_yaw_sin = 0, gate_yaw_cos = 0
- Frontal valid gate: gate_orientation_present = 1, gate_yaw_sin ≈ 0, gate_yaw_cos ≈ 1
- Maintains dvl_present.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import numpy as np

from marine_race_arena.controllers.vision import (
    select_default_visual_target,
    select_visual_target_for_beacon,
    vision_targets_from_camera,
)
from marine_race_arena.learning.config import ACTION_DIM
from marine_race_arena.learning.config_local_transition_27d import (
    BEACON_BEARING_RATE_SCALE_DEG_S,
    BEACON_ELEVATION_RATE_SCALE_DEG_S,
    BEACON_RANGE_RATE_SCALE_M_S,
    DEPTH_SCALE_M,
    ELEVATION_SCALE_DEG,
    FEATURE_BOUNDS_LOCAL_TRANSITION_27D,
    FEATURE_NAMES_LOCAL_TRANSITION_27D,
    OBS_DIM_LOCAL_TRANSITION_27D,
    RANGE_SCALE_M,
    VELOCITY_SCALE_MPS,
    VISION_CENTER_RATE_SCALE_S,
    YAW_RATE_SCALE_RPS,
    LocalTransition27dLearningContext,
)
from marine_race_arena.learning.observation_encoder import (
    _clip,
    _depth_m,
    _dvl_body_velocity,
    _finite,
    _imu_yaw_rate,
    _select_beacon_packet,
)


def _finite_clip(value: Any, scale: float) -> float:
    try:
        converted = float(value) / float(scale)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0
    if not math.isfinite(converted):
        return 0.0
    return float(np.clip(converted, -1.0, 1.0))


def encode_observation_local_transition_27d(
    observation: Mapping[str, Any],
    context: Optional[LocalTransition27dLearningContext] = None,
) -> np.ndarray:
    """Return the exact finite, clipped 27-value Gen-2 observation vector."""

    if context is None:
        context = LocalTransition27dLearningContext()
    if not isinstance(observation, Mapping):
        observation = {}

    sensors = observation.get("sensors")
    if not isinstance(sensors, Mapping):
        sensors = {}
    beacons = observation.get("beacons")
    if not isinstance(beacons, (list, tuple)):
        beacons = []

    features: list[float] = []

    # --------------------------------------------------------------------------
    # 1-5: Beacon block
    # --------------------------------------------------------------------------
    packet = _select_beacon_packet(beacons, context.expected_beacon_id)
    beacon_bearing_deg: Optional[float] = None
    beacon_range_m: Optional[float] = None
    if packet is None:
        features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        bearing_deg = _finite(packet.get("bearing_deg"))
        elevation_deg = _finite(packet.get("elevation_deg"))
        range_m = max(0.0, _finite(packet.get("range_m")))
        bearing_rad = math.radians(bearing_deg)
        beacon_bearing_deg = bearing_deg
        beacon_range_m = range_m
        features.extend([
            1.0,                                                    # 1: beacon_present
            _clip(math.sin(bearing_rad), -1.0, 1.0),               # 2: beacon_bearing_sin
            _clip(math.cos(bearing_rad), -1.0, 1.0),               # 3: beacon_bearing_cos
            _clip(elevation_deg / ELEVATION_SCALE_DEG, -1.0, 1.0),  # 4: beacon_elevation_norm
            _clip(range_m / RANGE_SCALE_M, 0.0, 1.0),              # 5: beacon_range_norm
        ])

    # --------------------------------------------------------------------------
    # 6-8: Vision block
    # --------------------------------------------------------------------------
    target = context.visual_target if context.use_tracked_visual_target else None
    if not context.use_tracked_visual_target:
        image = sensors.get("FrontCamera")
        if image is not None:
            try:
                targets = vision_targets_from_camera(image)
                if beacon_bearing_deg is not None:
                    target = select_visual_target_for_beacon(
                        targets, beacon_bearing_deg, beacon_range_m
                    )
                else:
                    target = select_default_visual_target(targets)
            except Exception:
                target = None

    if target is None:
        features.extend([0.0, 0.0, 0.0])  # 6: center_x, 7: center_y, 8: area_fraction
    else:
        features.extend([
            _clip(_finite(target.center_x), -1.0, 1.0),         # 6: vision_center_x
            _clip(_finite(target.center_y), -1.0, 1.0),         # 7: vision_center_y
            _clip(_finite(target.area_fraction), 0.0, 1.0),      # 8: vision_area_fraction
        ])

    # --------------------------------------------------------------------------
    # 9: Depth block
    # --------------------------------------------------------------------------
    depth = _depth_m(sensors)
    if depth is None:
        features.append(0.0)                                    # 9: depth_norm
    else:
        features.append(_clip(depth / DEPTH_SCALE_M, 0.0, 1.0)) # 9: depth_norm

    # --------------------------------------------------------------------------
    # 10-13: DVL block
    # --------------------------------------------------------------------------
    dvl = _dvl_body_velocity(sensors)
    if dvl is None:
        features.extend([0.0, 0.0, 0.0, 0.0])
    else:
        features.extend([
            _clip(dvl[0] / VELOCITY_SCALE_MPS, -1.0, 1.0),     # 10: dvl_surge_norm
            _clip(dvl[1] / VELOCITY_SCALE_MPS, -1.0, 1.0),     # 11: dvl_sway_norm
            _clip(dvl[2] / VELOCITY_SCALE_MPS, -1.0, 1.0),     # 12: dvl_heave_norm
            1.0,                                                # 13: dvl_present
        ])

    # --------------------------------------------------------------------------
    # 14: IMU yaw rate
    # --------------------------------------------------------------------------
    yaw_rate = _imu_yaw_rate(sensors)
    if yaw_rate is None:
        features.append(0.0)                                    # 14: imu_yaw_rate_norm
    else:
        features.append(_clip(yaw_rate / YAW_RATE_SCALE_RPS, -1.0, 1.0))

    # --------------------------------------------------------------------------
    # 15-18: Previous action
    # --------------------------------------------------------------------------
    prev = list(context.prev_action) if context.prev_action is not None else []
    for i in range(ACTION_DIM):
        value = _finite(prev[i]) if i < len(prev) else 0.0
        features.append(_clip(value, -1.0, 1.0))
        # 15: previous_surge, 16: previous_sway, 17: previous_heave, 18: previous_yaw

    # --------------------------------------------------------------------------
    # 19-20: Vision rates
    # --------------------------------------------------------------------------
    if context.vision_rate_valid:
        features.append(_finite_clip(context.vision_center_x_rate_s, VISION_CENTER_RATE_SCALE_S)) # 19
        features.append(_finite_clip(context.vision_center_y_rate_s, VISION_CENTER_RATE_SCALE_S)) # 20
    else:
        features.extend([0.0, 0.0])

    # --------------------------------------------------------------------------
    # 21-23: Beacon rates
    # --------------------------------------------------------------------------
    if context.beacon_rate_valid:
        features.append(_finite_clip(context.beacon_bearing_rate_deg_s, BEACON_BEARING_RATE_SCALE_DEG_S))   # 21
        features.append(_finite_clip(context.beacon_elevation_rate_deg_s, BEACON_ELEVATION_RATE_SCALE_DEG_S)) # 22
        features.append(_finite_clip(context.beacon_range_rate_m_s, BEACON_RANGE_RATE_SCALE_M_S))         # 23
    else:
        features.extend([0.0, 0.0, 0.0])

    # --------------------------------------------------------------------------
    # 24: Target changed recently
    # --------------------------------------------------------------------------
    features.append(1.0 if context.target_changed_recently else 0.0)  # 24

    # --------------------------------------------------------------------------
    # 25-27: Gate orientation
    # --------------------------------------------------------------------------
    present = bool(context.gate_orientation_present)
    try:
        yaw_deg = float(context.gate_yaw_deg) if context.gate_yaw_deg is not None else float("nan")
    except (TypeError, ValueError):
        yaw_deg = float("nan")
    present = present and math.isfinite(yaw_deg)
    if present:
        yaw_rad = math.radians(yaw_deg)
        features.extend([
            1.0,                                    # 25: gate_orientation_present
            _clip(math.sin(yaw_rad), -1.0, 1.0),   # 26: gate_yaw_sin
            _clip(math.cos(yaw_rad), -1.0, 1.0),   # 27: gate_yaw_cos
        ])
    else:
        features.extend([0.0, 0.0, 0.0])

    vector = np.asarray(features, dtype=np.float32)
    vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)
    if vector.shape != (OBS_DIM_LOCAL_TRANSITION_27D,):
        raise ValueError(
            f"27-D observation has shape {vector.shape}, expected ({OBS_DIM_LOCAL_TRANSITION_27D},)"
        )
    low = np.asarray([bound[0] for bound in FEATURE_BOUNDS_LOCAL_TRANSITION_27D], dtype=np.float32)
    high = np.asarray([bound[1] for bound in FEATURE_BOUNDS_LOCAL_TRANSITION_27D], dtype=np.float32)
    return np.clip(vector, low, high).astype(np.float32, copy=False)
