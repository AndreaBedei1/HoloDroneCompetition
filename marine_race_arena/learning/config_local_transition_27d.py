"""Sequence-length-independent onboard observation contract (27-D Gen-2).

The 27-D contract preserves only essential onboard signals, short derivatives,
previous actions, and visual gate orientation. All uninformative confidence,
presence-redundant flags, and ground-truth leaks are removed.

Exact feature ordering (1-indexed):
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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple


OBS_ENCODING_VERSION_LOCAL_TRANSITION_27D = "onboard_local_transition_27d_v1"

BEACON_BEARING_RATE_SCALE_DEG_S = 180.0
BEACON_ELEVATION_RATE_SCALE_DEG_S = 90.0
BEACON_RANGE_RATE_SCALE_M_S = 5.0
VISION_CENTER_RATE_SCALE_S = 4.0
TARGET_CHANGED_RECENT_WINDOW_S = 1.0

DEPTH_SCALE_M = 10.0
VELOCITY_SCALE_MPS = 1.5
YAW_RATE_SCALE_RPS = 1.5
ELEVATION_SCALE_DEG = 45.0
RANGE_SCALE_M = 20.0

FEATURE_NAMES_LOCAL_TRANSITION_27D: Tuple[str, ...] = (
    "beacon_present",
    "beacon_bearing_sin",
    "beacon_bearing_cos",
    "beacon_elevation_norm",
    "beacon_range_norm",
    "vision_center_x",
    "vision_center_y",
    "vision_area_fraction",
    "depth_norm",
    "dvl_surge_norm",
    "dvl_sway_norm",
    "dvl_heave_norm",
    "dvl_present",
    "imu_yaw_rate_norm",
    "previous_surge",
    "previous_sway",
    "previous_heave",
    "previous_yaw",
    "vision_center_x_rate",
    "vision_center_y_rate",
    "beacon_bearing_rate",
    "beacon_elevation_rate",
    "beacon_range_rate",
    "target_changed_recently",
    "gate_orientation_present",
    "gate_yaw_sin",
    "gate_yaw_cos",
)

_PM1 = (-1.0, 1.0)
_01 = (0.0, 1.0)

FEATURE_BOUNDS_LOCAL_TRANSITION_27D: Tuple[Tuple[float, float], ...] = (
    _01,   # 1 beacon_present
    _PM1,  # 2 beacon_bearing_sin
    _PM1,  # 3 beacon_bearing_cos
    _PM1,  # 4 beacon_elevation_norm
    _01,   # 5 beacon_range_norm
    _PM1,  # 6 vision_center_x
    _PM1,  # 7 vision_center_y
    _01,   # 8 vision_area_fraction
    _01,   # 9 depth_norm
    _PM1,  # 10 dvl_surge_norm
    _PM1,  # 11 dvl_sway_norm
    _PM1,  # 12 dvl_heave_norm
    _01,   # 13 dvl_present
    _PM1,  # 14 imu_yaw_rate_norm
    _PM1,  # 15 previous_surge
    _PM1,  # 16 previous_sway
    _PM1,  # 17 previous_heave
    _PM1,  # 18 previous_yaw
    _PM1,  # 19 vision_center_x_rate
    _PM1,  # 20 vision_center_y_rate
    _PM1,  # 21 beacon_bearing_rate
    _PM1,  # 22 beacon_elevation_rate
    _PM1,  # 23 beacon_range_rate
    _01,   # 24 target_changed_recently
    _01,   # 25 gate_orientation_present
    _PM1,  # 26 gate_yaw_sin
    _PM1,  # 27 gate_yaw_cos
)

OBS_DIM_LOCAL_TRANSITION_27D = len(FEATURE_NAMES_LOCAL_TRANSITION_27D)
assert OBS_DIM_LOCAL_TRANSITION_27D == 27
assert len(FEATURE_BOUNDS_LOCAL_TRANSITION_27D) == 27
assert len(set(FEATURE_NAMES_LOCAL_TRANSITION_27D)) == 27

# Features explicitly removed from 35-D:
REMOVED_FEATURES_FROM_35D = (
    "beacon_signal_strength",
    "beacon_age_norm",
    "vision_present",
    "vision_confidence",
    "depth_present",
    "depth_error_norm",
    "depth_reference_present",
    "imu_present",
    "vision_rate_valid",
    "beacon_rate_valid",
    "time_since_target_change_norm",
)
assert len(REMOVED_FEATURES_FROM_35D) == 11

# New orientation features added to 27-D:
NEW_ORIENTATION_FEATURES_27D = (
    "gate_orientation_present",
    "gate_yaw_sin",
    "gate_yaw_cos",
)
assert len(NEW_ORIENTATION_FEATURES_27D) == 3


@dataclass
class LocalTransition27dLearningContext:
    """Legal controller-local state consumed by the 27-feature encoder."""

    expected_beacon_id: Optional[str] = None
    depth_reference_m: Optional[float] = None
    visual_target: Optional[Any] = None
    use_tracked_visual_target: bool = False
    prev_action: Sequence[float] = field(
        default_factory=lambda: (0.0, 0.0, 0.0, 0.0)
    )
    vision_center_x_rate_s: float = 0.0
    vision_center_y_rate_s: float = 0.0
    vision_rate_valid: bool = False
    beacon_bearing_rate_deg_s: float = 0.0
    beacon_elevation_rate_deg_s: float = 0.0
    beacon_range_rate_m_s: float = 0.0
    beacon_rate_valid: bool = False
    target_changed_recently: bool = False
    time_since_target_change_s: float = 0.0
    gate_orientation_present: bool = False
    gate_yaw_deg: Optional[float] = None
