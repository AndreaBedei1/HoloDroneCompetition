"""Sequence-length-independent onboard observation contract.

The policy sees only the current expected beacon, the current visual target,
onboard depth/DVL/IMU measurements, its previous action, and short local
derivatives.  Gate number, mission length, tracker phase, simulator pose,
future geometry, and referee state are deliberately absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


OBS_ENCODING_VERSION_LOCAL_TRANSITION = "onboard_local_transition_v1"

BEACON_BEARING_RATE_SCALE_DEG_S = 180.0
BEACON_ELEVATION_RATE_SCALE_DEG_S = 90.0
BEACON_RANGE_RATE_SCALE_M_S = 5.0
VISION_CENTER_RATE_SCALE_S = 4.0
TARGET_CHANGED_RECENT_WINDOW_S = 1.0
TARGET_CHANGE_TIME_SCALE_S = 3.0

FEATURE_NAMES_LOCAL_TRANSITION = (
    "beacon_present",
    "beacon_bearing_sin",
    "beacon_bearing_cos",
    "beacon_elevation_norm",
    "beacon_range_norm",
    "beacon_signal_strength",
    "beacon_age_norm",
    "vision_present",
    "vision_center_x",
    "vision_center_y",
    "vision_area_fraction",
    "vision_confidence",
    "depth_norm",
    "depth_present",
    "depth_error_norm",
    "depth_reference_present",
    "dvl_surge_norm",
    "dvl_sway_norm",
    "dvl_heave_norm",
    "dvl_present",
    "imu_yaw_rate_norm",
    "imu_present",
    "previous_surge",
    "previous_sway",
    "previous_heave",
    "previous_yaw",
    "vision_center_x_rate",
    "vision_center_y_rate",
    "vision_rate_valid",
    "beacon_bearing_rate",
    "beacon_elevation_rate",
    "beacon_range_rate",
    "beacon_rate_valid",
    "target_changed_recently",
    "time_since_target_change_norm",
)

_PM1 = (-1.0, 1.0)
_01 = (0.0, 1.0)
FEATURE_BOUNDS_LOCAL_TRANSITION = (
    _01, _PM1, _PM1, _PM1, _01, _01, _01,
    _01, _PM1, _PM1, _01, _01,
    _01, _01, _PM1, _01,
    _PM1, _PM1, _PM1, _01, _PM1, _01,
    _PM1, _PM1, _PM1, _PM1,
    _PM1, _PM1, _01,
    _PM1, _PM1, _PM1, _01,
    _01, _01,
)
OBS_DIM_LOCAL_TRANSITION = len(FEATURE_NAMES_LOCAL_TRANSITION)
assert OBS_DIM_LOCAL_TRANSITION == 35
assert len(FEATURE_BOUNDS_LOCAL_TRANSITION) == OBS_DIM_LOCAL_TRANSITION
assert len(set(FEATURE_NAMES_LOCAL_TRANSITION)) == OBS_DIM_LOCAL_TRANSITION


# Explicit semantic mapping used by the selective 65 -> 35 warm start.  The
# nine local transition/dynamics columns are intentionally absent and therefore
# start with zero input weights.
LOCAL_TO_SEQUENCE_FEATURE_MAP = {
    "beacon_present": "beacon_present",
    "beacon_bearing_sin": "beacon_bearing_sin",
    "beacon_bearing_cos": "beacon_bearing_cos",
    "beacon_elevation_norm": "beacon_elevation_norm",
    "beacon_range_norm": "beacon_range_norm",
    "beacon_signal_strength": "beacon_signal_strength",
    "beacon_age_norm": "beacon_age_norm",
    "vision_present": "vision_present",
    "vision_center_x": "vision_center_x",
    "vision_center_y": "vision_center_y",
    "vision_area_fraction": "vision_area_fraction",
    "vision_confidence": "vision_confidence",
    "depth_norm": "depth_norm",
    "depth_present": "depth_present",
    "depth_error_norm": "depth_error_norm",
    "depth_reference_present": "depth_ref_present",
    "dvl_surge_norm": "dvl_surge_norm",
    "dvl_sway_norm": "dvl_sway_norm",
    "dvl_heave_norm": "dvl_heave_norm",
    "dvl_present": "dvl_present",
    "imu_yaw_rate_norm": "imu_yaw_rate_norm",
    "imu_present": "imu_present",
    "previous_surge": "prev_surge",
    "previous_sway": "prev_sway",
    "previous_heave": "prev_heave",
    "previous_yaw": "prev_yaw",
}


@dataclass
class LocalTransitionLearningContext:
    """Legal controller-local state consumed by the 35-feature encoder."""

    expected_beacon_id: Optional[str] = None
    depth_reference_m: Optional[float] = None
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

