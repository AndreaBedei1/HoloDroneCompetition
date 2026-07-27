"""Observation contract for the learned multi-gate controller.

``onboard_only_v1`` is frozen in :mod:`marine_race_arena.learning.config`.
This module appends controller-local temporal features without changing the
first 36 v1 columns.  Every value is derived from official onboard
observations, the local gate-progress estimator, or the policy's previously
applied action.
"""

from __future__ import annotations

from dataclasses import dataclass

from marine_race_arena.learning.config import (
    FEATURE_BOUNDS as V1_FEATURE_BOUNDS,
    FEATURE_NAMES as V1_FEATURE_NAMES,
    LearningContext,
)

OBS_ENCODING_VERSION_V3 = "onboard_multigate_rl_v3"

STEPS_SINCE_SEEN_SCALE = 50.0
STEPS_SINCE_BEACON_CHANGE_SCALE = 50.0
FORWARD_DISPLACEMENT_SCALE_M = 5.0
RANGE_DELTA_SCALE_M = 1.0
RECENT_VISION_WINDOW_STEPS = 10
RECENT_LOSS_WINDOW_STEPS = 15
RECENT_GATE_HISTORY_STEPS = 20

TEMPORAL_FEATURE_NAMES = (
    "camera_present",
    "vision_recently_seen",
    "vision_recently_lost",
    "steps_since_gate_seen_norm",
    "steps_since_gate_seen_present",
    "last_seen_center_x",
    "last_seen_center_y",
    "last_seen_area_fraction",
    "last_seen_confidence",
    "last_seen_present",
    "gate_was_recently_centered",
    "gate_was_recently_large",
    "forward_displacement_since_visual_loss_norm",
    "forward_displacement_since_visual_loss_present",
    "beacon_range_delta_norm",
    "beacon_range_delta_present",
    "beacon_range_min_recent_norm",
    "beacon_range_min_recent_present",
    "beacon_now_receding",
    "expected_beacon_changed",
    "steps_since_beacon_change_norm",
    "previous_gate_in_rear_sector",
    "previous_gate_bearing_present",
)

FEATURE_NAMES_V3 = V1_FEATURE_NAMES + TEMPORAL_FEATURE_NAMES
OBS_DIM_V3 = len(FEATURE_NAMES_V3)

_PM1 = (-1.0, 1.0)
_01 = (0.0, 1.0)
TEMPORAL_FEATURE_BOUNDS = (
    _01,  # camera_present
    _01, _01, _01, _01,  # recently seen/lost, steps + mask
    _PM1, _PM1, _01, _01, _01,  # last seen values + mask
    _01, _01,  # recently centered / large
    _01, _01,  # DVL forward displacement + mask
    _PM1, _01,  # range delta + mask
    _01, _01, _01,  # recent minimum + mask, receding
    _01, _01,  # beacon changed + steps
    _01, _01,  # previous beacon rear + bearing mask
)
FEATURE_BOUNDS_V3 = V1_FEATURE_BOUNDS + TEMPORAL_FEATURE_BOUNDS
assert len(FEATURE_BOUNDS_V3) == OBS_DIM_V3


@dataclass
class MultiGateLearningContext(LearningContext):
    """Legal temporal state appended to the frozen v1 context."""

    camera_present: bool = False
    vision_recently_seen: bool = False
    vision_recently_lost: bool = False
    steps_since_gate_seen: int = 0
    steps_since_gate_seen_present: bool = False
    last_seen_center_x: float = 0.0
    last_seen_center_y: float = 0.0
    last_seen_area_fraction: float = 0.0
    last_seen_confidence: float = 0.0
    last_seen_present: bool = False
    gate_was_recently_centered: bool = False
    gate_was_recently_large: bool = False
    forward_displacement_since_visual_loss_m: float = 0.0
    forward_displacement_since_visual_loss_present: bool = False
    beacon_range_delta_m: float = 0.0
    beacon_range_delta_present: bool = False
    beacon_range_min_recent_m: float = 0.0
    beacon_range_min_recent_present: bool = False
    beacon_now_receding: bool = False
    expected_beacon_changed: bool = False
    steps_since_beacon_change: int = 0
    previous_gate_in_rear_sector: bool = False
    previous_gate_bearing_present: bool = False
