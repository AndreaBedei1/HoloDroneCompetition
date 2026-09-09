"""Minimal gate-plane orientation extension of the frozen 35-D contract.

The first 35 columns are exactly :data:`onboard_local_transition_v1`.  Only
three camera-derived values are appended.  Keeping the prefix byte-for-byte
compatible is what makes an exact warm start of the existing recurrent policy
possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from marine_race_arena.learning.config_local_transition import (
    FEATURE_BOUNDS_LOCAL_TRANSITION,
    FEATURE_NAMES_LOCAL_TRANSITION,
    LocalTransitionLearningContext,
)


OBS_ENCODING_VERSION_LOCAL_TRANSITION_GATE_YAW = (
    "onboard_local_transition_gate_yaw_v2"
)

GATE_YAW_FEATURE_NAMES = (
    "gate_orientation_present",
    "gate_yaw_sin",
    "gate_yaw_cos",
)

FEATURE_NAMES_LOCAL_TRANSITION_GATE_YAW = (
    *FEATURE_NAMES_LOCAL_TRANSITION,
    *GATE_YAW_FEATURE_NAMES,
)
FEATURE_BOUNDS_LOCAL_TRANSITION_GATE_YAW = (
    *FEATURE_BOUNDS_LOCAL_TRANSITION,
    (0.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
)
OBS_DIM_LOCAL_TRANSITION_GATE_YAW = len(
    FEATURE_NAMES_LOCAL_TRANSITION_GATE_YAW
)

assert OBS_DIM_LOCAL_TRANSITION_GATE_YAW == 38
assert FEATURE_NAMES_LOCAL_TRANSITION_GATE_YAW[:35] == FEATURE_NAMES_LOCAL_TRANSITION
assert (
    FEATURE_BOUNDS_LOCAL_TRANSITION_GATE_YAW[:35]
    == FEATURE_BOUNDS_LOCAL_TRANSITION
)
assert len(set(FEATURE_NAMES_LOCAL_TRANSITION_GATE_YAW)) == 38


@dataclass
class LocalTransitionGateYawLearningContext(LocalTransitionLearningContext):
    """The v1 legal state plus one filtered FrontCamera plane angle."""

    gate_orientation_present: bool = False
    gate_yaw_deg: Optional[float] = None

