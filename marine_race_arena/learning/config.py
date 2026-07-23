"""Fixed, documented configuration for the learning observation/action contract.

The observation encoding produced by :mod:`observation_encoder` is a fixed-size,
normalized, finite, clipped vector derived *only* from legal onboard information
(the official controller observation) plus controller-local state (a
``LearningContext``). No privileged simulator or referee state is ever encoded.

Every feature is listed in :data:`FEATURE_NAMES` in order; :data:`OBS_DIM` is
derived from it, so the network input size and the encoder stay in lock-step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

# --- Action contract (normalized body-frame high-level command) ---------------
ACTION_AXES = ("surge", "sway", "heave", "yaw")
ACTION_DIM = len(ACTION_AXES)
ACTION_LOW = -1.0
ACTION_HIGH = 1.0

# Bumped when the action axes or their range change; recorded alongside the
# observation-encoding version in run/eval manifests so a model is never resumed
# or evaluated against an incompatible action contract.
ACTION_CONTRACT_VERSION = "surge_sway_heave_yaw_pm1_v1"

# --- LocalCourseTracker phases (mirror of the controller-side constants) -------
# Duplicated here as plain strings so the encoding layout is stable and does not
# import controller internals; a test cross-checks these against the tracker.
TRACKER_PHASES = (
    "SEARCH",
    "APPROACH",
    "VISUAL_ALIGN",
    "COMMIT",
    "VERIFY_EXIT",
    "ADVANCE",
    "FINISHED",
)

# --- Normalization scales (declared ranges; the encoder clips to these) --------
RANGE_SCALE_M = 30.0          # beacon range -> [0, 1]
PACKET_AGE_SCALE_S = 5.0      # packet age (local_time - received_at) -> [0, 1]
ELEVATION_SCALE_DEG = 90.0    # beacon elevation -> [-1, 1]
DEPTH_SCALE_M = 20.0          # absolute depth -> ~[0, 1] (positive down)
DEPTH_ERROR_SCALE_M = 3.0     # depth error vs local reference -> [-1, 1]
VELOCITY_SCALE_MPS = 1.5      # DVL body velocity -> [-1, 1]
YAW_RATE_SCALE_RPS = 2.0      # IMU body-z angular rate -> [-1, 1]
DEFAULT_TOTAL_BEACONS = 12    # fallback normalization when mission size unknown
DEFAULT_LAPS = 1

# --- Ordered feature layout ---------------------------------------------------
# The encoder MUST fill features in exactly this order.
FEATURE_NAMES = (
    # Expected-beacon acoustic packet (7)
    "beacon_present",
    "beacon_bearing_sin",
    "beacon_bearing_cos",
    "beacon_elevation_norm",
    "beacon_range_norm",
    "beacon_signal_strength",
    "beacon_age_norm",
    # FrontCamera-derived vision (5)
    "vision_present",
    "vision_center_x",
    "vision_center_y",
    "vision_area_fraction",
    "vision_confidence",
    # Depth and motion (10)
    "depth_norm",
    "depth_present",
    "depth_error_norm",
    "depth_ref_present",
    "dvl_surge_norm",
    "dvl_sway_norm",
    "dvl_heave_norm",
    "dvl_present",
    "imu_yaw_rate_norm",
    "imu_present",
    # Controller-local state: tracker phase one-hot (7)
    "phase_SEARCH",
    "phase_APPROACH",
    "phase_VISUAL_ALIGN",
    "phase_COMMIT",
    "phase_VERIFY_EXIT",
    "phase_ADVANCE",
    "phase_FINISHED",
    # Controller-local state: progress + lock + previous action (7)
    "local_beacon_index_norm",
    "local_lap_norm",
    "visual_lock",
    "prev_surge",
    "prev_sway",
    "prev_heave",
    "prev_yaw",
)

OBS_DIM = len(FEATURE_NAMES)

# Bumped when the observation feature layout changes; recorded in run metadata so a
# checkpoint is never resumed or evaluated against an incompatible encoding.
OBS_ENCODING_VERSION = "onboard_only_v1"

# Per-feature declared bounds, used both to clip in the encoder and to assert
# bounds in tests. Order matches FEATURE_NAMES.
_PM1 = (-1.0, 1.0)
_01 = (0.0, 1.0)
FEATURE_BOUNDS = (
    _01, _PM1, _PM1, _PM1, _01, _01, _01,          # beacon
    _01, _PM1, _PM1, _01, _01,                      # vision
    _01, _01, _PM1, _01, _PM1, _PM1, _PM1, _01, _PM1, _01,  # depth/motion
    _01, _01, _01, _01, _01, _01, _01,              # phase one-hot
    _01, _01, _01, _PM1, _PM1, _PM1, _PM1,          # progress/lock/prev-action
)
assert len(FEATURE_BOUNDS) == OBS_DIM, "FEATURE_BOUNDS must match FEATURE_NAMES"


@dataclass
class LearningContext:
    """Controller-local state used by the encoder — all legal, none privileged.

    These values are produced by a controller-side ``LocalCourseTracker`` (or an
    equivalent legal estimator) and the policy's own previous action. They are
    NOT simulator/referee state.
    """

    expected_beacon_id: Optional[str] = None
    tracker_phase: Optional[str] = None
    local_beacon_index: int = 0
    local_lap: int = 0
    total_beacons: int = DEFAULT_TOTAL_BEACONS
    laps: int = DEFAULT_LAPS
    depth_reference_m: Optional[float] = None
    visual_lock: bool = False
    prev_action: Sequence[float] = field(default_factory=lambda: (0.0, 0.0, 0.0, 0.0))
