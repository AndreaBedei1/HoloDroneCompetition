"""Legal, fixed-size observation encoding for learning controllers.

The encoder turns the *official controller observation* (the same onboard-only
dict the rule-based controllers receive) plus controller-local state
(:class:`~marine_race_arena.learning.config.LearningContext`) into a fixed-size,
normalized, finite, clipped float32 vector.

Legality (enforced by construction and by tests):
  * it reads only ``observation`` keys that the official contract exposes
    (``local_time_s``, ``sensors``, ``beacons``, ``comms``) and the controller-local
    ``LearningContext``;
  * it never reads simulator pose, world-frame velocity, true gate geometry,
    referee progress/target/status, current vectors or any other privileged key —
    injecting such keys into ``observation`` does not change the output;
  * missing beacon / camera / sensor data is signalled by explicit ``*_present``
    mask features, never silently replaced by ground truth.

Sensor parsing mirrors the official rule-based controllers (DepthSensor negated
to positive-down depth, DVLSensor as a body-frame 3-vector ``[vx, vy, vz]``,
IMUSensor body-z angular rate); a test cross-checks this against the official
helpers to catch drift.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from marine_race_arena.controllers.vision import (
    select_default_visual_target,
    select_visual_target_for_beacon,
    vision_targets_from_camera,
)
from marine_race_arena.learning.config import (
    ACTION_DIM,
    DEPTH_ERROR_SCALE_M,
    DEPTH_SCALE_M,
    ELEVATION_SCALE_DEG,
    OBS_DIM,
    PACKET_AGE_SCALE_S,
    RANGE_SCALE_M,
    TRACKER_PHASES,
    VELOCITY_SCALE_MPS,
    YAW_RATE_SCALE_RPS,
    LearningContext,
)

_PHASE_INDEX = {phase: i for i, phase in enumerate(TRACKER_PHASES)}


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(converted):
        return default
    return converted


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _as_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _depth_m(sensors: Mapping[str, Any]) -> Optional[float]:
    """Positive-down depth from DepthSensor (z position, negative underwater)."""
    reading = sensors.get("DepthSensor")
    reading = _as_list(reading)
    if isinstance(reading, (list, tuple)):
        if not reading:
            return None
        reading = reading[0]
    try:
        converted = float(reading)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted):
        return None
    return -converted


def _dvl_body_velocity(sensors: Mapping[str, Any]) -> Optional[tuple[float, float, float]]:
    """Body-frame DVL velocity [vx (surge), vy (sway), vz (heave)]."""
    dvl = _as_list(sensors.get("DVLSensor"))
    if dvl is None:
        return None
    if isinstance(dvl, Mapping):
        keys = (("vx", "x", "velocity_x"), ("vy", "y", "velocity_y"), ("vz", "z", "velocity_z"))
        out = []
        for group in keys:
            found = None
            for key in group:
                if key in dvl:
                    found = _finite(dvl.get(key))
                    break
            if found is None:
                return None
            out.append(found)
        return (out[0], out[1], out[2])
    if isinstance(dvl, (list, tuple)) and len(dvl) >= 3:
        return (_finite(dvl[0]), _finite(dvl[1]), _finite(dvl[2]))
    return None


def _imu_yaw_rate(sensors: Mapping[str, Any]) -> Optional[float]:
    """Body-z angular rate from IMUSensor.

    HoloOcean 2.3.0 with ReturnBias returns
    ``[acceleration, angular_velocity, accel_bias, angular_velocity_bias]``.
    """
    imu = _as_list(sensors.get("IMUSensor"))
    if imu is None:
        return None
    if isinstance(imu, Mapping):
        angular = imu.get("angular_velocity", imu.get("gyro"))
        angular = _as_list(angular)
        if isinstance(angular, Mapping):
            for key in ("z", "yaw", "wz"):
                if key in angular:
                    return _finite(angular.get(key))
            return None
        if isinstance(angular, (list, tuple)) and len(angular) >= 3:
            return _finite(angular[2])
        return None
    if isinstance(imu, (list, tuple)) and len(imu) >= 2:
        angular = _as_list(imu[1])
        if isinstance(angular, (list, tuple)) and len(angular) >= 3:
            return _finite(angular[2])
    return None


def _select_beacon_packet(
    beacons: Sequence[Mapping[str, Any]], expected_beacon_id: Optional[str]
) -> Optional[Mapping[str, Any]]:
    """Pick the packet for the expected beacon, else the strongest received.

    When several packets share the expected id, the freshest (largest
    ``received_at_s``) wins. This is a legal, onboard selection: it uses only
    received-packet fields.
    """
    valid = [p for p in beacons if isinstance(p, Mapping)]
    if not valid:
        return None
    if expected_beacon_id is not None:
        matching = [p for p in valid if p.get("beacon_id") == expected_beacon_id]
        if matching:
            return max(matching, key=lambda p: _finite(p.get("received_at_s")))
        return None
    return max(valid, key=lambda p: _finite(p.get("signal_strength")))


def encode_observation(
    observation: Mapping[str, Any],
    context: Optional[LearningContext] = None,
) -> np.ndarray:
    """Encode one official observation into the fixed learning feature vector.

    Returns a ``float32`` array of shape ``(OBS_DIM,)`` with every value finite
    and clipped to its declared bound.
    """
    if context is None:
        context = LearningContext()
    if not isinstance(observation, Mapping):
        observation = {}

    local_time_s = _finite(observation.get("local_time_s"), 0.0)
    sensors = observation.get("sensors")
    if not isinstance(sensors, Mapping):
        sensors = {}
    beacons = observation.get("beacons")
    if not isinstance(beacons, (list, tuple)):
        beacons = []

    features: list[float] = []

    # --- Beacon block ---------------------------------------------------------
    packet = _select_beacon_packet(beacons, context.expected_beacon_id)
    beacon_bearing_deg: Optional[float] = None
    beacon_range_m: Optional[float] = None
    if packet is None:
        features += [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        bearing_deg = _finite(packet.get("bearing_deg"))
        elevation_deg = _finite(packet.get("elevation_deg"))
        range_m = _finite(packet.get("range_m"))
        signal = _finite(packet.get("signal_strength"))
        received_at = _finite(packet.get("received_at_s"))
        age = _clip((local_time_s - received_at) / PACKET_AGE_SCALE_S, 0.0, 1.0)
        bearing_rad = math.radians(bearing_deg)
        beacon_bearing_deg = bearing_deg
        beacon_range_m = range_m
        features += [
            1.0,
            _clip(math.sin(bearing_rad), -1.0, 1.0),
            _clip(math.cos(bearing_rad), -1.0, 1.0),
            _clip(elevation_deg / ELEVATION_SCALE_DEG, -1.0, 1.0),
            _clip(range_m / RANGE_SCALE_M, 0.0, 1.0),
            _clip(signal, 0.0, 1.0),
            age,
        ]

    # --- Vision block ---------------------------------------------------------
    image = sensors.get("FrontCamera")
    target = None
    if image is not None:
        try:
            targets = vision_targets_from_camera(image)
            if beacon_bearing_deg is not None:
                target = select_visual_target_for_beacon(targets, beacon_bearing_deg, beacon_range_m)
            else:
                target = select_default_visual_target(targets)
        except Exception:  # pragma: no cover - defensive; encoder must not crash
            target = None
    if target is None:
        features += [0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        features += [
            1.0,
            _clip(_finite(target.center_x), -1.0, 1.0),
            _clip(_finite(target.center_y), -1.0, 1.0),
            _clip(_finite(target.area_fraction), 0.0, 1.0),
            _clip(_finite(target.confidence), 0.0, 1.0),
        ]

    # --- Depth and motion block ----------------------------------------------
    depth = _depth_m(sensors)
    if depth is None:
        features += [0.0, 0.0]  # depth_norm, depth_present
        depth_present = False
    else:
        features += [_clip(depth / DEPTH_SCALE_M, 0.0, 1.0), 1.0]
        depth_present = True
    if depth is not None and context.depth_reference_m is not None:
        err = (depth - _finite(context.depth_reference_m)) / DEPTH_ERROR_SCALE_M
        features += [_clip(err, -1.0, 1.0), 1.0]
    else:
        features += [0.0, 0.0]  # depth_error_norm masked

    dvl = _dvl_body_velocity(sensors)
    if dvl is None:
        features += [0.0, 0.0, 0.0, 0.0]  # surge, sway, heave, present
    else:
        features += [
            _clip(dvl[0] / VELOCITY_SCALE_MPS, -1.0, 1.0),
            _clip(dvl[1] / VELOCITY_SCALE_MPS, -1.0, 1.0),
            _clip(dvl[2] / VELOCITY_SCALE_MPS, -1.0, 1.0),
            1.0,
        ]

    yaw_rate = _imu_yaw_rate(sensors)
    if yaw_rate is None:
        features += [0.0, 0.0]  # imu_yaw_rate_norm, imu_present
    else:
        features += [_clip(yaw_rate / YAW_RATE_SCALE_RPS, -1.0, 1.0), 1.0]

    # --- Controller-local state: tracker phase one-hot ------------------------
    phase_onehot = [0.0] * len(TRACKER_PHASES)
    idx = _PHASE_INDEX.get(context.tracker_phase) if context.tracker_phase else None
    if idx is not None:
        phase_onehot[idx] = 1.0
    features += phase_onehot

    # --- Controller-local state: progress, lock, previous action --------------
    total = max(1, int(context.total_beacons or 1))
    laps = max(1, int(context.laps or 1))
    features.append(_clip(float(context.local_beacon_index) / float(total), 0.0, 1.0))
    features.append(_clip(float(context.local_lap) / float(laps), 0.0, 1.0))
    features.append(1.0 if context.visual_lock else 0.0)
    prev = list(context.prev_action) if context.prev_action is not None else []
    for i in range(ACTION_DIM):
        value = _finite(prev[i]) if i < len(prev) else 0.0
        features.append(_clip(value, -1.0, 1.0))

    vector = np.asarray(features, dtype=np.float32)
    # Defensive: guarantee finiteness and exact shape regardless of inputs.
    vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)
    if vector.shape != (OBS_DIM,):  # pragma: no cover - guarded by tests
        raise ValueError(f"encoded observation has shape {vector.shape}, expected ({OBS_DIM},)")
    return vector
