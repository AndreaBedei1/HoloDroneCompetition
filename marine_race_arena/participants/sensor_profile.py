"""Official controller observation construction.

The official observation is built exclusively from onboard information:

    {
        "local_time_s": <participant-local elapsed time since release>,
        "sensors": {<allowed onboard sensors only>},
        "beacons": [<physically received beacon packets>],
        "comms": {"inbox": [...]}      # only when inter-rover comms is enabled
    }

It never contains referee state, race progress, expected targets, ground-truth
pose/heading/depth, world-frame velocity, or environment metadata. The builder
deliberately has no access to the referee: isolation is structural, not a
filter. A defensive strip of forbidden keys is still applied to the sensor
dictionary as a last line of defense, but adapters are required to remove
privileged fields at their source.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

#: Keys the official observation may contain, and nothing else.
OFFICIAL_OBSERVATION_FIELDS = ("local_time_s", "sensors", "beacons", "comms")

#: Onboard sensors permitted in an official observation.
OFFICIAL_SENSOR_ALLOWLIST = frozenset(
    {
        "FrontCamera",
        "DepthSensor",
        "IMUSensor",
        "DVLSensor",
        "CollisionSensor",
    }
)

#: Sensor keys that must never reach a controller in official mode. Kept as a
#: defensive blacklist on top of source-level removal in the adapters.
FORBIDDEN_SENSOR_KEYS = frozenset(
    {
        "PoseSensor",
        "LocationSensor",
        "RotationSensor",
        "DynamicsSensor",
        "VelocitySensor",
        "GPSSensor",
        "heading_yaw_deg",
        "depth_m",
        "environment_current_m_s",
        "current_physical_coupling_active",
        "current_coupling_method",
        "control_mode",
        "ground_truth",
        "debug_ground_truth",
        "exact_pose",
        "exact_position",
    }
)


def build_observation(
    *,
    local_time_s: float,
    sensor_data: Mapping[str, Any],
    beacon_packets: Iterable[Mapping[str, Any]],
    official_mode: bool,
    comms_inbox: Optional[List[Dict[str, Any]]] = None,
    debug_ground_truth: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the controller observation for one step.

    ``debug_ground_truth`` is attached only in non-official debug runs for
    explicitly ground-truth controllers (e.g. the oracle); it never appears in
    official mode.
    """
    sensors = _sanitized_sensors(sensor_data, official_mode)
    observation: Dict[str, Any] = {
        "local_time_s": float(local_time_s),
        "sensors": sensors,
        "beacons": [dict(packet) for packet in beacon_packets],
    }
    if comms_inbox is not None:
        observation["comms"] = {"inbox": list(comms_inbox)}
    if debug_ground_truth is not None and not official_mode:
        observation["debug_ground_truth"] = dict(debug_ground_truth)
    return observation


def _sanitized_sensors(sensor_data: Mapping[str, Any], official_mode: bool) -> Dict[str, Any]:
    sensors: Dict[str, Any] = {}
    for name, value in sensor_data.items():
        if name in FORBIDDEN_SENSOR_KEYS:
            continue
        if official_mode and name not in OFFICIAL_SENSOR_ALLOWLIST:
            continue
        sensors[name] = value
    return sensors
