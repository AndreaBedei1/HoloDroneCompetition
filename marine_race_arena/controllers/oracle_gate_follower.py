"""Debug-only no-yaw oracle gate follower.

This controller intentionally uses ground truth and is not competition-valid.
It is a simple feasibility tool: it translates the BlueROV2 through each gate
without commanding yaw rotation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from marine_race_arena.participants.controller_interface import BaseController


Vector3 = Tuple[float, float, float]


class OracleGateFollowerController(BaseController):
    """Cheating feasibility controller that translates through gate centers."""

    debug_only = True
    uses_ground_truth = True

    def reset(self, race_info: Dict[str, Any]) -> None:
        self.max_speed = min(float(race_info.get("max_command", 0.95)), 0.65)
        bounds = race_info.get("bounds", {})
        self.z_min = float(bounds.get("z_min", -8.0))
        self.z_max = float(bounds.get("z_max", -1.0))
        self.exit_distance_m = 2.6
        self.approach_distance_m = 1.2
        self._last_gate_id: Optional[str] = None
        self._last_gate_geometry: Optional[Dict[str, Any]] = None
        self._exit_hold: Optional[Dict[str, Any]] = None
        self.debug_state: Dict[str, Any] = {}

    def step(self, observation: Dict[str, Any]) -> Dict[str, float]:
        debug = observation.get("debug_ground_truth")
        if not debug:
            return _zero_command()

        own_position = _vector3(debug["own_position"])
        own_yaw_deg = float(tuple(debug.get("own_rotation_rpy_deg", (0.0, 0.0, 0.0)))[2])
        target_gate_id = str(observation.get("race", {}).get("target_gate_id", ""))
        gate_geometry = {
            "gate_id": target_gate_id,
            "center": _vector3(debug["target_gate_center"]),
            "normal": _normalize(_vector3(debug.get("target_gate_normal", (1.0, 0.0, 0.0)))),
            "right": _normalize(_vector3(debug.get("target_gate_right_axis", (0.0, 1.0, 0.0)))),
            "up": _normalize(_vector3(debug.get("target_gate_up_axis", (0.0, 0.0, 1.0)))),
        }

        if (
            self._last_gate_id is not None
            and target_gate_id != self._last_gate_id
            and self._last_gate_geometry is not None
        ):
            self._exit_hold = dict(self._last_gate_geometry)

        active_geometry = gate_geometry
        phase = "TRANSLATE"
        if self._exit_hold is not None and not self._exit_hold_complete(own_position, self._exit_hold):
            active_geometry = self._exit_hold
            phase = "EXIT_HOLD"
        else:
            self._exit_hold = None

        gate_center = active_geometry["center"]
        gate_normal = active_geometry["normal"]
        gate_right = active_geometry["right"]
        gate_up = active_geometry["up"]

        signed_distance = _dot(_subtract(own_position, gate_center), gate_normal)
        if phase == "EXIT_HOLD":
            target = _add(gate_center, _scale(gate_normal, self.exit_distance_m))
        elif signed_distance < -self.approach_distance_m:
            target = _subtract(gate_center, _scale(gate_normal, self.approach_distance_m))
            phase = "APPROACH"
        else:
            target = _add(gate_center, _scale(gate_normal, self.exit_distance_m))
            phase = "TRANSIT"

        error = _subtract(target, own_position)
        lateral_right = _dot(_subtract(own_position, gate_center), gate_right)
        lateral_up = _dot(_subtract(own_position, gate_center), gate_up)
        lateral_error = math.hypot(lateral_right, lateral_up)

        desired_world = _limit_vector(error, self.max_speed)
        command = _world_velocity_to_body_command(desired_world, own_yaw_deg)
        command["heave"] = _clamp(command["heave"], -0.45, 0.45)
        if own_position[2] < self.z_min + 0.4:
            command["heave"] = max(command["heave"], 0.25)
        if own_position[2] > self.z_max - 0.4:
            command["heave"] = min(command["heave"], -0.25)
        command["yaw"] = 0.0

        self.debug_state = {
            "phase": phase,
            "target_gate_id": target_gate_id,
            "active_gate_id": active_geometry["gate_id"],
            "signed_gate_distance": signed_distance,
            "lateral_error": lateral_error,
            "yaw_command": 0.0,
        }
        self._last_gate_id = target_gate_id
        self._last_gate_geometry = gate_geometry
        return command

    def close(self) -> None:
        pass

    def _exit_hold_complete(self, own_position: Vector3, geometry: Dict[str, Any]) -> bool:
        signed_distance = _dot(_subtract(own_position, geometry["center"]), geometry["normal"])
        return signed_distance >= self.exit_distance_m - 0.15


def _zero_command() -> Dict[str, float]:
    return {"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}


def _world_velocity_to_body_command(world_velocity: Vector3, yaw_deg: float) -> Dict[str, float]:
    yaw_rad = math.radians(yaw_deg)
    surge = math.cos(yaw_rad) * world_velocity[0] + math.sin(yaw_rad) * world_velocity[1]
    sway = -math.sin(yaw_rad) * world_velocity[0] + math.cos(yaw_rad) * world_velocity[1]
    return {
        "surge": _clamp(surge, -0.85, 0.85),
        "sway": _clamp(sway, -0.85, 0.85),
        "heave": _clamp(world_velocity[2], -0.85, 0.85),
        "yaw": 0.0,
    }


def _limit_vector(vector: Vector3, max_length: float) -> Vector3:
    length = math.sqrt(_dot(vector, vector))
    if length <= 1e-9:
        return (0.0, 0.0, 0.0)
    scale = min(max_length, length) / length
    return _scale(vector, scale)


def _vector3(value: Any) -> Vector3:
    return (float(value[0]), float(value[1]), float(value[2]))


def _add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _subtract(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(vector: Vector3, scalar: float) -> Vector3:
    return (vector[0] * scalar, vector[1] * scalar, vector[2] * scalar)


def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _normalize(vector: Vector3) -> Vector3:
    length = math.sqrt(_dot(vector, vector))
    if length <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
