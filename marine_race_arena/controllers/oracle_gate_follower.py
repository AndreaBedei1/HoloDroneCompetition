"""Debug-only no-yaw oracle gate follower.

This controller intentionally uses ground truth and is not competition-valid.
It is a simple feasibility tool: it translates the BlueROV2 through each gate
without commanding yaw rotation.

Under the onboard-only architecture the oracle receives no referee feedback
either: ``debug_ground_truth`` carries its own pose and the full ordered gate
geometry, and the oracle tracks its own progression by detecting its gate-plane
crossings from that ground truth. It runs only in non-official debug mode.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from marine_race_arena.participants.controller_interface import BaseController


Vector3 = Tuple[float, float, float]


class OracleGateFollowerController(BaseController):
    """Cheating feasibility controller that translates through gate centers."""

    debug_only = True
    uses_ground_truth = True

    def reset(self, mission_info: Dict[str, Any]) -> None:
        self.max_speed = 0.65
        self.laps = max(1, int(mission_info.get("laps", 1)))
        self.exit_distance_m = 2.6
        self.approach_distance_m = 1.2
        self.approach_transition_tolerance_m = 0.08
        self._gate_index = 0
        self._lap = 1
        self._finished = False
        self._previous_position: Optional[Vector3] = None
        self._exit_hold: Optional[Dict[str, Any]] = None
        self.debug_state: Dict[str, Any] = {}

    def step(self, observation: Dict[str, Any]) -> Dict[str, float]:
        debug = observation.get("debug_ground_truth")
        if not debug:
            return _zero_command()
        gates = _parse_gates(debug.get("gates"))
        if not gates or self._finished:
            return _zero_command()

        own_position = _vector3(debug["own_position"])
        own_yaw_deg = float(tuple(debug.get("own_rotation_rpy_deg", (0.0, 0.0, 0.0)))[2])
        bounds = debug.get("bounds", {})
        z_min = float(bounds.get("z_min", -8.0))
        z_max = float(bounds.get("z_max", -1.0))

        self._advance_on_crossing(gates, own_position)
        if self._finished:
            self._previous_position = own_position
            return _zero_command()

        gate_geometry = gates[self._gate_index]

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
        lateral_right = _dot(_subtract(own_position, gate_center), gate_right)
        lateral_up = _dot(_subtract(own_position, gate_center), gate_up)
        lateral_error = math.hypot(lateral_right, lateral_up)
        if phase == "EXIT_HOLD":
            target = _add(gate_center, _scale(gate_normal, self.exit_distance_m))
        elif signed_distance < -0.2 and (
            signed_distance < -(self.approach_distance_m + self.approach_transition_tolerance_m)
            or lateral_error > 0.20
        ):
            target = _subtract(gate_center, _scale(gate_normal, self.approach_distance_m))
            phase = "APPROACH"
        else:
            target = _add(gate_center, _scale(gate_normal, self.exit_distance_m))
            phase = "TRANSIT"

        error = _subtract(target, own_position)

        desired_world = _limit_vector(error, self.max_speed)
        command = _world_velocity_to_body_command(desired_world, own_yaw_deg)
        command["heave"] = _clamp(command["heave"], -0.45, 0.45)
        if own_position[2] < z_min + 0.4:
            command["heave"] = max(command["heave"], 0.25)
        if own_position[2] > z_max - 0.4:
            command["heave"] = min(command["heave"], -0.25)
        command["yaw"] = 0.0

        self.debug_state = {
            "phase": phase,
            "gate_index": self._gate_index,
            "lap": self._lap,
            "signed_gate_distance": signed_distance,
            "lateral_error": lateral_error,
            "yaw_command": 0.0,
        }
        self._previous_position = own_position
        return command

    def close(self) -> None:
        pass

    def _advance_on_crossing(self, gates: List[Dict[str, Any]], own_position: Vector3) -> None:
        """Advance the oracle's own gate index when it crosses the current plane."""
        if self._previous_position is None:
            return
        gate = gates[self._gate_index]
        d0 = _dot(_subtract(self._previous_position, gate["center"]), gate["normal"])
        d1 = _dot(_subtract(own_position, gate["center"]), gate["normal"])
        if not (d0 <= 0.0 < d1):
            return
        self._exit_hold = dict(gate)
        if self._gate_index == len(gates) - 1:
            if self._lap >= self.laps:
                self._finished = True
                return
            self._lap += 1
            self._gate_index = 0
        else:
            self._gate_index += 1

    def _exit_hold_complete(self, own_position: Vector3, geometry: Dict[str, Any]) -> bool:
        signed_distance = _dot(_subtract(own_position, geometry["center"]), geometry["normal"])
        return signed_distance >= self.exit_distance_m - 0.15


def _parse_gates(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    gates: List[Dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            gates.append(
                {
                    "gate_id": str(entry.get("gate_id", "")),
                    "center": _vector3(entry["center"]),
                    "normal": _normalize(_vector3(entry.get("normal", (1.0, 0.0, 0.0)))),
                    "right": _normalize(_vector3(entry.get("right_axis", (0.0, 1.0, 0.0)))),
                    "up": _normalize(_vector3(entry.get("up_axis", (0.0, 0.0, 1.0)))),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return gates


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
