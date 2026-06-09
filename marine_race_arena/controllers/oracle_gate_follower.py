"""Debug-only oracle gate follower.

This controller intentionally uses ground truth and is not competition-valid.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

from marine_race_arena.participants.controller_interface import BaseController


class OracleGateFollowerController(BaseController):
    """Cheating feasibility controller that follows exact gate centers."""

    debug_only = True
    uses_ground_truth = True

    def reset(self, race_info: Dict[str, Any]) -> None:
        self.max_speed = float(race_info.get("max_command", 0.95))
        self.z_min = float(race_info.get("bounds", {}).get("z_min", -8.0))

    def step(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        debug = observation.get("debug_ground_truth")
        if not debug:
            return {"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}

        own_position = tuple(debug["own_position"])
        own_rotation = tuple(debug.get("own_rotation_rpy_deg", (0.0, 0.0, 0.0)))
        target_center = tuple(debug["target_gate_center"])
        target_normal = tuple(debug.get("target_gate_normal", (1.0, 0.0, 0.0)))
        target_right = tuple(debug.get("target_gate_right_axis", (0.0, 1.0, 0.0)))
        target_up = tuple(debug.get("target_gate_up_axis", (0.0, 0.0, 1.0)))

        aim_point = self._aim_point(own_position, target_center, target_normal, target_right, target_up)
        vector = (
            aim_point[0] - own_position[0],
            aim_point[1] - own_position[1],
            aim_point[2] - own_position[2],
        )
        distance = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
        if distance <= 1e-6:
            return {"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}

        yaw_rad = math.radians(float(own_rotation[2]))
        body_x = math.cos(yaw_rad) * vector[0] + math.sin(yaw_rad) * vector[1]
        body_y = -math.sin(yaw_rad) * vector[0] + math.cos(yaw_rad) * vector[1]
        heading_to_target = math.atan2(vector[1], vector[0])
        yaw_error = _wrap_radians(heading_to_target - yaw_rad)

        gate_distance = self._signed_gate_distance(own_position, target_center, target_normal)
        near_gate_speed = 0.55 if abs(gate_distance) < 2.0 else self.max_speed
        speed = min(near_gate_speed, self.max_speed, max(0.25, distance / 4.0))
        norm_xy = max(1e-6, math.hypot(body_x, body_y))
        heave = max(-0.55, min(0.55, vector[2] / max(distance, 1e-6) * speed))
        if own_position[2] < self.z_min + 0.4:
            heave = max(heave, 0.25)

        return {
            "surge": max(-1.0, min(1.0, speed * body_x / norm_xy)),
            "sway": max(-0.8, min(0.8, speed * body_y / norm_xy)),
            "heave": heave,
            "yaw": max(-1.0, min(1.0, yaw_error / math.radians(50.0))),
        }

    def close(self) -> None:
        pass

    def _aim_point(
        self,
        own_position: Tuple[float, float, float],
        target_center: Tuple[float, float, float],
        target_normal: Tuple[float, float, float],
        target_right: Tuple[float, float, float],
        target_up: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        relative = (
            own_position[0] - target_center[0],
            own_position[1] - target_center[1],
            own_position[2] - target_center[2],
        )
        signed_distance = self._signed_gate_distance(own_position, target_center, target_normal)
        lateral_right = (
            relative[0] * target_right[0]
            + relative[1] * target_right[1]
            + relative[2] * target_right[2]
        )
        lateral_up = relative[0] * target_up[0] + relative[1] * target_up[1] + relative[2] * target_up[2]
        lateral_error = math.hypot(lateral_right, lateral_up)
        if signed_distance < -1.0 and lateral_error > 0.25:
            return (
                target_center[0] - target_normal[0] * 1.2,
                target_center[1] - target_normal[1] * 1.2,
                target_center[2] - target_normal[2] * 1.2,
            )
        if signed_distance < 0.4:
            return (
                target_center[0] + target_normal[0] * 0.9,
                target_center[1] + target_normal[1] * 0.9,
                target_center[2] + target_normal[2] * 0.9,
            )
        # If the gate is still expected after the vehicle is in front of it, recover
        # by returning to the entry side and attempting a cleaner pass.
        return (
            target_center[0] - target_normal[0] * 1.5,
            target_center[1] - target_normal[1] * 1.5,
            target_center[2] - target_normal[2] * 1.5,
        )

    def _signed_gate_distance(
        self,
        own_position: Tuple[float, float, float],
        target_center: Tuple[float, float, float],
        target_normal: Tuple[float, float, float],
    ) -> float:
        return (
            (own_position[0] - target_center[0]) * target_normal[0]
            + (own_position[1] - target_center[1]) * target_normal[1]
            + (own_position[2] - target_center[2]) * target_normal[2]
        )


def _wrap_radians(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
