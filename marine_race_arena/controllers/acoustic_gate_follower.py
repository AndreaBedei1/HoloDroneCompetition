"""Competition-safe acoustic beacon baseline controller."""

from __future__ import annotations

import math
from typing import Any, Dict

from marine_race_arena.participants.controller_interface import BaseController


class AcousticGateFollowerController(BaseController):
    """Simple baseline that follows bearing/range/elevation beacon observations."""

    debug_only = False
    uses_ground_truth = False

    def reset(self, race_info: Dict[str, Any]) -> None:
        self.max_speed = float(race_info.get("max_command", 0.85))

    def step(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        beacon = observation.get("beacon", {})
        if not beacon.get("valid"):
            return {"surge": 0.10, "sway": 0.0, "heave": 0.0, "yaw": 0.15}

        bearing_deg = float(beacon.get("bearing_deg") or 0.0)
        elevation_deg = float(beacon.get("elevation_deg") or 0.0)
        range_m = float(beacon.get("range_m") or 0.0)
        bearing = math.radians(bearing_deg)
        elevation = math.radians(elevation_deg)
        speed = min(self.max_speed, max(0.2, range_m / 8.0))

        yaw_command = max(-1.0, min(1.0, bearing / math.radians(45.0)))
        return {
            "surge": max(-1.0, min(1.0, speed * math.cos(bearing) * math.cos(elevation))),
            "sway": max(-0.6, min(0.6, 0.45 * speed * math.sin(bearing))),
            "heave": max(-0.5, min(0.5, speed * math.sin(elevation))),
            "yaw": yaw_command,
        }

    def close(self) -> None:
        pass

