"""Student controller template.

This template uses only competition-safe observations. It does not access ground
truth gate positions or referee internals.
"""

from __future__ import annotations

import math
from typing import Any, Dict

from marine_race_arena.participants.controller_interface import BaseController


class StudentController(BaseController):
    debug_only = False
    uses_ground_truth = False

    def reset(self, race_info: Dict[str, Any]) -> None:
        self.last_target_gate_id = race_info.get("initial_target_gate_id")

    def step(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        beacon = observation.get("beacon", {})
        if not beacon.get("valid"):
            return {"surge": 0.15, "sway": 0.0, "heave": 0.0, "yaw": 0.0}

        bearing = math.radians(float(beacon.get("bearing_deg") or 0.0))
        elevation = math.radians(float(beacon.get("elevation_deg") or 0.0))
        range_m = float(beacon.get("range_m") or 0.0)
        speed = max(0.15, min(0.75, range_m / 6.0))

        return {
            "surge": speed * math.cos(bearing) * math.cos(elevation),
            "sway": speed * math.sin(bearing),
            "heave": max(-0.45, min(0.45, speed * math.sin(elevation))),
            "yaw": max(-0.7, min(0.7, bearing)),
        }

    def close(self) -> None:
        pass

