"""Student controller template.

This template uses only the official onboard observation contract:

* ``observation["local_time_s"]`` — your own elapsed time since release;
* ``observation["sensors"]`` — FrontCamera, DepthSensor, IMUSensor, DVLSensor;
* ``observation["beacons"]`` — the beacon packets you physically received.

Nobody tells you which beacon to chase or when you passed a gate: you must
track your own progress. The provided
:class:`~marine_race_arena.controllers.local_course_tracker.LocalCourseTracker`
does that from your camera, DVL and received packets; this template shows the
intended usage. Replace the simple steering with your own logic.
"""

from __future__ import annotations

import math
from typing import Any, Dict

from marine_race_arena.controllers.local_course_tracker import LocalCourseTracker
from marine_race_arena.participants.controller_interface import BaseController


class StudentController(BaseController):
    debug_only = False
    uses_ground_truth = False

    def reset(self, mission_info: Dict[str, Any]) -> None:
        self.tracker = LocalCourseTracker(
            initial_beacon_id=str(mission_info.get("initial_beacon_id", "B01")),
            total_beacons=int(mission_info.get("total_beacons", 1)),
            laps=int(mission_info.get("laps", 1)),
        )

    def step(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        sensors = observation.get("sensors", {})
        tracker_step = self.tracker.update(
            local_time_s=float(observation.get("local_time_s", 0.0)),
            beacons=observation.get("beacons", []),
            camera_image=sensors.get("FrontCamera"),
            dvl_velocity=sensors.get("DVLSensor"),
        )
        if tracker_step.finished:
            return {"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}

        beacon = tracker_step.beacon  # your expected beacon, if recently heard
        if beacon is None:
            return {"surge": 0.15, "sway": 0.0, "heave": 0.0, "yaw": 0.10}

        bearing = math.radians(beacon.bearing_deg)
        elevation = math.radians(beacon.elevation_deg)
        speed = max(0.15, min(0.75, beacon.range_m / 6.0))
        return {
            "surge": speed * math.cos(bearing) * math.cos(elevation),
            "sway": speed * math.sin(bearing),
            "heave": max(-0.45, min(0.45, speed * math.sin(elevation))),
            "yaw": max(-0.7, min(0.7, bearing)),
        }

    def close(self) -> None:
        pass
