"""A minimal Marine Race Arena controller.

It demonstrates the participant contract and nothing else: read the acoustic
packets you actually received, steer towards the beacon you are chasing, return
a body-frame command.

Deliberately incomplete: it never advances past its first beacon, because
deciding that you have passed a gate is the participant's job. See
``docs/controllers.md`` and ``marine_race_arena/controllers/student_template.py``
for the version that tracks course progression with ``LocalCourseTracker``.

Run it:

    python -m marine_race_arena.scripts.run_marine_race \
      --track marine_race_arena/tracks/marine_race_horseshoe_bay.json \
      --benchmark-task clean_gate \
      --participant-controller docs/examples/minimal_controller.py \
      --controller-class MinimalBeaconController \
      --adapter holoocean --official --headless \
      --seed 0 --dt 0.1 --duration 120 \
      --log-dir results/minimal_controller
"""

from __future__ import annotations

import math
from typing import Any, Dict

from marine_race_arena.participants.controller_interface import BaseController


class MinimalBeaconController(BaseController):
    # Declares that this controller never reads privileged simulator state, so
    # it is allowed in official mode.
    debug_only = False
    uses_ground_truth = False

    def reset(self, mission_info: Dict[str, Any]) -> None:
        """Called once before the race. Only the mission assignment is given."""
        self.expected_beacon = str(mission_info.get("initial_beacon_id", "B01"))
        limits = mission_info.get("command_limits", {})
        self.max_command = float(limits.get("surge", [-0.95, 0.95])[1])

    def step(self, observation: Dict[str, Any]) -> Dict[str, float]:
        """Called once per control step. Return surge / sway / heave / yaw."""
        packets = [p for p in observation.get("beacons", [])
                   if p["beacon_id"] == self.expected_beacon]
        if not packets:
            # Nothing heard: creep forward and rotate to search.
            return self._clamp({"surge": 0.15, "sway": 0.0, "heave": 0.0, "yaw": 0.15})

        packet = max(packets, key=lambda p: p["received_at_s"])
        bearing = math.radians(packet["bearing_deg"])
        elevation = math.radians(packet["elevation_deg"])
        speed = max(0.15, min(0.6, packet["range_m"] / 8.0))

        return self._clamp({
            "surge": speed * math.cos(bearing),
            "sway": 0.0,
            "heave": 0.6 * math.sin(elevation),
            "yaw": 1.2 * bearing,
        })

    def close(self) -> None:
        """Called once after the race. Release anything you opened."""

    def _clamp(self, command: Dict[str, float]) -> Dict[str, float]:
        limit = self.max_command
        return {axis: max(-limit, min(limit, float(value)))
                for axis, value in command.items()}
