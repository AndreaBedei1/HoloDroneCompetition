"""Observation building with official/debug sensor separation."""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_observation(
    participant_id: str,
    time_s: float,
    sensor_data: Dict[str, Any],
    beacon_observation: Dict[str, Any],
    race_progress: Dict[str, Any],
    official_mode: bool,
    debug_ground_truth: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    observation: Dict[str, Any] = {
        "participant_id": participant_id,
        "time_s": time_s,
        "sensors": dict(sensor_data),
        "beacon": dict(beacon_observation),
        "race": dict(race_progress),
    }
    if debug_ground_truth is not None and not official_mode:
        observation["debug_ground_truth"] = dict(debug_ground_truth)
    return observation

