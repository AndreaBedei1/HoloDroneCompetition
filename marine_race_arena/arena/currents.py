"""Configurable marine current fields."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, List

from marine_race_arena.config.schema import CurrentConfig, Vector3

LOGGER = logging.getLogger(__name__)


class CurrentFieldManager:
    def __init__(self, currents: Iterable[CurrentConfig]):
        self.currents = list(currents)
        self._warned_physical_coupling = False

    def get_current_at(self, position: Vector3, time_s: float) -> Vector3:
        velocity = (0.0, 0.0, 0.0)
        for current in self.currents:
            velocity = _add(velocity, self._evaluate_current(current, position, time_s))
        return velocity

    def apply_current_to_vehicle(self, sim_interface: Any, participant_id: str, position: Vector3, time_s: float) -> bool:
        """Apply current to a simulator vehicle when an adapter supports it."""

        current = self.get_current_at(position, time_s)
        if hasattr(sim_interface, "apply_current_to_vehicle"):
            sim_interface.apply_current_to_vehicle(participant_id, current)
            return True
        if not self._warned_physical_coupling:
            LOGGER.warning(
                "No simulator current-force adapter is available; currents are exposed to logs, "
                "controllers, and the fallback kinematic runner only."
            )
            self._warned_physical_coupling = True
        return False

    def _evaluate_current(self, current: CurrentConfig, position: Vector3, time_s: float) -> Vector3:
        params = current.params
        if current.type == "constant":
            return _vector(params.get("velocity", [0.0, 0.0, 0.0]))
        if current.type == "localized_jet":
            center = _vector(params.get("center", [0.0, 0.0, 0.0]))
            radius = float(params.get("radius", 1.0))
            velocity = _vector(params.get("velocity", [0.0, 0.0, 0.0]))
            distance = _distance(position, center)
            if distance > radius:
                return (0.0, 0.0, 0.0)
            falloff = str(params.get("falloff", "gaussian"))
            if falloff == "linear":
                weight = max(0.0, 1.0 - distance / radius)
            else:
                sigma = max(radius / 2.0, 1e-6)
                weight = math.exp(-(distance**2) / (2.0 * sigma**2))
            return _scale(velocity, weight)
        if current.type == "sinusoidal":
            base_velocity = _vector(params.get("base_velocity", [0.0, 0.0, 0.0]))
            axis = str(params.get("axis", "x"))
            amplitude = float(params.get("amplitude", 0.0))
            frequency = float(params.get("frequency", 0.1))
            phase = float(params.get("phase", 0.0))
            value = amplitude * math.sin(2.0 * math.pi * frequency * time_s + phase)
            components = list(base_velocity)
            axis_index = {"x": 0, "y": 1, "z": 2}.get(axis, 0)
            components[axis_index] += value
            return (components[0], components[1], components[2])
        if current.type == "vortex":
            LOGGER.debug("Vortex current placeholder evaluated as zero velocity.")
            return (0.0, 0.0, 0.0)
        return (0.0, 0.0, 0.0)


def _vector(value: Any) -> Vector3:
    values: List[float] = [float(component) for component in value]
    if len(values) != 3:
        return (0.0, 0.0, 0.0)
    return (values[0], values[1], values[2])


def _add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(vector: Vector3, scalar: float) -> Vector3:
    return (vector[0] * scalar, vector[1] * scalar, vector[2] * scalar)


def _distance(a: Vector3, b: Vector3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

