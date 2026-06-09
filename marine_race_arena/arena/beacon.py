"""Acoustic beacon models for gate guidance."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

from marine_race_arena.arena.gate import Gate
from marine_race_arena.config.schema import BeaconConfig, Vector3


@dataclass(frozen=True)
class Beacon:
    id: str
    gate_id: str
    mode: str
    position: Vector3
    range_m: float
    noise_std: float
    dropout_probability: float
    update_rate_hz: float
    message: Dict[str, Any]
    gate_center: Vector3
    gate_normal: Vector3
    enabled: bool = True

    @classmethod
    def from_gate(cls, gate: Gate, config: Optional[BeaconConfig]) -> Optional["Beacon"]:
        if config is None or not config.enabled or not config.id:
            return None
        offset = config.position_offset
        position = _add(
            gate.center,
            _add(
                _scale(gate.normal_vector, offset[0]),
                _add(_scale(gate.right_axis, offset[1]), _scale(gate.up_axis, offset[2])),
            ),
        )
        return cls(
            id=config.id,
            gate_id=gate.id,
            mode=config.mode,
            position=position,
            range_m=config.range_m,
            noise_std=config.noise_std,
            dropout_probability=config.dropout_probability,
            update_rate_hz=config.update_rate_hz,
            message=dict(config.message),
            gate_center=gate.center,
            gate_normal=gate.normal_vector,
            enabled=True,
        )

    def observe(
        self,
        receiver_position: Vector3,
        receiver_yaw_deg: float,
        target_sequence_index: int,
        observation_mode: str,
        official_mode: bool,
        rng: random.Random,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return self._invalid(target_sequence_index, "disabled")

        delta = _subtract(self.position, receiver_position)
        distance = _norm(delta)
        if distance > self.range_m:
            observation = self._invalid(target_sequence_index, "out_of_range")
            observation["range_m"] = distance
            observation["signal_strength"] = 0.0
            return observation
        if self.dropout_probability > 0.0 and rng.random() < self.dropout_probability:
            return self._invalid(target_sequence_index, "dropout")

        horizontal_distance = math.hypot(delta[0], delta[1])
        global_bearing = math.degrees(math.atan2(delta[1], delta[0]))
        relative_bearing = _wrap_degrees(global_bearing - receiver_yaw_deg)
        elevation = math.degrees(math.atan2(delta[2], horizontal_distance))
        range_m = distance

        noise_level = 0.0
        if observation_mode == "acoustic_noisy" and self.noise_std > 0.0:
            noise_level = self.noise_std
            relative_bearing += rng.gauss(0.0, self.noise_std)
            elevation += rng.gauss(0.0, self.noise_std)
            range_m = max(0.0, range_m + rng.gauss(0.0, self.noise_std))

        observation: Dict[str, Any] = {
            "valid": True,
            "reason": "ok",
            "active_beacon_id": self.id,
            "target_gate_id": self.gate_id,
            "sequence_index": target_sequence_index,
            "bearing_deg": _wrap_degrees(relative_bearing),
            "elevation_deg": elevation,
            "range_m": range_m,
            "signal_strength": max(0.0, 1.0 - distance / self.range_m),
            "noise_level": noise_level,
            "mode": observation_mode,
            "message": dict(self.message),
        }
        if observation_mode == "oracle" and not official_mode:
            observation["exact_gate_center"] = self.gate_center
            observation["exact_gate_normal"] = self.gate_normal
            observation["exact_beacon_position"] = self.position
        return observation

    def _invalid(self, target_sequence_index: int, reason: str) -> Dict[str, Any]:
        return {
            "valid": False,
            "reason": reason,
            "active_beacon_id": self.id,
            "target_gate_id": self.gate_id,
            "sequence_index": target_sequence_index,
            "bearing_deg": None,
            "elevation_deg": None,
            "range_m": None,
            "signal_strength": 0.0,
            "noise_level": self.noise_std,
            "mode": self.mode,
            "message": dict(self.message),
        }


def _add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _subtract(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(vector: Vector3, scalar: float) -> Vector3:
    return (vector[0] * scalar, vector[1] * scalar, vector[2] * scalar)


def _norm(vector: Vector3) -> float:
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


def _wrap_degrees(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0

