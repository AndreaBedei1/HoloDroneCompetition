"""Independent acoustic beacon transmitters.

Every gate carries one beacon with a sequential ID (``B01`` ... ``BN``) that
matches the official gate ordering. Beacons transmit periodically at their
configured update rate regardless of any participant's race progress: the
referee never activates a beacon and never selects which beacon a controller
should listen to. A receiver either physically hears a transmission (in range,
not dropped) and gets a noisy relative measurement packet, or it hears nothing
at all.

A delivered packet carries only physically observable fields:

    beacon_id, bearing_deg, elevation_deg, range_m, signal_strength,
    received_at_s

It never carries target/progress information, exact positions, dropout or
range-rejection reasons, noise levels, or gate geometry.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

from marine_race_arena.arena.gate import Gate
from marine_race_arena.config.schema import BeaconConfig, Vector3

#: Exactly the keys an official beacon packet may contain.
BEACON_PACKET_FIELDS = (
    "beacon_id",
    "bearing_deg",
    "elevation_deg",
    "range_m",
    "signal_strength",
    "received_at_s",
)


@dataclass(frozen=True)
class Beacon:
    """A single independent beacon transmitter mounted on one gate."""

    id: str
    gate_id: str
    position: Vector3
    range_m: float
    angular_noise_std_deg: float
    range_noise_std_m: float
    dropout_probability: float
    update_rate_hz: float
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
            position=position,
            range_m=config.range_m,
            angular_noise_std_deg=config.angular_noise_std_deg,
            range_noise_std_m=config.range_noise_std_m,
            dropout_probability=config.dropout_probability,
            update_rate_hz=config.update_rate_hz,
            enabled=True,
        )

    def transmission_index(self, time_s: float) -> int:
        """Index of the most recent periodic transmission at or before ``time_s``.

        Transmissions happen at t = k / update_rate_hz for k = 0, 1, 2, ...
        Returns -1 before the first transmission.
        """
        if self.update_rate_hz <= 0.0:
            return -1
        if time_s < -1e-9:
            return -1
        return int(math.floor(time_s * self.update_rate_hz + 1e-9))

    def receive(
        self,
        *,
        receiver_position: Vector3,
        receiver_yaw_deg: float,
        received_at_s: float,
        rng: random.Random,
    ) -> Optional[Dict[str, Any]]:
        """Physically receive one transmission, or return ``None``.

        ``None`` means no packet was delivered (disabled, out of physical
        range, or a dropout). The caller must not distinguish these cases;
        diagnostic reasons are the transmitter's business, not the receiver's.
        """
        if not self.enabled:
            return None

        delta = _subtract(self.position, receiver_position)
        distance = _norm(delta)
        if distance > self.range_m:
            return None
        if self.dropout_probability > 0.0 and rng.random() < self.dropout_probability:
            return None

        horizontal_distance = math.hypot(delta[0], delta[1])
        global_bearing = math.degrees(math.atan2(delta[1], delta[0]))
        relative_bearing = _wrap_degrees(global_bearing - receiver_yaw_deg)
        elevation = math.degrees(math.atan2(delta[2], horizontal_distance))
        range_measurement = distance

        # Draw order is fixed: bearing, then elevation (both with the angular
        # sigma, in degrees), then range (with the range sigma, in metres).
        # Range noise is applied exactly once. When both sigmas equal the
        # legacy scalar this reproduces the pre-refactor packet stream bit for
        # bit (see tests/test_beacon_noise_migration.py).
        if self.angular_noise_std_deg > 0.0 or self.range_noise_std_m > 0.0:
            relative_bearing += rng.gauss(0.0, self.angular_noise_std_deg)
            elevation += rng.gauss(0.0, self.angular_noise_std_deg)
            range_measurement = max(0.0, range_measurement + rng.gauss(0.0, self.range_noise_std_m))

        # Signal strength is derived from the noisy range so the packet never
        # carries a second, cleaner distance estimate as a side channel.
        signal_strength = max(0.0, 1.0 - range_measurement / self.range_m)
        return {
            "beacon_id": self.id,
            "bearing_deg": _wrap_degrees(relative_bearing),
            "elevation_deg": elevation,
            "range_m": range_measurement,
            "signal_strength": signal_strength,
            "received_at_s": received_at_s,
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
