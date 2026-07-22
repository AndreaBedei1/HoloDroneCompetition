"""Deterministic, seeded start-pose and beacon-noise randomization for training.

Used by the curriculum (Stage 2 onward) to make a learned policy robust to varied
initial conditions and sensor noise, without modifying the official tracks or
relaxing the referee. Given a fixed seed and spec, the perturbation is fully
reproducible, and the applied values are returned so they can be logged.

Only the participant's start pose and the beacon-noise parameters are perturbed;
the gate geometry, referee, observation contract and action mapping are untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Decorrelate the randomization stream from the simulator/beacon seed so the two
# do not move in lock-step, while staying deterministic in the episode seed.
_RANDOMIZATION_SALT = 0x5741524D  # "WARM"


@dataclass(frozen=True)
class StartRandomization:
    """Ranges for seeded start-pose and beacon-noise randomization.

    Offsets are sampled uniformly in ``[-value, +value]``. Beacon-noise fields
    override the track's values when set (``None`` keeps the track value).
    """

    lateral_offset_m: float = 0.0        # body-lateral (y) start offset
    depth_offset_m: float = 0.0          # vertical (z) start offset
    yaw_offset_deg: float = 0.0          # yaw start offset
    longitudinal_offset_m: float = 0.0   # along-approach (x) start offset
    beacon_angular_noise_std_deg: Optional[float] = None
    beacon_range_noise_std_m: Optional[float] = None
    beacon_dropout_probability: Optional[float] = None

    def is_noop(self) -> bool:
        return (
            self.lateral_offset_m == 0.0
            and self.depth_offset_m == 0.0
            and self.yaw_offset_deg == 0.0
            and self.longitudinal_offset_m == 0.0
            and self.beacon_angular_noise_std_deg is None
            and self.beacon_range_noise_std_m is None
            and self.beacon_dropout_probability is None
        )


def _uniform(rng, magnitude: float) -> float:
    magnitude = abs(float(magnitude))
    if magnitude == 0.0:
        return 0.0
    return float(rng.uniform(-magnitude, magnitude))


def apply_start_randomization(
    config: Any,
    position: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    spec: StartRandomization,
    seed: int,
) -> Tuple[Any, Tuple[float, float, float], Tuple[float, float, float], Dict[str, float]]:
    """Return (config, position, rotation, applied) with the seeded perturbation.

    ``config`` is returned unchanged unless a beacon-noise override is set, in which
    case a copy with the overridden ``beacon`` is returned (the input is frozen).
    """
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), _RANDOMIZATION_SALT]))
    # Sample offsets in the vehicle's initial *body* frame.
    d_longitudinal = _uniform(rng, spec.longitudinal_offset_m)  # along initial forward
    d_lateral = _uniform(rng, spec.lateral_offset_m)            # along initial body-right/left
    dz = _uniform(rng, spec.depth_offset_m)                     # vertical (world z)
    dyaw = _uniform(rng, spec.yaw_offset_deg)                   # relative to initial yaw

    # Rotate the body-frame (longitudinal, lateral) offset into the world frame using
    # the initial yaw. Project convention (matches the adapter body->world transform):
    #   forward f = [cos psi, sin psi],  lateral r = [-sin psi, cos psi].
    psi = math.radians(float(rotation[2]))
    cos_psi, sin_psi = math.cos(psi), math.sin(psi)
    world_dx = d_longitudinal * cos_psi - d_lateral * sin_psi
    world_dy = d_longitudinal * sin_psi + d_lateral * cos_psi

    new_position = (float(position[0]) + world_dx, float(position[1]) + world_dy, float(position[2]) + dz)
    new_rotation = (float(rotation[0]), float(rotation[1]), float(rotation[2]) + dyaw)

    beacon = config.beacon
    beacon_updates: Dict[str, float] = {}
    if spec.beacon_angular_noise_std_deg is not None:
        beacon_updates["angular_noise_std_deg"] = float(spec.beacon_angular_noise_std_deg)
    if spec.beacon_range_noise_std_m is not None:
        beacon_updates["range_noise_std_m"] = float(spec.beacon_range_noise_std_m)
    if spec.beacon_dropout_probability is not None:
        beacon_updates["dropout_probability"] = float(spec.beacon_dropout_probability)
    new_config = config
    if beacon_updates:
        new_config = replace(config, beacon=replace(beacon, **beacon_updates))

    applied = {
        "seed": int(seed),
        "initial_yaw_deg": float(rotation[2]),
        # Sampled body-frame offsets (what the spec ranges refer to).
        "longitudinal_offset_m": d_longitudinal,
        "lateral_offset_m": d_lateral,
        "depth_offset_m": dz,
        "yaw_offset_deg": dyaw,
        # Resulting world-frame translation actually applied to the spawn position.
        "world_dx_m": world_dx,
        "world_dy_m": world_dy,
        "world_dz_m": dz,
        "beacon_angular_noise_std_deg": beacon_updates.get("angular_noise_std_deg", float(beacon.angular_noise_std_deg)),
        "beacon_range_noise_std_m": beacon_updates.get("range_noise_std_m", float(beacon.range_noise_std_m)),
        "beacon_dropout_probability": beacon_updates.get("dropout_probability", float(beacon.dropout_probability)),
    }
    return new_config, new_position, new_rotation, applied
