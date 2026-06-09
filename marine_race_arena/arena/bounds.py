"""Arena bounds and safety checks."""

from __future__ import annotations

from dataclasses import dataclass

from marine_race_arena.config.schema import BoundsConfig, Vector3


@dataclass(frozen=True)
class ArenaBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @classmethod
    def from_config(cls, config: BoundsConfig) -> "ArenaBounds":
        return cls(
            x_min=config.x_min,
            x_max=config.x_max,
            y_min=config.y_min,
            y_max=config.y_max,
            z_min=config.z_min,
            z_max=config.z_max,
        )

    def contains(self, position: Vector3) -> bool:
        x, y, z = position
        return (
            self.x_min <= x <= self.x_max
            and self.y_min <= y <= self.y_max
            and self.z_min <= z <= self.z_max
        )

    def violation_reason(self, position: Vector3) -> str | None:
        x, y, z = position
        if z < self.z_min:
            return "below_z_min"
        if z > self.z_max:
            return "above_z_max"
        if x < self.x_min or x > self.x_max:
            return "x_out_of_bounds"
        if y < self.y_min or y > self.y_max:
            return "y_out_of_bounds"
        return None

