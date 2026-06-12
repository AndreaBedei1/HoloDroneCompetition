"""Abstract gate geometry used by the referee."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from marine_race_arena.config.schema import GateConfig, Vector3


@dataclass(frozen=True)
class GateCrossingResult:
    valid: bool
    reason: str
    intersection: Optional[Vector3] = None
    signed_distance_previous: Optional[float] = None
    signed_distance_current: Optional[float] = None


@dataclass
class Gate:
    id: str
    type: str
    center: Vector3
    rotation_rpy_deg: Vector3
    inner_width_m: float
    inner_height_m: float
    bar_thickness_m: float
    color: Any
    passage_direction: Vector3
    linked_gate: Optional[str] = None
    beacon_id: Optional[str] = None

    @classmethod
    def from_config(cls, config: GateConfig) -> "Gate":
        return cls(
            id=config.id,
            type=config.type,
            center=config.position,
            rotation_rpy_deg=config.rotation_rpy_deg,
            inner_width_m=config.inner_size_m[0],
            inner_height_m=config.inner_size_m[1],
            bar_thickness_m=config.bar_thickness_m,
            color=config.color,
            passage_direction=config.passage_direction,
            linked_gate=config.linked_gate,
            beacon_id=config.beacon.id if config.beacon else None,
        )

    @property
    def normal_vector(self) -> Vector3:
        return _normalize(self.passage_direction)

    @property
    def right_axis(self) -> Vector3:
        return canonical_gate_frame(self.passage_direction)[1]

    @property
    def up_axis(self) -> Vector3:
        return canonical_gate_frame(self.passage_direction)[2]

    @property
    def visual_rotation_rpy_deg(self) -> Vector3:
        return rotation_from_axes(*visual_gate_frame(self.passage_direction, self.rotation_rpy_deg))

    @property
    def visual_right_axis(self) -> Vector3:
        return visual_gate_frame(self.passage_direction, self.rotation_rpy_deg)[1]

    @property
    def visual_up_axis(self) -> Vector3:
        return visual_gate_frame(self.passage_direction, self.rotation_rpy_deg)[2]

    def signed_distance_to_plane(self, point: Vector3) -> float:
        return _dot(_subtract(point, self.center), self.normal_vector)

    def project_point_to_gate_plane(self, point: Vector3) -> Vector3:
        distance = self.signed_distance_to_plane(point)
        return _subtract(point, _scale(self.normal_vector, distance))

    def local_aperture_coordinates(self, point: Vector3) -> tuple[float, float]:
        projected = self.project_point_to_gate_plane(point)
        relative = _subtract(projected, self.center)
        return (_dot(relative, self.right_axis), _dot(relative, self.up_axis))

    def is_point_inside_aperture(self, point: Vector3, margin_m: float = 0.0) -> bool:
        right, up = self.local_aperture_coordinates(point)
        clearance = max(0.0, margin_m)
        half_width = max(0.0, self.inner_width_m / 2.0 - clearance)
        half_height = max(0.0, self.inner_height_m / 2.0 - clearance)
        return abs(right) <= half_width and abs(up) <= half_height

    def crossed_between(self, previous_position: Vector3, current_position: Vector3) -> bool:
        d0 = self.signed_distance_to_plane(previous_position)
        d1 = self.signed_distance_to_plane(current_position)
        if abs(d0 - d1) <= 1e-9:
            return False
        return (d0 <= 0.0 <= d1) or (d1 <= 0.0 <= d0)

    def crossed_in_correct_direction(
        self, previous_position: Vector3, current_position: Vector3
    ) -> bool:
        d0 = self.signed_distance_to_plane(previous_position)
        d1 = self.signed_distance_to_plane(current_position)
        movement = _subtract(current_position, previous_position)
        return d1 > d0 and _dot(movement, self.normal_vector) > 0.0

    def intersection_point(
        self, previous_position: Vector3, current_position: Vector3
    ) -> Optional[Vector3]:
        d0 = self.signed_distance_to_plane(previous_position)
        d1 = self.signed_distance_to_plane(current_position)
        denominator = d0 - d1
        if abs(denominator) <= 1e-12:
            return None
        t = d0 / denominator
        if t < -1e-9 or t > 1.0 + 1e-9:
            return None
        clamped_t = min(1.0, max(0.0, t))
        segment = _subtract(current_position, previous_position)
        return _add(previous_position, _scale(segment, clamped_t))

    def validate_crossing(
        self,
        previous_position: Vector3,
        current_position: Vector3,
        clearance_margin_m: float = 0.0,
    ) -> GateCrossingResult:
        d0 = self.signed_distance_to_plane(previous_position)
        d1 = self.signed_distance_to_plane(current_position)
        if not self.crossed_between(previous_position, current_position):
            return GateCrossingResult(False, "no_plane_crossing", None, d0, d1)
        if not self.crossed_in_correct_direction(previous_position, current_position):
            return GateCrossingResult(False, "wrong_direction", None, d0, d1)
        intersection = self.intersection_point(previous_position, current_position)
        if intersection is None:
            return GateCrossingResult(False, "no_segment_intersection", None, d0, d1)
        if not self.is_point_inside_aperture(intersection, margin_m=clearance_margin_m):
            return GateCrossingResult(False, "outside_aperture", intersection, d0, d1)
        return GateCrossingResult(True, "valid", intersection, d0, d1)


def canonical_gate_frame(passage_direction: Vector3) -> tuple[Vector3, Vector3, Vector3]:
    """Return the canonical gate frame: normal, right, up.

    The passage direction is the source of truth. For normal horizontal gates,
    up is world vertical. For sloped or vertical gates, up is the closest
    orthonormal axis that remains perpendicular to the gate normal.
    """

    normal = _normalize(passage_direction)
    world_up = (0.0, 0.0, 1.0)
    right = _cross(world_up, normal)
    if _norm(right) <= 1e-9:
        right = (1.0, 0.0, 0.0)
    else:
        right = _normalize(right)
    up = _normalize(_cross(normal, right))
    return (normal, right, up)


def visual_gate_frame(
    passage_direction: Vector3,
    fallback_rotation_rpy_deg: Vector3 = (0.0, 0.0, 0.0),
) -> tuple[Vector3, Vector3, Vector3]:
    """Return a HoloOcean visual frame for the full 3D gate orientation.

    The visual frame and the referee frame now use the same source of truth:
    ``passage_direction``. Local X follows the gate normal/depth, local Y follows
    the right/opening width axis, and local Z follows the gate up/opening height
    axis. This supports yaw-only gates and pitched gates without changing the
    four-bar assembly rules.
    """

    del fallback_rotation_rpy_deg
    return canonical_gate_frame(passage_direction)


def rotation_from_axes(normal: Vector3, right: Vector3, up: Vector3) -> Vector3:
    """Convert local X/Y/Z axes to HoloOcean roll/pitch/yaw degrees."""

    r00, r01, r02 = normal[0], right[0], up[0]
    r10, r11, r12 = normal[1], right[1], up[1]
    r20, r21, r22 = normal[2], right[2], up[2]
    pitch = math.asin(max(-1.0, min(1.0, -r20)))
    if abs(math.cos(pitch)) > 1e-9:
        roll = math.atan2(r21, r22)
        yaw = math.atan2(r10, r00)
    else:
        roll = math.atan2(-r12, r11)
        yaw = 0.0
    return (
        _wrap_degrees(math.degrees(roll)),
        _wrap_degrees(math.degrees(pitch)),
        _wrap_degrees(math.degrees(yaw)),
    )


def _rotation_axes(rotation_rpy_deg: Vector3) -> tuple[Vector3, Vector3, Vector3]:
    roll, pitch, yaw = [math.radians(value) for value in rotation_rpy_deg]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    matrix = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    local_x = (matrix[0][0], matrix[1][0], matrix[2][0])
    local_y = (matrix[0][1], matrix[1][1], matrix[2][1])
    local_z = (matrix[0][2], matrix[1][2], matrix[2][2])
    return (local_x, local_y, local_z)


def _add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _subtract(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(vector: Vector3, scalar: float) -> Vector3:
    return (vector[0] * scalar, vector[1] * scalar, vector[2] * scalar)


def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(vector: Vector3) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalize(vector: Vector3) -> Vector3:
    length = _norm(vector)
    if length <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _wrap_degrees(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0
