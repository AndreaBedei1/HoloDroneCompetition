"""Synthetic FrontCamera for the engine-free fallback adapter.

The kinematic fallback adapter has no rendering engine, but the onboard-only
controller contract confirms gate passage visually. This module renders a
minimal simulated camera image of the gate frames (bright bars on a dark water
background) from the true simulator state, exactly as a rendering engine
would. It is a debug/test substrate: official experiments use the real
HoloOcean camera.

The rendered pixels satisfy the same coarse color statistics the built-in gate
detector looks for (bright, saturated bars), so the full visual confirmation
pipeline can be exercised in unit and integration tests without HoloOcean.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

try:  # numpy is an optional dependency of the fallback camera
    import numpy as _np
except ImportError:  # pragma: no cover - exercised only without numpy
    _np = None

Vector3 = Tuple[float, float, float]

#: Bright, saturated gate-bar color (passes the detector's bar-pixel test).
GATE_COLOR = (40, 235, 150)
#: Dark underwater background (fails the detector's bar-pixel test).
WATER_COLOR = (10, 32, 58)


class SyntheticGateCamera:
    """Pinhole projection of gate frames onto a small RGB image."""

    def __init__(
        self,
        gates: Iterable[object],
        width: int = 160,
        height: int = 120,
        fov_deg: float = 90.0,
        max_render_distance_m: float = 16.0,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fov_deg = float(fov_deg)
        self.max_render_distance_m = float(max_render_distance_m)
        self.focal_px = (self.width / 2.0) / math.tan(math.radians(self.fov_deg) / 2.0)
        self._segments: List[Tuple[Vector3, Vector3, float]] = []
        for gate in gates:
            self._segments.extend(_gate_frame_segments(gate))

    @property
    def available(self) -> bool:
        return _np is not None

    def render(self, position: Vector3, rotation_rpy_deg: Vector3):
        """Render the camera image for a camera at ``position`` looking along yaw.

        Returns an ``HxWx3`` uint8 numpy array, or ``None`` when numpy is not
        installed.
        """
        if _np is None:  # pragma: no cover - exercised only without numpy
            return None
        image = _np.empty((self.height, self.width, 3), dtype=_np.uint8)
        image[:, :] = WATER_COLOR

        yaw_rad = math.radians(rotation_rpy_deg[2])
        cos_yaw, sin_yaw = math.cos(yaw_rad), math.sin(yaw_rad)

        for start, end, thickness_m in self._segments:
            self._draw_segment(image, position, cos_yaw, sin_yaw, start, end, thickness_m)
        return image

    def _draw_segment(
        self,
        image,
        position: Vector3,
        cos_yaw: float,
        sin_yaw: float,
        start: Vector3,
        end: Vector3,
        thickness_m: float,
    ) -> None:
        camera_start = _world_to_camera(start, position, cos_yaw, sin_yaw)
        camera_end = _world_to_camera(end, position, cos_yaw, sin_yaw)
        # Sample points along the 3D segment and splat a small square per point.
        length = _distance(camera_start, camera_end)
        samples = max(2, int(length * 14.0))
        for index in range(samples + 1):
            t = index / samples
            point = (
                camera_start[0] + t * (camera_end[0] - camera_start[0]),
                camera_start[1] + t * (camera_end[1] - camera_start[1]),
                camera_start[2] + t * (camera_end[2] - camera_start[2]),
            )
            forward = point[0]
            if forward < 0.25 or forward > self.max_render_distance_m:
                continue
            pixel_x = self.width / 2.0 - self.focal_px * (point[1] / forward)
            pixel_y = self.height / 2.0 - self.focal_px * (point[2] / forward)
            half = max(1, int(round(self.focal_px * thickness_m / forward / 2.0)))
            x0 = int(pixel_x) - half
            y0 = int(pixel_y) - half
            x1 = int(pixel_x) + half + 1
            y1 = int(pixel_y) + half + 1
            if x1 <= 0 or y1 <= 0 or x0 >= self.width or y0 >= self.height:
                continue
            image[max(0, y0):min(self.height, y1), max(0, x0):min(self.width, x1)] = GATE_COLOR


def _gate_frame_segments(gate) -> List[Tuple[Vector3, Vector3, float]]:
    """Four thick frame segments (top/bottom/left/right) around the aperture."""
    center = gate.center
    right = gate.right_axis
    up = gate.up_axis
    half_width = gate.inner_width_m / 2.0 + gate.bar_thickness_m / 2.0
    half_height = gate.inner_height_m / 2.0 + gate.bar_thickness_m / 2.0
    corner_a = _add(center, _add(_scale(right, -half_width), _scale(up, half_height)))
    corner_b = _add(center, _add(_scale(right, half_width), _scale(up, half_height)))
    corner_c = _add(center, _add(_scale(right, half_width), _scale(up, -half_height)))
    corner_d = _add(center, _add(_scale(right, -half_width), _scale(up, -half_height)))
    thickness = max(0.05, float(gate.bar_thickness_m))
    return [
        (corner_a, corner_b, thickness),
        (corner_c, corner_d, thickness),
        (corner_a, corner_d, thickness),
        (corner_b, corner_c, thickness),
    ]


def _world_to_camera(
    point: Vector3,
    position: Vector3,
    cos_yaw: float,
    sin_yaw: float,
) -> Vector3:
    dx = point[0] - position[0]
    dy = point[1] - position[1]
    dz = point[2] - position[2]
    # Camera frame: x forward (body surge axis), y left, z up.
    forward = cos_yaw * dx + sin_yaw * dy
    left = -sin_yaw * dx + cos_yaw * dy
    return (forward, left, dz)


def _add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(vector: Sequence[float], scalar: float) -> Vector3:
    return (vector[0] * scalar, vector[1] * scalar, vector[2] * scalar)


def _distance(a: Vector3, b: Vector3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
