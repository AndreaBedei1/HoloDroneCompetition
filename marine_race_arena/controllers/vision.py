"""Shared onboard gate detection from FrontCamera images.

A deliberately simple, deterministic color/blob detector: it samples the image
on a coarse grid, keeps bright/saturated pixels (gate bars), groups them into
connected components and scores rectangular, centered clusters. This is the
only visual front end used by the official controllers and the local course
tracker; it consumes nothing but the camera image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

try:  # OpenCV is optional; the pure-Python color fallback remains available.
    import cv2 as _cv2
    import numpy as _np
except ImportError:  # pragma: no cover - exercised only in minimal installs
    _cv2 = None
    _np = None


@dataclass(frozen=True)
class VisionTarget:
    center_x: float
    center_y: float
    confidence: float
    area_fraction: float = 0.0
    width_fraction: float = 0.0
    height_fraction: float = 0.0

    def with_confidence(self, confidence: float) -> "VisionTarget":
        return VisionTarget(
            center_x=self.center_x,
            center_y=self.center_y,
            confidence=confidence,
            area_fraction=self.area_fraction,
            width_fraction=self.width_fraction,
            height_fraction=self.height_fraction,
        )


def vision_target_from_camera(image: Any) -> VisionTarget | None:
    return select_default_visual_target(vision_targets_from_camera(image))


def vision_targets_from_camera(image: Any) -> list[VisionTarget]:
    shape = getattr(image, "shape", None)
    if shape is not None and len(shape) >= 2:
        height = int(shape[0])
        width = int(shape[1])
    elif isinstance(image, list) and image and isinstance(image[0], list):
        height = len(image)
        width = len(image[0])
    else:
        return []
    if width <= 0 or height <= 0:
        return []

    targets = _rectangular_frame_targets(image, width, height)

    step = max(1, int(max(width, height) / 80))
    sampled = 0
    selected_cells: dict[tuple[int, int], tuple[int, int]] = {}

    for y in range(0, height, step):
        grid_y = y // step
        for x in range(0, width, step):
            pixel = _pixel_channels(image, x, y)
            if pixel is None:
                continue
            sampled += 1
            if not _looks_like_gate_bar_pixel(pixel):
                continue
            selected_cells[(x // step, grid_y)] = (x, y)

    if sampled == 0 or len(selected_cells) < max(8, int(0.004 * sampled)):
        return []

    visited: set[tuple[int, int]] = set()
    for cell in selected_cells:
        if cell in visited:
            continue
        stack = [cell]
        visited.add(cell)
        component_points: list[tuple[int, int]] = []
        while stack:
            current = stack.pop()
            component_points.append(selected_cells[current])
            cx, cy = current
            for nx in range(cx - 1, cx + 2):
                for ny in range(cy - 1, cy + 2):
                    neighbor = (nx, ny)
                    if neighbor in visited or neighbor not in selected_cells:
                        continue
                    visited.add(neighbor)
                    stack.append(neighbor)
        if len(component_points) < max(6, int(0.0015 * sampled)):
            continue
        target = _vision_target_from_points(component_points, sampled, width, height)
        if target is not None:
            targets.append(target)

    deduped: list[VisionTarget] = []
    for target in sorted(targets, key=lambda item: item.confidence, reverse=True):
        if any(
            abs(target.center_x - existing.center_x) < 0.04
            and abs(target.center_y - existing.center_y) < 0.04
            for existing in deduped
        ):
            continue
        deduped.append(target)
    return deduped[:8]


def _rectangular_frame_targets(image: Any, width: int, height: int) -> list[VisionTarget]:
    """Detect four-bar frames without assuming a rendered RGB gate color.

    HoloOcean's public ``white`` prop material is strongly tinted by the water
    and can be darker than the water column and seabed. A generic
    bright/saturated mask therefore selects the entire scene. Canny contours
    expose the near-rectangular frame itself; the bounded perimeter ratio
    rejects filled background regions and irregular surface reflections.
    """

    if _cv2 is None or _np is None:
        return []
    try:
        frame = _np.asarray(image)
        if frame.ndim < 3 or frame.shape[2] < 3:
            return []
        frame = frame[:, :, :3]
        if frame.dtype != _np.uint8:
            finite = _np.nan_to_num(frame, nan=0.0, posinf=255.0, neginf=0.0)
            if float(_np.max(finite)) <= 1.0:
                finite = finite * 255.0
            frame = _np.clip(finite, 0.0, 255.0).astype(_np.uint8)
        gray = _cv2.cvtColor(frame, _cv2.COLOR_RGB2GRAY)
        edges = _cv2.Canny(gray, 30, 80)
        contours_result = _cv2.findContours(edges, _cv2.RETR_LIST, _cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_result[-2]
    except Exception:
        return []

    min_width = max(12, int(round(width * 0.05)))
    min_height = max(12, int(round(height * 0.05)))
    frame_area = max(1.0, float(width * height))
    targets: list[VisionTarget] = []
    for contour in contours:
        x, y, box_width_px, box_height_px = _cv2.boundingRect(contour)
        if box_width_px < min_width or box_height_px < min_height:
            continue
        width_fraction = box_width_px / max(1.0, float(width))
        height_fraction = box_height_px / max(1.0, float(height))
        if width_fraction >= 0.92 or height_fraction >= 0.92:
            continue
        aspect_ratio = box_width_px / max(1.0, float(box_height_px))
        if not 0.45 <= aspect_ratio <= 2.40:
            continue

        rectangle_perimeter = 2.0 * (box_width_px + box_height_px)
        contour_perimeter = float(_cv2.arcLength(contour, True))
        perimeter_ratio = contour_perimeter / max(1.0, rectangle_perimeter)
        if not 0.50 <= perimeter_ratio <= 2.20:
            continue

        area_fraction = (box_width_px * box_height_px) / frame_area
        if area_fraction < 0.008:
            continue
        center_x = _clamp(
            ((x + 0.5 * box_width_px) - width * 0.5) / max(1.0, width * 0.5),
            -1.0,
            1.0,
        )
        center_y = _clamp(
            ((y + 0.5 * box_height_px) - height * 0.5) / max(1.0, height * 0.5),
            -1.0,
            1.0,
        )
        perimeter_score = _clamp(1.0 - abs(perimeter_ratio - 1.0) / 1.20, 0.0, 1.0)
        aspect_score = _clamp(
            1.0 - abs(math.log(max(1e-6, aspect_ratio))) / math.log(3.0),
            0.0,
            1.0,
        )
        size_score = _clamp(area_fraction / 0.10, 0.0, 1.0)
        center_score = _clamp(
            1.0 - 0.45 * abs(center_x) - 0.30 * abs(center_y),
            0.0,
            1.0,
        )
        confidence = _clamp(
            0.38 * perimeter_score
            + 0.24 * aspect_score
            + 0.20 * size_score
            + 0.18 * center_score,
            0.0,
            1.0,
        )
        if confidence < 0.38:
            continue
        targets.append(
            VisionTarget(
                center_x=center_x,
                center_y=center_y,
                confidence=confidence,
                area_fraction=area_fraction,
                width_fraction=width_fraction,
                height_fraction=height_fraction,
            )
        )
    return targets


def select_visual_target_for_beacon(
    targets: list[VisionTarget],
    bearing_deg: float | None,
    range_m: float | None,
) -> VisionTarget | None:
    """Pick the detection most consistent with the expected beacon's bearing.

    ``bearing_deg``/``range_m`` come from the controller's own filtered packet
    for its locally expected beacon (or ``None`` when it has none).
    """
    if not targets:
        return None
    if bearing_deg is None:
        return select_default_visual_target(targets)

    bearing_abs = abs(bearing_deg)
    range_value = 999.0 if range_m is None else max(0.0, range_m)

    scored: list[tuple[float, VisionTarget]] = []
    for target in targets:
        if target.confidence < 0.38:
            continue
        centered_score = _clamp(1.0 - 0.75 * abs(target.center_x) - 0.25 * abs(target.center_y), 0.0, 1.0)
        size_score = _clamp(target.area_fraction / 0.12, 0.0, 1.0)
        # Consistency between where the detection sits in the image and where
        # the expected beacon says the gate should be. Penalizing mismatch
        # keeps a blob merged across several visible gates from outscoring the
        # actual expected gate.
        bearing_mismatch_deg = _bearing_mismatch_deg(target, bearing_deg)
        # In a multi-gate view, the most centered/high-confidence rectangle can
        # be the *next* gate. Keep the expected beacon's noisy bearing as the
        # dominant association cue at every range instead of bypassing it in
        # the final 3.2 m.
        bearing_penalty = 0.050 * bearing_mismatch_deg
        if bearing_abs <= 25.0:
            scored.append(
                (target.confidence + 0.35 * centered_score + 0.20 * size_score - bearing_penalty, target)
            )
            continue

        # Positive acoustic bearing means the expected gate is to camera-left.
        expected_side_amount = -math.copysign(1.0, bearing_deg) * target.center_x
        min_offset = 0.10 if bearing_abs < 45.0 else 0.18
        if expected_side_amount < min_offset:
            continue
        if bearing_abs > 70.0 and range_value > 3.0 and target.confidence < 0.86:
            continue
        side_score = _clamp(expected_side_amount / 0.55, 0.0, 1.0)
        scored.append((target.confidence + 0.65 * side_score + 0.15 * size_score - bearing_penalty, target))
    if not scored:
        return None
    return max(scored, key=lambda item: item[0])[1]


def _bearing_mismatch_deg(target: VisionTarget, bearing_deg: float) -> float:
    """Mismatch between a detection's implied camera bearing and the beacon bearing.

    For the 90-degree FOV FrontCamera, a normalized image position ``center_x``
    corresponds to a body bearing of ``-atan(center_x)`` (positive bearing is
    camera-left).
    """
    implied_bearing_deg = -math.degrees(math.atan(target.center_x))
    return abs(implied_bearing_deg - bearing_deg)


def select_default_visual_target(targets: list[VisionTarget]) -> VisionTarget | None:
    if not targets:
        return None
    return max(
        targets,
        key=lambda target: (
            target.confidence
            + 0.30 * _clamp(1.0 - abs(target.center_x), 0.0, 1.0)
            + 0.15 * _clamp(1.0 - abs(target.center_y), 0.0, 1.0)
        ),
    )


def vision_conflicts_with_beacon(target: VisionTarget, bearing_deg: float) -> bool:
    if abs(bearing_deg) <= 25.0 or abs(target.center_x) <= 0.10:
        return False
    return bearing_deg * target.center_x > 0.0


def _vision_target_from_points(
    points: list[tuple[int, int]],
    sampled: int,
    width: int,
    height: int,
) -> VisionTarget | None:
    if not points:
        return None
    selected = len(points)
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    box_width = max_x - min_x + 1
    box_height = max_y - min_y + 1
    if box_width < width * 0.025 or box_height < height * 0.025:
        return None

    center_x_px = sum(point[0] for point in points) / selected
    center_y_px = sum(point[1] for point in points) / selected
    center_x = _clamp((center_x_px - (width - 1) * 0.5) / max(1.0, (width - 1) * 0.5), -1.0, 1.0)
    center_y = _clamp((center_y_px - (height - 1) * 0.5) / max(1.0, (height - 1) * 0.5), -1.0, 1.0)
    coverage = selected / sampled
    box_area = (box_width * box_height) / max(1.0, width * height)
    width_fraction = box_width / max(1.0, width)
    height_fraction = box_height / max(1.0, height)
    # Reject components spanning almost the full view: they represent
    # illuminated water, seabed or surface texture rather than four gate bars.
    if (
        width_fraction >= 0.85
        or height_fraction >= 0.85
        or box_area >= 0.45
        or box_area < 0.008
    ):
        return None
    aspect_ratio = width_fraction / max(1e-6, height_fraction)
    aspect_score = _clamp(1.0 - abs(math.log(max(1e-6, aspect_ratio))) / math.log(3.0), 0.0, 1.0)
    center_score = _clamp(1.0 - 0.45 * abs(center_x) - 0.30 * abs(center_y), 0.0, 1.0)
    confidence = _clamp(
        0.30 * min(1.0, coverage / 0.04)
        + 0.30 * min(1.0, box_area / 0.20)
        + 0.20 * aspect_score
        + 0.20 * center_score,
        0.0,
        1.0,
    )
    return VisionTarget(
        center_x=center_x,
        center_y=center_y,
        confidence=confidence,
        area_fraction=box_area,
        width_fraction=width_fraction,
        height_fraction=height_fraction,
    )


def _pixel_channels(image: Any, x: int, y: int) -> tuple[float, float, float] | None:
    try:
        pixel = image[y][x]
    except (TypeError, IndexError, KeyError):
        return None
    if hasattr(pixel, "tolist"):
        pixel = pixel.tolist()
    if isinstance(pixel, (int, float)):
        value = float(pixel)
        return (value, value, value)
    if not isinstance(pixel, (list, tuple)) or len(pixel) < 3:
        return None
    try:
        return (float(pixel[0]), float(pixel[1]), float(pixel[2]))
    except (TypeError, ValueError):
        return None


def _looks_like_gate_bar_pixel(pixel: tuple[float, float, float]) -> bool:
    high = max(pixel)
    low = min(pixel)
    mean = sum(pixel) / 3.0
    saturation = high - low
    return (high >= 115.0 and saturation >= 35.0) or mean >= 190.0


def collision_active_from_sensors(sensors: Mapping[str, Any]) -> bool:
    for key, value in sensors.items():
        if "collision" not in str(key).lower() and "contact" not in str(key).lower():
            continue
        if hasattr(value, "any"):
            try:
                return bool(value.any())
            except Exception:
                return bool(value)
        if isinstance(value, Mapping):
            return any(bool(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(bool(item) for item in value)
        return bool(value)
    return False


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
