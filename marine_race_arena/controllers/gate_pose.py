"""Onboard visual gate-pose estimation from FrontCamera images.

Extends the deterministic gate detector (:mod:`vision`) to estimate the four inner
aperture corners and, when possible, the relative pose of the gate plane (translation +
orientation) via PnP, with a robust projective-orientation fallback and a center-only
degradation path. It consumes ONLY the camera image and the known public gate aperture
(1.5 x 1.5 m); it never reads the simulator gate pose or world geometry.

Camera convention (OpenCV): +x right, +y down, +z forward (into the scene). The reported
translation is (lateral_x right+, vertical_y down+, forward_z away+).

Gate-plane yaw uses a single documented *canonical* convention (see
:func:`canonical_gate_plane_yaw`). A gate is a planar square, so its orientation is defined
only *modulo 180 deg* (the normal ``n`` and ``-n`` describe the same plane). The canonical
yaw first orients the plane normal toward the camera and then measures its horizontal
obliqueness from the optical axis, wrapped to ``[-90, +90]`` deg:

* ``0`` == frontal (gate plane perpendicular to the line of sight);
* ``> 0`` == the gate's left edge is nearer the camera (image left bar taller) ==
  ``rotated_left``;
* ``< 0`` == the gate's right edge is nearer (image right bar taller) == ``rotated_right``.

This is distinct from the gate's *allowed passage direction* (an oriented ``+/-`` normal used
by the referee), which is never inferred here. Angular error between two plane yaws must use
:func:`plane_angle_distance_deg` (modulo-180), not a raw subtraction, and the coarse
left/right/frontal call uses :func:`orientation_class`. All angles are in degrees.

Ground truth may be used only offline to score this detector, to make synthetic labels, or
in evaluation-only tests -- never as a runtime input or fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

try:  # OpenCV/numpy are optional; without them only center-only detection is available.
    import cv2 as _cv2
    import numpy as _np
except ImportError:  # pragma: no cover
    _cv2 = None
    _np = None

# Public competition constant: the square inner aperture side length (metres).
GATE_INNER_SIZE_M = 1.5


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics derived from the configured resolution and horizontal FOV."""

    width: int = 640
    height: int = 480
    fov_deg: float = 90.0  # horizontal field of view (HoloOcean FovAngle)

    @property
    def fx(self) -> float:
        return self.width / (2.0 * math.tan(math.radians(self.fov_deg) / 2.0))

    @property
    def fy(self) -> float:
        return self.fx  # square pixels (single HoloOcean focal length)

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    def matrix(self):
        return _np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=_np.float64)

    def as_dict(self) -> dict:
        return {"width": self.width, "height": self.height, "fov_deg": self.fov_deg,
                "fx": round(self.fx, 3), "fy": round(self.fy, 3), "cx": self.cx, "cy": self.cy}


DEFAULT_INTRINSICS = CameraIntrinsics(640, 480, 90.0)


# --------------------------------------------------------------- angle conventions
def canonical_gate_plane_yaw(rotation_matrix) -> float:
    """Canonical horizontal obliqueness of the gate plane from frontal, in ``[-90, 90]`` deg.

    ``rotation_matrix`` is the gate->camera rotation ``R`` (so the gate normal, object
    ``+z``, is ``R[:, 2]`` in camera coordinates). A gate is a plane, whose orientation is
    defined only modulo 180 deg, so:

    1. the normal is oriented consistently *toward the camera* (the camera looks along ``+z``,
       so a normal facing it has ``n_z <= 0``); then
    2. the yaw is the angle of the normal's horizontal projection from the optical axis.

    Returns ``0`` for a frontal gate, ``> 0`` when the gate's left edge is nearer (image left
    bar taller, ``rotated_left``) and ``< 0`` when the right edge is nearer (``rotated_right``).
    The result lies in ``[-90, 90]`` and is invariant to reversing the normal (``n`` vs ``-n``).
    """
    n = _np.asarray(rotation_matrix, dtype=_np.float64)[:, 2]
    if n[2] > 0.0:  # orient toward the camera (n_z <= 0)
        n = -n
    return math.degrees(math.atan2(float(n[0]), float(-n[2])))


def canonical_gate_plane_pitch(rotation_matrix) -> float:
    """Canonical vertical obliqueness of the gate plane, in ``[-90, 90]`` deg (0 = frontal).

    Same construction as :func:`canonical_gate_plane_yaw` but for the normal's vertical
    component (camera ``+y`` is down): ``> 0`` when the gate's top edge is farther.
    """
    n = _np.asarray(rotation_matrix, dtype=_np.float64)[:, 2]
    if n[2] > 0.0:
        n = -n
    return math.degrees(math.atan2(float(n[1]), float(-n[2])))


def wrap_plane_yaw_deg(yaw_deg: float) -> float:
    """Fold an oriented yaw (any range) into the unoriented-plane interval ``[-90, 90]``."""
    y = ((float(yaw_deg) + 90.0) % 180.0) - 90.0
    return 90.0 if y == -90.0 else y  # keep +90 and -90 collapsed to a single endpoint


def plane_angle_distance_deg(a: float, b: float) -> float:
    """Angular distance between two UNORIENTED plane angles, modulo 180 deg (``0..90``).

    A plane at ``+85`` deg and one at ``-85`` deg differ by only 10 deg, not 170. Use this
    for gate-plane yaw error instead of ``abs(a - b)``.
    """
    d = abs((float(a) - float(b)) % 180.0)
    return min(d, 180.0 - d)


def orientation_class(yaw_deg: Optional[float], *, frontal_threshold_deg: float = 8.0) -> str:
    """Coarse gate-plane orientation label from a canonical yaw: frontal/left/right/unknown."""
    if yaw_deg is None:
        return "unknown"
    if abs(float(yaw_deg)) < frontal_threshold_deg:
        return "frontal"
    return "rotated_left" if float(yaw_deg) > 0.0 else "rotated_right"


@dataclass
class GatePoseTarget:
    """Visual gate-pose estimate. Fields beyond the center are optional / masked."""

    center_x: float
    center_y: float
    area_fraction: float = 0.0
    width_fraction: float = 0.0
    height_fraction: float = 0.0
    confidence: float = 0.0
    # Ordered inner aperture corners in normalized image coords [-1, 1]: tl, tr, br, bl.
    corners_normalized: Optional[List[Tuple[float, float]]] = None
    pose_present: bool = False
    translation_camera_m: Optional[Tuple[float, float, float]] = None  # (lateral_x, vertical_y, forward_z)
    gate_plane_yaw_deg: Optional[float] = None
    gate_plane_pitch_deg: Optional[float] = None
    gate_plane_roll_deg: Optional[float] = None
    reprojection_error_px: Optional[float] = None
    pose_confidence: float = 0.0
    detection_source: str = "center_only"  # center_only | corners | pnp | projective
    quadrilateral_skew: float = 0.0
    orientation_hint: str = "unknown"  # frontal | rotated_left | rotated_right | unknown
    orientation_present: bool = False
    predicted: bool = False
    track_age_frames: int = 0


# --------------------------------------------------------------------------- corners
def order_corners(pts) -> "list":
    """Order 4 image points as top-left, top-right, bottom-right, bottom-left."""
    pts = _np.asarray(pts, dtype=_np.float64).reshape(-1, 2)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    tl = pts[_np.argmin(s)]
    br = pts[_np.argmax(s)]
    tr = pts[_np.argmax(d)]
    bl = pts[_np.argmin(d)]
    return [tuple(tl), tuple(tr), tuple(br), tuple(bl)]


def _polygon_area(pts) -> float:
    p = _np.asarray(pts, dtype=_np.float64)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(_np.dot(x, _np.roll(y, -1)) - _np.dot(y, _np.roll(x, -1))))


def _is_convex(pts) -> bool:
    p = _np.asarray(pts, dtype=_np.float64)
    n = len(p)
    signs = []
    for i in range(n):
        a, b, c = p[i], p[(i + 1) % n], p[(i + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        signs.append(cross > 0)
    return all(signs) or not any(signs)


def validate_quad(corners, *, image_area: float, min_area_frac: float = 0.004,
                  max_aspect: float = 4.0, min_side_ratio: float = 0.25) -> bool:
    """Reject implausible quadrilaterals (non-convex, tiny, degenerate, wrong aspect)."""
    c = [tuple(pt) for pt in corners]
    if len(c) != 4:
        return False
    if not _is_convex(c):
        return False
    area = _polygon_area(c)
    if area < max(1.0, min_area_frac * image_area):
        return False
    tl, tr, br, bl = c
    top = math.hypot(tr[0] - tl[0], tr[1] - tl[1])
    bottom = math.hypot(br[0] - bl[0], br[1] - bl[1])
    left = math.hypot(bl[0] - tl[0], bl[1] - tl[1])
    right = math.hypot(br[0] - tr[0], br[1] - tr[1])
    sides = [top, bottom, left, right]
    if min(sides) <= 1e-3:
        return False
    # Opposite sides must not differ wildly (perspective is allowed but bounded).
    if min(top, bottom) / max(top, bottom) < min_side_ratio:
        return False
    if min(left, right) / max(left, right) < min_side_ratio:
        return False
    aspect = max(top, bottom) / max(1e-3, max(left, right))
    if aspect > max_aspect or aspect < 1.0 / max_aspect:
        return False
    return True


def _bar_mask(image):
    """Binary mask of gate-bar pixels.

    Matches the proven per-pixel classifier used by the frozen v1 detector
    (:func:`vision._looks_like_gate_bar_pixel`), which is validated on real HoloOcean gates:
    a bright, *colorful* pixel (high >= 115 and high-low >= 35 -- e.g. the #00ff88 gate) OR a
    very bright pixel (mean >= 190). The earlier whitish/low-saturation heuristic wrongly
    rejected the saturated green gate on real images.
    """
    arr = _np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[:, :, :3].astype(_np.float32)
    else:
        rgb = _np.stack([arr] * 3, axis=-1).astype(_np.float32)
    high = rgb.max(axis=2)
    low = rgb.min(axis=2)
    mean = rgb.mean(axis=2)
    sat = high - low
    mask = (((high >= 115.0) & (sat >= 35.0)) | (mean >= 190.0)).astype(_np.uint8) * 255
    return mask


def _mask_aperture_candidates(image):
    """Return aperture candidates from the original colour/bar mask."""
    h, w = image.shape[0], image.shape[1]
    mask = _bar_mask(image)
    mask = _cv2.morphologyEx(mask, _cv2.MORPH_CLOSE, _np.ones((5, 5), _np.uint8))
    contours, hierarchy = _cv2.findContours(mask, _cv2.RETR_CCOMP, _cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    candidates = []
    hier = hierarchy[0] if hierarchy is not None else [(-1, -1, -1, -1)] * len(contours)
    for idx, cnt in enumerate(contours):
        area = _cv2.contourArea(cnt)
        if area < 0.003 * w * h:
            continue
        peri = _cv2.arcLength(cnt, True)
        approx = _cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) != 4:
            continue
        corners = order_corners(approx.reshape(-1, 2))
        if not validate_quad(corners, image_area=w * h):
            continue
        source = "corners" if hier[idx][3] != -1 else "corners_outer"
        candidates.append((corners, source, float(area)))
    return candidates


def _line_angle_deg(line) -> float:
    x1, y1, x2, y2, _length = line
    angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
    return min(angle, 180.0 - angle)


def _line_value_at(line, *, x: Optional[float] = None, y: Optional[float] = None) -> Optional[float]:
    """Evaluate x(y) or y(x) on a finite Hough segment's infinite line."""
    x1, y1, x2, y2, _length = line
    if y is not None:
        dy = y2 - y1
        if abs(dy) < 1e-6:
            return None
        return float(x1 + (y - y1) * (x2 - x1) / dy)
    if x is not None:
        dx = x2 - x1
        if abs(dx) < 1e-6:
            return None
        return float(y1 + (x - x1) * (y2 - y1) / dx)
    return None


def _line_intersection(a, b):
    x1, y1, x2, y2, _ = a
    x3, y3, x4, y4, _ = b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-6:
        return None
    det1 = x1 * y2 - y1 * x2
    det2 = x3 * y4 - y3 * x4
    return (
        (det1 * (x3 - x4) - (x1 - x2) * det2) / den,
        (det1 * (y3 - y4) - (y1 - y2) * det2) / den,
    )


def _dedupe_hough_lines(lines, *, vertical: bool, reference: float):
    """Collapse the many nearly-identical Hough segments along one gate edge."""
    ranked = sorted(lines, key=lambda line: line[4], reverse=True)
    kept = []
    for line in ranked:
        value = _line_value_at(line, y=reference) if vertical else _line_value_at(line, x=reference)
        if value is None:
            continue
        duplicate = False
        for existing in kept:
            other = (_line_value_at(existing, y=reference) if vertical
                     else _line_value_at(existing, x=reference))
            if other is not None and abs(value - other) < 5.0 \
                    and abs(_line_angle_deg(line) - _line_angle_deg(existing)) < 5.0:
                duplicate = True
                break
        if not duplicate:
            kept.append(line)
    return kept[:12]


def _hough_aperture_candidates(image, visual_target=None):
    """Find four inner bar edges inside today's tracked-gate ROI.

    HoloOcean's underwater lighting often makes the white gate darker than the
    water behind it, so colour thresholding alone is unreliable.  The inner
    aperture still has two long near-vertical and two long near-horizontal
    edges.  This deliberately small fallback searches only around the gate
    selected by :class:`TemporalVisionTracker`; scene-wide Hough lines are not
    accepted as controller targets.
    """
    h, w = image.shape[:2]
    if visual_target is None:
        return []
    cx = (float(visual_target.center_x) + 1.0) * 0.5 * w
    cy = (float(visual_target.center_y) + 1.0) * 0.5 * h
    bw = max(24.0, float(visual_target.width_fraction) * w)
    bh = max(24.0, float(visual_target.height_fraction) * h)
    half_w = max(0.18 * w, 0.80 * bw)
    half_h = max(0.18 * h, 0.80 * bh)
    x0, x1 = max(0, int(cx - half_w)), min(w, int(cx + half_w) + 1)
    y0, y1 = max(0, int(cy - half_h)), min(h, int(cy + half_h) + 1)
    if x1 - x0 < 40 or y1 - y0 < 40:
        return []

    frame = _np.asarray(image)
    if frame.ndim != 3 or frame.shape[2] < 3:
        return []
    frame = frame[:, :, :3]
    if frame.dtype != _np.uint8:
        finite = _np.nan_to_num(frame, nan=0.0, posinf=255.0, neginf=0.0)
        if float(_np.max(finite)) <= 1.0:
            finite *= 255.0
        frame = _np.clip(finite, 0.0, 255.0).astype(_np.uint8)
    gray = _cv2.cvtColor(frame[y0:y1, x0:x1], _cv2.COLOR_BGR2GRAY)
    gray = _cv2.GaussianBlur(gray, (3, 3), 0.0)
    edges = _cv2.Canny(gray, 20, 60)
    roi_w, roi_h = x1 - x0, y1 - y0
    lines = _cv2.HoughLinesP(
        edges, 1.0, math.pi / 180.0,
        threshold=max(24, int(0.08 * max(roi_w, roi_h))),
        minLineLength=max(28, int(0.14 * min(roi_w, roi_h))),
        maxLineGap=max(10, int(0.06 * max(roi_w, roi_h))),
    )
    if lines is None:
        return []
    vertical, horizontal = [], []
    for raw in lines:
        lx1, ly1, lx2, ly2 = [int(v) for v in raw.reshape(-1)]
        line = (lx1 + x0, ly1 + y0, lx2 + x0, ly2 + y0,
                math.hypot(lx2 - lx1, ly2 - ly1))
        angle = _line_angle_deg(line)
        if angle >= 65.0:
            vertical.append(line)
        elif angle <= 25.0:
            horizontal.append(line)
    vertical = _dedupe_hough_lines(vertical, vertical=True, reference=cy)
    horizontal = _dedupe_hough_lines(horizontal, vertical=False, reference=cx)
    if len(vertical) < 2 or len(horizontal) < 2:
        return []

    vertical_positions = [
        value for value in (_line_value_at(line, y=cy) for line in vertical)
        if value is not None
    ]
    horizontal_positions = [
        value for value in (_line_value_at(line, x=cx) for line in horizontal)
        if value is not None
    ]

    candidates = []
    min_w = max(35.0, 0.20 * bw)
    min_h = max(35.0, 0.20 * bh)
    for left in vertical:
        left_x = _line_value_at(left, y=cy)
        if left_x is None:
            continue
        for right in vertical:
            right_x = _line_value_at(right, y=cy)
            if right_x is None or right_x - left_x < min_w:
                continue
            if abs(_line_angle_deg(left) - _line_angle_deg(right)) > 15.0:
                continue
            for top in horizontal:
                top_y = _line_value_at(top, x=cx)
                if top_y is None:
                    continue
                for bottom in horizontal:
                    bottom_y = _line_value_at(bottom, x=cx)
                    if bottom_y is None or bottom_y - top_y < min_h:
                        continue
                    if abs(_line_angle_deg(top) - _line_angle_deg(bottom)) > 18.0:
                        continue
                    corners = [
                        _line_intersection(left, top),
                        _line_intersection(right, top),
                        _line_intersection(right, bottom),
                        _line_intersection(left, bottom),
                    ]
                    if any(point is None for point in corners):
                        continue
                    if any(not (-8 <= px < w + 8 and -8 <= py < h + 8) for px, py in corners):
                        continue
                    corners = order_corners(corners)
                    if not validate_quad(corners, image_area=w * h, min_area_frac=0.003):
                        continue
                    side_lengths = [
                        math.hypot(corners[(i + 1) % 4][0] - corners[i][0],
                                   corners[(i + 1) % 4][1] - corners[i][1])
                        for i in range(4)
                    ]
                    support = sum(min(1.0, line[4] / max(1.0, side)) for line, side in zip(
                        (top, right, bottom, left), side_lengths
                    )) / 4.0
                    area = _polygon_area(corners)
                    aperture_w = max(1.0, right_x - left_x)
                    aperture_h = max(1.0, bottom_y - top_y)
                    # An inner aperture edge normally has the other side of
                    # the physical bar just outside it.  This small cue picks
                    # the inside edge rather than the equally strong outer
                    # silhouette, without relying on gate colour.
                    inner_evidence = sum((
                        any(0.025 * aperture_w <= left_x - x <= 0.20 * aperture_w
                            for x in vertical_positions),
                        any(0.025 * aperture_w <= x - right_x <= 0.20 * aperture_w
                            for x in vertical_positions),
                        any(0.025 * aperture_h <= top_y - y <= 0.20 * aperture_h
                            for y in horizontal_positions),
                        any(0.025 * aperture_h <= y - bottom_y <= 0.20 * aperture_h
                            for y in horizontal_positions),
                    )) / 4.0
                    candidates.append((
                        corners, "hough_inner", float(area), support, inner_evidence
                    ))
    return candidates


def _aperture_candidate_score(candidate, visual_target, width: int, height: int) -> float:
    corners, source, area = candidate[:3]
    support = float(candidate[3]) if len(candidate) > 3 else 0.75
    inner_evidence = float(candidate[4]) if len(candidate) > 4 else 0.0
    cx = sum(point[0] for point in corners) / 4.0
    cy = sum(point[1] for point in corners) / 4.0
    if visual_target is None:
        source_bonus = 1.0 if source == "corners" else (0.4 if source == "hough_inner" else 0.0)
        return source_bonus + support + 0.9 * inner_evidence + area / max(1.0, width * height)
    tx = (float(visual_target.center_x) + 1.0) * 0.5 * width
    ty = (float(visual_target.center_y) + 1.0) * 0.5 * height
    tw = max(24.0, float(visual_target.width_fraction) * width)
    th = max(24.0, float(visual_target.height_fraction) * height)
    center_error = math.hypot((cx - tx) / tw, (cy - ty) / th)
    expected_area = max(64.0, 0.62 * float(visual_target.area_fraction) * width * height)
    area_error = abs(math.log(max(1.0, area) / expected_area))
    source_bonus = 0.45 if source in ("corners", "hough_inner") else 0.0
    return (1.4 * support + 0.9 * inner_evidence + source_bonus
            - 2.4 * center_error - 0.65 * area_error)


def detect_aperture_corners(
    image,
    intr: CameraIntrinsics = DEFAULT_INTRINSICS,
    *,
    visual_target=None,
):
    """Estimate the four inner aperture corners (tl,tr,br,bl) in pixels, or None.

    Uses the frame's inner hole (RETR_CCOMP child contours) and, when today's
    tracked visual target is supplied, a local four-line fallback robust to
    underwater colour loss. Returns ``(corners_px, source)`` or ``None``.
    """
    if _cv2 is None:
        return None
    h, w = image.shape[0], image.shape[1]
    candidates = _mask_aperture_candidates(image)
    candidates.extend(_hough_aperture_candidates(image, visual_target))
    if not candidates:
        return None
    best = max(candidates, key=lambda candidate: _aperture_candidate_score(
        candidate, visual_target, w, h
    ))
    return best[0], best[1]


# --------------------------------------------------------------------------- pose
def _gate_object_points(width_m: float = GATE_INNER_SIZE_M, height_m: Optional[float] = None):
    """Object-frame aperture corners (tl, tr, br, bl) for a ``width_m`` x ``height_m`` gate.

    The gate aperture size is a *public* competition constant per track (not privileged pose
    information); pass the track's declared ``inner_width_m`` / ``inner_height_m``.
    """
    hw = float(width_m) / 2.0
    hh = float(height_m if height_m is not None else width_m) / 2.0
    # tl, tr, br, bl in the gate plane (x right, y down, z=0), matching order_corners.
    return _np.array([[-hw, -hh, 0.0], [hw, -hh, 0.0], [hw, hh, 0.0], [-hw, hh, 0.0]],
                     dtype=_np.float64)


def estimate_pose_pnp(corners_px, intr: CameraIntrinsics = DEFAULT_INTRINSICS, *, yaw_sign_hint: int = 0,
                      gate_width_m: float = GATE_INNER_SIZE_M, gate_height_m: Optional[float] = None):
    """Estimate gate pose from 4 aperture corners via planar PnP; select the best of the
    (up to two) solutions. The square-planar sign ambiguity is resolved with
    ``yaw_sign_hint`` (the unambiguous image bar-height ratio: -1/0/+1). ``gate_width_m`` /
    ``gate_height_m`` are the track's public aperture size. Returns a dict or None."""
    if _cv2 is None:
        return None
    obj = _gate_object_points(gate_width_m, gate_height_m)
    img = _np.asarray(corners_px, dtype=_np.float64).reshape(-1, 1, 2)
    K = intr.matrix()
    dist = _np.zeros((4, 1))
    try:
        # SOLVEPNP_IPPE: general planar solver (up to two solutions), agnostic to the
        # marker-specific vertex convention that IPPE_SQUARE assumes.
        retval, rvecs, tvecs, reproj = _cv2.solvePnPGeneric(
            obj, img, K, dist, flags=_cv2.SOLVEPNP_IPPE)
    except Exception:  # pragma: no cover
        return None
    if not retval:
        return None
    best = None
    for rvec, tvec, err in zip(rvecs, tvecs, (reproj.ravel() if reproj is not None else [None] * len(rvecs))):
        t = tvec.reshape(3)
        if not _np.all(_np.isfinite(t)) or not _np.all(_np.isfinite(rvec)):
            continue  # reject degenerate/non-finite solutions
        if t[2] <= 0:  # gate must be in front of the camera
            continue
        R, _ = _cv2.Rodrigues(rvec)
        # Gate normal (object +z) expressed in the camera frame.
        normal_cam = R[:, 2]
        # A visible gate faces roughly toward the camera: its normal has negative z.
        facing = -float(normal_cam[2])
        # Canonical, normal-reversal-invariant plane obliqueness (0 = frontal, +/-90 wrap).
        yaw = canonical_gate_plane_yaw(R)
        pitch = canonical_gate_plane_pitch(R)
        roll = math.degrees(math.atan2(R[1, 0], R[1, 1]))
        cand = {
            "translation": (float(t[0]), float(t[1]), float(t[2])),
            "yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll,
            "reprojection_error_px": (float(err) if err is not None else None),
            "facing": facing,
        }
        # Lowest reprojection error is the reliable discriminator between the two planar
        # solutions when there is perspective; the bar-height hint and facing only break
        # near-degenerate (near-frontal) ties.
        matches_hint = int(yaw_sign_hint != 0 and math.copysign(1, yaw) == yaw_sign_hint)
        reproj = cand["reprojection_error_px"] or 1e3
        score = (-round(reproj, 3), matches_hint, facing)
        if best is None or score > best[0]:
            best = (score, cand)
    return best[1] if best else None


def projective_orientation(
    corners_px,
    intr: CameraIntrinsics = DEFAULT_INTRINSICS,
    *,
    gate_width_m: float = GATE_INNER_SIZE_M,
    gate_height_m: Optional[float] = None,
):
    """Robust projective orientation from a quad when metric PnP is unavailable/unstable.

    Uses the left/right apparent side-height ratio and horizontal skew to decide
    frontal / rotated-left / rotated-right, plus a skew magnitude. Returns a dict.
    """
    tl, tr, br, bl = [(_np.asarray(p, dtype=_np.float64)) for p in corners_px]
    left_h = float(_np.hypot(*(bl - tl)))
    right_h = float(_np.hypot(*(br - tr)))
    ratio = (left_h - right_h) / max(1e-3, left_h + right_h)  # >0 => left bar taller/closer
    # Horizontal skew: difference of top vs bottom edge horizontal centers, normalized.
    top_cx = 0.5 * (tl[0] + tr[0])
    bot_cx = 0.5 * (bl[0] + br[0])
    width = 0.5 * (abs(tr[0] - tl[0]) + abs(br[0] - bl[0]))
    skew = float((top_cx - bot_cx) / max(1e-3, width))
    # Perspective geometry for a vertical rectangular gate gives exactly
    #   ratio = half_width * sin(yaw) / forward_distance.
    # Estimate forward distance from the mean apparent vertical side length.
    # The previous fixed ``ratio * 1.5`` conversion severely underestimated
    # distant yaw and made planar PnP choose the mirrored sign in haze.
    aperture_h_m = float(gate_height_m if gate_height_m is not None else gate_width_m)
    mean_side_h_px = 0.5 * (left_h + right_h)
    forward_from_height = intr.fy * aperture_h_m / max(1.0, mean_side_h_px)
    sin_yaw = ratio * forward_from_height / max(1e-6, 0.5 * float(gate_width_m))
    yaw_proxy = math.degrees(math.asin(max(-1.0, min(1.0, sin_yaw))))
    hint = orientation_class(yaw_proxy)
    return {"orientation_hint": hint, "quadrilateral_skew": skew, "yaw_proxy_deg": yaw_proxy,
            "left_right_height_ratio": ratio}


def _normalize_point(px, py, w, h):
    return ((px - w / 2.0) / (w / 2.0), (py - h / 2.0) / (h / 2.0))


def estimate_gate_pose(image, *, intr: CameraIntrinsics = DEFAULT_INTRINSICS,
                       beacon_bearing_deg: Optional[float] = None,
                       visual_target=None,
                       max_reprojection_px: float = 8.0,
                       gate_width_m: float = GATE_INNER_SIZE_M,
                       gate_height_m: Optional[float] = None) -> Optional[GatePoseTarget]:
    """Full onboard estimate: aperture corners -> PnP (with projective fallback). Returns a
    :class:`GatePoseTarget` or None if no gate quadrilateral is found (callers should then
    fall back to the center-only :class:`vision.VisionTarget`).

    ``gate_width_m`` / ``gate_height_m`` are the track's *public* aperture dimensions (default
    the 1.5 m official gate). Passing the correct size is required for a correct metric scale:
    modelling a 2.0 m aperture as 1.5 m scales every PnP/size distance by 0.75."""
    if _cv2 is None or _np is None:
        return None
    h, w = int(image.shape[0]), int(image.shape[1])
    found = detect_aperture_corners(image, intr, visual_target=visual_target)
    if found is None:
        return None
    corners_px, source = found
    xs = [p[0] for p in corners_px]
    ys = [p[1] for p in corners_px]
    cx = sum(xs) / 4.0
    cy = sum(ys) / 4.0
    width_frac = (max(xs) - min(xs)) / w
    height_frac = (max(ys) - min(ys)) / h
    area_frac = _polygon_area(corners_px) / (w * h)
    corners_norm = [_normalize_point(px, py, w, h) for px, py in corners_px]
    proj = projective_orientation(
        corners_px,
        intr,
        gate_width_m=gate_width_m,
        gate_height_m=gate_height_m,
    )
    target = GatePoseTarget(
        center_x=(cx - w / 2.0) / (w / 2.0),
        center_y=(cy - h / 2.0) / (h / 2.0),
        area_fraction=float(area_frac), width_fraction=float(width_frac),
        height_fraction=float(height_frac), confidence=min(1.0, 4.0 * area_frac + 0.3),
        corners_normalized=corners_norm, detection_source=source,
        quadrilateral_skew=proj["quadrilateral_skew"], orientation_hint=proj["orientation_hint"],
        orientation_present=True,
    )
    # The bar-height ratio is an unambiguous image cue for the yaw sign; use it to resolve
    # the square-planar PnP ambiguity. In the canonical convention a taller LEFT bar
    # (ratio > 0) means the gate's left edge is nearer == positive canonical yaw.
    ratio = proj["left_right_height_ratio"]
    projective_yaw = float(proj["yaw_proxy_deg"])
    yaw_sign_hint = (
        int(math.copysign(1, projective_yaw)) if abs(projective_yaw) >= 5.0 else 0
    )
    pose = estimate_pose_pnp(corners_px, intr, yaw_sign_hint=yaw_sign_hint,
                             gate_width_m=gate_width_m, gate_height_m=gate_height_m)
    beacon_ok = True
    if pose is not None and beacon_bearing_deg is not None:
        # Bearing consistency: PnP lateral sign should agree with the beacon bearing sign.
        lateral = pose["translation"][0]
        # Camera +x is image-right, while the acoustic convention reports a
        # right/starboard target with a *negative* bearing.  Equal signs are
        # therefore inconsistent (the previous test was reversed).
        if abs(beacon_bearing_deg) > 8.0 and (lateral * beacon_bearing_deg) > 0 and abs(lateral) > 0.3:
            beacon_ok = False
    if pose is not None and beacon_ok and (pose["reprojection_error_px"] or 0.0) <= max_reprojection_px:
        target.pose_present = True
        target.translation_camera_m = pose["translation"]
        pnp_yaw = float(pose["yaw_deg"])
        projective_yaw = float(proj["yaw_proxy_deg"])
        # Square-planar PnP can choose a strongly oblique solution with a tiny
        # reprojection error even when noisy real-image corners show two bars
        # of practically equal height.  The bar-height cue is coarse in
        # magnitude but reliable for frontal-vs-oblique and for sign.  Use it
        # only as a conservative veto, not as a second general pose solver.
        # Near-frontal square PnP is two-solution ambiguous: one-pixel corner
        # noise can produce alternating +/-10..20 degree poses with an equally
        # tiny reprojection error. When the directly visible side-height cue
        # says frontal, suppress even a moderate PnP obliqueness. Stronger
        # projective evidence still leaves PnP free to estimate the magnitude.
        frontal_conflict = abs(projective_yaw) < 8.0 and abs(pnp_yaw) > 8.0
        sign_conflict = (
            abs(projective_yaw) >= 5.0
            and abs(pnp_yaw) >= 8.0
            and math.copysign(1.0, projective_yaw) != math.copysign(1.0, pnp_yaw)
        )
        magnitude_conflict = abs(pnp_yaw) > max(
            12.0, 1.8 * abs(projective_yaw) + 5.0
        )
        if frontal_conflict or sign_conflict or magnitude_conflict:
            target.gate_plane_yaw_deg = projective_yaw
            target.detection_source = "pnp_projective_yaw"
        else:
            target.gate_plane_yaw_deg = pnp_yaw
            target.detection_source = "pnp"
        target.gate_plane_pitch_deg = pose["pitch_deg"]
        target.gate_plane_roll_deg = pose["roll_deg"]
        target.reprojection_error_px = pose["reprojection_error_px"]
        target.pose_confidence = float(max(0.0, min(1.0, 1.0 - (pose["reprojection_error_px"] or 0.0) / max_reprojection_px)))
        if frontal_conflict or sign_conflict or magnitude_conflict:
            target.pose_confidence *= 0.75
        # Refine the orientation hint from the canonical metric yaw when available.
        if target.gate_plane_yaw_deg is not None:
            target.orientation_hint = orientation_class(target.gate_plane_yaw_deg)
    else:
        target.detection_source = "projective"
        target.gate_plane_yaw_deg = proj["yaw_proxy_deg"]  # soft projective yaw
        target.pose_confidence = 0.3
        # Size-based metric fallback: the aperture is a known gate_height_m tall square, so its
        # apparent pixel height gives forward distance without PnP (onboard-legal: known gate
        # size + image only). Lateral/vertical follow from the center offset at that distance.
        pixel_h = max(1.0, height_frac * h)
        aperture_h_m = float(gate_height_m if gate_height_m is not None else gate_width_m)
        z_est = intr.fy * aperture_h_m / pixel_h
        if math.isfinite(z_est) and 0.3 < z_est < 60.0:
            lateral = (cx - intr.cx) / intr.fx * z_est
            vertical = (cy - intr.cy) / intr.fy * z_est
            target.translation_camera_m = (float(lateral), float(vertical), float(z_est))
    return target


def _ema(prev: Optional[float], new: float, alpha: float) -> float:
    return new if prev is None else (alpha * new + (1.0 - alpha) * prev)


def _ema_angle_deg(prev: Optional[float], new: float, alpha: float) -> float:
    """EMA that handles angular wrapping by blending unit vectors."""
    if prev is None:
        return new
    pr, nr = math.radians(prev), math.radians(new)
    x = alpha * math.cos(nr) + (1 - alpha) * math.cos(pr)
    y = alpha * math.sin(nr) + (1 - alpha) * math.sin(pr)
    return math.degrees(math.atan2(y, x))


def _ema_plane_angle_deg(prev: Optional[float], new: float, alpha: float) -> float:
    """EMA for an unoriented plane angle (period 180, not 360 degrees)."""
    new = wrap_plane_yaw_deg(new)
    if prev is None:
        return new
    pr, nr = 2.0 * math.radians(prev), 2.0 * math.radians(new)
    x = alpha * math.cos(nr) + (1.0 - alpha) * math.cos(pr)
    y = alpha * math.sin(nr) + (1.0 - alpha) * math.sin(pr)
    return wrap_plane_yaw_deg(0.5 * math.degrees(math.atan2(y, x)))


class GatePoseTracker:
    """Controller-side temporal filter over past onboard gate-pose detections only.

    Exponentially smooths the pose fields (with angular wrapping for yaw/pitch), tracks
    detection age and consecutive valid frames, and expires a stale estimate after
    ``max_age_steps`` frames without a fresh pose. Uses no future or privileged information.
    """

    def __init__(self, alpha: float = 0.4, max_age_steps: int = 4):
        self.alpha = float(alpha)
        self.max_age_steps = int(max_age_steps)
        self.reset()

    def reset(self) -> None:
        self._lat = self._vert = self._fwd = None
        self._yaw = self._pitch = None
        self._reproj = self._conf = None
        self._center_x = self._center_y = None
        self._area = self._width = self._height = None
        self._corners = None
        self._metric_pose_present = False
        self._orientation_present = False
        self.age = 0
        self.consecutive_valid = 0
        self.track_age_frames = 0
        self.source = "none"

    @property
    def has_pose(self) -> bool:
        return self._fwd is not None and self.age <= self.max_age_steps

    @property
    def has_orientation(self) -> bool:
        return self._yaw is not None and self.age <= self.max_age_steps

    def update(self, target: Optional[GatePoseTarget]) -> Optional[GatePoseTarget]:
        fresh = (
            target is not None
            and (target.orientation_present or target.pose_present)
            and target.gate_plane_yaw_deg is not None
        )
        if not fresh:
            self.age += 1
            self.consecutive_valid = 0
            if self._yaw is None or self.age > self.max_age_steps:
                self.reset()
                self.source = "lost"
                return None
            return self._as_target(stale=True, base=target)
        a = self.alpha
        self._center_x = _ema(self._center_x, float(target.center_x), a)
        self._center_y = _ema(self._center_y, float(target.center_y), a)
        self._area = _ema(self._area, float(target.area_fraction), a)
        self._width = _ema(self._width, float(target.width_fraction), a)
        self._height = _ema(self._height, float(target.height_fraction), a)
        if target.corners_normalized is not None:
            if self._corners is None:
                self._corners = [tuple(point) for point in target.corners_normalized]
            else:
                self._corners = [
                    (_ema(old[0], new[0], a), _ema(old[1], new[1], a))
                    for old, new in zip(self._corners, target.corners_normalized)
                ]
        t = target.translation_camera_m
        if t is not None:
            self._lat = _ema(self._lat, float(t[0]), a)
            self._vert = _ema(self._vert, float(t[1]), a)
            self._fwd = _ema(self._fwd, float(t[2]), a)
        if target.gate_plane_yaw_deg is not None:
            yaw = float(target.gate_plane_yaw_deg)
            if abs(yaw) <= 90.0 and (self._yaw is None or abs(self._yaw) <= 90.0):
                self._yaw = _ema_plane_angle_deg(self._yaw, yaw, a)
            else:  # backward-compatible handling for legacy non-canonical inputs
                self._yaw = _ema_angle_deg(self._yaw, yaw, a)
        if target.gate_plane_pitch_deg is not None:
            self._pitch = _ema_angle_deg(self._pitch, float(target.gate_plane_pitch_deg), a)
        self._reproj = _ema(self._reproj, target.reprojection_error_px or 0.0, a)
        self._conf = _ema(self._conf, float(target.pose_confidence), a)
        self._metric_pose_present = bool(target.pose_present)
        self._orientation_present = True
        self.age = 0
        self.consecutive_valid += 1
        self.track_age_frames += 1
        self.source = target.detection_source
        return self._as_target(stale=False, base=target)

    def _as_target(self, *, stale: bool, base: Optional[GatePoseTarget]) -> GatePoseTarget:
        conf = (self._conf or 0.0) * (0.6 ** self.age if stale else 1.0)  # decay stale confidence
        out = GatePoseTarget(
            center_x=float(self._center_x or 0.0),
            center_y=float(self._center_y or 0.0),
            area_fraction=float(self._area or 0.0),
            width_fraction=float(self._width or 0.0),
            height_fraction=float(self._height or 0.0),
            confidence=(base.confidence if base is not None else float(conf)),
            corners_normalized=list(self._corners) if self._corners is not None else None,
            pose_present=self._metric_pose_present,
            translation_camera_m=(
                (float(self._lat), float(self._vert), float(self._fwd))
                if self._lat is not None and self._vert is not None and self._fwd is not None
                else None
            ),
            gate_plane_yaw_deg=self._yaw, gate_plane_pitch_deg=self._pitch,
            reprojection_error_px=self._reproj, pose_confidence=float(max(0.0, min(1.0, conf))),
            detection_source=("stale" if stale else self.source),
            quadrilateral_skew=(base.quadrilateral_skew if base is not None else 0.0),
            orientation_hint=orientation_class(self._yaw),
            orientation_present=self._orientation_present,
            predicted=stale,
            track_age_frames=self.track_age_frames + self.age,
        )
        return out
