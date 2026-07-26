"""Onboard visual gate-pose estimation from FrontCamera images.

Extends the deterministic gate detector (:mod:`vision`) to estimate the four inner
aperture corners and, when possible, the relative pose of the gate plane (translation +
orientation) via PnP, with a robust projective-orientation fallback and a center-only
degradation path. It consumes ONLY the camera image and the known public gate aperture
(1.5 x 1.5 m); it never reads the simulator gate pose or world geometry.

Camera convention (OpenCV): +x right, +y down, +z forward (into the scene). The reported
translation is (lateral_x right+, vertical_y down+, forward_z away+). Gate-plane yaw is the
rotation of the gate about the camera vertical axis: 0 = frontal, positive = the gate's
right edge is farther (rover must turn/strafe to face it). All angles in degrees.

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
    """Binary mask of bright/whitish gate-bar pixels (robust to water tint)."""
    arr = _np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[:, :, :3].astype(_np.float32)
    else:
        rgb = _np.stack([arr] * 3, axis=-1).astype(_np.float32)
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    brightness = mx / 255.0
    # Gate props render bright and relatively low-saturation vs the blue water.
    sat = (mx - mn) / (mx + 1e-3)
    mask = ((brightness > 0.45) & (sat < 0.55)).astype(_np.uint8) * 255
    return mask


def detect_aperture_corners(image, intr: CameraIntrinsics = DEFAULT_INTRINSICS):
    """Estimate the four inner aperture corners (tl,tr,br,bl) in pixels, or None.

    Uses the frame's inner hole (RETR_CCOMP child contours) approximated to a quad, else
    the largest convex quad. Returns ``(corners_px, source)`` or ``None``.
    """
    if _cv2 is None:
        return None
    h, w = image.shape[0], image.shape[1]
    mask = _bar_mask(image)
    mask = _cv2.morphologyEx(mask, _cv2.MORPH_CLOSE, _np.ones((5, 5), _np.uint8))
    contours, hierarchy = _cv2.findContours(mask, _cv2.RETR_CCOMP, _cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    holes, outers = [], []
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
        (holes if hier[idx][3] != -1 else outers).append((area, corners))
    # The inner opening of the gate frame is a child contour (hole); it IS the aperture.
    # Strongly prefer the largest valid hole; only fall back to the outer quad if no hole.
    if holes:
        holes.sort(key=lambda t: t[0], reverse=True)
        return holes[0][1], "corners"
    if outers:
        outers.sort(key=lambda t: t[0], reverse=True)
        return outers[0][1], "corners_outer"
    return None


# --------------------------------------------------------------------------- pose
def _gate_object_points():
    half = GATE_INNER_SIZE_M / 2.0
    # tl, tr, br, bl in the gate plane (x right, y down, z=0), matching order_corners.
    return _np.array([[-half, -half, 0.0], [half, -half, 0.0], [half, half, 0.0], [-half, half, 0.0]],
                     dtype=_np.float64)


def estimate_pose_pnp(corners_px, intr: CameraIntrinsics = DEFAULT_INTRINSICS, *, yaw_sign_hint: int = 0):
    """Estimate gate pose from 4 aperture corners via planar PnP; select the best of the
    (up to two) solutions. The square-planar sign ambiguity is resolved with
    ``yaw_sign_hint`` (the unambiguous image bar-height ratio: -1/0/+1). Returns a dict or None."""
    if _cv2 is None:
        return None
    obj = _gate_object_points()
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
        yaw = math.degrees(math.atan2(R[0, 2], R[2, 2]))   # rotation about camera vertical
        pitch = math.degrees(math.atan2(-R[1, 2], math.hypot(R[0, 2], R[2, 2])))
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


def projective_orientation(corners_px):
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
    if abs(ratio) < 0.06:
        hint = "frontal"
    elif ratio > 0:
        hint = "rotated_left"   # gate's left edge appears larger/closer
    else:
        hint = "rotated_right"
    # Coarse yaw proxy from the height ratio (bounded, monotone) for a soft signal.
    yaw_proxy = max(-45.0, min(45.0, math.degrees(math.asin(max(-1.0, min(1.0, ratio * 1.5))))))
    return {"orientation_hint": hint, "quadrilateral_skew": skew, "yaw_proxy_deg": yaw_proxy,
            "left_right_height_ratio": ratio}


def _normalize_point(px, py, w, h):
    return ((px - w / 2.0) / (w / 2.0), (py - h / 2.0) / (h / 2.0))


def estimate_gate_pose(image, *, intr: CameraIntrinsics = DEFAULT_INTRINSICS,
                       beacon_bearing_deg: Optional[float] = None,
                       max_reprojection_px: float = 8.0) -> Optional[GatePoseTarget]:
    """Full onboard estimate: aperture corners -> PnP (with projective fallback). Returns a
    :class:`GatePoseTarget` or None if no gate quadrilateral is found (callers should then
    fall back to the center-only :class:`vision.VisionTarget`)."""
    if _cv2 is None or _np is None:
        return None
    h, w = int(image.shape[0]), int(image.shape[1])
    found = detect_aperture_corners(image, intr)
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
    proj = projective_orientation(corners_px)
    target = GatePoseTarget(
        center_x=(cx - w / 2.0) / (w / 2.0),
        center_y=(cy - h / 2.0) / (h / 2.0),
        area_fraction=float(area_frac), width_fraction=float(width_frac),
        height_fraction=float(height_frac), confidence=min(1.0, 4.0 * area_frac + 0.3),
        corners_normalized=corners_norm, detection_source=source,
        quadrilateral_skew=proj["quadrilateral_skew"], orientation_hint=proj["orientation_hint"],
    )
    # The bar-height ratio is an unambiguous image cue for the yaw sign; use it to resolve
    # the square-planar PnP ambiguity. ratio>0 (left bar taller) => PnP yaw is negative.
    ratio = proj["left_right_height_ratio"]
    yaw_sign_hint = int(-math.copysign(1, ratio)) if abs(ratio) > 0.06 else 0
    pose = estimate_pose_pnp(corners_px, intr, yaw_sign_hint=yaw_sign_hint)
    beacon_ok = True
    if pose is not None and beacon_bearing_deg is not None:
        # Bearing consistency: PnP lateral sign should agree with the beacon bearing sign.
        lateral = pose["translation"][0]
        if abs(beacon_bearing_deg) > 8.0 and (lateral * beacon_bearing_deg) < 0 and abs(lateral) > 0.3:
            beacon_ok = False
    if pose is not None and beacon_ok and (pose["reprojection_error_px"] or 0.0) <= max_reprojection_px:
        target.pose_present = True
        target.translation_camera_m = pose["translation"]
        target.gate_plane_yaw_deg = pose["yaw_deg"]
        target.gate_plane_pitch_deg = pose["pitch_deg"]
        target.gate_plane_roll_deg = pose["roll_deg"]
        target.reprojection_error_px = pose["reprojection_error_px"]
        target.pose_confidence = float(max(0.0, min(1.0, 1.0 - (pose["reprojection_error_px"] or 0.0) / max_reprojection_px)))
        target.detection_source = "pnp"
        # Refine the orientation hint from metric yaw when available.
        if pose["yaw_deg"] is not None:
            if abs(pose["yaw_deg"]) < 5.0:
                target.orientation_hint = "frontal"
            else:
                target.orientation_hint = "rotated_left" if pose["yaw_deg"] > 0 else "rotated_right"
    else:
        target.detection_source = "projective"
        target.gate_plane_yaw_deg = proj["yaw_proxy_deg"]  # soft projective yaw
        target.pose_confidence = 0.3
        # Size-based metric fallback: the aperture is a known GATE_INNER_SIZE_M square, so its
        # apparent pixel height gives forward distance without PnP (onboard-legal: known gate
        # size + image only). Lateral/vertical follow from the center offset at that distance.
        pixel_h = max(1.0, height_frac * h)
        z_est = intr.fy * GATE_INNER_SIZE_M / pixel_h
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


class GatePoseTracker:
    """Controller-side temporal filter over past onboard gate-pose detections only.

    Exponentially smooths the pose fields (with angular wrapping for yaw/pitch), tracks
    detection age and consecutive valid frames, and expires a stale estimate after
    ``max_age_steps`` frames without a fresh pose. Uses no future or privileged information.
    """

    def __init__(self, alpha: float = 0.4, max_age_steps: int = 8):
        self.alpha = float(alpha)
        self.max_age_steps = int(max_age_steps)
        self._lat = self._vert = self._fwd = None
        self._yaw = self._pitch = None
        self._reproj = self._conf = None
        self.age = 0
        self.consecutive_valid = 0
        self.source = "none"

    @property
    def has_pose(self) -> bool:
        return self._fwd is not None and self.age <= self.max_age_steps

    def update(self, target: Optional[GatePoseTarget]) -> Optional[GatePoseTarget]:
        fresh = target is not None and target.pose_present and target.translation_camera_m is not None
        if not fresh:
            self.age += 1
            self.consecutive_valid = 0
            if self._fwd is None or self.age > self.max_age_steps:
                self._lat = self._vert = self._fwd = self._yaw = self._pitch = None
                self.source = "lost"
                return None
            return self._as_target(stale=True, base=target)
        t = target.translation_camera_m
        a = self.alpha
        self._lat = _ema(self._lat, float(t[0]), a)
        self._vert = _ema(self._vert, float(t[1]), a)
        self._fwd = _ema(self._fwd, float(t[2]), a)
        if target.gate_plane_yaw_deg is not None:
            self._yaw = _ema_angle_deg(self._yaw, float(target.gate_plane_yaw_deg), a)
        if target.gate_plane_pitch_deg is not None:
            self._pitch = _ema_angle_deg(self._pitch, float(target.gate_plane_pitch_deg), a)
        self._reproj = _ema(self._reproj, target.reprojection_error_px or 0.0, a)
        self._conf = _ema(self._conf, float(target.pose_confidence), a)
        self.age = 0
        self.consecutive_valid += 1
        self.source = target.detection_source
        return self._as_target(stale=False, base=target)

    def _as_target(self, *, stale: bool, base: Optional[GatePoseTarget]) -> GatePoseTarget:
        conf = (self._conf or 0.0) * (0.6 ** self.age if stale else 1.0)  # decay stale confidence
        out = GatePoseTarget(
            center_x=(base.center_x if base is not None else 0.0),
            center_y=(base.center_y if base is not None else 0.0),
            area_fraction=(base.area_fraction if base is not None else 0.0),
            width_fraction=(base.width_fraction if base is not None else 0.0),
            height_fraction=(base.height_fraction if base is not None else 0.0),
            confidence=(base.confidence if base is not None else 0.0),
            corners_normalized=(base.corners_normalized if base is not None else None),
            pose_present=True,
            translation_camera_m=(self._lat, self._vert, self._fwd),
            gate_plane_yaw_deg=self._yaw, gate_plane_pitch_deg=self._pitch,
            reprojection_error_px=self._reproj, pose_confidence=float(max(0.0, min(1.0, conf))),
            detection_source=("stale" if stale else self.source),
            orientation_hint=(base.orientation_hint if base is not None else "unknown"),
        )
        return out
