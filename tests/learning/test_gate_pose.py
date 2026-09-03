"""Tests for onboard visual gate-pose estimation (synthetic gates at known poses)."""

import math

import pytest

pytest.importorskip("cv2")
pytest.importorskip("numpy")

import cv2
import numpy as np

from marine_race_arena.controllers.gate_pose import (
    DEFAULT_INTRINSICS,
    GATE_INNER_SIZE_M,
    CameraIntrinsics,
    GatePoseTarget,
    GatePoseTracker,
    canonical_gate_plane_yaw,
    detect_aperture_corners,
    estimate_gate_pose,
    estimate_pose_pnp,
    order_corners,
    orientation_class,
    plane_angle_distance_deg,
    projective_orientation,
    validate_quad,
    wrap_plane_yaw_deg,
)
from marine_race_arena.controllers.vision import VisionTarget


def render_gate(intr, tx, ty, tz, yaw_deg, bar_frac=0.18, size_m=GATE_INNER_SIZE_M):
    """Render a white square gate frame at a known camera-frame pose (evaluation-only)."""
    theta = math.radians(yaw_deg)
    right = np.array([math.cos(theta), 0.0, math.sin(theta)])
    up = np.array([0.0, 1.0, 0.0])
    center = np.array([tx, ty, tz])
    half = size_m / 2.0
    outer = half + bar_frac * size_m

    def corners(hh):
        signs = [(-1, -1), (1, -1), (1, 1), (-1, 1)]  # tl, tr, br, bl
        return [center + right * (sx * hh) + up * (sy * hh) for sx, sy in signs]

    def project(pts):
        uv = []
        for p in pts:
            uv.append([intr.fx * p[0] / p[2] + intr.cx, intr.fy * p[1] / p[2] + intr.cy])
        return np.array(uv, dtype=np.int32)

    img = np.empty((intr.height, intr.width, 3), np.uint8)
    img[:] = (70, 40, 20)  # dark bluish water
    cv2.fillPoly(img, [project(corners(outer))], (235, 238, 240))  # white frame
    cv2.fillPoly(img, [project(corners(half))], (70, 40, 20))       # inner opening (aperture)
    return img


def test_intrinsics_from_fov():
    intr = CameraIntrinsics(640, 480, 90.0)
    assert intr.fx == pytest.approx(320.0) and intr.fy == pytest.approx(320.0)
    assert intr.cx == 320.0 and intr.cy == 240.0


def test_order_corners_canonical():
    pts = [(100, 100), (200, 100), (200, 200), (100, 200)]
    tl, tr, br, bl = order_corners([pts[2], pts[0], pts[3], pts[1]])  # shuffled
    assert tl == (100, 100) and tr == (200, 100) and br == (200, 200) and bl == (100, 200)


def test_validate_quad_rejects_degenerate():
    good = [(100, 100), (300, 100), (300, 300), (100, 300)]
    assert validate_quad(good, image_area=640 * 480)
    thin = [(100, 100), (300, 100), (300, 105), (100, 105)]  # aspect too extreme
    assert not validate_quad(thin, image_area=640 * 480)
    nonconvex = [(100, 100), (300, 100), (150, 150), (100, 300)]
    assert not validate_quad(nonconvex, image_area=640 * 480)


def test_detect_corners_on_frontal_gate():
    img = render_gate(DEFAULT_INTRINSICS, 0.0, 0.0, 5.0, 0.0)
    found = detect_aperture_corners(img, DEFAULT_INTRINSICS)
    assert found is not None
    corners, source = found
    assert len(corners) == 4
    # center of the detected quad should be near the image center for a centered gate
    cx = sum(p[0] for p in corners) / 4
    cy = sum(p[1] for p in corners) / 4
    assert abs(cx - 320) < 25 and abs(cy - 240) < 25


@pytest.mark.parametrize("tz", [3.0, 6.0, 9.0])  # avoid the exact-frontal tz=5.0 IPPE degeneracy
def test_pnp_recovers_distance(tz):
    img = render_gate(DEFAULT_INTRINSICS, 0.0, 0.0, tz, 0.0)
    corners = detect_aperture_corners(img, DEFAULT_INTRINSICS)[0]
    pose = estimate_pose_pnp(corners, DEFAULT_INTRINSICS)
    assert pose is not None
    assert pose["translation"][2] == pytest.approx(tz, abs=0.4)  # forward distance within 0.4 m
    assert abs(pose["yaw_deg"]) < 6.0  # frontal


def test_pnp_recovers_lateral_offset():
    img = render_gate(DEFAULT_INTRINSICS, 0.8, 0.0, 5.0, 0.0)  # gate shifted right in camera frame
    corners = detect_aperture_corners(img, DEFAULT_INTRINSICS)[0]
    pose = estimate_pose_pnp(corners, DEFAULT_INTRINSICS)
    assert pose is not None
    assert pose["translation"][0] == pytest.approx(0.8, abs=0.4)


@pytest.mark.parametrize("yaw", [20.0, -20.0, 35.0, -35.0])
def test_pnp_yaw_sign_is_consistent(yaw):
    img = render_gate(DEFAULT_INTRINSICS, 0.0, 0.0, 5.0, yaw)
    found = detect_aperture_corners(img, DEFAULT_INTRINSICS)
    assert found is not None
    pose = estimate_pose_pnp(found[0], DEFAULT_INTRINSICS)
    assert pose is not None
    # Canonical yaw matches the render sign directly (render theta -> +theta).
    assert abs(abs(pose["yaw_deg"]) - abs(yaw)) < 10.0
    assert math.copysign(1, pose["yaw_deg"]) == math.copysign(1, yaw)


# --------------------------------------------------------- canonical yaw convention
def _rmat_from_render_yaw(yaw_deg):
    """Gate->camera rotation for the render's yaw convention (columns = right, up, normal)."""
    theta = math.radians(yaw_deg)
    right = np.array([math.cos(theta), 0.0, math.sin(theta)])
    up = np.array([0.0, 1.0, 0.0])
    normal = np.cross(right, up)  # object +z
    return np.column_stack([right, up, normal])


def test_canonical_yaw_is_zero_for_frontal():
    # A frontal gate's normal faces the camera (n ~ (0,0,-1)); the naive atan2(R02,R22)
    # would read ~180 deg. The canonical convention must read ~0.
    assert abs(canonical_gate_plane_yaw(_rmat_from_render_yaw(0.0))) < 1e-6


@pytest.mark.parametrize("yaw", [0.0, 10.0, -10.0, 25.0, -25.0, 45.0, -45.0, 80.0, -80.0])
def test_canonical_yaw_matches_render_angle(yaw):
    assert canonical_gate_plane_yaw(_rmat_from_render_yaw(yaw)) == pytest.approx(yaw, abs=1e-6)


@pytest.mark.parametrize("yaw", [0.0, 15.0, -30.0, 60.0])
def test_canonical_yaw_invariant_to_normal_reversal(yaw):
    R = _rmat_from_render_yaw(yaw)
    R_flip = R.copy()
    R_flip[:, 2] = -R_flip[:, 2]  # reverse the plane normal (same physical plane)
    assert canonical_gate_plane_yaw(R_flip) == pytest.approx(canonical_gate_plane_yaw(R), abs=1e-6)


def test_wrap_plane_yaw_folds_to_interval():
    assert wrap_plane_yaw_deg(170.0) == pytest.approx(-10.0)
    assert wrap_plane_yaw_deg(-170.0) == pytest.approx(10.0)
    assert wrap_plane_yaw_deg(30.0) == pytest.approx(30.0)


def test_plane_angle_distance_is_modulo_180():
    assert plane_angle_distance_deg(85.0, -85.0) == pytest.approx(10.0)  # not 170
    assert plane_angle_distance_deg(20.0, -20.0) == pytest.approx(40.0)
    assert plane_angle_distance_deg(10.0, 10.0) == pytest.approx(0.0)
    assert plane_angle_distance_deg(0.0, 90.0) == pytest.approx(90.0)


def test_orientation_class_labels():
    assert orientation_class(2.0) == "frontal"
    assert orientation_class(25.0) == "rotated_left"
    assert orientation_class(-25.0) == "rotated_right"
    assert orientation_class(None) == "unknown"


@pytest.mark.parametrize("size_m", [1.5, 2.0])
def test_pnp_distance_scales_with_gate_size(size_m):
    # A larger aperture at the same distance projects larger; PnP must use the correct size.
    img = render_gate(DEFAULT_INTRINSICS, 0.0, 0.0, 6.0, 0.0, size_m=size_m)
    corners = detect_aperture_corners(img, DEFAULT_INTRINSICS)[0]
    correct = estimate_pose_pnp(corners, DEFAULT_INTRINSICS, gate_width_m=size_m, gate_height_m=size_m)
    assert correct is not None
    assert correct["translation"][2] == pytest.approx(6.0, abs=0.5)
    # Modelling the wrong size mis-scales the distance by exactly the size ratio.
    wrong_size = 1.5 if size_m == 2.0 else 2.0
    wrong = estimate_pose_pnp(corners, DEFAULT_INTRINSICS, gate_width_m=wrong_size, gate_height_m=wrong_size)
    assert wrong is not None
    assert wrong["translation"][2] == pytest.approx(6.0 * wrong_size / size_m, abs=0.5)


def test_estimate_gate_pose_end_to_end():
    img = render_gate(DEFAULT_INTRINSICS, 0.3, 0.0, 6.0, 25.0)
    t = estimate_gate_pose(img, intr=DEFAULT_INTRINSICS)
    assert t is not None and t.pose_present
    assert t.corners_normalized is not None and len(t.corners_normalized) == 4
    assert t.detection_source == "pnp"
    assert t.orientation_hint in ("rotated_left", "rotated_right")
    assert t.translation_camera_m[2] == pytest.approx(6.0, abs=0.6)


def test_tracked_roi_selects_the_matching_gate_when_two_are_visible():
    background = np.array([70, 40, 20], dtype=np.uint8)
    left = render_gate(DEFAULT_INTRINSICS, -1.7, 0.0, 6.0, 20.0)
    right = render_gate(DEFAULT_INTRINSICS, 1.2, 0.0, 5.0, -25.0)
    image = np.empty_like(left)
    image[:] = background
    left_mask = np.any(left != background, axis=2)
    right_mask = np.any(right != background, axis=2)
    image[left_mask] = left[left_mask]
    image[right_mask] = right[right_mask]
    target = VisionTarget(
        center_x=0.25,
        center_y=0.0,
        confidence=0.9,
        area_fraction=0.05,
        width_fraction=0.25,
        height_fraction=0.30,
    )
    found = detect_aperture_corners(image, DEFAULT_INTRINSICS, visual_target=target)
    assert found is not None
    center_x = sum(point[0] for point in found[0]) / 4.0
    assert center_x > DEFAULT_INTRINSICS.cx


def test_projective_consensus_vetoes_implausible_oblique_pnp(monkeypatch):
    import marine_race_arena.controllers.gate_pose as module

    corners = [(200.0, 140.0), (440.0, 140.0), (440.0, 340.0), (200.0, 340.0)]
    monkeypatch.setattr(module, "detect_aperture_corners", lambda *args, **kwargs: (corners, "hough_inner"))
    monkeypatch.setattr(module, "projective_orientation", lambda _corners, *args, **kwargs: {
        "orientation_hint": "frontal",
        "quadrilateral_skew": 0.0,
        "yaw_proxy_deg": 0.5,
        "left_right_height_ratio": 0.0,
    })
    monkeypatch.setattr(module, "estimate_pose_pnp", lambda *args, **kwargs: {
        "translation": (0.0, 0.0, 3.0),
        "yaw_deg": 38.0,
        "pitch_deg": 0.0,
        "roll_deg": 0.0,
        "reprojection_error_px": 1.0,
        "facing": 1.0,
    })
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    target = estimate_gate_pose(image)
    assert target is not None and target.pose_present
    assert target.detection_source == "pnp_projective_yaw"
    assert target.gate_plane_yaw_deg == pytest.approx(0.5)


def test_projective_consensus_stabilizes_moderate_near_frontal_pnp(monkeypatch):
    import marine_race_arena.controllers.gate_pose as module

    corners = [(200.0, 140.0), (440.0, 140.0), (440.0, 340.0), (200.0, 340.0)]
    monkeypatch.setattr(module, "detect_aperture_corners", lambda *args, **kwargs: (corners, "hough_inner"))
    monkeypatch.setattr(module, "projective_orientation", lambda _corners, *args, **kwargs: {
        "orientation_hint": "frontal",
        "quadrilateral_skew": 0.0,
        "yaw_proxy_deg": -0.8,
        "left_right_height_ratio": -0.01,
    })
    monkeypatch.setattr(module, "estimate_pose_pnp", lambda *args, **kwargs: {
        "translation": (0.0, 0.0, 4.0),
        "yaw_deg": 14.0,
        "pitch_deg": 0.0,
        "roll_deg": 0.0,
        "reprojection_error_px": 0.2,
        "facing": 1.0,
    })

    target = estimate_gate_pose(np.zeros((480, 640, 3), dtype=np.uint8))

    assert target is not None
    assert target.detection_source == "pnp_projective_yaw"
    assert target.gate_plane_yaw_deg == pytest.approx(-0.8)


def test_projective_orientation_frontal_vs_rotated():
    frontal = [(100, 100), (300, 100), (300, 300), (100, 300)]
    assert projective_orientation(frontal)["orientation_hint"] == "frontal"
    # left bar taller (closer) than right -> rotated_left
    rot = [(100, 90), (300, 130), (300, 270), (100, 310)]
    assert projective_orientation(rot)["orientation_hint"] in ("rotated_left", "rotated_right")


@pytest.mark.parametrize("yaw", [20.0, -20.0, 35.0, -35.0])
def test_projective_orientation_uses_apparent_distance_for_yaw(yaw):
    image = render_gate(DEFAULT_INTRINSICS, 0.0, 0.0, 5.0, yaw)
    corners = detect_aperture_corners(image, DEFAULT_INTRINSICS)[0]

    estimate = projective_orientation(corners, DEFAULT_INTRINSICS)

    assert estimate["yaw_proxy_deg"] == pytest.approx(yaw, abs=7.0)


def test_no_gate_returns_none():
    img = np.empty((480, 640, 3), np.uint8)
    img[:] = (70, 40, 20)  # empty water, no gate
    assert estimate_gate_pose(img, intr=DEFAULT_INTRINSICS) is None


def _pose(fwd, lat=0.0, yaw=0.0):
    return GatePoseTarget(center_x=0.0, center_y=0.0, pose_present=True,
                          translation_camera_m=(lat, 0.0, fwd), gate_plane_yaw_deg=yaw,
                          gate_plane_pitch_deg=0.0, reprojection_error_px=1.0, pose_confidence=0.9,
                          detection_source="pnp")


def test_tracker_smooths_and_tracks_validity():
    tr = GatePoseTracker(alpha=0.5, max_age_steps=4)
    out = tr.update(_pose(6.0))
    assert out.pose_present and tr.consecutive_valid == 1
    out = tr.update(_pose(8.0))  # EMA(6, 8, 0.5) = 7.0
    assert out.translation_camera_m[2] == pytest.approx(7.0) and tr.consecutive_valid == 2


def test_tracker_expires_stale_pose():
    tr = GatePoseTracker(alpha=0.5, max_age_steps=3)
    tr.update(_pose(5.0))
    for _ in range(3):  # 3 misses -> still within max_age, returns decayed stale pose
        assert tr.update(None) is not None
    assert tr.update(None) is None  # 4th miss exceeds max_age -> expired
    assert not tr.has_pose


def test_tracker_handles_angular_wrap():
    tr = GatePoseTracker(alpha=0.5)
    tr.update(_pose(5.0, yaw=170.0))
    out = tr.update(_pose(5.0, yaw=-170.0))  # wrap across +-180; mean should be ~180, not ~0
    assert abs(abs(out.gate_plane_yaw_deg) - 180.0) < 5.0


def test_tracker_handles_canonical_plane_wrap_at_90_degrees():
    tr = GatePoseTracker(alpha=0.5)
    tr.update(_pose(5.0, yaw=89.0))
    out = tr.update(_pose(5.0, yaw=-89.0))
    assert plane_angle_distance_deg(out.gate_plane_yaw_deg, 90.0) < 2.0


def test_ground_truth_relative_pose_for_yaw_tracks():
    """Offline GT used only to score the detector: a yaw-rotated vehicle sees a known
    relative gate pose (distance 4 m, gate normal +x)."""
    from marine_race_arena.learning.vision_pose_capture import gt_relative_pose

    gt = gt_relative_pose([-4.0, 0.0, -4.0], 25.0, [0.0, 0.0, -4.0], [1.0, 0.0, 0.0])
    assert gt["distance"] == pytest.approx(4.0, abs=1e-6)
    assert gt["forward_z"] == pytest.approx(4.0 * math.cos(math.radians(25)), abs=1e-3)
    assert gt["lateral_x"] == pytest.approx(4.0 * math.sin(math.radians(25)), abs=1e-3)
    assert gt["gate_yaw_deg"] == pytest.approx(-25.0, abs=1e-3)  # relative gate-plane yaw
