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
    detect_aperture_corners,
    estimate_gate_pose,
    estimate_pose_pnp,
    order_corners,
    projective_orientation,
    validate_quad,
)


def render_gate(intr, tx, ty, tz, yaw_deg, bar_frac=0.18):
    """Render a white square gate frame at a known camera-frame pose (evaluation-only)."""
    theta = math.radians(yaw_deg)
    right = np.array([math.cos(theta), 0.0, math.sin(theta)])
    up = np.array([0.0, 1.0, 0.0])
    center = np.array([tx, ty, tz])
    half = GATE_INNER_SIZE_M / 2.0
    outer = half + bar_frac * GATE_INNER_SIZE_M

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
    # magnitude recovered within tolerance; sign monotone & opposite for opposite renders.
    assert abs(abs(pose["yaw_deg"]) - abs(yaw)) < 10.0
    assert math.copysign(1, pose["yaw_deg"]) == math.copysign(1, -yaw)  # render convention


def test_estimate_gate_pose_end_to_end():
    img = render_gate(DEFAULT_INTRINSICS, 0.3, 0.0, 6.0, 25.0)
    t = estimate_gate_pose(img, intr=DEFAULT_INTRINSICS)
    assert t is not None and t.pose_present
    assert t.corners_normalized is not None and len(t.corners_normalized) == 4
    assert t.detection_source == "pnp"
    assert t.orientation_hint in ("rotated_left", "rotated_right")
    assert t.translation_camera_m[2] == pytest.approx(6.0, abs=0.6)


def test_projective_orientation_frontal_vs_rotated():
    frontal = [(100, 100), (300, 100), (300, 300), (100, 300)]
    assert projective_orientation(frontal)["orientation_hint"] == "frontal"
    # left bar taller (closer) than right -> rotated_left
    rot = [(100, 90), (300, 130), (300, 270), (100, 310)]
    assert projective_orientation(rot)["orientation_hint"] in ("rotated_left", "rotated_right")


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


def test_ground_truth_relative_pose_for_yaw_tracks():
    """Offline GT used only to score the detector: a yaw-rotated vehicle sees a known
    relative gate pose (distance 4 m, gate normal +x)."""
    from marine_race_arena.learning.vision_pose_capture import gt_relative_pose

    gt = gt_relative_pose([-4.0, 0.0, -4.0], 25.0, [0.0, 0.0, -4.0], [1.0, 0.0, 0.0])
    assert gt["distance"] == pytest.approx(4.0, abs=1e-6)
    assert gt["forward_z"] == pytest.approx(4.0 * math.cos(math.radians(25)), abs=1e-3)
    assert gt["lateral_x"] == pytest.approx(4.0 * math.sin(math.radians(25)), abs=1e-3)
    assert gt["gate_yaw_deg"] == pytest.approx(-25.0, abs=1e-3)  # relative gate-plane yaw
