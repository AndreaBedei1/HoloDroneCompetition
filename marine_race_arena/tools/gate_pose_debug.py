"""Human-readable, onboard-only gate-pose debugger for real HoloOcean frames.

The online path is deliberately narrow:

* the existing rule controller receives the official observation unchanged;
* its current :class:`TemporalVisionTracker` target selects the gate ROI;
* :func:`estimate_gate_pose` derives aperture corners and plane orientation
  from ``FrontCamera`` pixels;
* :class:`GatePoseTracker` filters only past camera-derived estimates.

No world position, configured gate pose, referee progress or simulator state is
read on the standard path.  ``--offline-ground-truth`` enables a separately
named scoring function and writes a separate JSONL file.  Those values never
enter the controller, detector, temporal filters, or online log.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from marine_race_arena.controllers.gate_pose import (
    DEFAULT_INTRINSICS,
    GatePoseTarget,
    GatePoseTracker,
    estimate_gate_pose,
    orientation_class,
    plane_angle_distance_deg,
    wrap_plane_yaw_deg,
)
from marine_race_arena.controllers.vision import VisionTarget, vision_targets_from_camera
from marine_race_arena.learning.config_local_transition_27d import (
    OBS_DIM_LOCAL_TRANSITION_27D,
    OBS_ENCODING_VERSION_LOCAL_TRANSITION_27D,
)
from marine_race_arena.learning.observation_encoder_local_transition_27d import (
    encode_observation_local_transition_27d,
)
from marine_race_arena.learning.tracker_context_local_transition_27d import (
    OnboardLocalTransition27dContextTracker,
)


TRACKS = {
    "horseshoe_bay": "marine_race_arena/tracks/marine_race_horseshoe_bay.json",
    "vertical_serpent": "marine_race_arena/tracks/marine_race_vertical_serpent.json",
    "mixed_endurance": "marine_race_arena/tracks/marine_race_mixed_endurance.json",
}

CAM_W, CAM_H = 640, 480
RECON_W, VALUES_W = 470, 450
STATUS_H = 56
WIN_W, WIN_H = CAM_W + RECON_W + VALUES_W, CAM_H + STATUS_H

BG = (24, 25, 30)
PANEL = (32, 34, 40)
FG = (235, 238, 242)
DIM = (155, 160, 170)
GREEN = (110, 225, 135)
YELLOW = (80, 220, 245)
ORANGE = (70, 155, 245)
RED = (90, 90, 245)
CYAN = (245, 205, 70)
PURPLE = (220, 120, 235)
FONT = 0


@dataclass
class OnlineFrame:
    track: str
    step: int
    time_s: float
    expected_beacon_id: str
    tracker_phase: str
    tracking_status: str
    raw_candidates: int
    detection_present: bool
    detection_predicted: bool
    detection_confidence: Optional[float]
    center_x: Optional[float]
    center_y: Optional[float]
    visual_area_fraction: Optional[float]
    corners_present: bool
    orientation_present: bool
    metric_pose_present: bool
    gate_yaw_deg: Optional[float]
    orientation: str
    pose_confidence: Optional[float]
    pose_source: str
    pose_predicted: bool
    pose_track_age_frames: int
    beacon_present: bool
    beacon_bearing_deg: Optional[float]
    beacon_elevation_deg: Optional[float]
    beacon_range_m: Optional[float]
    association_error: Optional[float]
    yaw_jump_deg: Optional[float]


@dataclass
class ViewState:
    paused: bool = False
    quit: bool = False
    realtime: bool = True


def _finite(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _fmt(value: Optional[float], fmt: str = ".3f") -> str:
    return "--" if value is None else format(float(value), fmt)


def _text(canvas, value: str, xy, colour=FG, scale=0.53, thickness=1) -> None:
    import cv2

    cv2.putText(canvas, value, xy, FONT, scale, colour, thickness, cv2.LINE_AA)


def _plate(canvas, value: str, xy, colour=FG, scale=0.53, thickness=1) -> None:
    import cv2

    (width, height), baseline = cv2.getTextSize(value, FONT, scale, thickness)
    x, y = xy
    cv2.rectangle(canvas, (x - 4, y - height - 4),
                  (x + width + 4, y + baseline + 4), (18, 20, 24), -1)
    _text(canvas, value, xy, colour, scale, thickness)


def _camera_bgr(raw: Mapping[str, Any]) -> np.ndarray:
    sensors = raw.get("sensors") if isinstance(raw.get("sensors"), Mapping) else {}
    image = sensors.get("FrontCamera")
    if image is None:
        canvas = np.full((CAM_H, CAM_W, 3), BG, dtype=np.uint8)
        _text(canvas, "FRONTCAMERA NON DISPONIBILE", (135, 245), RED, 0.75, 2)
        return canvas
    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[2] < 3:
        return np.full((CAM_H, CAM_W, 3), BG, dtype=np.uint8)
    frame = np.ascontiguousarray(frame[:, :, :3].astype(np.uint8))
    if frame.shape[:2] != (CAM_H, CAM_W):
        import cv2

        frame = cv2.resize(frame, (CAM_W, CAM_H))
    return frame


def _norm_to_px(point, width: int = CAM_W, height: int = CAM_H):
    return (
        int(round((float(point[0]) + 1.0) * 0.5 * width)),
        int(round((float(point[1]) + 1.0) * 0.5 * height)),
    )


def _draw_camera(
    frame: np.ndarray,
    visual: Optional[VisionTarget],
    pose: Optional[GatePoseTarget],
    online: OnlineFrame,
) -> np.ndarray:
    import cv2

    canvas = frame.copy()
    h, w = canvas.shape[:2]
    image_center = (w // 2, h // 2)
    cv2.line(canvas, (image_center[0], 0), (image_center[0], h), (75, 78, 84), 1)
    cv2.line(canvas, (0, image_center[1]), (w, image_center[1]), (75, 78, 84), 1)
    cv2.drawMarker(canvas, image_center, FG, cv2.MARKER_CROSS, 24, 2)

    if visual is not None:
        center = _norm_to_px((visual.center_x, visual.center_y), w, h)
        bw = max(10, int(visual.width_fraction * w))
        bh = max(10, int(visual.height_fraction * h))
        p0 = (max(0, center[0] - bw // 2), max(0, center[1] - bh // 2))
        p1 = (min(w - 1, center[0] + bw // 2), min(h - 1, center[1] + bh // 2))
        cv2.rectangle(canvas, p0, p1, CYAN, 1)
        _plate(canvas, "TRACKED GATE ROI", (p0[0] + 5, max(22, p0[1] - 7)), CYAN, 0.43)

    if pose is not None and pose.corners_normalized is not None:
        points = np.asarray([_norm_to_px(point, w, h) for point in pose.corners_normalized],
                            dtype=np.int32)
        colour = ORANGE if pose.predicted else GREEN
        cv2.polylines(canvas, [points], True, colour, 3, cv2.LINE_AA)
        names = ("TL", "TR", "BR", "BL")
        corner_colours = (CYAN, YELLOW, PURPLE, ORANGE)
        for name, point, corner_colour in zip(names, points, corner_colours):
            p = (int(point[0]), int(point[1]))
            cv2.circle(canvas, p, 7, corner_colour, -1, cv2.LINE_AA)
            cv2.circle(canvas, p, 9, (18, 20, 24), 2, cv2.LINE_AA)
            _plate(canvas, name, (p[0] + 10, max(18, p[1] - 8)), corner_colour, 0.42, 1)
        center = tuple(np.rint(points.mean(axis=0)).astype(int))
        cv2.drawMarker(canvas, center, YELLOW, cv2.MARKER_TILTED_CROSS, 24, 3)
        cv2.line(canvas, image_center, center, YELLOW, 1, cv2.LINE_AA)
        yaw = pose.gate_plane_yaw_deg
        if yaw is not None:
            direction = -1 if yaw > 0.0 else (1 if yaw < 0.0 else 0)
            if direction:
                end = (int(center[0] + direction * 70), int(center[1]))
                cv2.arrowedLine(canvas, center, end, ORANGE, 3, cv2.LINE_AA, tipLength=0.22)
        _plate(
            canvas,
            "%s  yaw=%s deg  conf=%s" % (
                online.orientation, _fmt(online.gate_yaw_deg, "+.1f"),
                _fmt(online.pose_confidence, ".2f"),
            ),
            (14, 70), colour, 0.60, 2,
        )
    else:
        _plate(canvas, "APERTURA: 4 CORNER NON DISPONIBILI", (14, 70), RED, 0.58, 2)

    status_colour = ORANGE if online.pose_predicted else (GREEN if online.corners_present else RED)
    _plate(canvas, online.tracking_status, (14, 38), status_colour, 0.62, 2)
    _plate(canvas, "VIEW 1  FRONTCAMERA -- STIMA SOLO ONBOARD", (14, h - 17), FG, 0.48, 1)
    return canvas


def _draw_reconstruction(pose: Optional[GatePoseTarget]) -> np.ndarray:
    """Top-down plane reconstruction: x right, z forward, camera at origin."""
    import cv2

    canvas = np.full((CAM_H, RECON_W, 3), PANEL, dtype=np.uint8)
    _text(canvas, "VIEW 2  PIANO DEL GATE (TOP VIEW)", (18, 30), FG, 0.57, 2)
    _text(canvas, "camera: origine   x: destra   z: avanti", (18, 54), DIM, 0.45)
    origin = (RECON_W // 2, CAM_H - 75)
    cv2.drawMarker(canvas, origin, FG, cv2.MARKER_TRIANGLE_UP, 25, 2)
    _text(canvas, "CAMERA / ROVER", (origin[0] - 72, origin[1] + 30), FG, 0.44)
    cv2.line(canvas, origin, (origin[0] - 150, 95), (62, 70, 82), 1)
    cv2.line(canvas, origin, (origin[0] + 150, 95), (62, 70, 82), 1)
    cv2.line(canvas, origin, (origin[0], 75), (70, 78, 88), 1)

    if pose is None or pose.translation_camera_m is None or pose.gate_plane_yaw_deg is None:
        _text(canvas, "Ricostruzione non disponibile", (85, 245), RED, 0.60, 2)
        _text(canvas, "Servono 4 corner validi", (115, 274), DIM, 0.50)
        return canvas

    lateral, _vertical, forward = pose.translation_camera_m
    forward = max(0.25, float(forward))
    lateral = float(lateral)
    world_extent = max(5.0, forward * 1.30, abs(lateral) * 2.8)
    scale = (CAM_H - 145) / world_extent

    def project(x: float, z: float):
        return int(round(origin[0] + x * scale)), int(round(origin[1] - z * scale))

    center = project(lateral, forward)
    cv2.arrowedLine(canvas, origin, center, YELLOW, 2, cv2.LINE_AA, tipLength=0.08)
    yaw = math.radians(float(pose.gate_plane_yaw_deg))
    half = 0.75
    dx, dz = half * math.cos(yaw), half * math.sin(yaw)
    left = project(lateral - dx, forward - dz)
    right = project(lateral + dx, forward + dz)
    colour = ORANGE if pose.predicted else GREEN
    cv2.line(canvas, left, right, colour, 7, cv2.LINE_AA)
    cv2.circle(canvas, center, 7, YELLOW, -1, cv2.LINE_AA)

    # Plane normal toward the camera; useful for seeing the sign immediately.
    nx, nz = math.sin(yaw), -math.cos(yaw)
    normal_end = project(lateral + 0.9 * nx, forward + 0.9 * nz)
    cv2.arrowedLine(canvas, center, normal_end, ORANGE, 3, cv2.LINE_AA, tipLength=0.25)
    _text(canvas, "gate plane", (max(8, center[0] - 45), max(78, center[1] - 18)), colour, 0.45)
    _text(canvas, "center direction", (20, CAM_H - 82), YELLOW, 0.45)
    _text(canvas, "plane normal", (20, CAM_H - 60), ORANGE, 0.45)
    _text(canvas, "Debug geometrico: non entra nel controller", (60, CAM_H - 18), DIM, 0.43)
    return canvas


def _draw_values(online: OnlineFrame) -> np.ndarray:
    import cv2

    canvas = np.full((CAM_H, VALUES_W, 3), PANEL, dtype=np.uint8)
    _text(canvas, "VIEW 3  VALORI ONBOARD", (18, 30), FG, 0.60, 2)
    cv2.line(canvas, (15, 42), (VALUES_W - 15, 42), (75, 78, 86), 1)
    rows = [
        ("center_x", _fmt(online.center_x), CYAN),
        ("center_y", _fmt(online.center_y), CYAN),
        ("beacon bearing", _fmt(online.beacon_bearing_deg, "+.1f") + " deg", GREEN),
        ("beacon elevation", _fmt(online.beacon_elevation_deg, "+.1f") + " deg", GREEN),
        ("beacon range", _fmt(online.beacon_range_m, ".2f") + " m", GREEN),
        ("gate yaw", _fmt(online.gate_yaw_deg, "+.1f") + " deg", ORANGE),
        ("orientation", online.orientation, ORANGE),
        ("detection conf", _fmt(online.detection_confidence, ".2f"), CYAN),
        ("pose conf", _fmt(online.pose_confidence, ".2f"), ORANGE),
        ("tracking", online.tracking_status, GREEN if online.detection_present else RED),
        ("pose source", online.pose_source, DIM),
        ("expected beacon", online.expected_beacon_id, FG),
        ("controller phase", online.tracker_phase, FG),
        ("raw candidates", str(online.raw_candidates), DIM),
    ]
    y = 74
    for label, value, colour in rows:
        _text(canvas, label, (20, y), DIM, 0.49)
        _text(canvas, value, (205, y), colour, 0.53, 2 if label in ("orientation", "tracking") else 1)
        y += 28
    cv2.line(canvas, (15, 448), (VALUES_W - 15, 448), (75, 78, 86), 1)
    _text(canvas, "Nessun ground truth in questa vista", (72, 469), GREEN, 0.47, 1)
    return canvas


def _compose(frame, visual, pose, online: OnlineFrame, state: ViewState) -> np.ndarray:
    import cv2

    top = np.hstack([
        _draw_camera(frame, visual, pose, online),
        _draw_reconstruction(pose),
        _draw_values(online),
    ])
    status = np.full((STATUS_H, WIN_W, 3), BG, dtype=np.uint8)
    mode = "PAUSA" if state.paused else "LIVE"
    _text(status, "[%s] step=%d  t=%.1fs" % (mode, online.step, online.time_s),
          (18, 24), YELLOW if state.paused else GREEN, 0.55, 2)
    _text(status, "SPACE pausa/riprendi   N o RIGHT frame successivo   S screenshot   Q/ESC esci",
          (18, 47), FG, 0.50)
    _text(status, "ONLINE: FrontCamera + beacon/DVL/IMU onboard; posa da camera",
          (900, 30), DIM, 0.43)
    cv2.line(status, (0, 0), (WIN_W, 0), (80, 82, 88), 1)
    return np.vstack([top, status])


def _packet_for_expected(raw: Mapping[str, Any], expected_id: str) -> Optional[Mapping[str, Any]]:
    for packet in raw.get("beacons") or []:
        if str(packet.get("beacon_id", "")) == expected_id:
            return packet
    return None


def _tracking_label(diag: Mapping[str, Any], visual: Optional[VisionTarget], pose) -> str:
    visual_diag = diag.get("visual_track") or {}
    if visual is None:
        return "LOST"
    if getattr(visual, "predicted", False):
        return "VISION PRED"
    if pose is not None and pose.predicted:
        return "POSE PRED"
    if visual_diag.get("locked"):
        return "LOCK"
    return "PROPOSAL"


def _association_error(visual: Optional[VisionTarget], pose: Optional[GatePoseTarget]) -> Optional[float]:
    if visual is None or pose is None:
        return None
    return math.hypot(float(pose.center_x) - float(visual.center_x),
                      float(pose.center_y) - float(visual.center_y))


def _online_frame(
    track: str,
    step: int,
    raw: Mapping[str, Any],
    tracker,
    visual: Optional[VisionTarget],
    pose: Optional[GatePoseTarget],
    raw_candidates: int,
    previous_yaw: Optional[float],
) -> OnlineFrame:
    diag = tracker.diagnostics()
    expected = str(tracker.expected_beacon_id)
    packet = _packet_for_expected(raw, expected)
    yaw = None if pose is None else _finite(pose.gate_plane_yaw_deg)
    jump = None if yaw is None or previous_yaw is None else plane_angle_distance_deg(yaw, previous_yaw)
    return OnlineFrame(
        track=track,
        step=step,
        time_s=float(raw.get("local_time_s", 0.0) or 0.0),
        expected_beacon_id=expected,
        tracker_phase=str(diag.get("phase", "--")),
        tracking_status=_tracking_label(diag, visual, pose),
        raw_candidates=raw_candidates,
        detection_present=visual is not None,
        detection_predicted=bool(visual is not None and visual.predicted),
        detection_confidence=None if visual is None else float(visual.confidence),
        center_x=None if pose is None else float(pose.center_x),
        center_y=None if pose is None else float(pose.center_y),
        visual_area_fraction=None if visual is None else float(visual.area_fraction),
        corners_present=bool(pose is not None and pose.corners_normalized is not None and not pose.predicted),
        orientation_present=bool(pose is not None and pose.orientation_present),
        metric_pose_present=bool(pose is not None and pose.pose_present),
        gate_yaw_deg=yaw,
        orientation=(orientation_class(yaw).replace("rotated_", "").upper()
                     if yaw is not None else "UNKNOWN"),
        pose_confidence=None if pose is None else float(pose.pose_confidence),
        pose_source="none" if pose is None else str(pose.detection_source),
        pose_predicted=bool(pose is not None and pose.predicted),
        pose_track_age_frames=0 if pose is None else int(pose.track_age_frames),
        beacon_present=packet is not None,
        beacon_bearing_deg=None if packet is None else _finite(packet.get("bearing_deg")),
        beacon_elevation_deg=None if packet is None else _finite(packet.get("elevation_deg")),
        beacon_range_m=None if packet is None else _finite(packet.get("range_m")),
        association_error=_association_error(visual, pose),
        yaw_jump_deg=jump,
    )


def _offline_ground_truth(episode, expected_online_id: str, pose: Optional[GatePoseTarget]) -> Dict[str, Any]:
    """Evaluation-only simulator comparison; never called on the standard path."""
    from marine_race_arena.learning.vision_pose_capture import gt_relative_pose

    referee_gate_id = episode.expected_gate_id()
    gate_id = (
        "G" + expected_online_id[1:]
        if expected_online_id.upper().startswith("B") else expected_online_id
    )
    gate = episode.context.arena.gate_map.get(gate_id)
    state = episode.context.adapter.get_participant_state(episode.participant_id)
    row: Dict[str, Any] = {
        "offline_gt_referee_gate_id": referee_gate_id,
        "offline_gt_online_target_matches": referee_gate_id == gate_id,
    }
    if gate is None:
        return row
    gt = gt_relative_pose(
        state.position,
        float(state.rotation_rpy_deg[2]),
        gate.center,
        gate.normal_vector,
    )
    gt_yaw = wrap_plane_yaw_deg(gt["gate_yaw_deg"])
    est_yaw = None if pose is None else _finite(pose.gate_plane_yaw_deg)
    forward_m = float(gt["forward_z"])
    lateral_m = float(gt["lateral_x"])
    expected_center_x = (
        lateral_m / forward_m / math.tan(math.radians(DEFAULT_INTRINSICS.fov_deg) / 2.0)
        if forward_m > 0.05 else None
    )
    row.update({
        "offline_gt_gate_yaw_deg": gt_yaw,
        "offline_gt_orientation": orientation_class(gt_yaw),
        "offline_gt_vehicle_position": [float(v) for v in state.position],
        "offline_gt_vehicle_yaw_deg": float(state.rotation_rpy_deg[2]),
        "offline_gt_forward_m": forward_m,
        "offline_gt_lateral_m": lateral_m,
        "offline_gt_expected_center_x": expected_center_x,
        "offline_gt_center_error": (
            None if pose is None or expected_center_x is None
            else abs(float(pose.center_x) - expected_center_x)
        ),
        "offline_gt_yaw_error_deg": (
            None if est_yaw is None else plane_angle_distance_deg(est_yaw, gt_yaw)
        ),
        "offline_gt_wrong_sign": (
            None if est_yaw is None or abs(gt_yaw) < 8.0 or abs(est_yaw) < 3.0
            else math.copysign(1.0, est_yaw) != math.copysign(1.0, gt_yaw)
        ),
    })
    return row


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    return round(float(np.percentile(np.asarray(values, dtype=float), percentile)), 3)


def summarize(online_rows: Sequence[Mapping[str, Any]], gt_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(online_rows)
    detected = [row for row in online_rows if row.get("detection_present")]
    corners = [row for row in online_rows if row.get("corners_present")]
    orient = [row for row in online_rows if row.get("orientation_present")]
    pnp = [row for row in online_rows if row.get("metric_pose_present")]
    multi = [
        row for row in online_rows
        if int(row.get("raw_candidates", 0)) > 1
        and row.get("detection_present")
        and not row.get("detection_predicted")
    ]
    assoc = [float(row["association_error"]) for row in multi
             if row.get("association_error") is not None]
    jumps = [float(row["yaw_jump_deg"]) for row in online_rows
             if row.get("yaw_jump_deg") is not None]
    yaw_errors = [float(row["offline_gt_yaw_error_deg"]) for row in gt_rows
                  if row.get("offline_gt_yaw_error_deg") is not None]
    sign_rows = [row for row in gt_rows if row.get("offline_gt_wrong_sign") is not None]
    strong_sign_rows = [
        row for row in sign_rows
        if abs(float(row.get("offline_gt_gate_yaw_deg", 0.0))) >= 15.0
    ]
    target_rows = [row for row in gt_rows if row.get("offline_gt_online_target_matches") is not None]
    in_view = [
        row for row in online_rows
        if row.get("beacon_present")
        and row.get("beacon_range_m") is not None
        and float(row["beacon_range_m"]) <= 6.0
        and row.get("beacon_bearing_deg") is not None
        and abs(float(row["beacon_bearing_deg"])) <= 45.0
    ]
    center_rows = [
        row for row in gt_rows
        if row.get("offline_gt_online_target_matches") is True
        and row.get("offline_gt_expected_center_x") is not None
        and abs(float(row["offline_gt_expected_center_x"])) <= 1.0
        and row.get("offline_gt_center_error") is not None
    ]
    mismatch_runs: List[int] = []
    mismatch_run = 0
    for row in target_rows:
        if not bool(row["offline_gt_online_target_matches"]):
            mismatch_run += 1
        elif mismatch_run:
            mismatch_runs.append(mismatch_run)
            mismatch_run = 0
    if mismatch_run:
        mismatch_runs.append(mismatch_run)
    target_changes = sum(
        online_rows[index].get("expected_beacon_id") != online_rows[index - 1].get("expected_beacon_id")
        for index in range(1, total)
    )
    return {
        "frames": total,
        "detection_availability": round(len(detected) / total, 4) if total else 0.0,
        "valid_four_corner_availability": round(len(corners) / total, 4) if total else 0.0,
        "orientation_availability": round(len(orient) / total, 4) if total else 0.0,
        "in_view_frames": len(in_view),
        "in_view_detection_availability": (
            round(sum(bool(row.get("detection_present")) for row in in_view) / len(in_view), 4)
            if in_view else None
        ),
        "in_view_four_corner_availability": (
            round(sum(bool(row.get("corners_present")) for row in in_view) / len(in_view), 4)
            if in_view else None
        ),
        "metric_pnp_availability": round(len(pnp) / total, 4) if total else 0.0,
        "multi_candidate_frames": len(multi),
        "multi_candidate_association_within_0_25": (
            round(sum(value <= 0.25 for value in assoc) / len(assoc), 4) if assoc else None
        ),
        "target_switches_online": target_changes,
        "target_match_rate_vs_offline_referee": (
            round(sum(bool(row["offline_gt_online_target_matches"]) for row in target_rows)
                  / len(target_rows), 4) if target_rows else None
        ),
        "offline_center_match_within_0_25": (
            round(sum(float(row["offline_gt_center_error"]) <= 0.25 for row in center_rows)
                  / len(center_rows), 4) if center_rows else None
        ),
        "offline_median_center_error": (
            round(statistics.median(float(row["offline_gt_center_error"])
                                    for row in center_rows), 4)
            if center_rows else None
        ),
        "max_target_switch_mismatch_run_frames": max(mismatch_runs, default=0),
        "median_orientation_error_deg": (
            round(statistics.median(yaw_errors), 3) if yaw_errors else None
        ),
        "p90_orientation_error_deg": _percentile(yaw_errors, 90),
        "gross_orientation_error_rate_gt_30deg": (
            round(sum(value > 30.0 for value in yaw_errors) / len(yaw_errors), 4)
            if yaw_errors else None
        ),
        "wrong_sign_rate": (
            round(sum(bool(row["offline_gt_wrong_sign"]) for row in sign_rows) / len(sign_rows), 4)
            if sign_rows else None
        ),
        "wrong_sign_rate_gt_abs_15deg": (
            round(sum(bool(row["offline_gt_wrong_sign"]) for row in strong_sign_rows)
                  / len(strong_sign_rows), 4) if strong_sign_rows else None
        ),
        "yaw_jump_rate_gt_20deg": (
            round(sum(value > 20.0 for value in jumps) / len(jumps), 4) if jumps else None
        ),
        "p90_frame_to_frame_yaw_change_deg": _percentile(jumps, 90),
        "failure_counts": {
            "no_detection": total - len(detected),
            "detection_without_four_corners": len(detected) - len(corners),
            "four_corners_without_metric_pnp": len(corners) - len(pnp),
            "large_yaw_jump": sum(value > 20.0 for value in jumps),
            "gross_orientation_error": sum(value > 30.0 for value in yaw_errors),
            "wrong_orientation_sign": sum(bool(row["offline_gt_wrong_sign"]) for row in sign_rows),
        },
        "ground_truth_separate_and_offline_only": bool(gt_rows),
    }


def _unique_output_dir(base: Optional[str | Path], track_name: str, seed: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    requested = Path(base) if base else Path("results/debug/gate_pose_visual")
    candidate = requested / f"{stamp}_{track_name}_seed{seed}"
    suffix = 1
    while candidate.exists():
        candidate = requested / f"{stamp}_{track_name}_seed{seed}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _aperture_size(config) -> tuple[float, float]:
    size = getattr(config.track, "gate_inner_size_m", None)
    if size is None:
        return 1.5, 1.5
    return float(size[0]), float(size[1])


def _handle_key(key: int, state: ViewState) -> str:
    low = key & 0xFF
    if low in (ord("q"), 27):
        state.quit = True
        return "quit"
    if low == ord(" "):
        state.paused = not state.paused
        return "toggle_pause"
    if low in (ord("n"), ord("N")) or key in (2555904, 65363):
        state.paused = True
        return "next"
    if low in (ord("s"), ord("S")):
        return "screenshot"
    return "none"


def run_track(
    track_name: str = "horseshoe_bay",
    *,
    seed: int = 67000,
    max_steps: int = 1200,
    warmup_steps: int = 10,
    show_window: bool = True,
    start_paused: bool = False,
    video: bool = False,
    offline_ground_truth: bool = False,
    out_root: Optional[str | Path] = None,
) -> Dict[str, Any]:
    import cv2

    from marine_race_arena.config.benchmark_tasks import BENCHMARK_TASK_CLEAN_GATE
    from marine_race_arena.controllers.official_baselines import RuleGateCenterThenCommitController
    from marine_race_arena.learning.episode import RaceEpisode
    from marine_race_arena.scripts.run_marine_race import _mission_info

    track_path = TRACKS.get(track_name, track_name)
    label = Path(track_path).stem.replace("marine_race_", "")
    out = _unique_output_dir(out_root, label, seed)
    screenshots = out / "screenshots"
    screenshots.mkdir()
    online_file = (out / "online_onboard.jsonl").open("w", encoding="utf-8")
    gt_file = ((out / "offline_ground_truth.jsonl").open("w", encoding="utf-8")
               if offline_ground_truth else None)

    episode = RaceEpisode(
        track_path,
        seed=seed,
        dt=0.1,
        adapter="holoocean",
        allow_fallback=False,
        headless=not show_window,
        max_steps=max_steps + warmup_steps,
        official=True,
        current_profile="none",
        benchmark_task=BENCHMARK_TASK_CLEAN_GATE,
    )
    expert = RuleGateCenterThenCommitController()
    pose_tracker = GatePoseTracker(alpha=0.38, max_age_steps=4)
    online_rows: List[Dict[str, Any]] = []
    gt_rows: List[Dict[str, Any]] = []
    writer = None
    state = ViewState(paused=start_paused)
    window = "Gate Pose Visual Validation -- ONBOARD ONLY"
    previous_expected: Optional[str] = None
    previous_yaw: Optional[float] = None
    previous_action = np.zeros(4, dtype=np.float32)
    saved_events: set[str] = set()

    try:
        raw = episode.reset(seed=seed)
        zero = {"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}
        # Runtime-spawned HoloOcean props need a few ticks before they become
        # visible to FrontCamera.  The rover remains stationary and no frame is
        # scored during this renderer warm-up.
        for _ in range(max(0, int(warmup_steps))):
            raw = episode.step(zero).observation

        expert.reset(dict(_mission_info(episode.context.config, episode.participant_id)))
        onboard_tracker = OnboardLocalTransition27dContextTracker(
            total_beacons=len(episode.context.config.track.gate_sequence),
            laps=int(episode.context.config.race.laps),
        )
        onboard_tracker.reset(raw)
        gate_width_m, gate_height_m = _aperture_size(episode.context.config)
        if show_window:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, WIN_W, WIN_H)
        if video:
            writer = cv2.VideoWriter(
                str(out / "gate_pose_debug.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (WIN_W, WIN_H),
            )

        for step in range(max_steps):
            command = expert.step(dict(raw))
            context = onboard_tracker.context(
                raw, dt=episode.dt, prev_action=previous_action.tolist()
            )
            encoded = encode_observation_local_transition_27d(raw, context)
            if encoded.shape != (OBS_DIM_LOCAL_TRANSITION_27D,) or not np.isfinite(encoded).all():
                raise RuntimeError("27-D diagnostic observation is invalid")
            tracker = onboard_tracker.tracker
            expected = tracker.expected_beacon_id
            if previous_expected is not None and expected != previous_expected:
                pose_tracker.reset()
                previous_yaw = None
            visual = context.visual_target
            sensors = raw.get("sensors") if isinstance(raw.get("sensors"), Mapping) else {}
            image = sensors.get("FrontCamera")
            raw_candidates = vision_targets_from_camera(image) if image is not None else []
            pose = onboard_tracker.last_gate_pose
            online = _online_frame(
                label, step, raw, tracker, visual, pose, len(raw_candidates), previous_yaw
            )
            online_row = asdict(online)
            online_row.update({
                "observation_contract": OBS_ENCODING_VERSION_LOCAL_TRANSITION_27D,
                "observation_dim": OBS_DIM_LOCAL_TRANSITION_27D,
                "observation_finite": True,
                "observation_27d": [float(value) for value in encoded],
                "gate_orientation_present_feature": float(encoded[24]),
                "gate_yaw_sin_feature": float(encoded[25]),
                "gate_yaw_cos_feature": float(encoded[26]),
            })
            online_rows.append(online_row)
            online_file.write(json.dumps(online_row, ensure_ascii=False) + "\n")
            online_file.flush()

            gt_row = None
            if offline_ground_truth:
                # This call is the only bridge to simulator truth.  Its result
                # is logged separately after the online estimate is complete.
                gt_row = {
                    "track": label,
                    "step": step,
                    "time_s": online.time_s,
                    **_offline_ground_truth(episode, expected, pose),
                }
                gt_rows.append(gt_row)
                assert gt_file is not None
                gt_file.write(json.dumps(gt_row, ensure_ascii=False) + "\n")
                gt_file.flush()

            frame = _compose(_camera_bgr(raw), visual, pose, online, state)
            event_names = []
            if online.corners_present:
                event_names.append(f"corners_{expected}")
            if online.detection_present and not online.corners_present:
                event_names.append("detection_without_corners")
            if online.raw_candidates > 1 and online.corners_present:
                event_names.append("multi_candidate_lock")
            if previous_expected is not None and expected != previous_expected:
                event_names.append(f"target_switch_{expected}")
            if online.orientation in ("LEFT", "RIGHT") and online.corners_present:
                event_names.append(f"orientation_{online.orientation.lower()}")
            if gt_row is not None and (gt_row.get("offline_gt_yaw_error_deg") or 0.0) > 30.0:
                event_names.append("offline_gt_gross_yaw_error")
            for event in event_names:
                if event not in saved_events:
                    cv2.imwrite(str(screenshots / f"{step:05d}_{event}.png"), frame)
                    saved_events.add(event)
            if writer is not None:
                writer.write(frame)

            if show_window:
                cv2.imshow(window, frame)
                while True:
                    wait_ms = 0 if state.paused else (100 if state.realtime else 1)
                    action = _handle_key(cv2.waitKeyEx(wait_ms), state)
                    if action == "screenshot":
                        path = screenshots / f"{step:05d}_manual.png"
                        cv2.imwrite(str(path), frame)
                        print(f"[gate-pose] screenshot: {path}", flush=True)
                        continue
                    if action == "toggle_pause" and state.paused:
                        continue
                    if action == "none" and state.paused:
                        continue
                    break
                if state.quit:
                    break

            previous_expected = expected
            if pose is not None and pose.gate_plane_yaw_deg is not None:
                previous_yaw = float(pose.gate_plane_yaw_deg)
            outcome = episode.step(command)
            raw = outcome.observation
            previous_action = np.asarray(
                [float(command.get(axis, 0.0)) for axis in ("surge", "sway", "heave", "yaw")],
                dtype=np.float32,
            )
            if outcome.terminated or outcome.truncated:
                break
    finally:
        online_file.close()
        if gt_file is not None:
            gt_file.close()
        if writer is not None:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()
        episode.close()

    metrics = summarize(online_rows, gt_rows)
    summary = {
        "track": label,
        "seed": seed,
        "adapter": "holoocean",
        "controller_inputs": "official onboard observation only",
        "pose_inputs": "FrontCamera pixels + camera-derived tracked ROI",
        "ground_truth_mode": "separate offline scoring only" if offline_ground_truth else "disabled",
        "diagnostic_only": True,
        "include_in_training_dataset": False,
        "observation_contract": OBS_ENCODING_VERSION_LOCAL_TRANSITION_27D,
        "observation_dim": OBS_DIM_LOCAL_TRANSITION_27D,
        "metrics": metrics,
        "output_dir": str(out.resolve()),
        "screenshots": [str(path.resolve()) for path in sorted(screenshots.glob("*.png"))],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[gate-pose] " + json.dumps(summary, indent=2), flush=True)
    return summary


def validate_all(args) -> Dict[str, Any]:
    summaries = []
    for index, name in enumerate(TRACKS):
        summaries.append(run_track(
            name,
            seed=args.seed + index,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            show_window=False,
            video=False,
            offline_ground_truth=True,
            out_root=args.out,
        ))
    combined = {
        "generated_local": datetime.now().isoformat(timespec="seconds"),
        "tracks": summaries,
        "training_started": False,
        "note": "Ground truth was logged separately and never entered an online estimate.",
    }
    base = Path(args.out or "results/debug/gate_pose_visual")
    base.mkdir(parents=True, exist_ok=True)
    report = base / ("validation_all_%s.json" % datetime.now().strftime("%Y%m%d_%H%M%S"))
    report.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    combined["report"] = str(report.resolve())
    print("[gate-pose-validation] " + json.dumps(combined, indent=2), flush=True)
    return combined


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visual gate-pose validation (onboard only)")
    parser.add_argument("--track", default="horseshoe_bay",
                        help="horseshoe_bay, vertical_serpent, mixed_endurance, or track JSON")
    parser.add_argument("--seed", type=int, default=67000)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--offline-ground-truth", action="store_true",
                        help="logga GT in un file separato, solo per scoring offline")
    parser.add_argument("--validate-all", action="store_true",
                        help="esegue headless i tre circuiti con scoring GT separato")
    parser.add_argument("--visual-smoke-all", action="store_true",
                        help="esegue un episodio diagnostico visibile per ciascun circuito")
    parser.add_argument("--out", default=None,
                        help="radice output; ogni esecuzione crea comunque una directory nuova")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.visual_smoke_all:
        summaries = []
        for index, name in enumerate(TRACKS):
            summaries.append(run_track(
                name,
                seed=args.seed + index,
                max_steps=args.max_steps,
                warmup_steps=args.warmup_steps,
                show_window=True,
                start_paused=args.paused,
                video=args.video,
                offline_ground_truth=True,
                out_root=args.out or "artifacts_gen2/visual_collection_smoke",
            ))
        report = {
            "generated_local": datetime.now().isoformat(timespec="seconds"),
            "diagnostic_only": True,
            "include_in_training_dataset": False,
            "tracks": summaries,
            "training_started": False,
        }
        base = Path(args.out or "artifacts_gen2/visual_collection_smoke")
        base.mkdir(parents=True, exist_ok=True)
        report_path = base / "visual_smoke_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("[visual-smoke-27d] " + json.dumps({"report": str(report_path.resolve())}, indent=2), flush=True)
    elif args.validate_all:
        validate_all(args)
    else:
        run_track(
            args.track,
            seed=args.seed,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            show_window=not args.headless,
            start_paused=args.paused,
            video=args.video,
            offline_ground_truth=args.offline_ground_truth,
            out_root=args.out,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
