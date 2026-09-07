"""Rare PPO screenshot monitor with onboard HUD overlay.

Requirements:
- Completely headless (no GUI windows or popups during training);
- Active on a single designated monitor worker (Worker 0);
- Fires rarely: approx 1 event every ~50 completed episodes;
- Max 1-2 snapshots per event:
  * one at mid/long distance (beacon range > 4.5m or small gate);
  * one during close approach (range 1.0-3.5m or gate area >= 0.05);
- Rich onboard HUD overlay (strictly NO Ground Truth):
  * track name, global step, episode;
  * target gate, detected gate;
  * center crosshair, aperture corners, quadrilateral polygon;
  * plane yaw, LEFT / FRONTAL / RIGHT classification;
  * gate_orientation_present flag;
  * beacon bearing and range;
  * fog configuration (density, start distance, color rgb);
  * current applied PPO action [surge, sway, heave, yaw];
- Saves into:
  * results/rl/gen2/<RUN>/monitor_snapshots/snapshot_ep{N}_step{S}_{distance}.png
  * results/rl/gen2/<RUN>/monitor_snapshots/latest_snapshot.png
  * results/rl/gen2/<RUN>/monitor_snapshots/latest_snapshot.json
- Zero extra env.step calls.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from marine_race_arena.learning.gen2.fog_contract import APPROVED_WATER_FOG


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def render_monitor_overlay(
    image: np.ndarray,
    *,
    track: str,
    step: int,
    episode: int,
    target_gate: str,
    detected_gate: str,
    center: Optional[Tuple[float, float]],
    corners_norm: Optional[List[Tuple[float, float]]],
    yaw_deg: Optional[float],
    orientation_class: str,
    orientation_present: bool,
    beacon_bearing_deg: Optional[float],
    beacon_range_m: Optional[float],
    fog_config: Mapping[str, Any],
    action: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Render the official onboard diagnostic HUD overlay onto the camera frame.

    NEVER displays ground truth, world pose, simulator coordinates, or referee secrets.
    """
    if cv2 is None:
        return np.asarray(image)

    frame = np.asarray(image).copy()
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 3:
        pass
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if frame.dtype != np.uint8:
        finite = np.nan_to_num(frame, nan=0.0, posinf=255.0, neginf=0.0)
        if float(np.max(finite)) <= 1.0:
            finite *= 255.0
        frame = np.clip(finite, 0.0, 255.0).astype(np.uint8)

    h, w = frame.shape[:2]

    # Draw gate aperture corners and quadrilateral if available
    if corners_norm is not None and len(corners_norm) == 4:
        pts = []
        for nx, ny in corners_norm:
            px = int(np.clip((nx + 1.0) * 0.5 * w, 0, w - 1))
            py = int(np.clip((ny + 1.0) * 0.5 * h, 0, h - 1))
            pts.append([px, py])
        pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts_arr], isClosed=True, color=(0, 255, 255), thickness=2)
        for p in pts:
            cv2.circle(frame, tuple(p), radius=4, color=(0, 165, 255), thickness=-1)

    # Draw gate center crosshair / circle
    if center is not None:
        cx_norm, cy_norm = center
        cx = int(np.clip((cx_norm + 1.0) * 0.5 * w, 0, w - 1))
        cy = int(np.clip((cy_norm + 1.0) * 0.5 * h, 0, h - 1))
        cv2.circle(frame, (cx, cy), radius=6, color=(0, 255, 0), thickness=2)
        cv2.line(frame, (cx - 10, cy), (cx + 10, cy), color=(0, 255, 0), thickness=1)
        cv2.line(frame, (cx, cy - 10), (cx, cy + 10), color=(0, 255, 0), thickness=1)

    # Semi-transparent HUD header background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 95), (15, 15, 25), -1)
    cv2.rectangle(overlay, (0, h - 45), (w, h), (15, 15, 25), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (240, 240, 240)
    accent_color = (0, 220, 255)
    green_color = (100, 255, 100)
    yellow_color = (100, 220, 255)

    # Line 1: Track, Step, Episode, Target
    cv2.putText(
        frame,
        f"TRACK: {track.upper()}  |  EPISODE: {episode:03d}  |  STEP: {step:05d}",
        (12, 22),
        font,
        0.48,
        text_color,
        1,
        cv2.LINE_AA,
    )

    # Line 2: Target Gate & Detection info
    bearing_str = f"{beacon_bearing_deg:+.1f} deg" if beacon_bearing_deg is not None else "N/A"
    range_str = f"{beacon_range_m:.2f} m" if beacon_range_m is not None else "N/A"
    cv2.putText(
        frame,
        f"TARGET: {target_gate} (BEACON: brg={bearing_str}, rng={range_str})  |  DETECTED: {detected_gate}",
        (12, 45),
        font,
        0.44,
        accent_color,
        1,
        cv2.LINE_AA,
    )

    # Line 3: Orientation & Yaw
    yaw_str = f"{yaw_deg:+.1f} deg ({orientation_class})" if (orientation_present and yaw_deg is not None) else "UNAVAILABLE"
    orient_color = green_color if orientation_present else (120, 120, 200)
    cv2.putText(
        frame,
        f"GATE ORIENTATION: {yaw_str}  [present={int(orientation_present)}]",
        (12, 68),
        font,
        0.44,
        orient_color,
        1,
        cv2.LINE_AA,
    )

    # Line 4: Mandatory Fog Config
    density = fog_config.get("density", 5.0)
    start_dist = fog_config.get("start_distance_m", 1.0)
    color_rgb = fog_config.get("color_rgb", [0.4, 0.6, 1.0])
    cv2.putText(
        frame,
        f"FOG: density={density} start={start_dist}m color={color_rgb} (APPROVED SSoT)",
        (12, 88),
        font,
        0.38,
        (180, 210, 255),
        1,
        cv2.LINE_AA,
    )

    # Bottom footer: Action
    if action is not None and len(action) >= 4:
        act_str = f"SURGE: {action[0]:+.2f}  |  SWAY: {action[1]:+.2f}  |  HEAVE: {action[2]:+.2f}  |  YAW: {action[3]:+.2f}"
    else:
        act_str = "ACTION: [0.0, 0.0, 0.0, 0.0]"
    cv2.putText(
        frame,
        f"PPO COMMAND: {act_str}",
        (12, h - 16),
        font,
        0.44,
        yellow_color,
        1,
        cv2.LINE_AA,
    )

    return frame


class Gen2MonitorSnapshotWrapper:
    """Gym wrapper for Worker 0 that captures rare snapshots during training.

    Completely headless: saves PNG and JSON directly to disk, no display windows.
    """

    def __init__(
        self,
        env,
        *,
        output_dir: str | Path,
        snapshot_interval_episodes: int = 50,
        run_name: str = "run",
    ) -> None:
        self.env = env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_interval_episodes = max(1, int(snapshot_interval_episodes))
        self.run_name = str(run_name)

        self.episode_count = 0
        self.global_step_count = 0
        self.active_event = False
        self.captured_mid_long = False
        self.captured_approach = False
        self._last_snapshot_path: Optional[str] = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, **kwargs):
        res = self.env.reset(**kwargs)
        self.episode_count += 1
        # Trigger an event every snapshot_interval_episodes (or episode 1 for smoke check)
        if self.episode_count == 1 or (self.episode_count % self.snapshot_interval_episodes == 0):
            self.active_event = True
            self.captured_mid_long = False
            self.captured_approach = False
        else:
            self.active_event = False
        return res

    def step(self, action):
        self.global_step_count += 1
        encoded, reward, terminated, truncated, info = self.env.step(action)

        if self.active_event and (not self.captured_mid_long or not self.captured_approach):
            self._maybe_capture(action, info)

        return encoded, reward, terminated, truncated, info

    def _maybe_capture(self, action: Sequence[float], info: Mapping[str, Any]) -> None:
        episode_obj = getattr(self.env, "episode", None) or getattr(self.env, "_episode", None)
        if episode_obj is None:
            return

        ctx_source = getattr(self.env, "_ctx_source", None)
        last_context = getattr(ctx_source, "last_context", None)
        if last_context is None:
            return

        # Get the FrontCamera frame from the episode's latest observation
        raw_obs = getattr(episode_obj, "_last_observation", None)
        if raw_obs is None:
            raw_obs = getattr(episode_obj, "latest_observation", None)
        if raw_obs is None and hasattr(episode_obj, "_require_ctx"):
            try:
                raw_obs = episode_obj._build_observation()
            except Exception:
                raw_obs = None

        if not isinstance(raw_obs, Mapping):
            return
        sensors = raw_obs.get("sensors") or {}
        image = sensors.get("FrontCamera")
        if image is None:
            return

        # Extract onboard features (strictly NO Ground Truth)
        visual_target = last_context.visual_target
        center = (
            (_safe_float(visual_target.center_x), _safe_float(visual_target.center_y))
            if visual_target is not None else None
        )
        area = _safe_float(getattr(visual_target, "area_fraction", 0.0))

        # Beacon info
        beacons = raw_obs.get("beacons") or []
        packet = None
        for b in beacons:
            if isinstance(b, Mapping) and b.get("beacon_id") == last_context.expected_beacon_id:
                packet = b
                break
        if packet is None and beacons:
            packet = beacons[0]

        beacon_bearing = _safe_float(packet.get("bearing_deg")) if packet else None
        beacon_range = _safe_float(packet.get("range_m")) if packet else None

        # Orientation
        orientation_present = bool(last_context.gate_orientation_present)
        yaw_deg = last_context.gate_yaw_deg

        orient_class = "UNKNOWN"
        if orientation_present and yaw_deg is not None:
            if abs(yaw_deg) < 15.0:
                orient_class = "FRONTAL"
            elif yaw_deg > 0:
                orient_class = "LEFT"
            else:
                orient_class = "RIGHT"

        last_gate_pose = getattr(ctx_source, "last_gate_pose", None)
        corners = getattr(last_gate_pose, "corners_normalized", None) if last_gate_pose else None

        # Determine if this step matches mid_long or approach
        capture_type = None
        if not self.captured_mid_long and (beacon_range is None or beacon_range >= 4.5 or area < 0.04):
            capture_type = "mid_distance"
            self.captured_mid_long = True
        elif not self.captured_approach and (area >= 0.04 or (beacon_range is not None and beacon_range <= 3.5)):
            capture_type = "approach"
            self.captured_approach = True

        if capture_type is None:
            return

        track_name = getattr(episode_obj, "track", "marine_race")
        rendered = render_monitor_overlay(
            image,
            track=str(track_name),
            step=self.global_step_count,
            episode=self.episode_count,
            target_gate=str(last_context.expected_beacon_id or "B01"),
            detected_gate="YES" if visual_target is not None else "NO",
            center=center,
            corners_norm=corners,
            yaw_deg=yaw_deg,
            orientation_class=orient_class,
            orientation_present=orientation_present,
            beacon_bearing_deg=beacon_bearing,
            beacon_range_m=beacon_range,
            fog_config=APPROVED_WATER_FOG,
            action=action,
        )

        filename = f"snapshot_ep{self.episode_count:04d}_step{self.global_step_count:06d}_{capture_type}.png"
        filepath = self.output_dir / filename
        if cv2 is not None:
            cv2.imwrite(str(filepath), rendered)
            # Update latest_snapshot.png
            latest_img = self.output_dir / "latest_snapshot.png"
            cv2.imwrite(str(latest_img), rendered)

        # Write latest_snapshot.json
        meta = {
            "step": self.global_step_count,
            "track": str(track_name),
            "episode": self.episode_count,
            "gate": str(last_context.expected_beacon_id or "B01"),
            "capture_type": capture_type,
            "fog_config": dict(APPROVED_WATER_FOG),
            "observation_validity": {
                "beacon_present": packet is not None,
                "vision_present": visual_target is not None,
                "orientation_present": orientation_present,
                "dvl_present": bool(info.get("step_distance_m", 0) >= 0),
            },
            "yaw": float(yaw_deg) if yaw_deg is not None else None,
            "orientation_availability": orientation_present,
            "orientation_class": orient_class,
            "gate_progress": {
                "gate_crossings": int(info.get("gate_crossings", 0)),
                "expected_gate_id": str(info.get("expected_gate_id", "")),
            },
            "safety_events": {
                "collision": bool(info.get("collision_contact_frame", False)),
                "obstacle_collision": bool(info.get("obstacle_collision_frame", False)),
                "out_of_bounds": bool(info.get("out_of_bounds_frame", False)),
                "safety_warning": bool(info.get("safety_warning_frame", False)),
            },
            "latest_image_file": str(filepath.name),
        }
        latest_json = self.output_dir / "latest_snapshot.json"
        latest_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._last_snapshot_path = str(filepath)
