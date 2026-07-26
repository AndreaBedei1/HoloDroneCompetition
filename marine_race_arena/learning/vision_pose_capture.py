"""Offline visual-pose detector validation on real HoloOcean gate images.

Drives the vehicle on the yaw-rotated single-gate test tracks (which place the gate at a
known relative orientation to the camera), grabs each FrontCamera frame, runs the onboard
:func:`gate_pose.estimate_gate_pose` detector, and scores it against the ground-truth relative
pose computed from the track config + the vehicle's ground-truth state.

Two capture modes isolate perception from motion:

* ``stationary`` -- hold a zero command and capture several frames of the fixed oblique gate
  (isolates the detector from any rover movement);
* ``moving`` -- apply a small forward surge and capture one frame per simulator step (checks
  temporal stability). Exactly ONE ``RaceEpisode.step`` advances the sim per recorded frame,
  so frame index == simulator step count.

Gate-plane yaw uses the canonical modulo-180 convention (:func:`gate_pose.canonical_gate_plane_yaw`):
error is :func:`gate_pose.plane_angle_distance_deg` (not a raw subtraction) and the coarse
left/right call is :func:`gate_pose.orientation_class`. The public per-track aperture size is
read from the track config and passed to the detector (some yaw tracks are 2.0 m, the official
gates 1.5 m); modelling the wrong size would scale every distance estimate.

Ground truth is used ONLY to score the detector offline -- never as a controller input. Run
from the marine_race_rl env with real HoloOcean (no fallback), currents disabled.

    python -m marine_race_arena.learning.vision_pose_capture
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from marine_race_arena.config.loader import load_track_config
from marine_race_arena.controllers.gate_pose import (
    DEFAULT_INTRINSICS,
    canonical_gate_plane_yaw,  # noqa: F401  (documented convention entry point)
    estimate_gate_pose,
    orientation_class,
    plane_angle_distance_deg,
    wrap_plane_yaw_deg,
)
from marine_race_arena.learning.episode import RaceEpisode
from marine_race_arena.learning.provenance import git_sha, now_utc, package_versions

TRACKS = {
    0: "marine_race_arena/tracks/tests/single_gate_yaw_0.json",
    25: "marine_race_arena/tracks/tests/single_gate_yaw_25.json",
    45: "marine_race_arena/tracks/tests/single_gate_yaw_45.json",
    -25: "marine_race_arena/tracks/tests/single_gate_yaw_neg25.json",
    -45: "marine_race_arena/tracks/tests/single_gate_yaw_neg45.json",
}
OUT = Path("results/rl_public/visual_pose_v2/vision_pose")
FRONTAL_THRESHOLD_DEG = 8.0


def gt_relative_pose(veh_pos, veh_yaw_deg: float, gate_pos, gate_normal) -> Dict:
    """Ground-truth relative gate pose in the camera frame (offline scoring only).

    ``gate_yaw_deg`` is the oriented relative angle between the vehicle forward direction and
    the gate normal; fold it with :func:`wrap_plane_yaw_deg` to get the canonical unoriented
    plane yaw comparable to the detector's :func:`canonical_gate_plane_yaw`.
    """
    P = np.asarray(veh_pos, dtype=float)
    G = np.asarray(gate_pos, dtype=float)
    psi = math.radians(veh_yaw_deg)
    fwd = np.array([math.cos(psi), math.sin(psi), 0.0])
    right = np.array([math.sin(psi), -math.cos(psi), 0.0])
    d = G - P
    n = np.asarray(gate_normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-9)
    rel_yaw = math.degrees(math.atan2(fwd[0] * n[1] - fwd[1] * n[0], fwd[0] * n[0] + fwd[1] * n[1]))
    return {
        "forward_z": float(np.dot(d, fwd)),
        "lateral_x": float(np.dot(d, right)),
        "gate_yaw_deg": rel_yaw,
        "distance": float(np.linalg.norm(d)),
    }


def _front_camera(obs) -> Optional[np.ndarray]:
    sensors = obs.get("sensors", {}) if isinstance(obs, dict) else {}
    img = sensors.get("FrontCamera")
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[:, :, :3].astype(np.uint8)
    return arr


def _gate_aperture_m(track: str):
    """Public per-track aperture (width_m, height_m); some yaw tracks are 2.0 m, official 1.5 m."""
    cfg = load_track_config(track)
    size = getattr(cfg.gates[0], "inner_size_m", None)
    if size is None:
        return 1.5, 1.5
    return float(size[0]), float(size[1])


def _score_frame(track_yaw: int, mode: str, step: int, adapter_name: str, obs,
                 veh_pos, veh_yaw, gate_pos, gate_normal, gate_w, gate_h) -> Dict:
    gt = gt_relative_pose(veh_pos, veh_yaw, gate_pos, gate_normal)
    gt_yaw_canon = wrap_plane_yaw_deg(gt["gate_yaw_deg"])
    img = _front_camera(obs)
    est = (estimate_gate_pose(img, intr=DEFAULT_INTRINSICS, gate_width_m=gate_w, gate_height_m=gate_h)
           if img is not None else None)
    row = {
        "track_yaw_deg": track_yaw, "mode": mode, "step": step, "adapter": adapter_name,
        "gate_width_m": gate_w, "gate_height_m": gate_h,
        "gt_forward_z": round(gt["forward_z"], 3), "gt_lateral_x": round(gt["lateral_x"], 3),
        "gt_gate_yaw_deg": round(gt["gate_yaw_deg"], 2), "gt_gate_yaw_canonical_deg": round(gt_yaw_canon, 2),
        "gt_orientation": orientation_class(gt_yaw_canon, frontal_threshold_deg=FRONTAL_THRESHOLD_DEG),
        "detected": est is not None,
    }
    if est is not None:
        row["pose_present"] = est.pose_present
        row["detection_source"] = est.detection_source
        row["orientation_hint"] = est.orientation_hint
        if est.translation_camera_m is not None:
            lat, _vert, fwd = est.translation_camera_m
            row["est_forward_z"] = round(float(fwd), 3)
            row["est_lateral_x"] = round(float(lat), 3)
            row["distance_error_m"] = round(abs(float(fwd) - gt["forward_z"]), 3)
            row["lateral_error_m"] = round(abs(float(lat) - gt["lateral_x"]), 3)
        if est.gate_plane_yaw_deg is not None:
            est_yaw = wrap_plane_yaw_deg(float(est.gate_plane_yaw_deg))
            row["est_yaw_deg"] = round(est_yaw, 2)
            # Correct modulo-180 plane-angle error (NOT abs(est - gt)).
            row["yaw_error_deg"] = round(plane_angle_distance_deg(est_yaw, gt_yaw_canon), 2)
            est_cls = orientation_class(est_yaw, frontal_threshold_deg=FRONTAL_THRESHOLD_DEG)
            row["est_orientation"] = est_cls
            row["orientation_ok"] = (est_cls == row["gt_orientation"])
            # Sign accuracy only where the gate is meaningfully oblique.
            if abs(gt_yaw_canon) >= FRONTAL_THRESHOLD_DEG:
                row["yaw_sign_ok"] = (math.copysign(1, est_yaw) == math.copysign(1, gt_yaw_canon))
    return row


def capture_track(track_yaw: int, track: str, *, mode: str = "moving",
                  n_frames: int = 6) -> List[Dict]:
    """Capture ``n_frames`` scored frames on ``track`` in ``stationary`` or ``moving`` mode."""
    rows: List[Dict] = []
    cfg = json.loads(Path(track).read_text(encoding="utf-8"))
    gate = cfg["gates"][0]
    gate_pos, gate_normal = gate["position"], gate["passage_direction"]
    gate_w, gate_h = _gate_aperture_m(track)
    command = ({"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0} if mode == "stationary"
               else {"surge": 0.4, "sway": 0.0, "heave": 0.0, "yaw": 0.0})
    ep = RaceEpisode(track, seed=1600 + track_yaw, dt=0.1, adapter="holoocean",
                     allow_fallback=False, max_steps=200, official=True, current_profile="none")
    try:
        obs = ep.reset()
        for step in range(n_frames):
            state = ep.context.adapter.get_participant_state(ep.participant_id)
            rows.append(_score_frame(
                track_yaw, mode, step, ep.context.adapter.name, obs,
                list(state.position), float(state.rotation_rpy_deg[2]),
                gate_pos, gate_normal, gate_w, gate_h,
            ))
            # Exactly ONE simulator step per recorded frame -> frame index == step count.
            if step < n_frames - 1:
                result = ep.step(command)
                obs = result.observation
                assert ep.step_count == step + 1, (
                    f"capture step misalignment: step_count={ep.step_count} expected={step + 1}")
    finally:
        ep.close()
    return rows


def _median(xs):
    xs = sorted(v for v in xs if v is not None)
    return round(float(np.median(xs)), 3) if xs else None


def _pct(xs, p):
    xs = sorted(v for v in xs if v is not None)
    return round(float(np.percentile(xs, p)), 3) if xs else None


def _summarize(rows: List[Dict], label: str) -> Dict:
    frames = [r for r in rows if "step" in r]
    detected = [r for r in frames if r.get("detected")]
    with_pose = [r for r in detected if r.get("pose_present")]
    yaw_errs = [r.get("yaw_error_deg") for r in with_pose]
    dist_errs = [r.get("distance_error_m") for r in with_pose]
    lat_errs = [r.get("lateral_error_m") for r in with_pose]
    class_ok = [r.get("orientation_ok") for r in with_pose if r.get("orientation_ok") is not None]
    sign_ok = [r.get("yaw_sign_ok") for r in with_pose if r.get("yaw_sign_ok") is not None]
    n = len(frames)
    return {
        "label": label,
        "n_frames": n,
        "detection_rate": round(len(detected) / n, 3) if n else 0.0,
        "pose_availability_rate": round(len(with_pose) / n, 3) if n else 0.0,
        "median_yaw_error_deg": _median(yaw_errs),
        "p90_yaw_error_deg": _pct(yaw_errs, 90),
        "orientation_class_accuracy": round(sum(1 for v in class_ok if v) / len(class_ok), 3) if class_ok else None,
        "yaw_sign_accuracy": round(sum(1 for v in sign_ok if v) / len(sign_ok), 3) if sign_ok else None,
        "median_distance_error_m": _median(dist_errs),
        "p90_distance_error_m": _pct(dist_errs, 90),
        "median_lateral_error_m": _median(lat_errs),
    }


def run(modes=("stationary", "moving"), n_frames: int = 6) -> Dict:
    all_rows: List[Dict] = []
    for mode in modes:
        for yaw, track in TRACKS.items():
            try:
                all_rows.extend(capture_track(yaw, track, mode=mode, n_frames=n_frames))
            except Exception as exc:  # pragma: no cover
                all_rows.append({"track_yaw_deg": yaw, "mode": mode, "error": f"{type(exc).__name__}: {exc}"})
    frames = [r for r in all_rows if "step" in r]
    detected = [r for r in frames if r.get("detected")]
    metrics = {
        "generated_utc": now_utc(), "git_sha": git_sha(),
        "adapter_actual": (detected[0]["adapter"] if detected else (frames[0]["adapter"] if frames else None)),
        "fallback_used": any(r.get("adapter") == "fallback" for r in frames),
        "camera_intrinsics": DEFAULT_INTRINSICS.as_dict(),
        "gate_apertures_m": {str(y): list(_gate_aperture_m(t)) for y, t in TRACKS.items()},
        "yaw_convention": "canonical modulo-180 plane yaw in [-90,90]; error = plane_angle_distance_deg",
        "overall": _summarize(frames, "overall"),
        "by_mode": {mode: _summarize([r for r in frames if r.get("mode") == mode], mode) for mode in modes},
        "note": "Ground truth used only to score the detector offline; never a controller input.",
    }
    return {"metrics": metrics, "rows": all_rows}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    result = run()
    metrics, all_rows = result["metrics"], result["rows"]
    (OUT / "pose_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUT / "metrics_by_condition.csv").write_text(_rows_to_csv(all_rows), encoding="utf-8")
    (OUT / "camera_intrinsics.json").write_text(json.dumps(DEFAULT_INTRINSICS.as_dict(), indent=2), encoding="utf-8")
    (OUT / "detector_config.json").write_text(json.dumps({
        "module": "marine_race_arena.controllers.gate_pose",
        "gate_apertures_m": metrics["gate_apertures_m"],
        "pnp": "SOLVEPNP_IPPE", "yaw_convention": "canonical modulo-180 plane yaw [-90,90]",
        "yaw_error_metric": "plane_angle_distance_deg", "orientation_metric": "orientation_class",
        "sign_disambiguation": "bar-height ratio + reprojection error",
        "fallback": "projective + known-size distance", "packages": package_versions(),
    }, indent=2), encoding="utf-8")
    print("[vision-pose]", json.dumps(metrics, indent=2))
    return 0


def _rows_to_csv(rows: List[Dict]) -> str:
    import csv
    import io

    flat = [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in rows]
    if not flat:
        return ""
    fields = sorted({k for f in flat for k in f})
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields)
    w.writeheader()
    for f in flat:
        w.writerow(f)
    return buf.getvalue()


if __name__ == "__main__":
    raise SystemExit(main())
