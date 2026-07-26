"""Offline visual-pose detector validation on real HoloOcean gate images.

Drives the vehicle a few steps on the yaw-rotated single-gate test tracks (which place the
gate at a known relative orientation to the camera), grabs each FrontCamera frame, runs the
onboard :func:`gate_pose.estimate_gate_pose` detector, and scores it against the ground-truth
relative pose computed from the track config + the vehicle's ground-truth state.

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

from marine_race_arena.controllers.gate_pose import DEFAULT_INTRINSICS, estimate_gate_pose
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


def gt_relative_pose(veh_pos, veh_yaw_deg: float, gate_pos, gate_normal) -> Dict:
    """Ground-truth relative gate pose in the camera frame (offline scoring only)."""
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


def capture_track(track_yaw: int, track: str, *, forward_steps: int = 3) -> List[Dict]:
    rows: List[Dict] = []
    cfg = json.loads(Path(track).read_text(encoding="utf-8"))
    gate = cfg["gates"][0]
    gate_pos, gate_normal = gate["position"], gate["passage_direction"]
    ep = RaceEpisode(track, seed=1600 + track_yaw, dt=0.1, adapter="holoocean",
                     allow_fallback=False, max_steps=200, official=True, current_profile="none")
    try:
        obs = ep.reset()
        for step in range(forward_steps + 1):
            state = ep.context.adapter.get_participant_state(ep.participant_id)
            veh_pos = list(state.position)
            veh_yaw = float(state.rotation_rpy_deg[2])
            gt = gt_relative_pose(veh_pos, veh_yaw, gate_pos, gate_normal)
            img = _front_camera(obs)
            est = estimate_gate_pose(img, intr=DEFAULT_INTRINSICS) if img is not None else None
            row = {"track_yaw_deg": track_yaw, "step": step, "adapter": ep.context.adapter.name,
                   "gt": gt, "detected": est is not None}
            if est is not None:
                row["pose_present"] = est.pose_present
                row["detection_source"] = est.detection_source
                row["orientation_hint"] = est.orientation_hint
                if est.translation_camera_m is not None:
                    lat, vert, fwd = est.translation_camera_m
                    row["est_forward_z"] = round(float(fwd), 3)
                    row["est_lateral_x"] = round(float(lat), 3)
                    row["distance_error_m"] = round(abs(float(fwd) - gt["forward_z"]), 3)
                    row["lateral_error_m"] = round(abs(float(lat) - gt["lateral_x"]), 3)
                if est.gate_plane_yaw_deg is not None:
                    row["est_yaw_deg"] = round(float(est.gate_plane_yaw_deg), 2)
                    row["yaw_error_deg"] = round(abs(float(est.gate_plane_yaw_deg) - gt["gate_yaw_deg"]), 2)
                    row["yaw_sign_ok"] = (abs(gt["gate_yaw_deg"]) < 5.0
                                          or math.copysign(1, est.gate_plane_yaw_deg) == math.copysign(1, gt["gate_yaw_deg"]))
            rows.append(row)
            if step < forward_steps:
                ep.step({"surge": 0.5, "sway": 0.0, "heave": 0.0, "yaw": 0.0})
                obs = ep.step({"surge": 0.5, "sway": 0.0, "heave": 0.0, "yaw": 0.0}).observation
    finally:
        ep.close()
    return rows


def _median(xs):
    xs = sorted(v for v in xs if v is not None)
    return round(float(np.median(xs)), 3) if xs else None


def _pct(xs, p):
    xs = sorted(v for v in xs if v is not None)
    return round(float(np.percentile(xs, p)), 3) if xs else None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict] = []
    for yaw, track in TRACKS.items():
        try:
            all_rows.extend(capture_track(yaw, track))
        except Exception as exc:  # pragma: no cover
            all_rows.append({"track_yaw_deg": yaw, "error": f"{type(exc).__name__}: {exc}"})
    detected = [r for r in all_rows if r.get("detected")]
    with_pose = [r for r in detected if r.get("pose_present")]
    yaw_errs = [r.get("yaw_error_deg") for r in with_pose]
    dist_errs = [r.get("distance_error_m") for r in with_pose]
    lat_errs = [r.get("lateral_error_m") for r in with_pose]
    sign_ok = [r.get("yaw_sign_ok") for r in with_pose if r.get("yaw_sign_ok") is not None]
    n_frames = len([r for r in all_rows if "step" in r])
    metrics = {
        "generated_utc": now_utc(), "git_sha": git_sha(),
        "adapter_actual": (detected[0]["adapter"] if detected else None),
        "fallback_used": any(r.get("adapter") == "fallback" for r in detected),
        "camera_intrinsics": DEFAULT_INTRINSICS.as_dict(),
        "n_frames": n_frames,
        "detection_rate": round(len(detected) / n_frames, 3) if n_frames else 0.0,
        "pose_availability_rate": round(len(with_pose) / n_frames, 3) if n_frames else 0.0,
        "median_yaw_error_deg": _median(yaw_errs),
        "p90_yaw_error_deg": _pct(yaw_errs, 90),
        "yaw_sign_accuracy": round(sum(1 for v in sign_ok if v) / len(sign_ok), 3) if sign_ok else None,
        "median_distance_error_m": _median(dist_errs),
        "p90_distance_error_m": _pct(dist_errs, 90),
        "median_lateral_error_m": _median(lat_errs),
        "note": "Ground truth used only to score the detector offline; never a controller input.",
    }
    (OUT / "pose_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUT / "metrics_by_angle.csv").write_text(_rows_to_csv(all_rows), encoding="utf-8")
    (OUT / "camera_intrinsics.json").write_text(json.dumps(DEFAULT_INTRINSICS.as_dict(), indent=2), encoding="utf-8")
    (OUT / "detector_config.json").write_text(json.dumps({
        "module": "marine_race_arena.controllers.gate_pose", "gate_inner_size_m": 1.5,
        "pnp": "SOLVEPNP_IPPE", "sign_disambiguation": "bar-height ratio + reprojection error",
        "fallback": "projective + known-size distance", "packages": package_versions(),
    }, indent=2), encoding="utf-8")
    print("[vision-pose]", json.dumps(metrics, indent=2))
    return 0


def _rows_to_csv(rows: List[Dict]) -> str:
    import csv
    import io

    flat = []
    for r in rows:
        f = {k: v for k, v in r.items() if not isinstance(v, dict)}
        if isinstance(r.get("gt"), dict):
            for k, v in r["gt"].items():
                f[f"gt_{k}"] = round(v, 3) if isinstance(v, float) else v
        flat.append(f)
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
