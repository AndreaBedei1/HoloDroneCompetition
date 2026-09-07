"""No-learning HoloOcean preflight for the 38-D gate-yaw observation.

The immutable 35-D parent drives the rover.  The new observation is recorded
beside it but cannot affect actions, which isolates perception validation from
learning and from policy changes.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from marine_race_arena.config.benchmark_tasks import BENCHMARK_TASK_CLEAN_GATE
from marine_race_arena.controllers.gate_pose import plane_angle_distance_deg
from marine_race_arena.controllers.vision import vision_targets_from_camera
from marine_race_arena.learning.config import ACTION_AXES, ACTION_DIM
from marine_race_arena.learning.config_local_transition_gate_yaw import (
    FEATURE_NAMES_LOCAL_TRANSITION_GATE_YAW,
)
from marine_race_arena.learning.episode import RaceEpisode
from marine_race_arena.learning.gen2 import track_fragments as tf
from marine_race_arena.learning.gen2.fog_contract import verify_fog_sources
from marine_race_arena.learning.observation_encoder import _select_beacon_packet
from marine_race_arena.learning.observation_encoder_local_transition_gate_yaw import (
    encode_observation_local_transition_gate_yaw,
)
from marine_race_arena.learning.tracker_context_local_transition_gate_yaw import (
    OnboardLocalTransitionGateYawContextTracker,
)


F = {
    name: index
    for index, name in enumerate(FEATURE_NAMES_LOCAL_TRANSITION_GATE_YAW)
}


def _finite(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _image_stats(image) -> Optional[Dict[str, Any]]:
    if image is None:
        return None
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        return None
    bgr = array[..., :3].astype(np.float64)
    means = np.mean(bgr, axis=(0, 1))
    return {
        "shape": list(array.shape),
        "mean_bgr": [round(float(v), 3) for v in means],
        "std_luma": round(float(np.std(np.mean(bgr, axis=2))), 3),
        "blue_minus_red": round(float(means[0] - means[2]), 3),
    }


def _save_frame(image, path: Path) -> Optional[str]:
    if image is None:
        return None
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.asarray(image)
    # HoloOcean FrontCamera is BGRA/BGR, matching OpenCV's channel order.
    if frame.ndim == 3 and frame.shape[2] >= 3:
        frame = frame[..., :3]
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"failed to save FrontCamera frame {path}")
    return str(path)


def _summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    count = len(rows)
    vision = [row for row in rows if row["vision_present"]]
    fresh_vision = [row for row in vision if not row["vision_predicted"]]
    associated_vision = [row for row in fresh_vision if row["visual_locked"]]
    orientation = [row for row in rows if row["gate_orientation_present"]]
    orientation_on_vision = [
        row for row in vision if row["gate_orientation_present"]
    ]
    fresh_orientation = [row for row in orientation if not row["pose_predicted"]]
    jumps = [row["yaw_jump_deg"] for row in rows if row["yaw_jump_deg"] is not None]
    mismatches = [
        row["beacon_visual_bearing_error_deg"]
        for row in associated_vision
        if row["beacon_visual_bearing_error_deg"] is not None
    ]
    return {
        "steps": count,
        "vision_availability": round(len(vision) / max(1, count), 4),
        "fresh_vision_availability": round(len(fresh_vision) / max(1, count), 4),
        "associated_vision_availability": round(
            len(associated_vision) / max(1, count), 4
        ),
        "orientation_availability": round(len(orientation) / max(1, count), 4),
        "orientation_given_vision": round(
            len(orientation_on_vision) / max(1, len(vision)), 4
        ),
        "fresh_orientation_availability": round(
            len(fresh_orientation) / max(1, count), 4
        ),
        "expected_target_switches": sum(bool(row["target_switched"]) for row in rows),
        "max_yaw_jump_deg": round(max(jumps), 3) if jumps else None,
        "p95_yaw_jump_deg": (
            round(float(np.percentile(jumps, 95)), 3) if jumps else None
        ),
        "max_fresh_beacon_visual_bearing_error_deg": (
            round(max(mismatches), 3) if mismatches else None
        ),
        # The online selector rejects fresh associations beyond 32 degrees.
        "next_gate_steal_suspects": sum(value > 32.001 for value in mismatches),
        "unconfirmed_proposal_bearing_outliers": sum(
            bool(row["beacon_visual_bearing_error_deg"] is not None)
            and row["beacon_visual_bearing_error_deg"] > 32.001
            for row in fresh_vision
            if not row["visual_locked"]
        ),
        "nan_or_inf_rows": sum(not bool(row["all_38_finite"]) for row in rows),
        "availability_encoding_disagreements": sum(
            not bool(row["orientation_encoding_agrees"]) for row in rows
        ),
        "stale_orientation_max_age": max(
            [int(row["pose_age_frames"]) for row in orientation] or [0]
        ),
        "pass": bool(
            count
            and not any(not bool(row["all_38_finite"]) for row in rows)
            and not any(
                not bool(row["orientation_encoding_agrees"]) for row in rows
            )
            and not any(value > 32.001 for value in mismatches)
            and len(orientation) > 0
        ),
    }


def audit_track(
    track: str,
    *,
    parent_path: str | Path,
    output_dir: str | Path,
    seed: int,
    steps: int,
) -> Dict[str, Any]:
    from sb3_contrib import RecurrentPPO

    from marine_race_arena.learning.gen2.recurrent_policy import (
        Gen2RecurrentController,
    )

    track_path = tf.track_path(track)
    out = Path(output_dir) / track
    out.mkdir(parents=True, exist_ok=True)
    episode = RaceEpisode(
        str(track_path),
        seed=int(seed),
        dt=0.1,
        adapter="holoocean",
        allow_fallback=False,
        max_steps=int(steps) + 5,
        official=True,
        current_profile="none",
        benchmark_task=BENCHMARK_TASK_CLEAN_GATE,
    )
    controller = Gen2RecurrentController(
        RecurrentPPO.load(str(parent_path), device="cpu"), deterministic=True
    )
    rows: List[Dict[str, Any]] = []
    screenshots: List[str] = []
    started = time.perf_counter()
    try:
        raw = episode.reset(seed=int(seed))
        config = episode.context.config
        context_source = OnboardLocalTransitionGateYawContextTracker(
            total_beacons=max(1, len(config.track.gate_sequence)),
            laps=max(1, int(config.race.laps)),
        )
        context_source.reset(raw)
        controller.reset()
        previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
        previous_expected: Optional[str] = None
        previous_yaw: Optional[float] = None

        log_path = out / "onboard_gate_yaw.jsonl"
        with log_path.open("w", encoding="utf-8") as stream:
            for index in range(int(steps)):
                context = context_source.context(
                    raw, dt=0.1, prev_action=previous_action.tolist()
                )
                encoded = encode_observation_local_transition_gate_yaw(raw, context)
                visual = context.visual_target
                pose = context_source.last_gate_pose
                expected = str(context.expected_beacon_id or "")
                packet = _select_beacon_packet(raw.get("beacons") or [], expected)
                bearing = None if packet is None else _finite(packet.get("bearing_deg"))
                implied = (
                    None
                    if visual is None
                    else -math.degrees(math.atan(float(visual.center_x)))
                )
                mismatch = (
                    None
                    if bearing is None or implied is None
                    else abs(implied - bearing)
                )
                yaw = None if pose is None else _finite(pose.gate_plane_yaw_deg)
                switched = bool(
                    previous_expected is not None and expected != previous_expected
                )
                if switched:
                    # Yaw discontinuity across two different planes is not a
                    # tracker jump.  Reset the diagnostic at identity changes.
                    previous_yaw = None
                jump = (
                    None
                    if yaw is None or previous_yaw is None
                    else plane_angle_distance_deg(yaw, previous_yaw)
                )
                present = bool(encoded[F["gate_orientation_present"]] > 0.5)
                suffix = encoded[35:]
                suffix_valid = (
                    bool(np.allclose(suffix, 0.0, atol=0.0, rtol=0.0))
                    if not present
                    else bool(
                        abs(float(suffix[1] ** 2 + suffix[2] ** 2) - 1.0)
                        <= 2e-6
                    )
                )
                sensors = raw.get("sensors") if isinstance(raw.get("sensors"), Mapping) else {}
                image = sensors.get("FrontCamera")
                raw_candidates = (
                    len(vision_targets_from_camera(image)) if image is not None else 0
                )
                visual_locked = bool(
                    (context_source.tracker.diagnostics().get("visual_track") or {}).get(
                        "locked"
                    )
                )
                row = {
                    "step": index,
                    "local_time_s": _finite(raw.get("local_time_s")),
                    "expected_beacon_id": expected,
                    "target_switched": switched,
                    "center_x": float(encoded[F["vision_center_x"]]),
                    "center_y": float(encoded[F["vision_center_y"]]),
                    "vision_present": bool(encoded[F["vision_present"]] > 0.5),
                    "vision_predicted": bool(
                        visual is not None and getattr(visual, "predicted", False)
                    ),
                    "visual_locked": visual_locked,
                    "raw_gate_candidates": raw_candidates,
                    "beacon_present": packet is not None,
                    "beacon_bearing_deg": bearing,
                    "beacon_elevation_deg": (
                        None if packet is None else _finite(packet.get("elevation_deg"))
                    ),
                    "beacon_range_m": (
                        None if packet is None else _finite(packet.get("range_m"))
                    ),
                    "beacon_visual_bearing_error_deg": mismatch,
                    "gate_orientation_present": present,
                    "gate_yaw_deg": yaw,
                    "gate_yaw_sin": float(encoded[F["gate_yaw_sin"]]),
                    "gate_yaw_cos": float(encoded[F["gate_yaw_cos"]]),
                    "pose_source": "none" if pose is None else pose.detection_source,
                    "pose_predicted": bool(pose is not None and pose.predicted),
                    "pose_age_frames": context_source.gate_orientation_age_steps,
                    "yaw_jump_deg": jump,
                    "all_38_finite": bool(np.isfinite(encoded).all()),
                    "orientation_encoding_agrees": bool(
                        present == bool(context.gate_orientation_present)
                        and suffix_valid
                    ),
                }
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

                if index in {0, max(0, int(steps) // 2), int(steps) - 1}:
                    saved = _save_frame(image, out / f"frontcamera_{index:04d}.png")
                    if saved:
                        screenshots.append(saved)

                action = controller.act(encoded[:35], first_step=(index == 0))
                step = episode.step(
                    {axis: float(action[i]) for i, axis in enumerate(ACTION_AXES)}
                )
                previous_action = action
                raw = step.observation
                previous_expected = expected
                if yaw is not None:
                    previous_yaw = yaw
                if step.terminated or step.truncated:
                    break

        summary = _summarize(rows)
        summary.update(
            {
                "track": track,
                "track_path": str(track_path),
                "seed": int(seed),
                "parent_drove_actions": True,
                "learning_updates": 0,
                "fog": verify_fog_sources([track_path])["sources"][0]["water_fog"],
                "first_image_stats": _image_stats(
                    (raw.get("sensors") or {}).get("FrontCamera")
                    if isinstance(raw, Mapping)
                    else None
                ),
                "screenshots": screenshots,
                "online_log": str(log_path),
                "wall_time_s": round(time.perf_counter() - started, 2),
            }
        )
        (out / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary
    finally:
        episode.close()


def run_preflight(
    *,
    parent_path: str | Path,
    output_dir: str | Path,
    steps: int = 300,
    seed: int = 8300,
) -> Dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [tf.track_path(track) for track in tf.OFFICIAL_TRACKS]
    fog = verify_fog_sources(paths)
    summaries = []
    for offset, track in enumerate(tf.OFFICIAL_TRACKS):
        print(f"[preflight] {track}: fog={fog['sources'][offset]['water_fog']}", flush=True)
        summary = audit_track(
            track,
            parent_path=parent_path,
            output_dir=out,
            seed=int(seed) + offset,
            steps=int(steps),
        )
        summaries.append(summary)
        print(f"[preflight] {track}: {json.dumps(summary, sort_keys=True)}", flush=True)
    report = {
        "schema_version": "gen2_gate_yaw_preflight_v1",
        "parent": str(parent_path),
        "observation_contract": "onboard_local_transition_gate_yaw_v2",
        "learning_updates": 0,
        "fog": fog,
        "tracks": summaries,
        "pass": bool(all(row["pass"] for row in summaries)),
    }
    report_path = out / "preflight_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="No-learning 38-D HoloOcean preflight")
    parser.add_argument(
        "--parent",
        default="results/rl/gen2/best/best_completion_policy.zip",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=8300)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_preflight(
        parent_path=args.parent,
        output_dir=args.out,
        steps=args.steps,
        seed=args.seed,
    )
    print(json.dumps({"pass": report["pass"], "report": report["report_path"]}, indent=2))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
