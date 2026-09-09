"""Isolated deterministic full-track evaluation with bounded watchdogs.

The parent process launches exactly one fresh Python child for each
controller/track pair.  A child owns one ``RaceEpisode`` and therefore at most
one HoloOcean process.  Progress is flushed while the episode runs, allowing
the parent to distinguish a slow but advancing episode from a dead simulator.
This utility is diagnostic only; it does not alter the 27-D controller input.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from marine_race_arena.config.benchmark_tasks import BENCHMARK_TASK_CLEAN_GATE
from marine_race_arena.controllers.official_baselines import RuleGateCenterThenCommitController
from marine_race_arena.learning.config import ACTION_AXES, ACTION_DIM
from marine_race_arena.learning.episode import RaceEpisode
from marine_race_arena.learning.gen2 import track_fragments as tf
from marine_race_arena.learning.gen2.expert_rollout import (
    _RecordingObservation,
    _mission_info_for,
    action_to_command,
    assert_observation_is_legal,
    command_to_action,
)
from marine_race_arena.learning.gen2.recurrent_policy import Gen2RecurrentController
from marine_race_arena.learning.tracker_context_local_transition_27d import (
    OnboardLocalTransition27dContextTracker,
)
from marine_race_arena.learning.observation_encoder_local_transition_27d import (
    encode_observation_local_transition_27d,
)
from marine_race_arena.learning.tracker_context_local_transition import _select_beacon_packet


DT = 0.1
SURGE_LOW = 0.1
STRONG_ALIGNMENT_BEARING_DEG = 8.0
STRONG_ROTATION_ACTION = 0.25


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    # Windows may briefly deny ReplaceFile while the watchdog has the prior
    # progress file open.  This is a normal reader/writer race, not a run
    # failure; retry briefly before falling back to an in-place write.
    encoded = json.dumps(payload, indent=2, allow_nan=False)
    for _ in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.05)
    path.write_text(encoded, encoding="utf-8")


def _finite_dt(current: float, previous: float | None) -> float:
    if previous is None:
        return DT
    value = float(current) - float(previous)
    return value if 0.0 < value < 2.0 else DT


class _ActionRecorder:
    def __init__(self, controller: Gen2RecurrentController) -> None:
        self.controller = controller
        self.model = controller.model
        self.actions: list[np.ndarray] = []

    def reset(self) -> None:
        self.actions.clear()
        self.controller.reset()

    def act(self, observation: np.ndarray, first_step: bool = False) -> np.ndarray:
        action = np.asarray(self.controller.act(observation, first_step=first_step), dtype=np.float32)
        self.actions.append(action.copy())
        return action


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(np.mean(array)), 6),
        "median": round(float(np.median(array)), 6),
        "p95": round(float(np.percentile(array, 95)), 6),
        "max": round(float(np.max(array)), 6),
    }


def _new_segment(index: int, start_time: float) -> dict[str, Any]:
    return {
        "gate_index": int(index + 1),
        "start_time_s": round(float(start_time), 3),
        "end_time_s": None,
        "elapsed_time_s": 0.0,
        "path_increment_m": 0.0,
        "surge": [],
        "abs_sway": [],
        "abs_yaw": [],
        "time_surge_lt_0_1_s": 0.0,
        "time_strong_alignment_s": 0.0,
        "time_strong_rotation_s": 0.0,
        "steps": 0,
    }


def _finish_segment(segment: dict[str, Any], end_time: float) -> dict[str, Any]:
    segment["end_time_s"] = round(float(end_time), 3)
    segment["elapsed_time_s"] = round(float(end_time) - float(segment["start_time_s"]), 3)
    surge = segment.pop("surge")
    sway = segment.pop("abs_sway")
    yaw = segment.pop("abs_yaw")
    segment["surge_stats"] = _stats(surge)
    segment["mean_abs_sway"] = round(float(np.mean(sway)), 6) if sway else 0.0
    segment["mean_abs_yaw"] = round(float(np.mean(yaw)), 6) if yaw else 0.0
    segment["time_surge_lt_0_1_s"] = round(float(segment["time_surge_lt_0_1_s"]), 3)
    segment["time_strong_alignment_s"] = round(float(segment["time_strong_alignment_s"]), 3)
    segment["time_strong_rotation_s"] = round(float(segment["time_strong_rotation_s"]), 3)
    segment["path_increment_m"] = round(float(segment["path_increment_m"]), 3)
    return segment


def _child(args: argparse.Namespace) -> int:
    track = str(args.track)
    gate_count = len(tf.load_track(track)["track"]["gate_sequence"])
    max_steps = max(2000, gate_count * 900)
    progress_path = Path(args.progress)
    result_path = Path(args.result)
    started_wall = time.monotonic()
    episode = None
    controller = None
    run_status = "TECHNICAL_INVALID"
    segments: list[dict[str, Any]] = []
    try:
        episode = RaceEpisode(
            str(tf.track_path(track)), seed=int(args.seed), dt=DT,
            adapter="holoocean", allow_fallback=False, max_steps=max_steps,
            official=True, current_profile="none", benchmark_task=BENCHMARK_TASK_CLEAN_GATE,
        )
        raw = episode.reset(seed=int(args.seed))
        assert_observation_is_legal(raw)
        tracker = OnboardLocalTransition27dContextTracker(
            total_beacons=max(1, gate_count),
            laps=max(1, int(episode.context.config.race.laps)),
        )
        tracker.reset(raw)
        if args.controller == "rules":
            controller = RuleGateCenterThenCommitController()
            controller.reset(_mission_info_for(episode))
        else:
            from sb3_contrib import RecurrentPPO

            model = RecurrentPPO.load(str(args.checkpoint), device="cpu")
            controller = _ActionRecorder(Gen2RecurrentController(model, deterministic=True))
        previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
        previous_position: np.ndarray | None = None
        previous_time: float | None = None
        current_gate = 0
        segment = _new_segment(0, 0.0)
        path_total = 0.0
        surge_all: list[float] = []
        sway_all: list[float] = []
        yaw_all: list[float] = []
        low_surge_time = alignment_time = rotation_time = 0.0
        steps = 0
        time_s = 0.0
        last_flush = 0

        while steps < max_steps:
            context = tracker.context(raw, dt=DT, prev_action=previous_action.tolist())
            if args.controller == "rules":
                action = command_to_action(controller.step(_RecordingObservation(raw)))
            else:
                encoded = encode_observation_local_transition_27d(raw, context)
                action = controller.act(encoded, first_step=(steps == 0))
            action = np.clip(np.nan_to_num(np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)), -1.0, 1.0)
            beacon = _select_beacon_packet(raw.get("beacons") or [], context.expected_beacon_id)
            bearing_abs = abs(float(beacon.get("bearing_deg", 999.0))) if beacon else 999.0
            step = episode.step(action_to_command(action))
            now = float(step.time_s)
            time_s = now
            dt = _finite_dt(now, previous_time)
            position = np.asarray(step.current_state.position, dtype=np.float64)
            increment = float(np.linalg.norm(position - previous_position)) if previous_position is not None else 0.0
            path_total += increment
            surge, sway, yaw = (float(action[0]), float(action[1]), float(action[3]))
            segment["path_increment_m"] += increment
            segment["surge"].append(surge)
            segment["abs_sway"].append(abs(sway))
            segment["abs_yaw"].append(abs(yaw))
            segment["steps"] += 1
            if surge < SURGE_LOW:
                segment["time_surge_lt_0_1_s"] += dt
                low_surge_time += dt
            if bearing_abs <= STRONG_ALIGNMENT_BEARING_DEG:
                segment["time_strong_alignment_s"] += dt
                alignment_time += dt
            if abs(yaw) >= STRONG_ROTATION_ACTION:
                segment["time_strong_rotation_s"] += dt
                rotation_time += dt
            surge_all.append(surge)
            sway_all.append(abs(sway))
            yaw_all.append(abs(yaw))
            previous_position, previous_action, previous_time = position, action, now
            raw = step.observation
            steps += 1
            crossings = int(episode.referee_progress()["valid_gate_crossings"])
            if crossings > current_gate:
                while current_gate < min(crossings, gate_count):
                    segments.append(_finish_segment(segment, now))
                    current_gate += 1
                    segment = _new_segment(current_gate, now)
            if steps - last_flush >= 10 or crossings > current_gate or step.terminated or step.truncated:
                _write(progress_path, {
                    "status": "RUNNING", "controller": args.controller, "track": track,
                    "seed": int(args.seed), "steps": steps, "time_s": now,
                    "gates": current_gate, "max_steps": max_steps,
                    "wall_elapsed_s": round(time.monotonic() - started_wall, 3),
                    "last_progress_wall": time.time(),
                    "segments_completed": len(segments),
                })
                last_flush = steps
            if step.terminated or step.truncated:
                break

        state = episode.context.referee.states[episode.participant_id]
        status = getattr(state.status, "value", str(state.status))
        if segment["steps"]:
            segments.append(_finish_segment(segment, float(time_s if 'time_s' in locals() else 0.0)))
        finished = bool(status == "FINISHED" and int(state.valid_gate_crossings) == gate_count)
        run_status = "VALID"
        row = {
            "status": status,
            "technical_status": run_status,
            "controller": args.controller,
            "track": track,
            "seed": int(args.seed),
            "gate_count": gate_count,
            "gates_completed": int(state.valid_gate_crossings),
            "finished": finished,
            "steps": steps,
            "completion_time_s": round(float(time_s if 'time_s' in locals() else 0.0), 3),
            "path_length_m": round(path_total, 3),
            "collisions": int(state.collision_events) + int(state.obstacle_collision_events),
            "out_of_bounds_events": int(state.out_of_bounds_events),
            "missed_gate_attempts": int(state.missed_gate_attempts),
            "surge": _stats(surge_all),
            "mean_abs_sway": round(float(np.mean(sway_all)), 6) if sway_all else 0.0,
            "mean_abs_yaw": round(float(np.mean(yaw_all)), 6) if yaw_all else 0.0,
            "time_surge_lt_0_1_s": round(low_surge_time, 3),
            "time_strong_alignment_s": round(alignment_time, 3),
            "time_strong_rotation_s": round(rotation_time, 3),
            "alignment_threshold_bearing_deg": STRONG_ALIGNMENT_BEARING_DEG,
            "rotation_threshold_abs_action": STRONG_ROTATION_ACTION,
            "per_gate": segments,
            "adapter": "holoocean",
            "fallback_used": False,
            "observation_contract": "onboard_local_transition_27d_v1",
            "fog": {"enabled": True, "density": 5.0, "start_distance_m": 1.0, "color_rgb": [0.4, 0.6, 1.0]},
            "wall_elapsed_s": round(time.monotonic() - started_wall, 3),
        }
        _write(result_path, row)
        _write(progress_path, {"status": "FINISHED", "steps": steps, "gates": int(state.valid_gate_crossings), "last_progress_wall": time.time()})
        return 0
    except BaseException as exc:
        _write(result_path, {
            "technical_status": "TECHNICAL_INVALID", "controller": args.controller,
            "track": track, "seed": int(args.seed), "error": repr(exc),
            "wall_elapsed_s": round(time.monotonic() - started_wall, 3),
        })
        _write(progress_path, {"status": "TECHNICAL_INVALID", "error": repr(exc), "last_progress_wall": time.time()})
        return 2
    finally:
        try:
            if controller is not None and hasattr(controller, "close"):
                controller.close()
        except Exception:
            pass
        try:
            if episode is not None:
                episode.close()
        except Exception:
            pass


def _kill_owned(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass


def _run_one(args: argparse.Namespace, controller: str, track: str, attempt: int) -> dict[str, Any]:
    key = f"{controller}_{track}_seed{args.seed}_attempt{attempt}"
    progress = Path(args.out) / f"{key}.progress.json"
    result = Path(args.out) / f"{key}.result.json"
    if bool(args.resume) and result.exists():
        try:
            cached = json.loads(result.read_text(encoding="utf-8"))
            if cached.get("technical_status") == "VALID":
                cached["attempt"] = attempt
                return cached
        except (OSError, json.JSONDecodeError):
            pass
    cmd = [
        sys.executable, "-m", "marine_race_arena.learning.gen2.evaluate_speed_robust",
        "--child", "--controller", controller, "--track", track, "--seed", str(args.seed),
        "--progress", str(progress), "--result", str(result),
    ]
    if controller != "rules":
        cmd.extend(["--checkpoint", str(args.checkpoint)])
    process = subprocess.Popen(cmd, cwd=str(Path.cwd()), env=os.environ.copy())
    started = time.monotonic()
    last_mtime = time.time()
    last_gate = -1
    last_gate_sim_s = 0.0
    technical_reason = None
    while process.poll() is None:
        time.sleep(2.0)
        now = time.monotonic()
        try:
            mtime = progress.stat().st_mtime
            last_mtime = max(last_mtime, mtime)
            progress_data = json.loads(progress.read_text(encoding="utf-8"))
            gate = int(progress_data.get("gates", 0))
            sim_s = float(progress_data.get("time_s", 0.0))
            if gate != last_gate:
                last_gate = gate
                last_gate_sim_s = sim_s
        except FileNotFoundError:
            progress_data = {}
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            progress_data = {}
        if now - started > float(args.wall_timeout_s):
            technical_reason = f"wall_timeout_{args.wall_timeout_s}s"
            break
        if time.time() - last_mtime > float(args.stall_timeout_s):
            technical_reason = f"no_progress_{args.stall_timeout_s}s"
            break
        if (
            progress_data
            and last_gate >= 0
            and float(progress_data.get("time_s", 0.0)) - last_gate_sim_s
            > float(args.gate_stall_timeout_s)
            and not bool(progress_data.get("status") == "FINISHED")
        ):
            technical_reason = f"no_gate_progress_{args.gate_stall_timeout_s}s_sim"
            break
    if technical_reason is not None:
        _kill_owned(process.pid)
        process.wait(timeout=20)
        return {
            "technical_status": "TECHNICAL_INVALID", "controller": controller, "track": track,
            "seed": int(args.seed), "attempt": attempt, "technical_reason": technical_reason,
            "result_file": str(result), "progress_file": str(progress),
        }
    if result.exists():
        row = json.loads(result.read_text(encoding="utf-8"))
        row["attempt"] = attempt
        return row
    return {
        "technical_status": "TECHNICAL_INVALID", "controller": controller, "track": track,
        "seed": int(args.seed), "attempt": attempt, "technical_reason": f"child_exit_{process.returncode}",
        "result_file": str(result), "progress_file": str(progress),
    }


def _orchestrate(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    # A filtered invocation is useful for an isolated fresh retry.  Without
    # this, ``--controller/--track`` silently expanded back to the full 3x3
    # matrix, which could start additional HoloOcean processes unexpectedly.
    controllers = [args.controller] if args.controller else ["retry6_50k", "candidate25k", "rules"]
    tracks = [args.track] if args.track else ["horseshoe_bay", "vertical_serpent", "mixed_endurance"]
    for track in tracks:
        for controller in controllers:
            for attempt in (0, 1):
                row = _run_one(args, controller, track, attempt)
                if row.get("technical_status") == "VALID":
                    break
                if row.get("technical_status") != "TECHNICAL_INVALID":
                    break
            rows.append(row)
            _write(out / "evaluation_incremental.json", {
                "schema_version": "gen2_robust_eval_v1", "seed": int(args.seed),
                "rows": rows, "one_process_per_episode": True,
                "max_one_holoocean": True, "retry_limit": 1,
            })
    _write(out / "evaluation_complete.json", {
        "schema_version": "gen2_robust_eval_v1", "seed": int(args.seed),
        "rows": rows, "one_process_per_episode": True, "max_one_holoocean": True,
        "retry_limit": 1, "adapter": "holoocean", "fallback_used": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--controller", choices=("retry6_50k", "candidate25k", "rules"))
    parser.add_argument("--track", choices=tuple(tf.OFFICIAL_TRACKS))
    parser.add_argument("--seed", type=int, default=66001)
    parser.add_argument("--checkpoint")
    parser.add_argument("--progress")
    parser.add_argument("--result")
    parser.add_argument("--out", default="artifacts_gen2/ppo_27d_speed_campaign_B_20260908_retry6/robust_eval_050000")
    parser.add_argument("--wall-timeout-s", type=float, default=1800.0)
    parser.add_argument("--stall-timeout-s", type=float, default=180.0)
    parser.add_argument("--gate-stall-timeout-s", type=float, default=120.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.child:
        if args.controller != "rules" and not args.checkpoint:
            raise SystemExit("--checkpoint is required for learned child")
        raise SystemExit(_child(args))
    args.checkpoint = args.checkpoint or "artifacts_gen2/ppo_27d_speed_campaign_B_20260908_retry6/checkpoint_050000.zip"
    _orchestrate(args)


if __name__ == "__main__":
    main()
