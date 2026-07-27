"""Multi-gate diagnostic for the frozen BC-v1 controller (and any onboard controller).

Runs a controller through the real race runner on a multi-gate track and records, per control
step, the onboard local-course-tracker state (phase, expected beacon, filtered range), the
referee's own progress (valid gate crossings, out-of-bounds / collisions / wrong-direction),
and the vehicle position. Position/referee events are ground truth used ONLY for offline
diagnosis and scoring -- they are never fed to the controller (the controller sees only the
official onboard observation, exactly as in a scored race).

From the timeline it classifies the *first* real failure using the taxonomy the task defines
(VISION_NOT_FOUND ... TIME_LIMIT), so a corrective iteration can target the earliest root
cause rather than a downstream symptom.

    python -m marine_race_arena.learning.multigate_diagnostic \
        --track marine_race_arena/tracks/tests/two_gate_straight.json \
        --model results/rl_public/stage1/bc/model/best_model.pt \
        --seeds 1760-1762 --out results/rl/multigate_v1/two_gate --current-profile none
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from marine_race_arena.learning.closed_loop_eval import currents_summary
from marine_race_arena.learning.episode import RaceEpisode
from marine_race_arena.learning.provenance import git_sha, now_utc, sha256_file
from marine_race_arena.participants.controller_loader import ControllerLoader
from marine_race_arena.scripts.run_marine_race import _mission_info

FAILURE_TAXONOMY = (
    "GATE_NOT_ACQUIRED",
    "FAILED_LONG_RANGE_BEACON_FOLLOWING",
    "FAILED_VISUAL_ALIGNMENT",
    "COMMIT_TOO_EARLY",
    "COMMIT_TOO_LATE",
    "FAILED_POST_GATE_FORWARD",
    "TRACKER_NO_ADVANCE",
    "TRACKER_FALSE_ADVANCE",
    "RETURN_TO_PREVIOUS_GATE",
    "NEXT_GATE_NOT_ACQUIRED",
    "NEXT_GATE_TURN_FAILED",
    "DEPTH_DRIFT",
    "COLLISION",
    "OUT_OF_BOUNDS",
    "WRONG_DIRECTION",
    "TIMEOUT",
    "POLICY_NUMERICAL_ERROR",
    "SIMULATOR_ERROR",
    "FINISHED",
)


def _parse_seeds(spec: str) -> List[int]:
    seeds: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def _json_list(value) -> List:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value) if isinstance(value, (list, tuple)) else []


def _tracker_of(controller):
    src = getattr(controller, "_ctx_source", None)
    if src is not None and getattr(src, "tracker", None) is not None:
        return src.tracker
    # Rule / hybrid controllers expose their LocalCourseTracker directly as `.tracker`.
    return getattr(controller, "tracker", None)


def _classify_first_failure(*, finished: bool, end_reason: str, gates: int, expected_gates: int,
                            events: List[Dict], tracker_completed: int) -> Dict:
    """Heuristically label the earliest real failure cause from the referee + tracker timeline."""
    if finished:
        return {"failure": "FINISHED", "detail": "referee status FINISHED; all gates completed"}
    # Referee terminal causes first (these actually ended the race).
    last = events[-1] if events else {}
    if any(e.get("collision_delta") for e in events):
        return {"failure": "COLLISION", "detail": "collision event(s) recorded"}
    if end_reason == "REFEREE_TERMINAL" and last.get("out_of_bounds", 0) > 0:
        return {"failure": "OUT_OF_BOUNDS", "detail": "referee ended the race out of bounds"}
    # Tracker vs referee disagreement (advanced locally without a real crossing).
    if tracker_completed > gates:
        return {"failure": "TRACKER_FALSE_ADVANCE",
                "detail": f"tracker advanced {tracker_completed} but referee validated {gates} gates"}
    if any(e.get("wrong_dir_delta") for e in events):
        return {"failure": "WRONG_DIRECTION",
                "detail": "referee recorded a wrong-direction crossing"}
    if last.get("out_of_bounds", 0) > 0:
        return {"failure": "OUT_OF_BOUNDS",
                "detail": f"{last.get('out_of_bounds')} out-of-bounds events; drifted outside the arena"}
    # Non-terminal end (ran out of time) -> where did progress stall?
    if end_reason in ("TIME_LIMIT", "MAX_STEPS"):
        if gates == 0:
            saw_visual = any(
                e.get("visual_detected")
                or e.get("phase") in ("VISUAL_ALIGN", "COMMIT", "VERIFY_EXIT")
                for e in events
            )
            return {
                "failure": (
                    "FAILED_VISUAL_ALIGNMENT" if saw_visual else "GATE_NOT_ACQUIRED"
                ),
                "detail": "never completed gate 1 within the time limit",
            }
        if gates >= 1 and gates < expected_gates:
            # Reached >=1 gate but stalled before the next: was it turning or not advancing?
            phase_after = [e.get("phase") for e in events if e.get("gates") == gates]
            searched = sum(1 for p in phase_after if p in ("SEARCH", "APPROACH"))
            aligned = sum(1 for p in phase_after if p in ("VISUAL_ALIGN", "COMMIT", "VERIFY_EXIT"))
            if searched > 4 * max(1, aligned):
                return {"failure": "NEXT_GATE_TURN_FAILED",
                        "detail": f"reached gate {gates}; then mostly SEARCH/APPROACH toward the next beacon"}
            if not any(
                e.get("gates") == gates
                and (
                    e.get("visual_detected")
                    or e.get("phase")
                    in ("VISUAL_ALIGN", "COMMIT", "VERIFY_EXIT")
                )
                for e in events
            ):
                return {
                    "failure": "NEXT_GATE_NOT_ACQUIRED",
                    "detail": f"reached gate {gates}; next gate was never detected",
                }
            return {"failure": "FAILED_VISUAL_ALIGNMENT",
                    "detail": f"reached gate {gates}; engaged the next gate but never completed the crossing"}
    return {"failure": "TIMEOUT" if end_reason in ("TIME_LIMIT", "MAX_STEPS") else end_reason,
            "detail": "see timeline"}


def run_diagnostic(track: str, model: Optional[str], seed: int, *, controller_name: str = "rl_gate_controller",
                   adapter: str = "holoocean", allow_fallback: bool = False,
                   current_profile: Optional[str] = "none", benchmark_task: Optional[str] = None,
                   dt: float = 0.1, heartbeat: int = 10) -> Dict:
    """Run one seed and return a compact diagnostic (summary + event timeline + classification)."""
    ep = RaceEpisode(track, seed=int(seed), dt=dt, adapter=adapter, allow_fallback=allow_fallback,
                     official=True, current_profile=current_profile, benchmark_task=benchmark_task)
    obs = ep.reset()
    ctx = ep.context
    pid = ep.participant_id
    controller = ControllerLoader().load(
        controller_name, constructor_kwargs={"model_path": model} if model else None)
    controller.reset(_mission_info(ctx.config, pid))
    tracker = None

    referee_state = ctx.referee.states[pid]
    expected_gates = len(ctx.referee.gate_sequence) * int(ctx.config.race.laps)
    prev = {"gates": 0, "oob": 0, "coll": 0, "wrong": 0, "missed": 0}
    events: List[Dict] = []
    phase_prev: Optional[str] = None
    t0 = time.time()

    terminated = truncated = False
    step_idx = 0
    while not (terminated or truncated):
        command = controller.step(obs)
        if tracker is None:
            tracker = _tracker_of(controller)
        tdiag = tracker.diagnostics() if tracker is not None else {}
        source = getattr(controller, "_context_source", None)
        temporal = getattr(source, "last_context", None)
        step = ep.step(command)
        step_idx += 1
        terminated, truncated = step.terminated, step.truncated
        obs = step.observation

        gates = int(referee_state.valid_gate_crossings)
        oob = int(referee_state.out_of_bounds_events)
        coll = int(referee_state.collision_events) + int(referee_state.obstacle_collision_events)
        wrong = int(referee_state.wrong_direction_crossings)
        missed = int(referee_state.missed_gate_attempts)
        phase = tdiag.get("phase")
        pos = list(step.current_state.position)  # GT: offline diagnostic only

        crossing = gates != prev["gates"]
        phase_change = phase != phase_prev
        is_hb = (step_idx % max(1, heartbeat) == 0)
        if crossing or phase_change or is_hb or oob != prev["oob"] or coll != prev["coll"] or wrong != prev["wrong"]:
            events.append({
                "step": step_idx, "t_s": round(step.time_s, 2),
                "gates": gates, "expected_gates": expected_gates,
                "phase": phase, "expected_beacon": tdiag.get("expected_beacon_id"),
                "local_completed": tdiag.get("local_completed"),
                "filtered_range_m": tdiag.get("filtered_range_m"),
                "visual_detected": tdiag.get("visual_detected"),
                "visual_center_x": tdiag.get("visual_center_x"),
                "visual_center_y": tdiag.get("visual_center_y"),
                "visual_area_fraction": tdiag.get("visual_area_fraction"),
                "steps_since_gate_seen": (
                    getattr(temporal, "steps_since_gate_seen", None)
                ),
                "vision_recently_lost": (
                    getattr(temporal, "vision_recently_lost", None)
                ),
                "forward_displacement_since_visual_loss_m": (
                    getattr(
                        temporal,
                        "forward_displacement_since_visual_loss_m",
                        None,
                    )
                ),
                "beacon_range_delta_m": (
                    getattr(temporal, "beacon_range_delta_m", None)
                ),
                "expected_beacon_changed": (
                    getattr(temporal, "expected_beacon_changed", None)
                ),
                "previous_gate_in_rear_sector": (
                    getattr(temporal, "previous_gate_in_rear_sector", None)
                ),
                "dvl_velocity": _json_list(
                    (obs.get("sensors") or {}).get("DVLSensor")
                    if isinstance(obs, dict)
                    else None
                ),
                "commit_displacement_m": tdiag.get("commit_displacement_m"),
                "out_of_bounds": oob, "collisions": coll, "wrong_dir": wrong, "missed": missed,
                "gate_crossed": crossing, "wrong_dir_delta": wrong != prev["wrong"],
                "collision_delta": coll != prev["coll"],
                "pos_xyz": [round(v, 2) for v in pos],
                "command": {k: round(float(v), 3) for k, v in command.items()},
            })
        prev = {"gates": gates, "oob": oob, "coll": coll, "wrong": wrong, "missed": missed}
        phase_prev = phase

    status = referee_state.status.value if hasattr(referee_state.status, "value") else str(referee_state.status)
    from marine_race_arena.learning.evaluate_policy import derive_evaluation_end_reason
    end_reason = derive_evaluation_end_reason(status, truncated_by_max_steps=truncated and status in ("RUNNING", "NOT_STARTED"))
    tracker_completed = int(tracker.local_completed) if tracker is not None else 0
    final_gates = int(referee_state.valid_gate_crossings)
    classification = _classify_first_failure(
        finished=(status == "FINISHED"), end_reason=end_reason, gates=final_gates,
        expected_gates=expected_gates, events=events, tracker_completed=tracker_completed)

    try:
        controller.close()
    except Exception:  # pragma: no cover
        pass
    ep.close()

    return {
        "seed": int(seed), "track": track, "adapter_used": ctx.adapter.name,
        "referee_status": status, "evaluation_end_reason": end_reason,
        "completed_gates": final_gates, "expected_gates": expected_gates,
        "tracker_local_completed": tracker_completed,
        "out_of_bounds_events": int(referee_state.out_of_bounds_events),
        "collision_events": int(referee_state.collision_events),
        "obstacle_collision_events": int(referee_state.obstacle_collision_events),
        "wrong_direction_crossings": int(referee_state.wrong_direction_crossings),
        "missed_gate_attempts": int(referee_state.missed_gate_attempts),
        "classification": classification, "steps": step_idx,
        "wall_s": round(time.time() - t0, 1), "events": events,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--controller", default="rl_gate_controller")
    parser.add_argument("--adapter", default="holoocean")
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--current-profile", default="none")
    parser.add_argument("--benchmark-task", default=None)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--heartbeat", type=int, default=15)
    args = parser.parse_args(argv)

    if args.controller == "rl_multigate_controller":
        from marine_race_arena.learning.model_contract_v3 import validate_v3_model

        validate_v3_model(args.model)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    currents = currents_summary(args.track, args.current_profile, args.benchmark_task)
    if str(args.current_profile).strip().lower() == "none" and not currents["currents_are_zero"]:
        print(f"[diag] ABORT: --current-profile none did not disable currents: {currents}")
        return 3

    seeds = _parse_seeds(args.seeds)
    runs: List[Dict] = []
    for seed in seeds:
        r = run_diagnostic(args.track, args.model, seed, controller_name=args.controller,
                           adapter=args.adapter, allow_fallback=args.allow_fallback,
                           current_profile=args.current_profile, benchmark_task=args.benchmark_task,
                           dt=args.dt, heartbeat=args.heartbeat)
        runs.append(r)
        c = r["classification"]
        print(f"[diag] seed={seed} gates={r['completed_gates']}/{r['expected_gates']} "
              f"tracker={r['tracker_local_completed']} status={r['referee_status']} "
              f"end={r['evaluation_end_reason']} FAIL={c['failure']} oob={r['out_of_bounds_events']} "
              f"wrongdir={r['wrong_direction_crossings']} wall={r['wall_s']}s")
        # Per-seed timeline (kept compact -- event rows only).
        (out_dir / f"timeline_seed{seed}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")

    max_gates = max((r["completed_gates"] for r in runs), default=0)
    fail_counts: Dict[str, int] = {}
    for r in runs:
        f = r["classification"]["failure"]
        fail_counts[f] = fail_counts.get(f, 0) + 1
    summary = {
        "generated_utc": now_utc(), "git_sha": git_sha(), "track": args.track,
        "track_sha256": sha256_file(args.track), "controller": args.controller,
        "model": args.model, "model_sha256": (sha256_file(args.model) if args.model else None),
        "adapter_requested": args.adapter, "adapter_actual": (runs[0]["adapter_used"] if runs else None),
        "fallback_used": (
            args.adapter != "fallback"
            and any(r["adapter_used"] == "fallback" for r in runs)
        ),
        "current_profile": args.current_profile, "benchmark_task_override": args.benchmark_task,
        "currents": currents, "n_seeds": len(runs), "seeds": seeds,
        "expected_gates": (runs[0]["expected_gates"] if runs else None),
        "max_gates_reached": max_gates,
        "completions": sum(1 for r in runs if r["referee_status"] == "FINISHED"),
        "mean_gates": round(sum(r["completed_gates"] for r in runs) / len(runs), 3) if runs else 0.0,
        "first_failure_counts": fail_counts,
        "per_seed": [{k: r[k] for k in ("seed", "completed_gates", "tracker_local_completed",
                                        "referee_status", "evaluation_end_reason",
                                        "out_of_bounds_events", "wrong_direction_crossings",
                                        "collision_events", "wall_s")} | {"failure": r["classification"]["failure"]}
                     for r in runs],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[diag] SUMMARY:", json.dumps({k: summary[k] for k in (
        "max_gates_reached", "mean_gates", "completions", "first_failure_counts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
