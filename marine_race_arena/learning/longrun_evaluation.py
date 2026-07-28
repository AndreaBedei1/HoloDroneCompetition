"""Deterministic curriculum evaluation and safe checkpoint selection."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from marine_race_arena.learning.config_v3 import OBS_ENCODING_VERSION_V3
from marine_race_arena.learning.gym_env import MarineRaceGymEnv
from marine_race_arena.learning.longrun_checkpoint import atomic_write_json
from marine_race_arena.learning.longrun_env import apply_observation_mode
from marine_race_arena.learning.multigate_curriculum import (
    SINGLE_GATE,
    SIX_GATE,
    THREE_GATE_S,
    TWO_GATE_STRAIGHT,
)
from marine_race_arena.learning.parametric_curriculum import (
    STAGE_BY_KEY,
    TransitionGeometry,
    generate_two_gate_track,
)
from marine_race_arena.learning.reward_v3 import (
    MultiGateRewardConfig,
    MultiGateTrainingReward,
)


def checkpoint_metric_key(metrics: Mapping[str, Any]) -> Tuple[float, ...]:
    """Reliability-first lexicographic ordering; reward is intentionally absent."""
    return (
        float(metrics.get("completion_rate", 0.0)),
        min(
            float(metrics.get("left_completion_rate", 0.0)),
            float(metrics.get("right_completion_rate", 0.0)),
        ),
        float(metrics.get("single_gate_completion_rate", 0.0)),
        float(metrics.get("straight_completion_rate", 0.0)),
        -float(metrics.get("episodes_with_any_safety", 0.0)),
        -float(metrics.get("previous_gate_returns", 0.0)),
        float(metrics.get("mean_gates", 0.0)),
        -float(
            metrics.get("mean_penalized_time_s")
            if metrics.get("mean_penalized_time_s") is not None
            else 1e9
        ),
        -float(
            metrics.get("mean_successful_time_s")
            if metrics.get("mean_successful_time_s") is not None
            else 1e9
        ),
        -float(metrics.get("mean_action_jerk", 0.0)),
        -float(metrics.get("mean_inference_ms", 1e9)),
    )


def bc_checkpoint_metric_key(metrics: Mapping[str, Any]) -> Tuple[float, ...]:
    """BC selection keeps retention categories ahead of safety and speed."""
    return (
        float(metrics.get("single_gate_completion_rate", 0.0)),
        float(metrics.get("straight_completion_rate", 0.0)),
        float(metrics.get("left_completion_rate", 0.0)),
        float(metrics.get("right_completion_rate", 0.0)),
        -float(metrics.get("safety_events", 0.0)),
        -float(metrics.get("previous_gate_returns", 0.0)),
        -float(
            metrics.get("mean_penalized_time_s")
            if metrics.get("mean_penalized_time_s") is not None
            else 1e9
        ),
        -float(metrics.get("mean_action_jerk", 0.0)),
    )


def directional_metric_key(
    metrics: Mapping[str, Any], direction: str
) -> Tuple[float, ...]:
    return (
        float(metrics.get(f"{direction}_completion_rate", 0.0)),
        float(metrics.get("completion_rate", 0.0)),
        float(metrics.get("mean_gates", 0.0)),
        -float(metrics.get("safety_events", 0.0)),
    )


def fast_reliable_metric_key(
    metrics: Mapping[str, Any], promotion_config: Any
) -> Optional[Tuple[float, ...]]:
    """Speed ordering that is defined only inside the reliable checkpoint set."""
    from marine_race_arena.learning.parametric_curriculum import (
        reliability_requirements_met,
    )

    if not reliability_requirements_met(metrics, promotion_config):
        return None
    return (
        -float(
            metrics.get("mean_penalized_time_s")
            if metrics.get("mean_penalized_time_s") is not None
            else 1e9
        ),
        -float(
            metrics.get("mean_successful_time_s")
            if metrics.get("mean_successful_time_s") is not None
            else 1e9
        ),
        -float(metrics.get("mean_action_jerk", 0.0)),
        -float(metrics.get("mean_inference_ms", 1e9)),
    )


def is_better(
    candidate: Mapping[str, Any],
    incumbent: Optional[Mapping[str, Any]],
    *,
    direction: Optional[str] = None,
) -> bool:
    if incumbent is None:
        return True
    key = (
        directional_metric_key(candidate, direction)
        if direction
        else checkpoint_metric_key(candidate)
    )
    old = (
        directional_metric_key(incumbent, direction)
        if direction
        else checkpoint_metric_key(incumbent)
    )
    return key > old


def official_evaluation_unlocked(metrics: Mapping[str, Any]) -> bool:
    return (
        float(metrics.get("straight_completion_rate", 0.0)) >= 0.9
        and float(metrics.get("left_completion_rate", 0.0)) >= 0.8
        and float(metrics.get("right_completion_rate", 0.0)) >= 0.8
        and float(metrics.get("three_gate_completion_rate", 0.0)) >= 0.8
        and int(metrics.get("safety_events", 0)) == 0
    )


def plateau_detected(
    history: Sequence[Mapping[str, Any]],
    *,
    current_timesteps: int,
    plateau_steps: int = 100_000,
    min_evaluations: int = 3,
) -> Optional[Dict[str, Any]]:
    window = [
        row
        for row in history
        if int(row.get("timesteps", 0)) >= current_timesteps - plateau_steps
    ]
    if len(window) < min_evaluations:
        return None
    covered_steps = int(window[-1].get("timesteps", 0)) - int(
        window[0].get("timesteps", 0)
    )
    if covered_steps < plateau_steps:
        return None
    first = window[0]
    best_completion = max(float(row.get("completion_rate", 0.0)) for row in window)
    best_gates = max(float(row.get("mean_gates", 0.0)) for row in window)
    completion_gain = best_completion - float(first.get("completion_rate", 0.0))
    gate_gain = best_gates - float(first.get("mean_gates", 0.0))
    low_kl = all(
        float(row.get("approx_kl", 0.0)) < 0.001
        for row in window[-min_evaluations:]
    )
    imbalance = max(
        abs(
            float(row.get("left_completion_rate", 0.0))
            - float(row.get("right_completion_rate", 0.0))
        )
        for row in window
    )
    if completion_gain <= 0 and gate_gain <= 0:
        return {
            "detected": True,
            "window_steps": plateau_steps,
            "evaluations": len(window),
            "low_kl": low_kl,
            "left_right_imbalance": imbalance,
            "recommendations": [
                "add DAgger corrective samples from failed rollouts",
                "rebalance left/right sampling",
                "increase transition-angle diversity",
                (
                    "increase PPO update strength once within bounds"
                    if low_kl
                    else "audit reward components before changing optimization"
                ),
                "consider short frame stacking only after feed-forward evidence remains flat",
            ],
        }
    return None


def _evaluation_cases(
    stage: str,
    *,
    mode: str,
    seeds: Sequence[int],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    limit = STAGE_BY_KEY[stage].max_abs_turn_deg
    if mode == "light":
        templates = [
            ("retention", "retention", 0.0),
            ("straight", "straight", 0.0),
            ("left", "left", max(5.0, limit)),
            ("right", "right", -max(5.0, limit)),
            ("current", "left", max(5.0, 0.75 * limit)),
        ]
    else:
        templates = (
            [("retention", "retention", 0.0)] * 2
            + [("straight", "straight", 0.0)] * 4
            + [("left", "left", max(5.0, limit))] * 5
            + [("right", "right", -max(5.0, limit))] * 5
            + [
                ("current", "left", max(5.0, 0.75 * limit)),
                ("current", "right", -max(5.0, 0.75 * limit)),
                ("current", "left", max(5.0, 0.5 * limit)),
                ("current", "right", -max(5.0, 0.5 * limit)),
            ]
        )
    cases: List[Dict[str, Any]] = []
    for index, (category, direction, angle) in enumerate(templates):
        if index >= len(seeds):
            break
        if category == "retention":
            track = SINGLE_GATE
            geometry = None
        elif category == "straight":
            track = TWO_GATE_STRAIGHT
            geometry = None
        else:
            geometry = TransitionGeometry(
                signed_turn_deg=float(angle),
                gate_separation_m=5.0 + (index % 3) * 0.5,
                lateral_displacement_m=(-0.4 + (index % 3) * 0.4),
                vertical_displacement_m=(-0.3 + (index % 3) * 0.3),
                starting_yaw_error_deg=(-6.0 + (index % 3) * 6.0),
                initial_lateral_offset_m=(-0.5 + (index % 3) * 0.5),
                source="held_out_evaluation",
                stage=stage,
            )
            track_path = output_dir / f"{mode}_{index:02d}_{direction}.json"
            track = str(generate_two_gate_track(geometry, track_path))
        cases.append(
            {
                "category": category,
                "direction": direction,
                "seed": int(seeds[index]),
                "track": track,
                "geometry": asdict(geometry) if geometry else None,
            }
        )
    # Sequence retention is evaluated once it becomes relevant.
    if mode == "full" and int(stage[1:]) >= 5 and len(seeds) > len(cases):
        cases.append(
            {
                "category": "three_gate",
                "direction": "mixed",
                "seed": int(seeds[len(cases)]),
                "track": THREE_GATE_S,
                "geometry": None,
            }
        )
    if mode == "full" and int(stage[1:]) >= 6 and len(seeds) > len(cases):
        cases.append(
            {
                "category": "six_gate",
                "direction": "mixed",
                "seed": int(seeds[len(cases)]),
                "track": SIX_GATE,
                "geometry": None,
            }
        )
    return cases


def evaluate_longrun_policy(
    model: Any,
    *,
    stage: str,
    mode: str,
    seeds: Sequence[int],
    output_dir: str | Path,
    env_kwargs: Mapping[str, Any],
    reward_config: Optional[MultiGateRewardConfig] = None,
    timesteps: int = 0,
    policy_mode: str = "feedforward",
    frame_stack: int = 1,
    progress_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Evaluate without expert/rule construction and persist a compact report."""
    if mode not in {"light", "full"}:
        raise ValueError("evaluation mode must be light or full")
    out = Path(output_dir)
    track_dir = out / "tracks"
    track_dir.mkdir(parents=True, exist_ok=True)
    cases = _evaluation_cases(stage, mode=mode, seeds=seeds, output_dir=track_dir)
    config = reward_config or MultiGateRewardConfig()
    rows: List[Dict[str, Any]] = []
    if progress_callback is not None:
        progress_callback(
            {
                "mode": mode,
                "stage": stage,
                "completed": 0,
                "total": len(cases),
                "timesteps": int(timesteps),
            }
        )
    for index, case in enumerate(cases):
        base_env = MarineRaceGymEnv(
            case["track"],
            seed=case["seed"],
            reward_fn=MultiGateTrainingReward(config),
            **dict(env_kwargs),
        )
        env = apply_observation_mode(
            base_env, policy_mode=policy_mode, frame_stack=frame_stack
        )
        actions: List[np.ndarray] = []
        inference_ms: List[float] = []
        previous_return_active = False
        previous_returns = 0
        component_sums: Dict[str, float] = {}
        safety_frames = {"collision": 0, "out_of_bounds": 0, "warning": 0}
        safety_contact_events = {"collision": 0, "out_of_bounds": 0, "warning": 0}
        safety_active = {"collision": False, "out_of_bounds": False, "warning": False}
        timing = {
            "approach_s": [],
            "visual_alignment_s": [],
            "crossing_s": [],
            "post_crossing_clearance_s": [],
            "next_gate_acquisition_s": [],
        }
        seen_phase = set()
        last_gate_count = 0
        last_crossing_s: Optional[float] = None
        clearance_recorded = True
        acquisition_recorded = True
        try:
            obs, _ = env.reset(seed=case["seed"])
            done = False
            while not done:
                inference_started = time.perf_counter()
                action, _ = model.predict(obs, deterministic=True)
                inference_ms.append((time.perf_counter() - inference_started) * 1000.0)
                action_array = np.asarray(action, dtype=np.float32).reshape(-1)
                actions.append(action_array)
                obs, reward, terminated, truncated, info = env.step(action_array)
                elapsed_s = base_env.episode.step_count * base_env.episode.dt
                phase = getattr(base_env.tracker, "phase", None)
                if phase == "APPROACH" and phase not in seen_phase:
                    timing["approach_s"].append(elapsed_s)
                    seen_phase.add(phase)
                if phase == "VISUAL_ALIGN" and phase not in seen_phase:
                    timing["visual_alignment_s"].append(elapsed_s)
                    seen_phase.add(phase)
                gate_count = int(info.get("gate_crossings", last_gate_count))
                if gate_count > last_gate_count:
                    timing["crossing_s"].append(elapsed_s)
                    last_crossing_s = elapsed_s
                    clearance_recorded = False
                    acquisition_recorded = False
                    last_gate_count = gate_count
                active = {
                    "collision": bool(
                        info.get("collision_contact_frame")
                        or info.get("obstacle_collision_frame")
                    ),
                    "out_of_bounds": bool(info.get("out_of_bounds_frame")),
                    "warning": bool(info.get("safety_warning_frame")),
                }
                for name, is_active in active.items():
                    if is_active:
                        safety_frames[name] += 1
                    if is_active and not safety_active[name]:
                        safety_contact_events[name] += 1
                    safety_active[name] = is_active
                components = info.get("reward_components", {})
                for name, value in components.items():
                    component_sums[name] = component_sums.get(name, 0.0) + float(
                        value
                    )
                if last_crossing_s is not None:
                    if (
                        not clearance_recorded
                        and float(components.get("previous_gate_behind", 0.0)) > 0
                    ):
                        timing["post_crossing_clearance_s"].append(
                            elapsed_s - last_crossing_s
                        )
                        clearance_recorded = True
                    if (
                        not acquisition_recorded
                        and (
                            float(
                                components.get("next_beacon_acquisition", 0.0)
                            )
                            > 0
                            or float(
                                components.get(
                                    "next_gate_visual_acquisition", 0.0
                                )
                            )
                            > 0
                        )
                    ):
                        timing["next_gate_acquisition_s"].append(
                            elapsed_s - last_crossing_s
                        )
                        acquisition_recorded = True
                returning = float(
                    components.get("previous_gate_return_penalty", 0.0)
                ) < 0
                if returning and not previous_return_active:
                    previous_returns += 1
                previous_return_active = returning
                done = bool(terminated or truncated)
            progress = base_env.episode.referee_progress()
            state = base_env.episode.context.referee.states[
                base_env.episode.participant_id
            ]
            action_matrix = (
                np.stack(actions)
                if actions
                else np.zeros((1, 4), dtype=np.float32)
            )
            jerk = (
                float(
                    np.mean(
                        np.linalg.norm(np.diff(action_matrix, axis=0), axis=1)
                    )
                )
                if len(action_matrix) > 1
                else 0.0
            )
            finished = progress["status"] == "FINISHED"
            raw_time = base_env.episode.step_count * base_env.episode.dt
            rows.append(
                {
                    **case,
                    "finished": finished,
                    "status": progress["status"],
                    "completed_gates": int(progress["valid_gate_crossings"]),
                    "collision_events": int(state.collision_events),
                    "collision_contact_events": int(
                        safety_contact_events["collision"]
                    ),
                    "collision_frames": int(safety_frames["collision"]),
                    "out_of_bounds_events": int(state.out_of_bounds_events),
                    "out_of_bounds_contact_events": int(
                        safety_contact_events["out_of_bounds"]
                    ),
                    "out_of_bounds_frames": int(
                        safety_frames["out_of_bounds"]
                    ),
                    "wrong_direction_crossings": int(state.wrong_direction_crossings),
                    "safety_warning_events": int(
                        safety_contact_events["warning"]
                    ),
                    "safety_warning_frames": int(safety_frames["warning"]),
                    "previous_gate_returns": previous_returns,
                    "raw_time_s": raw_time if finished else None,
                    "penalized_time_s": (
                        raw_time + float(state.penalties_s) if finished else None
                    ),
                    "action_jerk": jerk,
                    "action_saturation": float(
                        np.mean(np.abs(action_matrix) > 0.98)
                    ),
                    "mean_inference_ms": (
                        float(np.mean(inference_ms)) if inference_ms else 0.0
                    ),
                    "actions_finite": bool(np.all(np.isfinite(action_matrix))),
                    "timing_diagnostics": timing,
                    "reward_components": component_sums,
                    "runtime_rule_actions": 0,
                }
            )
        finally:
            env.close()
        if progress_callback is not None:
            progress_callback(
                {
                    "mode": mode,
                    "stage": stage,
                    "completed": index + 1,
                    "total": len(cases),
                    "timesteps": int(timesteps),
                }
            )
    aggregate = aggregate_evaluation(rows)
    report = {
        "schema_version": "multigate_longrun_evaluation_v1",
        "timesteps": int(timesteps),
        "stage": stage,
        "mode": mode,
        "deterministic": True,
        "observation_version": OBS_ENCODING_VERSION_V3,
        "policy_mode": policy_mode,
        "frame_stack": int(frame_stack),
        "runtime_rule_controller_instantiated": False,
        "rows": rows,
        **aggregate,
    }
    atomic_write_json(out / f"{mode}_{int(timesteps):09d}.json", report)
    return report


def aggregate_evaluation(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    completed = sum(bool(row.get("finished")) for row in rows)
    categories = ("retention", "straight", "left", "right", "three_gate", "six_gate")
    category_metrics: Dict[str, Dict[str, Any]] = {}
    for category in categories:
        subset = [row for row in rows if row.get("category") == category]
        category_metrics[category] = {
            "n": len(subset),
            "successes": sum(bool(row.get("finished")) for row in subset),
            "completion_rate": (
                sum(bool(row.get("finished")) for row in subset) / len(subset)
                if subset
                else 0.0
            ),
        }
    finished_times = [
        float(row["penalized_time_s"])
        for row in rows
        if row.get("penalized_time_s") is not None
    ]
    component_totals: Dict[str, float] = {}
    for row in rows:
        for name, value in row.get("reward_components", {}).items():
            component_totals[name] = component_totals.get(name, 0.0) + float(value)
    collision_events = sum(int(row.get("collision_events", 0)) for row in rows)
    out_of_bounds_events = sum(
        int(row.get("out_of_bounds_events", 0)) for row in rows
    )
    wrong_direction_count = sum(
        int(row.get("wrong_direction_crossings", 0)) for row in rows
    )
    safety = (
        collision_events
        + out_of_bounds_events
        + wrong_direction_count
    )
    collision_frames = sum(int(row.get("collision_frames", 0)) for row in rows)
    out_of_bounds_frames = sum(
        int(row.get("out_of_bounds_frames", 0)) for row in rows
    )
    warning_events = sum(
        int(row.get("safety_warning_events", 0)) for row in rows
    )
    warning_frames = sum(
        int(row.get("safety_warning_frames", 0)) for row in rows
    )
    timing_values: Dict[str, List[float]] = {}
    for row in rows:
        for name, values in row.get("timing_diagnostics", {}).items():
            timing_values.setdefault(name, []).extend(float(value) for value in values)
    episodes_with_collision = sum(
        int(row.get("collision_events", 0)) > 0 for row in rows
    )
    episodes_with_out_of_bounds = sum(
        int(row.get("out_of_bounds_events", 0)) > 0 for row in rows
    )
    episodes_with_wrong_direction = sum(
        int(row.get("wrong_direction_crossings", 0)) > 0 for row in rows
    )
    episodes_with_any_safety = sum(
        int(row.get("collision_events", 0))
        + int(row.get("out_of_bounds_events", 0))
        + int(row.get("wrong_direction_crossings", 0))
        > 0
        for row in rows
    )
    return {
        "n_eval": n,
        "completions": int(completed),
        "completion_rate": completed / n if n else 0.0,
        "mean_gates": (
            float(np.mean([row.get("completed_gates", 0) for row in rows]))
            if rows
            else 0.0
        ),
        "left_n": category_metrics["left"]["n"],
        "left_successes": category_metrics["left"]["successes"],
        "left_completion_rate": category_metrics["left"]["completion_rate"],
        "right_n": category_metrics["right"]["n"],
        "right_successes": category_metrics["right"]["successes"],
        "right_completion_rate": category_metrics["right"]["completion_rate"],
        "straight_completion_rate": category_metrics["straight"][
            "completion_rate"
        ],
        "single_gate_completion_rate": category_metrics["retention"][
            "completion_rate"
        ],
        "three_gate_completion_rate": category_metrics["three_gate"][
            "completion_rate"
        ],
        "six_gate_completion_rate": category_metrics["six_gate"][
            "completion_rate"
        ],
        "safety_events": int(safety),
        "collision_events": int(collision_events),
        "collision_event_count": int(collision_events),
        "collision_frames": int(collision_frames),
        "collision_frame_count": int(collision_frames),
        "out_of_bounds_events": int(out_of_bounds_events),
        "out_of_bounds_event_count": int(out_of_bounds_events),
        "out_of_bounds_frames": int(out_of_bounds_frames),
        "out_of_bounds_frame_count": int(out_of_bounds_frames),
        "wrong_direction_count": int(wrong_direction_count),
        "wrong_direction_crossing_count": int(wrong_direction_count),
        "safety_warning_events": int(warning_events),
        "safety_warning_event_count": int(warning_events),
        "safety_warning_frames": int(warning_frames),
        "safety_warning_frame_count": int(warning_frames),
        "episodes_with_collision": int(episodes_with_collision),
        "episodes_with_out_of_bounds": int(episodes_with_out_of_bounds),
        "episodes_with_wrong_direction": int(episodes_with_wrong_direction),
        "episodes_with_any_safety": int(episodes_with_any_safety),
        "episodes_with_any_safety_event": int(episodes_with_any_safety),
        "previous_gate_returns": int(
            sum(int(row.get("previous_gate_returns", 0)) for row in rows)
        ),
        "mean_penalized_time_s": (
            float(np.mean(finished_times)) if finished_times else None
        ),
        "mean_successful_time_s": (
            float(
                np.mean(
                    [
                        float(row["raw_time_s"])
                        for row in rows
                        if row.get("raw_time_s") is not None
                    ]
                )
            )
            if finished_times
            else None
        ),
        "mean_action_jerk": (
            float(np.mean([row.get("action_jerk", 0.0) for row in rows]))
            if rows
            else 0.0
        ),
        "mean_action_saturation": (
            float(np.mean([row.get("action_saturation", 0.0) for row in rows]))
            if rows
            else 0.0
        ),
        "mean_inference_ms": (
            float(np.mean([row.get("mean_inference_ms", 0.0) for row in rows]))
            if rows
            else 0.0
        ),
        "timing_diagnostics": {
            name: {
                "count": len(values),
                "mean_s": float(np.mean(values)) if values else None,
                "max_s": float(np.max(values)) if values else None,
            }
            for name, values in sorted(timing_values.items())
        },
        "all_actions_finite": all(bool(row.get("actions_finite", True)) for row in rows),
        "reward_component_sums": component_totals,
        "category_metrics": category_metrics,
    }
