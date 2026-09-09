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


#: Categories the suite can contain. A category that produced no episode is
#: reported as ``None`` (not evaluated), never as ``0.0`` (evaluated and failed).
EVALUATION_CATEGORIES = ("retention", "straight", "left", "right", "three_gate", "six_gate")

#: The parametric stage at which each conditional category first enters the suite.
CATEGORY_FIRST_STAGE_INDEX = {"three_gate": 5, "six_gate": 6}

#: Category name -> the aggregate's top-level completion-rate key. The retention
#: category is published as ``single_gate_completion_rate`` for historical reasons.
CATEGORY_RATE_KEY = {
    "retention": "single_gate_completion_rate",
    "straight": "straight_completion_rate",
    "left": "left_completion_rate",
    "right": "right_completion_rate",
    "three_gate": "three_gate_completion_rate",
    "six_gate": "six_gate_completion_rate",
}

#: Category name -> the aggregate's top-level sample-count key.
CATEGORY_COUNT_KEY = {
    "retention": "single_gate_n",
    "straight": "straight_n",
    "left": "left_n",
    "right": "right_n",
    "three_gate": "three_gate_n",
    "six_gate": "six_gate_n",
}


def rate_or(metrics: Mapping[str, Any], key: str, default: float) -> float:
    """Read a completion rate, mapping "not evaluated" (``None``) to ``default``.

    ``metrics.get(key, default)`` is not enough: the key is present with a
    ``None`` value when the category was never part of the suite, and ``float``
    would raise. Callers that need a number state what an unmeasured category
    should count as, instead of silently reading it as a 0% success rate.
    """
    value = metrics.get(key)
    return float(default) if value is None else float(value)


def category_evaluated(metrics: Mapping[str, Any], category: str) -> bool:
    """True only when the suite actually ran at least one episode of ``category``.

    Prefers the per-category sample count; falls back to the published rate so
    reports written before this schema (which always emitted a float) still read
    as evaluated rather than silently disappearing.
    """
    categories = metrics.get("category_metrics")
    if isinstance(categories, Mapping):
        entry = categories.get(category)
        if isinstance(entry, Mapping) and "n" in entry:
            return int(entry.get("n", 0)) > 0
    count_key = CATEGORY_COUNT_KEY.get(category)
    if count_key is not None and metrics.get(count_key) is not None:
        return int(metrics[count_key]) > 0
    rate_key = CATEGORY_RATE_KEY.get(category, f"{category}_completion_rate")
    return metrics.get(rate_key) is not None


def category_not_evaluated_reason(category: str, stage: Any) -> Optional[str]:
    """Explain why a conditional category is absent from a stage's suite."""
    first = CATEGORY_FIRST_STAGE_INDEX.get(category)
    if first is None:
        return None
    try:
        stage_index = int(str(stage)[1:])
    except (TypeError, ValueError):
        return None
    if stage_index < first:
        return (
            f"{category} cases enter the evaluation suite at stage C{first}; "
            f"the run is at stage {stage}, so the category was not evaluated"
        )
    return None


def checkpoint_metric_key(metrics: Mapping[str, Any]) -> Tuple[float, ...]:
    """Reliability-first lexicographic ordering; reward is intentionally absent."""
    return (
        rate_or(metrics, "completion_rate", 0.0),
        min(
            rate_or(metrics, "left_completion_rate", 0.0),
            rate_or(metrics, "right_completion_rate", 0.0),
        ),
        rate_or(metrics, "single_gate_completion_rate", 0.0),
        rate_or(metrics, "straight_completion_rate", 0.0),
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
        rate_or(metrics, "single_gate_completion_rate", 0.0),
        rate_or(metrics, "straight_completion_rate", 0.0),
        rate_or(metrics, "left_completion_rate", 0.0),
        rate_or(metrics, "right_completion_rate", 0.0),
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
        rate_or(metrics, f"{direction}_completion_rate", 0.0),
        rate_or(metrics, "completion_rate", 0.0),
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
    """Official circuits unlock only on *measured* three-gate competence.

    A suite that never ran a three-gate case cannot unlock the circuits: an
    unmeasured category is not evidence either way.
    """
    if not category_evaluated(metrics, "three_gate"):
        return False
    return (
        rate_or(metrics, "straight_completion_rate", 0.0) >= 0.9
        and rate_or(metrics, "left_completion_rate", 0.0) >= 0.8
        and rate_or(metrics, "right_completion_rate", 0.0) >= 0.8
        and rate_or(metrics, "three_gate_completion_rate", 0.0) >= 0.8
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
    trace_case_seeds: Sequence[int] = (),
    case_indices: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Evaluate without expert/rule construction and persist a compact report."""
    if mode not in {"light", "full"}:
        raise ValueError("evaluation mode must be light or full")
    out = Path(output_dir)
    track_dir = out / "tracks"
    track_dir.mkdir(parents=True, exist_ok=True)
    cases = _evaluation_cases(stage, mode=mode, seeds=seeds, output_dir=track_dir)
    if case_indices is not None:
        selected = {int(index) for index in case_indices}
        cases = [
            case for index, case in enumerate(cases) if index in selected
        ]
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
        trace_steps: List[Dict[str, Any]] = []
        trace_enabled = int(case["seed"]) in {
            int(seed) for seed in trace_case_seeds
        }
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
                if trace_enabled:
                    participant_state = (
                        base_env.episode.context.adapter.get_participant_state(
                            base_env.episode.participant_id
                        )
                    )
                    referee_state = (
                        base_env.episode.context.referee.states[
                            base_env.episode.participant_id
                        ]
                    )
                    previous_action = (
                        actions[-2]
                        if len(actions) > 1
                        else np.zeros_like(action_array)
                    )
                    trace_steps.append(
                        {
                            "step": int(base_env.episode.step_count),
                            "time_s": float(elapsed_s),
                            "phase": phase,
                            "action": action_array.tolist(),
                            "action_delta_norm": float(
                                np.linalg.norm(
                                    action_array - previous_action
                                )
                            ),
                            "position": list(participant_state.position),
                            "rotation_rpy_deg": list(
                                participant_state.rotation_rpy_deg
                            ),
                            "gate_crossings": int(
                                info.get("gate_crossings", 0)
                            ),
                            "collision_contact_frame": bool(
                                info.get("collision_contact_frame")
                                or info.get("obstacle_collision_frame")
                            ),
                            "collision_events": int(
                                referee_state.collision_events
                            ),
                            "out_of_bounds_frame": bool(
                                info.get("out_of_bounds_frame")
                            ),
                            "wrong_direction_crossings": int(
                                referee_state.wrong_direction_crossings
                            ),
                        }
                    )
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
                    **({"trace": trace_steps} if trace_enabled else {}),
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
    not_evaluated_reasons = {
        category: reason
        for category in aggregate["not_evaluated_categories"]
        for reason in [category_not_evaluated_reason(category, stage)]
        if reason
    }
    report = {
        "schema_version": "multigate_longrun_evaluation_v1",
        "timesteps": int(timesteps),
        "stage": stage,
        "mode": mode,
        "not_evaluated_reasons": not_evaluated_reasons,
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
    category_metrics: Dict[str, Dict[str, Any]] = {}
    for category in EVALUATION_CATEGORIES:
        subset = [row for row in rows if row.get("category") == category]
        successes = sum(bool(row.get("finished")) for row in subset)
        category_metrics[category] = {
            "n": len(subset),
            "successes": successes,
            # ``None`` means the suite never ran this category. Reporting 0.0 here
            # is indistinguishable from "ran and failed every episode", which is
            # exactly the ambiguity that made three_gate_success look like a
            # failure while three-gate cases were simply not in the suite.
            "completion_rate": (successes / len(subset) if subset else None),
            "evaluated": bool(subset),
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
        "straight_n": category_metrics["straight"]["n"],
        "straight_completion_rate": category_metrics["straight"][
            "completion_rate"
        ],
        "single_gate_n": category_metrics["retention"]["n"],
        "single_gate_completion_rate": category_metrics["retention"][
            "completion_rate"
        ],
        "three_gate_n": category_metrics["three_gate"]["n"],
        "three_gate_evaluated": category_metrics["three_gate"]["evaluated"],
        "three_gate_completion_rate": category_metrics["three_gate"][
            "completion_rate"
        ],
        "six_gate_n": category_metrics["six_gate"]["n"],
        "six_gate_evaluated": category_metrics["six_gate"]["evaluated"],
        "six_gate_completion_rate": category_metrics["six_gate"][
            "completion_rate"
        ],
        "evaluated_categories": [
            name
            for name in EVALUATION_CATEGORIES
            if category_metrics[name]["evaluated"]
        ],
        "not_evaluated_categories": [
            name
            for name in EVALUATION_CATEGORIES
            if not category_metrics[name]["evaluated"]
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
