"""Rich Stage-2 (randomized-start) evaluation of a policy under the unchanged referee.

Completion alone saturates for the BC-initialized policy, so Stage-2 evaluation also
reports robustness: gates, collisions, out-of-bounds (events and episodes), wrong
direction, evaluation end reasons, mean/median/penalized time, action saturation and
smoothness, per-axis mean |action|, inference time, and -- crucially -- completion split
into the interior of the randomization envelope versus its extreme corners (where the
frozen BC failures occurred). A documented lexicographic key selects the safest, most
robust checkpoint rather than the fastest one.
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from marine_race_arena.learning.evaluate_policy import derive_evaluation_end_reason
from marine_race_arena.learning.gym_env import MarineRaceGymEnv
from marine_race_arena.learning.reward import TrainingReward

# Extreme-corner subset of the Stage-2 envelope (the region where frozen BC failed).
EXTREME_LATERAL_M = 0.8
EXTREME_YAW_DEG = 12.0


def _wilson(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def is_extreme_corner(applied: Optional[Dict[str, Any]]) -> bool:
    """True if the applied randomization sits in the extreme corner of the envelope."""
    if not applied:
        return False
    lat = abs(float(applied.get("lateral_offset_m", 0.0)))
    yaw = abs(float(applied.get("yaw_offset_deg", 0.0)))
    return lat >= EXTREME_LATERAL_M and yaw >= EXTREME_YAW_DEG


def evaluate_stage2(
    model,
    track: str,
    eval_seeds: Sequence[int],
    *,
    env_kwargs: Dict[str, Any],
    reward_config,
) -> Dict[str, Any]:
    """Evaluate ``model`` deterministically over ``eval_seeds`` and return per-seed rows
    plus an aggregate with interior/extreme-corner completion split."""
    rows: List[Dict[str, Any]] = []
    for seed in eval_seeds:
        env = MarineRaceGymEnv(track, seed=int(seed), reward_fn=TrainingReward(reward_config), **dict(env_kwargs or {}))
        actions: List[np.ndarray] = []
        infer_ms: List[float] = []
        try:
            obs, _ = env.reset(seed=int(seed))
            done = False
            truncated_flag = False
            while not done:
                t0 = time.perf_counter()
                action, _ = model.predict(obs, deterministic=True)
                infer_ms.append(1000.0 * (time.perf_counter() - t0))
                actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
                obs, _, terminated, truncated, _ = env.step(action)
                truncated_flag = bool(truncated)
                done = terminated or truncated
            progress = env.episode.referee_progress()
            state = env.episode.context.referee.states[env.episode.participant_id]
            applied = getattr(env.episode.context, "applied_randomization", None)
            status = progress["status"]
            finished = status == "FINISHED"
            steps = env.episode.step_count
            acts = np.asarray(actions, dtype=np.float32) if actions else np.zeros((1, 4), np.float32)
            smooth = float(np.mean(np.abs(np.diff(acts, axis=0)))) if len(acts) > 1 else 0.0
            max_steps = int(env_kwargs.get("max_steps") or 0)
            rows.append({
                "seed": int(seed),
                "finished": finished,
                "referee_status": status,
                "evaluation_end_reason": derive_evaluation_end_reason(
                    status, truncated_by_max_steps=(truncated_flag and max_steps and steps >= max_steps)),
                "completed_gates": int(state.valid_gate_crossings),
                "collision_events": int(state.collision_events),
                "out_of_bounds_events": int(state.out_of_bounds_events),
                "out_of_bounds_episode": int(state.out_of_bounds_events) > 0,
                "wrong_direction_crossings": int(state.wrong_direction_crossings),
                "missed_gate_attempts": int(state.missed_gate_attempts),
                "time_s": (steps * env.episode.dt if finished else None),
                "penalized_time_s": round(float(state.penalties_s), 3),
                "action_saturation": float(np.mean(np.abs(acts) > 0.98)),
                "action_smoothness": smooth,
                "mean_abs_action": {ax: float(np.mean(np.abs(acts[:, i]))) for i, ax in enumerate(("surge", "sway", "heave", "yaw"))},
                "inference_ms": round(float(np.mean(infer_ms)), 4) if infer_ms else None,
                "applied_randomization": applied,
                "is_extreme_corner": is_extreme_corner(applied),
                "actions_finite": bool(np.all(np.isfinite(acts))),
            })
        finally:
            env.close()
    return {"rows": rows, **aggregate_stage2(rows)}


def aggregate_stage2(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    finished = [r for r in rows if r["finished"]]
    rate = len(finished) / n if n else 0.0
    lo, hi = _wilson(rate, n)
    interior = [r for r in rows if not r["is_extreme_corner"]]
    extreme = [r for r in rows if r["is_extreme_corner"]]
    finish_times = [r["time_s"] for r in finished if r["time_s"] is not None]
    end_reasons: Dict[str, int] = {}
    for r in rows:
        end_reasons[r["evaluation_end_reason"]] = end_reasons.get(r["evaluation_end_reason"], 0) + 1

    def _completion(subset):
        return (sum(1 for r in subset if r["finished"]) / len(subset)) if subset else None

    return {
        "n_eval": n,
        "completion_rate": round(rate, 4),
        "completion_wilson95": [round(lo, 4), round(hi, 4)],
        "mean_gates": round(float(np.mean([r["completed_gates"] for r in rows])), 4) if rows else 0.0,
        "total_collisions": int(sum(r["collision_events"] for r in rows)),
        "oob_episodes": int(sum(1 for r in rows if r["out_of_bounds_episode"])),
        "total_out_of_bounds": int(sum(r["out_of_bounds_events"] for r in rows)),
        "total_wrong_direction": int(sum(r["wrong_direction_crossings"] for r in rows)),
        "total_missed_gate_attempts": int(sum(r["missed_gate_attempts"] for r in rows)),
        "mean_time_finished": round(float(np.mean(finish_times)), 4) if finish_times else None,
        "median_time_finished": round(float(statistics.median(finish_times)), 4) if finish_times else None,
        "mean_penalized_time": round(float(np.mean([r["penalized_time_s"] for r in rows])), 4) if rows else 0.0,
        "mean_action_saturation": round(float(np.mean([r["action_saturation"] for r in rows])), 4) if rows else 0.0,
        "mean_action_smoothness": round(float(np.mean([r["action_smoothness"] for r in rows])), 4) if rows else 0.0,
        "mean_inference_ms": round(float(np.mean([r["inference_ms"] for r in rows if r["inference_ms"] is not None])), 4) if any(r["inference_ms"] is not None for r in rows) else None,
        "interior_n": len(interior),
        "interior_completion": (round(_completion(interior), 4) if _completion(interior) is not None else None),
        "extreme_n": len(extreme),
        "extreme_completion": (round(_completion(extreme), 4) if _completion(extreme) is not None else None),
        "end_reason_counts": end_reasons,
        "all_actions_finite": all(r["actions_finite"] for r in rows) if rows else True,
    }


def stage2_best_metric_key(agg: Dict[str, Any]) -> Tuple:
    """Documented lexicographic best-model key (higher is better):
    completion, extreme-corner completion, gates, fewer OOB episodes, fewer wrong-direction,
    fewer collisions, lower penalized time. A faster but less robust policy never wins."""
    extreme = agg.get("extreme_completion")
    extreme = extreme if extreme is not None else -1.0  # no extreme samples ranks below any measured value
    penalized = agg.get("mean_penalized_time")
    return (
        float(agg.get("completion_rate", 0.0)),
        float(extreme),
        float(agg.get("mean_gates", 0.0)),
        -int(agg.get("oob_episodes", 0)),
        -int(agg.get("total_wrong_direction", 0)),
        -int(agg.get("total_collisions", 0)),
        -float(penalized if penalized is not None else 0.0),
    )


def stage2_is_better(new_agg: Dict[str, Any], best_agg: Optional[Dict[str, Any]]) -> bool:
    if best_agg is None:
        return True
    return stage2_best_metric_key(new_agg) > stage2_best_metric_key(best_agg)


def log_reward_components(model, track: str, eval_seeds, *, env_kwargs, reward_config) -> Dict[str, Any]:
    """Aggregate training-reward components by episode outcome (diagnostic; does NOT modify
    the reward). Verifies gate/progress dominate successes and failures are penalized."""
    categories: Dict[str, Dict[str, Any]] = {}
    for seed in eval_seeds:
        env = MarineRaceGymEnv(track, seed=int(seed), reward_fn=TrainingReward(reward_config), **dict(env_kwargs or {}))
        try:
            obs, _ = env.reset(seed=int(seed))
            done, ret, comp_sum = False, 0.0, {}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, terminated, truncated, info = env.step(action)
                ret += float(r)
                for k, v in info.get("reward_components", {}).items():
                    comp_sum[k] = comp_sum.get(k, 0.0) + float(v)
                done = terminated or truncated
            progress = env.episode.referee_progress()
            state = env.episode.context.referee.states[env.episode.participant_id]
            applied = getattr(env.episode.context, "applied_randomization", None)
            finished = progress["status"] == "FINISHED"
            if finished and is_extreme_corner(applied):
                cat = "success_extreme_corner"
            elif finished:
                cat = "success_interior"
            elif int(state.out_of_bounds_events) > 0:
                cat = "failure_out_of_bounds"
            elif int(state.wrong_direction_crossings) > 0:
                cat = "failure_wrong_direction"
            else:
                cat = "failure_time_limit"
            c = categories.setdefault(cat, {"episodes": 0, "total_return": 0.0, "component_sums": {}})
            c["episodes"] += 1
            c["total_return"] += ret
            for k, v in comp_sum.items():
                c["component_sums"][k] = c["component_sums"].get(k, 0.0) + v
        finally:
            env.close()

    def _avg(cat):
        c = categories.get(cat)
        if not c or not c["episodes"]:
            return None
        n = c["episodes"]
        return {"episodes": n, "mean_return": round(c["total_return"] / n, 4),
                "mean_components": {k: round(v / n, 4) for k, v in sorted(c["component_sums"].items())}}

    per_cat = {cat: _avg(cat) for cat in categories}
    succ = per_cat.get("success_interior") or per_cat.get("success_extreme_corner")
    oob = per_cat.get("failure_out_of_bounds")
    wrong = per_cat.get("failure_wrong_direction")
    tl = per_cat.get("failure_time_limit")
    checks = {
        "gate_completion_dominates_success": (succ is not None and
            (succ["mean_components"].get("gate_bonus", 0) + succ["mean_components"].get("completion_bonus", 0)) > 0),
        "forward_progress_rewarded_in_success": (succ is not None and succ["mean_components"].get("progress", 0) >= 0),
        "oob_penalized": (oob is None or oob["mean_components"].get("out_of_bounds_penalty", 0) < 0),
        "wrong_direction_penalized": (wrong is None or wrong["mean_components"].get("wrong_direction_penalty", 0) < 0),
        "action_magnitude_not_rewarded": (succ is None or succ["mean_components"].get("action_magnitude_penalty", 0) <= 0),
        "time_limit_return_not_large_positive": (tl is None or tl["mean_return"] <= (succ["mean_return"] if succ else 0.0)),
    }
    return {"by_outcome": per_cat, "checks": checks,
            "note": "Diagnostic only; the reward was not modified. A failing check flags a possible defect."}
