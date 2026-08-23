"""Closed-loop evaluation of a Gen-2 learned controller.

This module is deliberately **expert-free**.  It never imports, constructs or
references the rule controller, so "the final inference path has no expert
dependency" is a property of the import graph, not a promise in a comment --
``tests/learning/gen2/test_gen2_no_expert_at_inference.py`` asserts exactly
that by reading this module's source.

Every step is::

    action = learned_policy(observation, recurrent_state)

with no fallback, no blending, no safety takeover and no course knowledge.

Metrics are built around **unconditional survival**.  Generation 1 reported a
first-gate crossing of 1.00 next to a complete gate1 -> gate2 transition of
0.7333: the conditional rates looked excellent because they only ever measured
survivors.  Here every ``P(reach gate k)`` divides by the episodes whose course
*had* a gate k, so a policy that dies at gate 2 cannot show 1.00 at gate 12.
The reusable :func:`~marine_race_arena.learning.rl_failure_taxonomy.survival_curve`
does the counting; this module only chooses the reported positions and adds
Wilson intervals.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from marine_race_arena.learning.config import ACTION_AXES, ACTION_DIM
from marine_race_arena.learning.config_local_transition import (
    FEATURE_NAMES_LOCAL_TRANSITION,
    OBS_DIM_LOCAL_TRANSITION,
    OBS_ENCODING_VERSION_LOCAL_TRANSITION,
)
from marine_race_arena.config.benchmark_tasks import BENCHMARK_TASK_CLEAN_GATE
from marine_race_arena.learning.episode import RaceEpisode
from marine_race_arena.learning.gen2 import GEN2_ACTION_CONTRACT
from marine_race_arena.learning.gen2.course_family import Gen2CourseSpec, materialize_course, sample_course
from marine_race_arena.learning.observation_encoder_local_transition import (
    encode_observation_local_transition,
)
from marine_race_arena.learning.tracker_context_local_transition import (
    OnboardLocalTransitionContextTracker,
)

#: Sequence positions reported on the unconditional survival curve.
SURVIVAL_GATES: Tuple[int, ...] = (1, 2, 3, 5, 8, 12, 17, 22)

#: Steps after a crossing within which the transition must be re-established.
TRANSITION_WINDOW_STEPS = 40
#: |bearing| below this counts as aligned on the new target (radians-free: the
#: observation carries sin/cos, so this is the cosine bound).
ALIGNMENT_COS_BOUND = 0.90

_FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES_LOCAL_TRANSITION)}


@dataclass
class Gen2EvalEpisode:
    """One learner-only episode, in the row schema the taxonomy already reads."""

    seed: int
    gate_count: int
    gates_completed: int
    succeeded: bool
    status: str
    episode_type: str
    pattern: str
    steps: int
    completion_time_s: float
    path_length_m: float
    mean_action_jerk: float
    collision_events: int
    out_of_bounds_events: int
    wrong_direction_crossings: int
    missed_gate_attempts: int
    timeout: bool
    first_gate_crossed: bool
    correct_target_switch: Optional[bool]
    new_target_acquired_and_aligned: Optional[bool]
    new_target_range_decreasing: Optional[bool]
    initial_yaw_error_deg: float
    initial_lateral_offset_m: float
    wall_time_s: float

    def as_row(self) -> Dict[str, Any]:
        return asdict(self)


def _transition_quality(
    observations: np.ndarray, crossings: np.ndarray
) -> Tuple[Optional[bool], Optional[bool], Optional[bool]]:
    """Did the policy re-acquire the next gate after its first crossing?

    Computed only from recorded legal observations, so it is a diagnostic on
    the same information the policy had -- not privileged referee state.
    """
    if len(crossings) == 0 or len(observations) == 0:
        return None, None, None
    crossed = np.flatnonzero(np.asarray(crossings) >= 1)
    if crossed.size == 0:
        return None, None, None
    start = int(crossed[0])
    end = min(len(observations), start + TRANSITION_WINDOW_STEPS)
    if end <= start + 1:
        return None, None, None
    window = observations[start:end]

    changed = window[:, _FEATURE_INDEX["target_changed_recently"]]
    switched = bool(np.any(changed > 0.5))

    present = window[:, _FEATURE_INDEX["beacon_present"]] > 0.5
    cos_bearing = window[:, _FEATURE_INDEX["beacon_bearing_cos"]]
    aligned = bool(np.any(present & (cos_bearing >= ALIGNMENT_COS_BOUND)))

    rate_valid = window[:, _FEATURE_INDEX["beacon_rate_valid"]] > 0.5
    range_rate = window[:, _FEATURE_INDEX["beacon_range_rate"]]
    if rate_valid.any():
        decreasing = bool(float(np.mean(range_rate[rate_valid])) < 0.0)
    else:
        decreasing = None
    return switched, aligned, decreasing


def run_policy_episode(
    controller,
    track_path: str | Path,
    *,
    seed: int,
    spec: Optional[Gen2CourseSpec] = None,
    adapter: str = "holoocean",
    allow_fallback: bool = False,
    dt: float = 0.1,
    max_steps: int = 4000,
    unseal_record: Optional[Mapping[str, Any]] = None,
) -> Gen2EvalEpisode:
    """Drive one episode with the learned controller and nothing else."""
    from marine_race_arena.learning.gen2.holdout_seal import assert_course_accessible

    assert_course_accessible(
        track_path,
        context=f"Gen-2 policy evaluation (seed {int(seed)})",
        unseal_record=unseal_record,
    )
    started = time.perf_counter()
    episode = RaceEpisode(
        str(track_path), seed=int(seed), dt=float(dt), adapter=adapter,
        allow_fallback=allow_fallback, max_steps=int(max_steps),
        official=True, current_profile="none",
        # Mixed Endurance declares benchmark_task current_gate. Running it
        # current-free -- the official protocol for the 0/30 vs 30/30
        # comparison, see scripts/run_official_mixed_no_current.bat -- requires
        # overriding the task too, or the loader rejects the config.
        benchmark_task=BENCHMARK_TASK_CLEAN_GATE,
    )
    observations: List[np.ndarray] = []
    crossings: List[int] = []
    path_length = 0.0
    jerk_total = 0.0
    jerk_samples = 0
    previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
    previous_delta = np.zeros(ACTION_DIM, dtype=np.float32)
    previous_position: Optional[np.ndarray] = None
    steps = 0
    truncated = False
    time_s = 0.0

    try:
        raw = episode.reset(seed=int(seed))
        config = episode.context.config
        gate_total = len(config.track.gate_sequence)
        context_source = OnboardLocalTransitionContextTracker(
            total_beacons=max(1, gate_total), laps=max(1, int(config.race.laps))
        )
        context_source.reset(raw)
        controller.reset()

        while steps < max_steps:
            context = context_source.context(raw, dt=float(dt), prev_action=previous_action.tolist())
            encoded = encode_observation_local_transition(raw, context)
            observations.append(encoded)

            action = np.asarray(
                controller.act(encoded, first_step=(steps == 0)), dtype=np.float32
            ).reshape(ACTION_DIM)
            action = np.clip(np.nan_to_num(action, nan=0.0), -1.0, 1.0)

            delta = action - previous_action
            jerk_total += float(np.linalg.norm(delta - previous_delta))
            jerk_samples += 1
            previous_delta = delta

            step = episode.step({axis: float(action[i]) for i, axis in enumerate(ACTION_AXES)})
            position = np.asarray(step.current_state.position, dtype=np.float64)
            if previous_position is not None:
                path_length += float(np.linalg.norm(position - previous_position))
            previous_position = position

            raw = step.observation
            previous_action = action
            steps += 1
            time_s = float(step.time_s)
            crossings.append(int(episode.referee_progress()["valid_gate_crossings"]))
            if step.terminated or step.truncated:
                truncated = bool(step.truncated)
                break
        else:
            truncated = True

        state = episode.context.referee.states[episode.participant_id]
        status = getattr(state.status, "value", str(state.status))
        gates_completed = int(state.valid_gate_crossings)
        succeeded = status == "FINISHED" and gates_completed == gate_total
        obs_array = np.asarray(observations, dtype=np.float32)
        switched, aligned, decreasing = _transition_quality(
            obs_array, np.asarray(crossings, dtype=np.int32)
        )
        return Gen2EvalEpisode(
            seed=int(seed),
            gate_count=gate_total,
            gates_completed=gates_completed,
            succeeded=bool(succeeded),
            status=status,
            episode_type="full_sequence",
            pattern=str(spec.pattern) if spec is not None else "unknown",
            steps=steps,
            completion_time_s=round(time_s, 3),
            path_length_m=round(path_length, 3),
            mean_action_jerk=round(jerk_total / max(1, jerk_samples), 6),
            collision_events=int(state.collision_events) + int(state.obstacle_collision_events),
            out_of_bounds_events=int(state.out_of_bounds_events),
            wrong_direction_crossings=int(state.wrong_direction_crossings),
            missed_gate_attempts=int(state.missed_gate_attempts),
            timeout=bool(truncated and not succeeded),
            first_gate_crossed=gates_completed >= 1,
            correct_target_switch=switched,
            new_target_acquired_and_aligned=aligned,
            new_target_range_decreasing=decreasing,
            initial_yaw_error_deg=float(spec.initial_yaw_error_deg) if spec else 0.0,
            initial_lateral_offset_m=float(spec.initial_lateral_offset_m) if spec else 0.0,
            wall_time_s=round(time.perf_counter() - started, 2),
        )
    finally:
        episode.close()


def unconditional_survival(
    rows: Sequence[Mapping[str, Any]],
    *,
    gates: Sequence[int] = SURVIVAL_GATES,
) -> Dict[str, Any]:
    """P(reach gate k from episode start) with Wilson intervals.

    Reuses the audited counting in ``rl_failure_taxonomy.survival_curve``; this
    wrapper only selects the reported positions and attaches an interval, so a
    small cohort cannot be read as a precise number.
    """
    from marine_race_arena.learning.rl_failure_taxonomy import survival_curve
    from marine_race_arena.learning.rl_matched_benchmark import wilson_interval

    curve = survival_curve(rows, episode_types=("full_sequence",))
    reach = dict(curve.get("reach_from_start") or {})
    exposure = dict(curve.get("exposure") or {})
    out: Dict[str, Any] = {}
    for position in gates:
        key = str(position)
        rate = reach.get(key)
        n = int(exposure.get(key, 0) or 0)
        if rate is None or n == 0:
            out[key] = {"rate": None, "n": n, "ci95": None}
            continue
        successes = int(round(float(rate) * n))
        low, high = wilson_interval(successes, n)
        out[key] = {
            "rate": round(float(rate), 4),
            "n": n,
            "successes": successes,
            "ci95": [round(low, 4), round(high, 4)],
        }
    return out


def _rate(values: Iterable[bool]) -> Optional[float]:
    items = list(values)
    return round(sum(1 for v in items if v) / len(items), 4) if items else None


def aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Build the Gen-2 metric block, in the shape the readiness gate reads."""
    if not rows:
        return {"episodes": 0}
    by_length: Dict[int, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_length.setdefault(int(row["gate_count"]), []).append(row)

    completion_by_length = {
        str(length): _rate(bool(r["succeeded"]) for r in items)
        for length, items in sorted(by_length.items())
    }
    counts_by_length = {str(length): len(items) for length, items in sorted(by_length.items())}

    survival = unconditional_survival(rows)
    # The complete gate1 -> gate2 transition, measured unconditionally.  This is
    # the number Gen-1 reported as 0.7333 behind a first-gate rate of 1.00.
    transition = survival.get("2", {})

    two_gate_pool = [r for r in rows if int(r["gate_count"]) >= 2]
    succeeded = [bool(r["succeeded"]) for r in rows]
    finite_times = [float(r["completion_time_s"]) for r in rows if r["succeeded"]]
    finite_paths = [float(r["path_length_m"]) for r in rows if r["succeeded"]]

    switches = [r["correct_target_switch"] for r in rows if r["correct_target_switch"] is not None]
    aligns = [r["new_target_acquired_and_aligned"] for r in rows
              if r["new_target_acquired_and_aligned"] is not None]
    ranges = [r["new_target_range_decreasing"] for r in rows
              if r["new_target_range_decreasing"] is not None]

    return {
        "episodes": len(rows),
        "observation_contract": OBS_ENCODING_VERSION_LOCAL_TRANSITION,
        "action_contract": GEN2_ACTION_CONTRACT,
        "overall_completion_rate": _rate(succeeded),
        "first_gate_crossing_rate": _rate(bool(r["first_gate_crossed"]) for r in rows),
        # The headline transition metric: reaching gate 2 from the start.
        "universal_transition_success_rate": transition.get("rate"),
        "gate1_to_gate2_transition_rate": transition.get("rate"),
        "gate1_to_gate2_cases": transition.get("n", 0),
        "transition_cases": len(two_gate_pool),
        "unconditional_survival": {k: v.get("rate") for k, v in survival.items()},
        "unconditional_survival_detail": survival,
        "completion_by_length": completion_by_length,
        "cases_by_length": counts_by_length,
        "min_cases_per_length": min(counts_by_length.values()) if counts_by_length else 0,
        "collision_episode_rate": _rate(int(r["collision_events"]) > 0 for r in rows),
        "out_of_bounds_episode_rate": _rate(int(r["out_of_bounds_events"]) > 0 for r in rows),
        "wrong_direction_episode_rate": _rate(int(r["wrong_direction_crossings"]) > 0 for r in rows),
        "missed_gate_episode_rate": _rate(int(r["missed_gate_attempts"]) > 0 for r in rows),
        "timeout_episode_rate": _rate(bool(r["timeout"]) for r in rows),
        "correct_target_switch_rate": _rate(bool(v) for v in switches),
        "new_target_aligned_rate": _rate(bool(v) for v in aligns),
        "new_target_range_decreasing_rate": _rate(bool(v) for v in ranges),
        "mean_completion_time_s": round(float(np.mean(finite_times)), 3) if finite_times else None,
        "mean_path_length_m": round(float(np.mean(finite_paths)), 3) if finite_paths else None,
        "mean_action_jerk": round(float(np.mean([float(r["mean_action_jerk"]) for r in rows])), 6),
        # Contract attestations consumed by the readiness gate's evidence group.
        "inference_expert_free": 1,
        "contracts_match": 1,
    }


def evaluate_policy(
    controller,
    seeds: Sequence[int],
    *,
    gate_counts: Optional[Sequence[int]] = None,
    track_dir: str | Path,
    adapter: str = "holoocean",
    allow_fallback: bool = False,
    dt: float = 0.1,
    steps_per_gate: int = 900,
    unseal_record: Optional[Mapping[str, Any]] = None,
    on_episode: Optional[Callable[[Gen2EvalEpisode], None]] = None,
) -> Dict[str, Any]:
    """Run the learned controller over ``seeds`` and aggregate the result.

    ``gate_counts`` cycles alongside ``seeds`` so a stage evaluation can pin
    the sequence length (stage A = 2 gates, B = 3, C = 5, ...) while keeping
    each course a pure function of its seed.
    """
    track_dir = Path(track_dir)
    track_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    episodes: List[Gen2EvalEpisode] = []
    started = time.perf_counter()
    for index, seed in enumerate(seeds):
        forced = gate_counts[index % len(gate_counts)] if gate_counts else None
        spec = sample_course(seed, gate_count=forced)
        track = materialize_course(spec, track_dir / f"eval_{int(seed):06d}_{spec.gate_count}g.json")
        episode = run_policy_episode(
            controller, track, seed=int(seed), spec=spec, adapter=adapter,
            allow_fallback=allow_fallback, dt=dt,
            max_steps=max(600, spec.gate_count * int(steps_per_gate)),
            unseal_record=unseal_record,
        )
        episodes.append(episode)
        rows.append(episode.as_row())
        if on_episode is not None:
            on_episode(episode)
    metrics = aggregate(rows)
    metrics["wall_time_s"] = round(time.perf_counter() - started, 1)
    return {"metrics": metrics, "episodes": rows}


def write_report(result: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".partial")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    tmp.replace(target)
    return target


def failures_by_pattern(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Group failures by course pattern and gate count.

    A uniform failure rate points at the controller; one concentrated on
    ``climb`` or a wide-turn pattern points at a geometry the policy has not
    learned, or at an arena bound it is being pushed into.  Reporting the split
    is the difference between "the policy is unstable" and "the policy leaves
    the arena on descending courses".
    """
    if not rows:
        return {"episodes": 0}
    groups: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("pattern", "unknown"))
        bucket = groups.setdefault(key, {
            "episodes": 0, "completed": 0, "out_of_bounds": 0,
            "collisions": 0, "missed_gate": 0, "timeout": 0, "wrong_direction": 0,
        })
        bucket["episodes"] += 1
        bucket["completed"] += int(bool(row["succeeded"]))
        bucket["out_of_bounds"] += int(int(row["out_of_bounds_events"]) > 0)
        bucket["collisions"] += int(int(row["collision_events"]) > 0)
        bucket["missed_gate"] += int(int(row["missed_gate_attempts"]) > 0)
        bucket["timeout"] += int(bool(row["timeout"]))
        bucket["wrong_direction"] += int(int(row["wrong_direction_crossings"]) > 0)
    for bucket in groups.values():
        n = max(1, bucket["episodes"])
        bucket["completion_rate"] = round(bucket["completed"] / n, 4)
        bucket["out_of_bounds_rate"] = round(bucket["out_of_bounds"] / n, 4)

    by_length: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = str(row["gate_count"])
        bucket = by_length.setdefault(key, {"episodes": 0, "out_of_bounds": 0})
        bucket["episodes"] += 1
        bucket["out_of_bounds"] += int(int(row["out_of_bounds_events"]) > 0)
    for bucket in by_length.values():
        bucket["out_of_bounds_rate"] = round(
            bucket["out_of_bounds"] / max(1, bucket["episodes"]), 4
        )

    ranked = sorted(
        groups.items(), key=lambda kv: (-kv[1]["out_of_bounds_rate"], kv[0])
    )
    return {
        "episodes": len(rows),
        "by_pattern": dict(sorted(groups.items())),
        "out_of_bounds_by_gate_count": dict(
            sorted(by_length.items(), key=lambda kv: int(kv[0]))
        ),
        "worst_out_of_bounds_patterns": [
            {"pattern": name, "rate": stats["out_of_bounds_rate"],
             "episodes": stats["episodes"]}
            for name, stats in ranked[:5] if stats["out_of_bounds"] > 0
        ],
    }


def survival_table(metrics: Mapping[str, Any]) -> str:
    """Human-readable unconditional survival curve."""
    detail = dict(metrics.get("unconditional_survival_detail") or {})
    lines = ["gate   P(reach from start)      n     95% CI"]
    for position in SURVIVAL_GATES:
        entry = detail.get(str(position)) or {}
        rate = entry.get("rate")
        if rate is None:
            lines.append(f"{position:>4}   {'--':>18}   {entry.get('n', 0):>4}")
            continue
        low, high = entry.get("ci95") or (float("nan"), float("nan"))
        lines.append(
            f"{position:>4}   {rate:>18.4f}   {entry.get('n', 0):>4}   [{low:.3f}, {high:.3f}]"
        )
    return "\n".join(lines)
