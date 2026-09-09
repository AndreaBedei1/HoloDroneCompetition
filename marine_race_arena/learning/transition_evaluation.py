"""Deterministic unseen transition and length-generalization benchmarks."""

from __future__ import annotations

import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from marine_race_arena.config.benchmark_tasks import BENCHMARK_TASK_CLEAN_GATE
from marine_race_arena.learning.config_local_transition import (
    FEATURE_NAMES_LOCAL_TRANSITION,
    OBS_ENCODING_VERSION_LOCAL_TRANSITION,
)
from marine_race_arena.learning.gym_env import MarineRaceGymEnv
from marine_race_arena.learning.reward_local_transition import (
    LocalTransitionTrainingReward,
)
from marine_race_arena.learning.transition_curriculum import (
    FULL_SEQUENCE_LENGTHS,
    TransitionGeometry,
    TransitionGeometrySampler,
    generate_transition_track,
)
from marine_race_arena.learning.transition_policy import (
    predict_local_transition_action,
)
from marine_race_arena.learning.transition_selection import (
    CompetenceThresholds,
    DEFAULT_COMPETENCE_THRESHOLDS,
    evaluate_competence_gate,
    transition_policy_rank_key,
)

# A commanded axis below this magnitude is indistinguishable from holding
# station; the fraction of steps above it separates an active policy from one
# that only appears safe because it never moves.
NONTRIVIAL_ACTION_THRESHOLD = 0.05


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def _configure_render_camera(track_path: str | Path, camera: str) -> str:
    """Add a video-only camera without exposing it to the policy profile."""

    mode = str(camera)
    if mode == "first_person":
        return "FrontCamera"
    offsets = {
        "chase": ([-3.0, 0.0, 1.2], [0.0, -12.0, 0.0]),
        "spectator": ([-6.0, 0.0, 4.0], [0.0, -28.0, 0.0]),
    }
    if mode not in offsets:
        raise ValueError(f"unknown render camera {mode!r}")
    location, rotation = offsets[mode]
    path = Path(track_path)
    track = json.loads(path.read_text(encoding="utf-8"))
    sensors = track["participants"][0]["sensors"]
    configured = sensors["holoocean_sensors"]
    configured[:] = [
        value for value in configured
        if value.get("sensor_name") != "RenderCamera"
    ]
    configured.append({
        "sensor_type": "RGBCamera",
        "sensor_name": "RenderCamera",
        "socket": "CameraSocket",
        "location": location,
        "rotation": rotation,
        "Hz": 30,
        "configuration": {
            "CaptureWidth": 640,
            "CaptureHeight": 480,
            "FovAngle": 90.0,
        },
    })
    # Deliberately do not add RenderCamera to allowed_sensors: it is consumed
    # only by FrameRecorder, never by the 35-feature policy encoder.
    _atomic_json(path, track)
    return "RenderCamera"


def _geometry_with_length(geometry: TransitionGeometry, gate_count: int) -> TransitionGeometry:
    if geometry.gate_count == gate_count:
        return geometry
    # Sampled arrays are regenerated deterministically by a dedicated sampler in
    # the caller; this guard exists for explicitly requested benchmark lengths.
    raise ValueError(f"geometry length {geometry.gate_count} != requested {gate_count}")


def _completed_case(
    output_dir: Path, geometry: TransitionGeometry
) -> Optional[Dict[str, Any]]:
    path = output_dir / "episode.json"
    if not path.exists():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if (
        int(row.get("seed", -1)) != int(geometry.seed)
        or row.get("episode_type") != geometry.episode_type
        or int(row.get("gate_count", -1)) != int(geometry.gate_count)
        or "universal_transition_success" not in row
    ):
        return None
    return row


def _benchmark_cases(
    *,
    output: Path,
    seed: int,
    difficulty: str,
    transition_cases: int,
    full_cases_per_length: int,
    dataset_split: str = "train",
) -> list[tuple[int, TransitionGeometry, Path, bool]]:
    """Generate the complete deterministic case list before any engine starts.

    ``dataset_split`` selects the seed band the cases are drawn from.  It stays
    "train" by default so in-training competence evaluations are unchanged, but
    model *selection* between checkpoints must pass "validation": choosing a
    checkpoint on the same band it was trained on is not a held-out comparison.
    """

    cases: list[tuple[int, TransitionGeometry, Path, bool]] = []
    sampler = TransitionGeometrySampler(
        seed=int(seed), difficulty=difficulty, transition_focus_fraction=1.0,
        dataset_split=dataset_split,
    )
    ordinal = 0
    for index in range(int(transition_cases)):
        geometry = sampler.sample(force_episode_type="transition_focus")
        cases.append((
            ordinal,
            geometry,
            output / "transitions" / f"case_{index:04d}",
            index == 0,
        ))
        ordinal += 1
    full_sampler = TransitionGeometrySampler(
        seed=int(seed) + 1_000_003,
        difficulty=difficulty,
        transition_focus_fraction=0.0,
        dataset_split=dataset_split,
    )
    for length in FULL_SEQUENCE_LENGTHS:
        completed = 0
        attempts = 0
        while completed < int(full_cases_per_length):
            geometry = full_sampler.sample(force_episode_type="full_sequence")
            attempts += 1
            if geometry.gate_count != length:
                if attempts > 10_000:
                    raise RuntimeError(
                        "could not deterministically sample requested sequence lengths"
                    )
                continue
            cases.append((
                ordinal,
                _geometry_with_length(geometry, length),
                output / "full_sequences" / f"length_{length}" / f"case_{completed:03d}",
                completed == 0,
            ))
            ordinal += 1
            completed += 1
    return cases


def _evaluate_case(
    model: Any,
    *,
    geometry: TransitionGeometry,
    output_dir: Path,
    adapter: str,
    frames_per_sec: bool | int,
    max_steps: int,
    record_trajectory: bool,
    action_source: str = "ppo_policy",
) -> Dict[str, Any]:
    completed = _completed_case(output_dir, geometry)
    if completed is not None:
        return completed
    row = evaluate_local_transition_episode(
        model,
        geometry=geometry,
        output_dir=output_dir,
        adapter=adapter,
        frames_per_sec=frames_per_sec,
        max_steps=max_steps,
        record_trajectory=record_trajectory,
        action_source=action_source,
    )
    _atomic_json(output_dir / "episode.json", row)
    return row


def _evaluate_checkpoint_batch(
    checkpoint: str,
    cases: Sequence[tuple[int, TransitionGeometry, Path, bool]],
    adapter: str,
    frames_per_sec: bool | int,
    max_steps: int,
    algorithm: str = "ppo",
) -> list[tuple[int, Dict[str, Any]]]:
    """Spawn-safe evaluator worker; owns one model and one engine at a time."""

    if algorithm == "ppo":
        from stable_baselines3 import PPO

        model = PPO.load(checkpoint, device="cpu")
    elif algorithm == "sac":
        from marine_race_arena.learning.sac_transition_policy import (
            load_sac_transition_policy,
        )

        model = load_sac_transition_policy(checkpoint, device="cpu")
    else:
        raise ValueError(f"unknown transition evaluation algorithm {algorithm!r}")
    rows = []
    for ordinal, geometry, output_dir, record_trajectory in cases:
        rows.append((ordinal, _evaluate_case(
            model,
            geometry=geometry,
            output_dir=output_dir,
            adapter=adapter,
            frames_per_sec=frames_per_sec,
            max_steps=max_steps,
            record_trajectory=record_trajectory,
            action_source=f"{algorithm}_policy",
        )))
    return rows


def evaluate_local_transition_episode(
    model: Any,
    *,
    geometry: TransitionGeometry,
    output_dir: str | Path,
    adapter: str = "holoocean",
    frames_per_sec: bool | int = False,
    max_steps: int = 3600,
    transition_window_steps: int = 30,
    record_trajectory: bool = False,
    frame_callback: Optional[Any] = None,
    action_source: str = "ppo_policy",
    render_camera: Optional[str] = None,
) -> Dict[str, Any]:
    output = Path(output_dir)
    track_path = generate_transition_track(
        geometry,
        output / "track.json",
        frames_per_sec=frames_per_sec,
    )
    if render_camera is not None:
        _configure_render_camera(track_path, render_camera)
    reward = LocalTransitionTrainingReward()
    env = MarineRaceGymEnv(
        str(track_path),
        seed=geometry.seed,
        adapter=adapter,
        allow_fallback=adapter == "fallback",
        current_profile="none",
        benchmark_task=BENCHMARK_TASK_CLEAN_GATE,
        max_steps=max_steps,
        observation_encoding_version=OBS_ENCODING_VERSION_LOCAL_TRANSITION,
        reward_fn=reward,
    )
    obs, _ = env.reset(seed=geometry.seed)
    if frame_callback is not None:
        frame_callback(env, 0)
    if adapter == "holoocean":
        setter = getattr(env.episode.context.adapter, "set_participant_body_velocity", None)
        if callable(setter):
            setter(env.episode.participant_id, geometry.initial_body_velocity_m_s)
    initial_expected = env.tracker.expected_beacon_id
    previous_action = np.zeros(4, dtype=np.float32)
    previous_delta = np.zeros(4, dtype=np.float32)
    jerk_sum = 0.0
    jerk_n = 0
    # Anti-inactivity accounting.  These never enter the observation; they exist
    # only so a motionless policy cannot be mistaken for a safe one.
    absolute_action_sum = np.zeros(4, dtype=np.float64)
    nontrivial_action_steps = 0
    distance_travelled_m = 0.0
    collision_entries = 0
    collision_contact_frames = 0
    in_contact = False
    first_cross_step: Optional[int] = None
    target_switch_step: Optional[int] = None
    acquisition_step: Optional[int] = None
    post_switch_previous_range: Optional[float] = None
    new_target_range_decreasing = False
    transition_positions: Dict[int, Dict[str, Any]] = {}
    active_transition_position: Optional[int] = None
    previous_active_transition_position: Optional[int] = None
    trajectory = []
    terminated = truncated = False
    try:
        while not (terminated or truncated):
            action = predict_local_transition_action(model, obs)
            obs, _, terminated, truncated, info = env.step(action)
            step_count = int(info["step_count"])
            if frame_callback is not None:
                frame_callback(env, step_count)
            delta = action - previous_action
            jerk_sum += float(np.linalg.norm(delta - previous_delta))
            jerk_n += 1
            previous_delta = delta
            previous_action = action.copy()
            absolute = np.abs(np.asarray(action, dtype=np.float64))
            absolute_action_sum += absolute
            nontrivial_action_steps += int(
                absolute.max() > NONTRIVIAL_ACTION_THRESHOLD
            )
            distance_travelled_m += float(info.get("step_distance_m", 0.0) or 0.0)
            contact_now = bool(info.get("collision_contact_frame")) or bool(
                info.get("obstacle_collision_frame")
            )
            if contact_now:
                collision_contact_frames += 1
                if not in_contact:
                    collision_entries += 1
            in_contact = contact_now
            gates = int(info.get("gate_crossings", 0))
            if gates > 0 and first_cross_step is None:
                first_cross_step = step_count
            if gates > 0:
                for position in range(1, min(gates, geometry.gate_count - 1) + 1):
                    transition_positions.setdefault(position, {
                        "gate_crossed": True,
                        "target_switched": False,
                        "aligned": False,
                        "range_decreased": False,
                    })
                if gates < geometry.gate_count:
                    active_transition_position = gates

            if active_transition_position != previous_active_transition_position:
                post_switch_previous_range = None
                previous_active_transition_position = active_transition_position

            context = env._ctx_source.last_context
            current_expected = context.expected_beacon_id
            if current_expected != initial_expected and target_switch_step is None:
                target_switch_step = step_count
            if active_transition_position is not None:
                row = transition_positions[active_transition_position]
                expected_id = f"B{active_transition_position + 1:02d}"
                if current_expected == expected_id:
                    row["target_switched"] = True
                    index = {name: offset for offset, name in enumerate(FEATURE_NAMES_LOCAL_TRANSITION)}
                    if float(obs[index["beacon_present"]]) > 0.5:
                        bearing = abs(math.degrees(math.atan2(
                            float(obs[index["beacon_bearing_sin"]]),
                            float(obs[index["beacon_bearing_cos"]]),
                        )))
                        elevation = abs(float(obs[index["beacon_elevation_norm"]]) * 90.0)
                        current_range = max(0.0, float(obs[index["beacon_range_norm"]]) * 30.0)
                        if bearing <= 20.0 and elevation <= 15.0:
                            row["aligned"] = True
                            if acquisition_step is None and active_transition_position == 1:
                                acquisition_step = step_count
                        if post_switch_previous_range is not None and current_range < post_switch_previous_range - 0.01:
                            row["range_decreased"] = True
                            if active_transition_position == 1:
                                new_target_range_decreasing = True
                        post_switch_previous_range = current_range
            if record_trajectory:
                trajectory.append({
                    "step": step_count,
                    "action": np.asarray(action).round(6).tolist(),
                    "gates": gates,
                    "expected_beacon": current_expected,
                })
            if (
                geometry.episode_type == "transition_focus"
                and first_cross_step is not None
                and step_count - first_cross_step >= transition_window_steps
            ):
                truncated = True

        referee = env.episode.context.referee.states[env.episode.participant_id]
        status = getattr(referee.status, "value", str(referee.status))
        first_transition = transition_positions.get(1, {})
        first_crossed = first_cross_step is not None
        switched = bool(first_transition.get("target_switched"))
        aligned = bool(first_transition.get("aligned"))
        range_decreased = bool(first_transition.get("range_decreased")) or new_target_range_decreasing
        no_previous_return = not reward.previous_gate_return_triggered
        no_missed = int(referee.missed_gate_attempts) == 0
        collision_events = int(referee.collision_events) + int(referee.obstacle_collision_events)
        tracker_diagnostics = env.tracker.diagnostics()
        success = all((
            first_crossed, switched, aligned, range_decreased,
            no_previous_return, no_missed, collision_events == 0,
            int(referee.out_of_bounds_events) == 0,
            int(referee.wrong_direction_crossings) == 0,
            not reward.acquisition_timeout_triggered,
        ))
        row = {
            "seed": geometry.seed,
            "difficulty": geometry.difficulty,
            "episode_type": geometry.episode_type,
            "geometry_group": geometry.geometry_group,
            "gate_count": geometry.gate_count,
            "first_gate_crossed": first_crossed,
            "correct_target_switch": switched,
            "new_target_acquired_and_aligned": aligned,
            "new_target_range_decreasing": range_decreased,
            "previous_gate_return": not no_previous_return,
            "missed_gate_dnf": int(referee.missed_gate_attempts),
            "collision_events": collision_events,
            "out_of_bounds_events": int(referee.out_of_bounds_events),
            "wrong_direction_events": int(referee.wrong_direction_crossings),
            "acquisition_timeout": bool(reward.acquisition_timeout_triggered),
            "acquisition_time_s": (
                None if acquisition_step is None or first_cross_step is None
                else round((acquisition_step - first_cross_step) * env.episode.dt, 3)
            ),
            "action_jerk": round(jerk_sum / max(1, jerk_n), 6),
            "collision_entries": collision_entries,
            "collision_contact_frames": collision_contact_frames,
            "action_steps": int(jerk_n),
            "nontrivial_action_steps": int(nontrivial_action_steps),
            "mean_absolute_action_per_axis": [
                round(float(value / max(1, jerk_n)), 6)
                for value in absolute_action_sum
            ],
            "distance_travelled_m": round(float(distance_travelled_m), 4),
            "universal_transition_success": success,
            "full_sequence_completion": (
                status == "FINISHED"
                and int(referee.valid_gate_crossings) == geometry.gate_count
            ),
            "gates_completed": int(referee.valid_gate_crossings),
            "transition_positions": transition_positions,
            "tracker_final_phase": tracker_diagnostics["phase"],
            "tracker_local_completed": int(tracker_diagnostics["local_completed"]),
            "tracker_commit_entry_reason": tracker_diagnostics["commit_entry_reason"],
            "tracker_last_advancement_evidence": tracker_diagnostics[
                "last_advancement_evidence"
            ],
            "referee_tracker_first_crossing_mismatch": bool(
                first_crossed and int(tracker_diagnostics["local_completed"]) < 1
            ),
            "action_source": str(action_source),
        }
        if record_trajectory:
            _atomic_json(output / "trajectory.json", {
                "episode": row,
                "trajectory": trajectory,
            })
        return row
    finally:
        env.close()


def _activity_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Anti-inactivity evidence, ``None`` where a report predates the metric.

    Motion is never treated as success.  These values exist only so a policy
    that produces few safety events by remaining almost stationary is detected
    and rejected instead of being ranked as the safest candidate.
    """

    rows = list(rows)
    n_eval = max(1, len(rows))
    completed_gates = sum(int(row.get("gates_completed", 0) or 0) for row in rows)
    reached_first = sum(bool(row.get("first_gate_crossed")) for row in rows)
    distances = [
        float(row["distance_travelled_m"])
        for row in rows
        if row.get("distance_travelled_m") is not None
    ]
    action_steps = sum(int(row.get("action_steps", 0) or 0) for row in rows)
    nontrivial_steps = sum(
        int(row.get("nontrivial_action_steps", 0) or 0) for row in rows
    )
    per_axis_rows = [
        row["mean_absolute_action_per_axis"]
        for row in rows
        if row.get("mean_absolute_action_per_axis")
        and int(row.get("action_steps", 0) or 0) > 0
    ]
    weights = [
        float(row.get("action_steps", 0) or 0)
        for row in rows
        if row.get("mean_absolute_action_per_axis")
        and int(row.get("action_steps", 0) or 0) > 0
    ]
    if per_axis_rows:
        per_axis = np.average(
            np.asarray(per_axis_rows, dtype=np.float64), axis=0, weights=weights
        )
        per_axis_metric = {
            name: float(per_axis[index])
            for index, name in enumerate(("surge", "sway", "heave", "yaw"))
        }
        mean_absolute_action = float(per_axis.mean())
    else:
        per_axis_metric = None
        mean_absolute_action = None
    return {
        "completed_gate_count": completed_gates,
        "mean_completed_gates_per_episode": completed_gates / n_eval,
        "fraction_of_episodes_reaching_first_gate": reached_first / n_eval,
        "mean_distance_travelled_m": (
            None if not distances else float(np.mean(distances))
        ),
        "mean_absolute_action_per_axis": per_axis_metric,
        "mean_absolute_action": mean_absolute_action,
        "nontrivial_action_fraction": (
            None if action_steps <= 0 else nontrivial_steps / action_steps
        ),
        "nontrivial_action_threshold": NONTRIVIAL_ACTION_THRESHOLD,
        "collision_entries": (
            None
            if not any("collision_entries" in row for row in rows)
            else sum(int(row.get("collision_entries", 0) or 0) for row in rows)
        ),
        "collision_contact_frames": (
            None
            if not any("collision_contact_frames" in row for row in rows)
            else sum(int(row.get("collision_contact_frames", 0) or 0) for row in rows)
        ),
    }


def aggregate_transition_benchmark(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    transition_rows = [row for row in rows if row.get("episode_type") == "transition_focus"]
    full_rows = [row for row in rows if row.get("episode_type") == "full_sequence"]
    acquisitions = [
        float(row["acquisition_time_s"])
        for row in transition_rows if row.get("acquisition_time_s") is not None
    ]
    by_length = {}
    for length in FULL_SEQUENCE_LENGTHS:
        selected = [row for row in full_rows if int(row.get("gate_count", 0)) == length]
        by_length[str(length)] = {
            "n": len(selected),
            "completion_rate": (
                None if not selected else sum(bool(row.get("full_sequence_completion")) for row in selected) / len(selected)
            ),
        }
    positions: Dict[str, Dict[str, Any]] = {}
    for row in full_rows:
        for position, result in dict(row.get("transition_positions") or {}).items():
            key = str(position)
            bucket = positions.setdefault(key, {"n": 0, "successes": 0})
            bucket["n"] += 1
            bucket["successes"] += int(all(bool(result.get(name)) for name in (
                "gate_crossed", "target_switched", "aligned", "range_decreased"
            )))
    for value in positions.values():
        value["success_rate"] = value["successes"] / max(1, value["n"])
    activity = _activity_metrics(rows)
    metrics = {
        "n_eval": len(rows),
        "transition_n": len(transition_rows),
        "universal_transition_success_rate": (
            None if not transition_rows else sum(
                bool(row.get("universal_transition_success")) for row in transition_rows
            ) / len(transition_rows)
        ),
        "first_gate_crossing_rate": (
            None if not transition_rows else sum(bool(row.get("first_gate_crossed")) for row in transition_rows) / len(transition_rows)
        ),
        "target_switch_rate": (
            None if not transition_rows else sum(bool(row.get("correct_target_switch")) for row in transition_rows) / len(transition_rows)
        ),
        "new_target_alignment_rate": (
            None if not transition_rows else sum(bool(row.get("new_target_acquired_and_aligned")) for row in transition_rows) / len(transition_rows)
        ),
        "new_target_range_decrease_rate": (
            None if not transition_rows else sum(bool(row.get("new_target_range_decreasing")) for row in transition_rows) / len(transition_rows)
        ),
        "previous_gate_returns": sum(bool(row.get("previous_gate_return")) for row in rows),
        "missed_gate_dnf": sum(int(row.get("missed_gate_dnf", 0)) for row in rows),
        "collision_episodes": sum(int(row.get("collision_events", 0)) > 0 for row in rows),
        "collision_events": sum(int(row.get("collision_events", 0)) for row in rows),
        "out_of_bounds_episodes": sum(int(row.get("out_of_bounds_events", 0)) > 0 for row in rows),
        "wrong_direction_events": sum(int(row.get("wrong_direction_events", 0)) for row in rows),
        "acquisition_timeouts": sum(bool(row.get("acquisition_timeout")) for row in rows),
        "mean_acquisition_time_s": None if not acquisitions else float(np.mean(acquisitions)),
        "mean_action_jerk": None if not rows else float(np.mean([
            float(row.get("action_jerk", 0.0)) for row in rows
        ])),
        "full_sequence_success_by_length": by_length,
        "transition_success_by_position": positions,
        "all_actions_policy_generated": all(
            row.get("action_source") in {"ppo_policy", "sac_policy"}
            for row in rows
        ),
        **activity,
    }
    metrics["acquisition_timeout_rate"] = (
        None if not rows else metrics["acquisition_timeouts"] / len(rows)
    )
    metrics["long_sequence_completion_score"] = float(np.mean([
        value["completion_rate"]
        for value in by_length.values() if value["completion_rate"] is not None
    ])) if any(value["completion_rate"] is not None for value in by_length.values()) else None
    metrics["safety_clean"] = all(int(metrics[key] or 0) == 0 for key in (
        "previous_gate_returns", "missed_gate_dnf", "collision_episodes",
        "out_of_bounds_episodes", "wrong_direction_events", "acquisition_timeouts",
    ))
    return metrics


def transition_checkpoint_rank_key(
    metrics: Mapping[str, Any],
    thresholds: CompetenceThresholds = DEFAULT_COMPETENCE_THRESHOLDS,
) -> tuple:
    """Rank checkpoints by competence qualification first, then by safety.

    The previous key summed safety events only, which ranked a policy that
    barely moved above one that actually crossed gates.  A checkpoint that fails
    the competence gate now sorts below every competent checkpoint regardless of
    how few events it produced.
    """

    return transition_policy_rank_key(metrics, thresholds)


def transition_checkpoint_is_competent(
    metrics: Mapping[str, Any],
    thresholds: CompetenceThresholds = DEFAULT_COMPETENCE_THRESHOLDS,
) -> bool:
    return evaluate_competence_gate(metrics, thresholds).passed


def evaluate_checkpoint_universal_transition_benchmark(
    checkpoint: str | Path,
    *,
    output_dir: str | Path,
    seed: int,
    difficulty: str = "G6",
    transition_cases: int = 1000,
    full_cases_per_length: int = 5,
    adapter: str = "holoocean",
    frames_per_sec: bool | int = False,
    max_steps: int = 3600,
    parallel_workers: int = 2,
    algorithm: str = "ppo",
    dataset_split: str = "train",
) -> Dict[str, Any]:
    """Resume-safe checkpoint benchmark across isolated evaluator processes.

    Cases are a pure function of ``(seed, difficulty, counts, dataset_split)``,
    so two checkpoints given the same arguments face byte-identical geometry.
    That is what makes a paired comparison between them valid -- but the caller
    must give each checkpoint its own ``output_dir``, because resume detection
    keys on the case geometry and cannot tell which policy produced a row.
    """

    if int(transition_cases) < 1:
        raise ValueError("transition benchmark requires at least one case")
    workers = max(1, int(parallel_workers))
    output = Path(output_dir)
    cases = _benchmark_cases(
        output=output,
        seed=int(seed),
        difficulty=difficulty,
        transition_cases=int(transition_cases),
        full_cases_per_length=int(full_cases_per_length),
        dataset_split=dataset_split,
    )
    rows_by_ordinal: Dict[int, Dict[str, Any]] = {}
    pending = []
    for case in cases:
        ordinal, geometry, case_output, _ = case
        completed = _completed_case(case_output, geometry)
        if completed is None:
            pending.append(case)
        else:
            rows_by_ordinal[ordinal] = completed

    if pending and workers == 1:
        for ordinal, row in _evaluate_checkpoint_batch(
            str(checkpoint), pending, adapter, frames_per_sec, int(max_steps),
            algorithm,
        ):
            rows_by_ordinal[ordinal] = row
    elif pending:
        chunks = [pending[index::workers] for index in range(workers)]
        chunks = [chunk for chunk in chunks if chunk]
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=len(chunks), mp_context=context
        ) as executor:
            futures = [
                executor.submit(
                    _evaluate_checkpoint_batch,
                    str(checkpoint),
                    chunk,
                    adapter,
                    frames_per_sec,
                    int(max_steps),
                    algorithm,
                )
                for chunk in chunks
            ]
            for future in as_completed(futures):
                for ordinal, row in future.result():
                    rows_by_ordinal[ordinal] = row

    if len(rows_by_ordinal) != len(cases):
        raise RuntimeError(
            f"benchmark completed {len(rows_by_ordinal)} of {len(cases)} cases"
        )
    rows = [rows_by_ordinal[index] for index in range(len(cases))]
    report = {
        "schema_version": "universal_transition_benchmark_v1",
        "observation_version": OBS_ENCODING_VERSION_LOCAL_TRANSITION,
        "seed": int(seed),
        "difficulty": difficulty,
        "transition_cases": int(transition_cases),
        "full_cases_per_length": int(full_cases_per_length),
        "parallel_workers": workers,
        "algorithm": algorithm,
        "dataset_split": dataset_split,
        "checkpoint": str(checkpoint),
        "metrics": aggregate_transition_benchmark(rows),
        "episodes": rows,
    }
    _atomic_json(output / "evaluation.json", report)
    return report
