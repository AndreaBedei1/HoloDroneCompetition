"""Deterministic evaluation and reliability-first ranking for sequence PPO."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from marine_race_arena.learning.config_sequence import OBS_ENCODING_VERSION_SEQUENCE
from marine_race_arena.learning.gym_env import MarineRaceGymEnv
from marine_race_arena.learning.reward_sequence import SequenceTrainingReward
from marine_race_arena.learning.sequence_curriculum import (
    SequenceCurriculumSampler,
    SequenceGeometry,
    generate_sequence_track,
)
from marine_race_arena.learning.sequence_policy import predict_sequence_action


OFFICIAL_TRACKS = (
    "marine_race_arena/tracks/marine_race_horseshoe_bay.json",
    "marine_race_arena/tracks/marine_race_vertical_serpent.json",
    "marine_race_arena/tracks/marine_race_mixed_endurance.json",
)
RETENTION_TRACKS = (
    "marine_race_arena/tracks/training/stage1_single_gate.json",
    "marine_race_arena/tracks/tests/two_gate_straight.json",
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    if not rows:
        return None
    return sum(bool(row.get(key)) for row in rows) / len(rows)


def aggregate_sequence_evaluation(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate measured categories; an absent category is always ``None``."""
    rows = list(rows)
    completion = _rate(rows, "full_sequence_completion")
    by_category: Dict[str, Any] = {}
    for category in ("single_gate", "two_gate", "vertical", "three_gate", "current_stage", "official"):
        selected = [row for row in rows if row.get("category") == category]
        by_category[f"{category}_completion_rate"] = _rate(selected, "full_sequence_completion")
        by_category[f"{category}_n"] = len(selected)
    acquisitions = [
        float(value)
        for row in rows
        for value in row.get("next_gate_acquisition_times_s", [])
    ]
    completed_gates = sum(int(row.get("gates_completed", 0)) for row in rows)
    metrics = {
        "n_eval": len(rows),
        "full_sequence_completion_rate": completion,
        "completed_sequences": sum(bool(row.get("full_sequence_completion")) for row in rows),
        "gates_completed": completed_gates,
        "longest_sequence_reliably_completed": max(
            (int(row.get("expected_gates", 0)) for row in rows
             if row.get("full_sequence_completion") and not row.get("any_safety")),
            default=0,
        ),
        "missed_gate_dnf": sum(int(row.get("missed_gate_dnf", 0)) for row in rows),
        "previous_gate_returns": sum(int(row.get("previous_gate_returns", 0)) for row in rows),
        "collision_episodes": sum(int(row.get("collisions", 0)) > 0 for row in rows),
        "collision_events": sum(int(row.get("collisions", 0)) for row in rows),
        "out_of_bounds_episodes": sum(int(row.get("out_of_bounds", 0)) > 0 for row in rows),
        "wrong_direction_events": sum(int(row.get("wrong_direction", 0)) for row in rows),
        "mean_next_gate_acquisition_time_s": (
            float(np.mean(acquisitions)) if acquisitions else None
        ),
        "mean_completion_time_s": (
            float(np.mean([row["completion_time_s"] for row in rows
                           if row.get("full_sequence_completion")]))
            if any(row.get("full_sequence_completion") for row in rows) else None
        ),
        "mean_time_per_gate_s": (
            float(np.mean([row["time_per_gate_s"] for row in rows])) if rows else None
        ),
        "mean_path_length_m": (
            float(np.mean([row["path_length_m"] for row in rows])) if rows else None
        ),
        "mean_action_jerk": (
            float(np.mean([row["action_jerk"] for row in rows])) if rows else None
        ),
        "all_actions_policy_generated": all(row.get("action_source") == "ppo_policy" for row in rows),
        **by_category,
    }
    metrics["safety_clean"] = all(metrics[key] == 0 for key in (
        "collision_episodes", "out_of_bounds_episodes", "wrong_direction_events",
        "previous_gate_returns", "missed_gate_dnf",
    ))
    return metrics


def checkpoint_rank_key(metrics: Mapping[str, Any]) -> tuple:
    """Long reliable sequences outrank shorter policies regardless of speed."""
    safety_clean = all(int(metrics.get(key, 0) or 0) == 0 for key in (
        "collision_episodes", "out_of_bounds_episodes", "wrong_direction_events",
    ))
    sequence_clean = all(int(metrics.get(key, 0) or 0) == 0 for key in (
        "missed_gate_dnf", "previous_gate_returns",
    ))
    completion = metrics.get("full_sequence_completion_rate")
    time = metrics.get("mean_completion_time_s")
    jerk = metrics.get("mean_action_jerk")
    return (
        int(metrics.get("longest_sequence_reliably_completed", 0)),
        -1.0 if completion is None else float(completion),
        int(safety_clean),
        int(sequence_clean),
        -float("inf") if time is None else -float(time),
        -float("inf") if jerk is None else -float(jerk),
    )


def is_better_sequence_checkpoint(candidate: Mapping[str, Any], incumbent: Optional[Mapping[str, Any]]) -> bool:
    return incumbent is None or checkpoint_rank_key(candidate) > checkpoint_rank_key(incumbent)


def _previous_gate_return(
    position: np.ndarray,
    gates: Sequence[Any],
    completed: int,
    armed: bool,
) -> tuple[bool, bool]:
    if completed <= 0:
        return False, armed
    gate = gates[min(completed - 1, len(gates) - 1)]
    center = np.asarray(gate.position, dtype=np.float64)
    normal = np.asarray(gate.passage_direction, dtype=np.float64)
    signed = float(np.dot(position - center, normal))
    if signed >= 0.75:
        return False, True
    if signed <= -0.25 and armed:
        return True, False
    return False, armed


def evaluate_episode(
    model: Any,
    *,
    architecture: str,
    track: str,
    seed: int,
    category: str,
    adapter: str,
    max_steps: int,
    trajectory_path: Optional[Path] = None,
) -> Dict[str, Any]:
    env = MarineRaceGymEnv(
        track,
        seed=seed,
        adapter=adapter,
        allow_fallback=adapter == "fallback",
        current_profile="none",
        max_steps=max_steps,
        observation_encoding_version=OBS_ENCODING_VERSION_SEQUENCE,
        reward_fn=SequenceTrainingReward(),
    )
    obs, _ = env.reset(seed=seed)
    recurrent_state = None
    episode_start = np.ones((1,), dtype=bool)
    previous_position = np.asarray(
        env.episode.context.adapter.get_participant_state(env.episode.participant_id).position,
        dtype=np.float64,
    )
    previous_action = np.zeros(4, dtype=np.float32)
    previous_delta = np.zeros(4, dtype=np.float32)
    path_length = jerk_sum = 0.0
    jerk_n = 0
    returns = 0
    return_armed = False
    last_gates = 0
    awaiting_acquisition: Optional[int] = None
    acquisition_times = []
    trajectory = []
    terminated = truncated = False
    try:
        while not (terminated or truncated):
            action, recurrent_state = predict_sequence_action(
                model, obs, architecture=architecture,
                recurrent_state=recurrent_state, episode_start=episode_start,
            )
            episode_start[:] = False
            obs, _, terminated, truncated, info = env.step(action)
            position = np.asarray(
                env.episode.context.adapter.get_participant_state(env.episode.participant_id).position,
                dtype=np.float64,
            )
            path_length += float(np.linalg.norm(position - previous_position))
            previous_position = position
            delta = action.reshape(4) - previous_action
            jerk_sum += float(np.linalg.norm(delta - previous_delta))
            jerk_n += 1
            previous_delta = delta
            previous_action = action.reshape(4).copy()
            gates = int(info.get("gate_crossings", 0))
            if gates > last_gates:
                awaiting_acquisition = int(info["step_count"])
                return_armed = True
            context = getattr(env._ctx_source, "last_context", None)
            if (awaiting_acquisition is not None and context is not None
                    and context.new_target_acquired_since_crossing):
                acquisition_times.append((int(info["step_count"]) - awaiting_acquisition) * env.episode.dt)
                awaiting_acquisition = None
            returned, return_armed = _previous_gate_return(
                position, env.episode.context.config.gates, gates, return_armed
            )
            returns += int(returned)
            last_gates = gates
            trajectory.append({
                "step": int(info["step_count"]),
                "position": position.round(5).tolist(),
                "action": action.reshape(4).round(6).tolist(),
                "gates": gates,
                "expected_beacon": getattr(env.tracker, "expected_beacon_id", None),
                "local_completed": getattr(env.tracker, "local_completed", None),
                "target_changed": getattr(context, "expected_beacon_changed", None),
                "target_acquired": getattr(context, "new_target_acquired_since_crossing", None),
            })
        referee = env.episode.context.referee.states[env.episode.participant_id]
        status = getattr(referee.status, "value", str(referee.status))
        expected = len(env.episode.context.config.track.gate_sequence)
        steps = int(env.episode.step_count)
        row = {
            "seed": int(seed), "track": track, "category": category,
            "status": status, "gates_completed": int(referee.valid_gate_crossings),
            "expected_gates": expected,
            "full_sequence_completion": status == "FINISHED" and int(referee.valid_gate_crossings) == expected,
            "missed_gate_dnf": int(referee.missed_gate_attempts),
            "previous_gate_returns": returns,
            "next_gate_acquisition_times_s": [round(v, 3) for v in acquisition_times],
            "collisions": int(referee.collision_events) + int(referee.obstacle_collision_events),
            "out_of_bounds": int(referee.out_of_bounds_events),
            "wrong_direction": int(referee.wrong_direction_crossings),
            "completion_time_s": round(steps * env.episode.dt, 3),
            "time_per_gate_s": round(steps * env.episode.dt / max(1, int(referee.valid_gate_crossings)), 3),
            "path_length_m": round(path_length, 4),
            "action_jerk": round(jerk_sum / max(1, jerk_n), 6),
            "action_source": "ppo_policy",
        }
        row["any_safety"] = bool(row["collisions"] or row["out_of_bounds"] or row["wrong_direction"])
        if trajectory_path is not None:
            _atomic_json(trajectory_path, {"episode": row, "trajectory": trajectory})
        return row
    finally:
        env.close()


def _procedural_track(stage: str, seed: int, output: Path) -> str:
    mixture = {
        "current_stage": 1.0, "three_gate": 0.0, "previous_stage": 0.0,
        "short_retention": 0.0, "targeted_failure": 0.0,
    }
    sampler = SequenceCurriculumSampler(
        seed=seed, initial_stage=stage, maximum_stage=stage, replay_mixture=mixture
    )
    geometry = sampler.sample()
    return str(generate_sequence_track(geometry, output))


def _vertical_retention_track(seed: int, output: Path) -> str:
    sign = 1.0 if seed % 2 else -1.0
    geometry = SequenceGeometry(
        stage="S1", source="evaluation_retention", pattern=(
            "high_to_low" if sign > 0 else "low_to_high"
        ), seed=int(seed), gate_count=2, spacings_m=(5.0,),
        turn_deltas_deg=(0.0,), vertical_deltas_m=(sign * 1.1,),
        initial_yaw_error_deg=float((seed % 7) - 3),
        initial_lateral_offset_m=float(((seed % 5) - 2) * 0.1),
    )
    return str(generate_sequence_track(geometry, output))


def evaluate_sequence_suite(
    model: Any,
    *,
    architecture: str,
    stage: str,
    seeds: Sequence[int],
    output_dir: str | Path,
    adapter: str = "holoocean",
    max_steps: int = 2400,
    official: bool = False,
) -> Dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    specs = []
    if official:
        for index, seed in enumerate(seeds):
            specs.append((OFFICIAL_TRACKS[index % len(OFFICIAL_TRACKS)], int(seed), "official"))
    else:
        for index, seed in enumerate(seeds):
            if index < max(2, len(seeds) // 3):
                case_stage, category = "S1", "three_gate"
            elif index < max(4, 2 * len(seeds) // 3):
                case_stage, category = stage, "current_stage"
            else:
                retention_index = index % 3
                if retention_index == 2:
                    track = _vertical_retention_track(
                        int(seed), output / "tracks" / f"vertical_{seed}.json"
                    )
                    category = "vertical"
                else:
                    track = RETENTION_TRACKS[retention_index]
                    category = "single_gate" if "single_gate" in track else "two_gate"
                specs.append((track, int(seed), category))
                continue
            track = _procedural_track(
                case_stage, int(seed), output / "tracks" / f"{category}_{seed}.json"
            )
            specs.append((track, int(seed), category))
    for index, (track, seed, category) in enumerate(specs):
        trajectory_path = output / "trajectories" / f"{category}_{seed}.json"
        rows.append(evaluate_episode(
            model, architecture=architecture, track=track, seed=seed,
            category=category, adapter=adapter, max_steps=max_steps,
            trajectory_path=trajectory_path,
        ))
    metrics = aggregate_sequence_evaluation(rows)
    report = {"stage": stage, "architecture": architecture, "official": official,
              "seeds": list(map(int, seeds)), "metrics": metrics, "episodes": rows}
    _atomic_json(output / "evaluation.json", report)
    _save_representative_plot(rows, output)
    return report


def _save_representative_plot(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    if not rows:
        return
    representative = next((row for row in rows if not row["full_sequence_completion"]), rows[0])
    path = output / "trajectories" / f"{representative['category']}_{representative['seed']}.json"
    if not path.exists():
        return
    try:
        import matplotlib.pyplot as plt
        value = json.loads(path.read_text(encoding="utf-8"))
        xyz = np.asarray([item["position"] for item in value["trajectory"]])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(xyz[:, 0], xyz[:, 1]); axes[0].set(xlabel="x (m)", ylabel="y (m)", title="Horizontal")
        axes[1].plot(xyz[:, 0], xyz[:, 2]); axes[1].set(xlabel="x (m)", ylabel="z (m)", title="Vertical")
        fig.suptitle(f"{representative['category']} seed {representative['seed']}")
        fig.tight_layout(); fig.savefig(output / "representative_trajectory.png", dpi=180); plt.close(fig)
        # Compact trajectory video (policy path + progress marker). Camera video
        # capture remains available to the official benchmark runner; this file
        # is deterministic and inexpensive enough for every full evaluation.
        import imageio.v2 as imageio
        frames = []
        stride = max(1, len(xyz) // 60)
        for cursor in range(1, len(xyz) + 1, stride):
            fig, axis = plt.subplots(figsize=(5, 4))
            axis.plot(xyz[:, 0], xyz[:, 1], color="#bbbbbb", linewidth=1)
            axis.plot(xyz[:cursor, 0], xyz[:cursor, 1], color="#0066cc", linewidth=2)
            axis.scatter(xyz[cursor - 1, 0], xyz[cursor - 1, 1], color="#cc2200", s=24)
            axis.set(xlabel="x (m)", ylabel="y (m)", title=f"step {cursor}")
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            frames.append(frame)
            plt.close(fig)
        if frames:
            try:
                imageio.mimsave(output / "representative_trajectory.mp4", frames, fps=10)
            except Exception:
                imageio.mimsave(output / "representative_trajectory.gif", frames, fps=10)
    except Exception:
        return
