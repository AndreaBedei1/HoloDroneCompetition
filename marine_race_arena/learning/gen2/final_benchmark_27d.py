"""Matched final benchmark for the frozen Gen-2 27-D PPO and rule baseline.

This runner intentionally keeps the two inference paths separate: the learned
controller uses :func:`run_policy_episode`, while the rule controller is driven
from the same legal onboard observation and the same ``RaceEpisode`` settings.
Ground truth is read only after the episode for offline score accounting.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from marine_race_arena.config.benchmark_tasks import BENCHMARK_TASK_CLEAN_GATE
from marine_race_arena.controllers.official_baselines import RuleGateCenterThenCommitController
from marine_race_arena.learning.config import ACTION_AXES, ACTION_DIM
from marine_race_arena.learning.episode import RaceEpisode
from marine_race_arena.learning.gen2.evaluation import run_policy_episode
from marine_race_arena.learning.gen2.expert_rollout import (
    _RecordingObservation,
    _mission_info_for,
    action_to_command,
    assert_observation_is_legal,
    command_to_action,
)
from marine_race_arena.learning.gen2.track_fragments import OFFICIAL_TRACKS
from marine_race_arena.learning.gen2 import track_fragments as tf
from marine_race_arena.learning.tracker_context_local_transition_27d import (
    OnboardLocalTransition27dContextTracker,
)


def _rule_episode(track: str, seed: int) -> dict[str, Any]:
    path = tf.track_path(track)
    started = time.perf_counter()
    episode = RaceEpisode(
        str(path), seed=int(seed), dt=0.1, adapter="holoocean",
        allow_fallback=False, max_steps=20000, official=True,
        current_profile="none", benchmark_task=BENCHMARK_TASK_CLEAN_GATE,
    )
    controller = RuleGateCenterThenCommitController()
    observations = []
    crossings = []
    previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
    previous_delta = np.zeros(ACTION_DIM, dtype=np.float32)
    previous_position = None
    path_length = 0.0
    jerk_total = 0.0
    vision_steps = 0
    orientation_steps = 0
    steps = 0
    time_s = 0.0
    try:
        raw = episode.reset(seed=int(seed))
        assert_observation_is_legal(raw)
        gate_total = len(episode.context.config.track.gate_sequence)
        tracker = OnboardLocalTransition27dContextTracker(
            total_beacons=max(1, gate_total),
            laps=max(1, int(episode.context.config.race.laps)),
        )
        tracker.reset(raw)
        controller.reset(_mission_info_for(episode))
        max_steps = max(2000, gate_total * 900)
        while steps < max_steps:
            context = tracker.context(raw, dt=0.1, prev_action=previous_action.tolist())
            vision_steps += int(getattr(context, "visual_target", None) is not None)
            orientation_steps += int(bool(getattr(context, "gate_orientation_present", False)))
            view = _RecordingObservation(raw)
            action = command_to_action(controller.step(view))
            delta = action - previous_action
            jerk_total += float(np.linalg.norm(delta - previous_delta))
            previous_delta = delta
            step = episode.step(action_to_command(action))
            position = np.asarray(step.current_state.position, dtype=np.float64)
            if previous_position is not None:
                path_length += float(np.linalg.norm(position - previous_position))
            previous_position = position
            previous_action = action
            raw = step.observation
            assert_observation_is_legal(raw)
            steps += 1
            time_s = float(step.time_s)
            crossings.append(int(episode.referee_progress()["valid_gate_crossings"]))
            if step.terminated or step.truncated:
                break
        state = episode.context.referee.states[episode.participant_id]
        status = getattr(state.status, "value", str(state.status))
        gates = int(state.valid_gate_crossings)
        completed = bool(status == "FINISHED" and gates == gate_total)
        return {
            "controller": "rule_gate_center_then_commit",
            "track": track, "seed": int(seed), "gate_count": gate_total,
            "gates_completed": gates, "succeeded": completed, "status": status,
            "steps": steps, "completion_time_s": round(time_s, 3),
            "path_length_m": round(path_length, 3),
            "mean_action_jerk": round(jerk_total / max(1, steps), 6),
            "collision_events": int(state.collision_events) + int(state.obstacle_collision_events),
            "out_of_bounds_events": int(state.out_of_bounds_events),
            "wrong_direction_crossings": int(state.wrong_direction_crossings),
            "missed_gate_attempts": int(state.missed_gate_attempts),
            "timeout": bool(not completed and steps >= max_steps),
            "vision_availability": round(vision_steps / max(1, steps), 4),
            "orientation_availability": round(orientation_steps / max(1, steps), 4),
            "adapter": "holoocean", "fallback_used": False,
            "wall_time_s": round(time.perf_counter() - started, 2),
            "observation_contract": "onboard_local_transition_27d_v1",
        }
    finally:
        try:
            controller.close()
        finally:
            episode.close()


def _learned_episode(checkpoint: str, track: str, seed: int) -> dict[str, Any]:
    from sb3_contrib import RecurrentPPO
    from marine_race_arena.learning.gen2.recurrent_policy import Gen2RecurrentController

    controller = Gen2RecurrentController(
        RecurrentPPO.load(checkpoint, device="cpu"), deterministic=True
    )
    result = run_policy_episode(
        controller, tf.track_path(track), seed=int(seed), spec=None,
        adapter="holoocean", allow_fallback=False,
        max_steps=max(2000, len(tf.load_track(track)["track"]["gate_sequence"]) * 900),
    )
    row = result.as_row()
    row.update({
        "controller": "ppo_25k",
        "track": track,
        "adapter": "holoocean", "fallback_used": False,
        "observation_contract": "onboard_local_transition_27d_v1",
    })
    return row


def run_job(controller: str, track: str, seeds: Iterable[int], out: Path, checkpoint: str | None) -> None:
    rows = []
    for seed in seeds:
        rows.append(_rule_episode(track, int(seed)) if controller == "rules" else _learned_episode(checkpoint or "", track, int(seed)))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=("rules", "ppo25k"), required=True)
    parser.add_argument("--track", choices=tuple(OFFICIAL_TRACKS), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str)
    args = parser.parse_args()
    run_job(args.controller, args.track, args.seeds, args.out, args.checkpoint)


if __name__ == "__main__":
    main()
