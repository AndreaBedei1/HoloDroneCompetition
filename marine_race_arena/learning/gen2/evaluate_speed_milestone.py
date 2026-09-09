"""Deterministic speed milestone evaluation (diagnostic only).

Runs matched full-track episodes for learned checkpoints and the official rule
controller while recording the legal action stream, including surge statistics.
No controller input is augmented with simulator/referee state.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

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
from marine_race_arena.learning.gen2.recurrent_policy import Gen2RecurrentController
from marine_race_arena.learning.gen2.track_fragments import OFFICIAL_TRACKS
from marine_race_arena.learning.gen2 import track_fragments as tf
from marine_race_arena.learning.tracker_context_local_transition_27d import (
    OnboardLocalTransition27dContextTracker,
)


class _RecordedController:
    """Delegate a learned controller while retaining its exact action stream."""

    def __init__(self, controller: Gen2RecurrentController) -> None:
        self._controller = controller
        self.model = controller.model
        self.actions: list[np.ndarray] = []

    def reset(self) -> None:
        self.actions.clear()
        self._controller.reset()

    def act(self, observation: np.ndarray, first_step: bool = False) -> np.ndarray:
        action = np.asarray(self._controller.act(observation, first_step=first_step), dtype=np.float32)
        self.actions.append(action.copy())
        return action


def _surge_stats(actions: list[np.ndarray]) -> dict[str, float]:
    if not actions:
        return {"mean_surge": 0.0, "max_surge": 0.0, "min_surge": 0.0}
    surge = np.asarray(actions, dtype=np.float64)[:, 0]
    return {
        "mean_surge": round(float(np.mean(surge)), 6),
        "max_surge": round(float(np.max(surge)), 6),
        "min_surge": round(float(np.min(surge)), 6),
    }


def _learned(checkpoint: str, track: str, seed: int, label: str, max_steps: int) -> dict[str, Any]:
    from sb3_contrib import RecurrentPPO

    model = RecurrentPPO.load(checkpoint, device="cpu")
    recorder = _RecordedController(Gen2RecurrentController(model, deterministic=True))
    try:
        result = run_policy_episode(
            recorder,
            tf.track_path(track),
            seed=int(seed),
            adapter="holoocean",
            allow_fallback=False,
            max_steps=int(max_steps),
        )
        row = result.as_row()
        row.update({
            "controller": label,
            "track": track,
            "adapter": "holoocean",
            "fallback_used": False,
            "observation_contract": "onboard_local_transition_27d_v1",
            **_surge_stats(recorder.actions),
        })
        return row
    finally:
        recorder._controller = None  # release recurrent model references promptly
        del model


def _rules(track: str, seed: int, max_steps: int) -> dict[str, Any]:
    path = tf.track_path(track)
    started = time.perf_counter()
    episode = RaceEpisode(
        str(path), seed=int(seed), dt=0.1, adapter="holoocean", allow_fallback=False,
        max_steps=int(max_steps), official=True, current_profile="none",
        benchmark_task=BENCHMARK_TASK_CLEAN_GATE,
    )
    controller = RuleGateCenterThenCommitController()
    actions: list[np.ndarray] = []
    previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
    previous_delta = np.zeros(ACTION_DIM, dtype=np.float32)
    previous_position = None
    path_length = 0.0
    jerk_total = 0.0
    vision_steps = orientation_steps = steps = 0
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
        max_steps = int(max_steps)
        while steps < max_steps:
            context = tracker.context(raw, dt=0.1, prev_action=previous_action.tolist())
            vision_steps += int(getattr(context, "visual_target", None) is not None)
            orientation_steps += int(bool(getattr(context, "gate_orientation_present", False)))
            action = command_to_action(controller.step(_RecordingObservation(raw)))
            action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
            actions.append(action.copy())
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
            steps += 1
            time_s = float(step.time_s)
            if step.terminated or step.truncated:
                break
        state = episode.context.referee.states[episode.participant_id]
        status = getattr(state.status, "value", str(state.status))
        gates = int(state.valid_gate_crossings)
        return {
            "controller": "rules",
            "track": track,
            "seed": int(seed),
            "gate_count": gate_total,
            "gates_completed": gates,
            "succeeded": bool(status == "FINISHED" and gates == gate_total),
            "status": status,
            "steps": steps,
            "completion_time_s": round(time_s, 3),
            "path_length_m": round(path_length, 3),
            "collision_events": int(state.collision_events) + int(state.obstacle_collision_events),
            "out_of_bounds_events": int(state.out_of_bounds_events),
            "wrong_direction_crossings": int(state.wrong_direction_crossings),
            "missed_gate_attempts": int(state.missed_gate_attempts),
            "timeout": bool(steps >= max_steps and status != "FINISHED"),
            "mean_action_jerk": round(jerk_total / max(1, steps), 6),
            "vision_availability": round(vision_steps / max(1, steps), 4),
            "orientation_availability": round(orientation_steps / max(1, steps), 4),
            "adapter": "holoocean",
            "fallback_used": False,
            "observation_contract": "onboard_local_transition_27d_v1",
            "wall_time_s": round(time.perf_counter() - started, 2),
            **_surge_stats(actions),
        }
    finally:
        try:
            controller.close()
        finally:
            episode.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retry6", required=True)
    parser.add_argument("--candidate25k", required=True)
    parser.add_argument("--seed", type=int, default=66001)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--track", choices=tuple(OFFICIAL_TRACKS), action="append")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    tracks = tuple(args.track) if args.track else tuple(OFFICIAL_TRACKS)
    for track in tracks:
        rows.append(_learned(args.retry6, track, args.seed, "ppo_retry6_50k", args.max_steps))
        rows.append(_learned(args.candidate25k, track, args.seed, "ppo_candidate_25k", args.max_steps))
        rows.append(_rules(track, args.seed, args.max_steps))
    payload = {
        "schema_version": "gen2_speed_milestone_v1",
        "seed": int(args.seed),
        "tracks": list(OFFICIAL_TRACKS),
        "rows": rows,
        "controller_observation": "onboard_only",
        "adapter": "holoocean",
        "fallback_used": False,
        "fog": {"enabled": True, "density": 5.0, "start_distance_m": 1.0, "color_rgb": [0.4, 0.6, 1.0]},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
