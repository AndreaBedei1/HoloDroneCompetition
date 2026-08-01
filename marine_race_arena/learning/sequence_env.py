"""Gym environment that rebuilds a procedural S1--S5 path each episode."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np

try:
    import gymnasium as gym
except Exception:  # pragma: no cover
    gym = None

from marine_race_arena.learning.config_sequence import OBS_ENCODING_VERSION_SEQUENCE
from marine_race_arena.learning.gym_env import MarineRaceGymEnv
from marine_race_arena.learning.sequence_curriculum import (
    SequenceCurriculumSampler,
    SequenceGeometry,
    generate_sequence_track,
)

_BASE = gym.Env if gym is not None else object


class SequenceCurriculumEnv(_BASE):
    """One policy-action-only environment with procedural episode geometry."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        sampler: SequenceCurriculumSampler,
        *,
        run_dir: str | Path,
        seed: int,
        reward_factory: Callable[[], Any],
        env_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if gym is None:  # pragma: no cover
            raise ImportError("gymnasium is required for sequence PPO")
        super().__init__()
        self.sampler = sampler
        self.run_dir = Path(run_dir)
        self.base_seed = int(seed)
        self.reward_factory = reward_factory
        self.env_kwargs = dict(env_kwargs or {})
        self.env_kwargs["observation_encoding_version"] = OBS_ENCODING_VERSION_SEQUENCE
        self.generated_dir = self.run_dir / "generated_tracks"
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self._child: Optional[MarineRaceGymEnv] = None
        self._episode_counter = 0
        self.current_geometry: Optional[SequenceGeometry] = None
        self.current_track: Optional[str] = None
        self._previous_position: Optional[np.ndarray] = None
        self._previous_action = np.zeros(4, dtype=np.float32)
        self._previous_action_delta = np.zeros(4, dtype=np.float32)
        self._path_length = 0.0
        self._jerk_sum = 0.0
        self._jerk_samples = 0
        self._next_acquisition_steps: list[int] = []
        self._awaiting_acquisition_at: Optional[int] = None
        self._previous_gate_returns = 0
        self._return_armed = False
        self._last_gate_count = 0

        bootstrap = self.sampler.sample()
        path = generate_sequence_track(bootstrap, self.generated_dir / "bootstrap.json")
        self._child = self._make_child(str(path), self.base_seed)
        self.observation_space = self._child.observation_space
        self.action_space = self._child.action_space

    @property
    def episode(self):
        if self._child is None:
            raise RuntimeError("sequence environment has no active episode")
        return self._child.episode

    @property
    def tracker(self):
        return None if self._child is None else self._child.tracker

    def _make_child(self, track: str, seed: int) -> MarineRaceGymEnv:
        return MarineRaceGymEnv(
            track,
            seed=seed,
            reward_fn=self.reward_factory(),
            **self.env_kwargs,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None):
        if self._child is not None:
            self._child.close()
        geometry = self.sampler.sample()
        self.current_geometry = geometry
        path = self.generated_dir / "active_sequence.json"
        generate_sequence_track(geometry, path)
        self.current_track = str(path)
        episode_seed = int(seed) if seed is not None else self.base_seed + self._episode_counter
        self._episode_counter += 1
        self._child = self._make_child(self.current_track, episode_seed)
        observation, info = self._child.reset(seed=episode_seed)
        race = self._child.episode
        position = race.context.adapter.get_participant_state(race.participant_id).position
        self._previous_position = np.asarray(position, dtype=np.float64)
        self._previous_action.fill(0.0)
        self._previous_action_delta.fill(0.0)
        self._path_length = self._jerk_sum = 0.0
        self._jerk_samples = 0
        self._next_acquisition_steps = []
        self._awaiting_acquisition_at = None
        self._previous_gate_returns = 0
        self._return_armed = False
        self._last_gate_count = 0
        info.update(self._context_info())
        self._append_geometry(episode_seed, geometry)
        return observation, info

    def step(self, action):
        if self._child is None:
            raise RuntimeError("reset must be called before step")
        action_array = np.asarray(action, dtype=np.float32).reshape(4)
        observation, reward, terminated, truncated, info = self._child.step(action_array)
        race = self._child.episode
        position = np.asarray(
            race.context.adapter.get_participant_state(race.participant_id).position,
            dtype=np.float64,
        )
        if self._previous_position is not None:
            self._path_length += float(np.linalg.norm(position - self._previous_position))
        self._previous_position = position
        delta = action_array - self._previous_action
        self._jerk_sum += float(np.linalg.norm(delta - self._previous_action_delta))
        self._jerk_samples += 1
        self._previous_action_delta = delta
        self._previous_action = action_array.copy()

        gate_count = int(info.get("gate_crossings", 0))
        if gate_count > self._last_gate_count:
            self._awaiting_acquisition_at = int(info["step_count"])
            self._return_armed = True
        context = getattr(self._child._ctx_source, "last_context", None)
        if (self._awaiting_acquisition_at is not None and context is not None
                and context.new_target_acquired_since_crossing):
            self._next_acquisition_steps.append(
                int(info["step_count"]) - self._awaiting_acquisition_at
            )
            self._awaiting_acquisition_at = None
        self._observe_previous_gate_return(position, gate_count)
        self._last_gate_count = gate_count
        info.update(self._context_info())
        if terminated or truncated:
            info.update(self._terminal_metrics())
        return observation, reward, terminated, truncated, info

    def _observe_previous_gate_return(self, position: np.ndarray, gate_count: int) -> None:
        if not self._return_armed or gate_count <= 0 or self.current_geometry is None:
            return
        gates = self._child.episode.context.config.gates
        previous = gates[min(gate_count - 1, len(gates) - 1)]
        center = np.asarray(previous.position, dtype=np.float64)
        normal = np.asarray(previous.passage_direction, dtype=np.float64)
        signed = float(np.dot(position - center, normal))
        if signed >= 0.75:
            self._return_armed = True
        elif signed <= -0.25 and self._return_armed:
            self._previous_gate_returns += 1
            self._return_armed = False

    def _context_info(self) -> Dict[str, Any]:
        geometry = self.current_geometry
        return {
            "sequence_stage": self.sampler.current_stage,
            "sequence_source": None if geometry is None else geometry.source,
            "sequence_pattern": None if geometry is None else geometry.pattern,
            "sequence_gate_count": None if geometry is None else geometry.gate_count,
            "sequence_geometry_group": None if geometry is None else geometry.geometry_group,
            "track": self.current_track,
        }

    def _terminal_metrics(self) -> Dict[str, Any]:
        referee = self.episode.context.referee.states[self.episode.participant_id]
        status = getattr(referee.status, "value", str(referee.status))
        expected = len(self.episode.context.config.track.gate_sequence)
        steps = max(1, int(self.episode.step_count))
        return {
            "episode_finished": status == "FINISHED",
            "full_sequence_completion": status == "FINISHED" and int(referee.valid_gate_crossings) == expected,
            "episode_status": status,
            "episode_gates_completed": int(referee.valid_gate_crossings),
            "episode_expected_gates": expected,
            "episode_missed_gate_dnf": int(referee.missed_gate_attempts),
            "episode_previous_gate_returns": self._previous_gate_returns,
            "episode_next_gate_acquisition_times_s": [round(v * self.episode.dt, 3) for v in self._next_acquisition_steps],
            "episode_collisions": int(referee.collision_events) + int(referee.obstacle_collision_events),
            "episode_out_of_bounds": int(referee.out_of_bounds_events),
            "episode_wrong_direction": int(referee.wrong_direction_crossings),
            "episode_completion_time_s": round(steps * self.episode.dt, 3),
            "episode_time_per_gate_s": round(steps * self.episode.dt / max(1, int(referee.valid_gate_crossings)), 3),
            "episode_path_length_m": round(self._path_length, 4),
            "episode_action_jerk": round(self._jerk_sum / max(1, self._jerk_samples), 6),
        }

    def _append_geometry(self, seed: int, geometry: SequenceGeometry) -> None:
        path = self.run_dir / "logs" / "sampled_sequences.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"episode": self._episode_counter, "seed": seed,
                                     **geometry.__dict__, "geometry_group": geometry.geometry_group},
                                    separators=(",", ":")) + "\n")

    def close(self) -> None:
        if self._child is not None:
            self._child.close()
            self._child = None
