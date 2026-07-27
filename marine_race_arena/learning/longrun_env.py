"""Gym wrapper that rebuilds a fresh HoloOcean episode from the curriculum."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np

try:
    import gymnasium as gym
except Exception:  # pragma: no cover - only the non-RL environment
    gym = None

from marine_race_arena.learning.gym_env import MarineRaceGymEnv
from marine_race_arena.learning.multigate_curriculum import (
    HORSESHOE,
    MIXED,
    SINGLE_GATE,
    SIX_GATE,
    THREE_GATE_S,
    TWO_GATE_STRAIGHT,
    VERTICAL,
)
from marine_race_arena.learning.parametric_curriculum import (
    CurriculumSampler,
    TransitionGeometry,
    generate_two_gate_track,
)

_BASE = gym.Env if gym is not None else object
_WRAPPER_BASE = gym.Wrapper if gym is not None else object


class ObservationFrameStack(_WRAPPER_BASE):
    """Explicit, manifest-stamped temporal fallback for a separate experiment."""

    def __init__(self, env: Any, frame_stack: int) -> None:
        if gym is None:  # pragma: no cover
            raise ImportError("gymnasium is required for observation stacking")
        if int(frame_stack) not in {3, 4}:
            raise ValueError("frame_stack must be 3 or 4")
        super().__init__(env)
        self.frame_stack = int(frame_stack)
        shape = tuple(env.observation_space.shape or ())
        if len(shape) != 1:
            raise ValueError("frame stacking requires a flat observation")
        self.single_observation_dim = int(shape[0])
        self.observation_space = gym.spaces.Box(
            low=np.tile(env.observation_space.low, self.frame_stack),
            high=np.tile(env.observation_space.high, self.frame_stack),
            dtype=env.observation_space.dtype,
        )
        self._frames = np.zeros(
            (self.frame_stack, self.single_observation_dim),
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        frame = np.asarray(observation, dtype=self.observation_space.dtype)
        self._frames[:] = frame
        return self._frames.reshape(-1).copy(), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._frames[:-1] = self._frames[1:]
        self._frames[-1] = np.asarray(
            observation, dtype=self.observation_space.dtype
        )
        return (
            self._frames.reshape(-1).copy(),
            reward,
            terminated,
            truncated,
            info,
        )


def apply_observation_mode(env: Any, *, policy_mode: str, frame_stack: int) -> Any:
    if policy_mode == "feedforward":
        if int(frame_stack) != 1:
            raise ValueError("feedforward observation mode requires one frame")
        return env
    if policy_mode == "frame_stack":
        return ObservationFrameStack(env, frame_stack)
    raise ValueError(f"unsupported policy mode {policy_mode!r}")


class CurriculumMarineRaceEnv(_BASE):
    """One SB3 environment whose underlying track changes between episodes.

    Every reset closes the old race and starts a new episode.  Simulator physical
    state is never claimed to survive a reset or process resume.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        sampler: CurriculumSampler,
        *,
        run_dir: str | Path,
        seed: int,
        reward_factory: Callable[[], Any],
        env_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if gym is None:  # pragma: no cover
            raise ImportError("gymnasium is required for long-run PPO")
        super().__init__()
        self.sampler = sampler
        self.run_dir = Path(run_dir)
        self.base_seed = int(seed)
        self.reward_factory = reward_factory
        self.env_kwargs = dict(env_kwargs or {})
        self.generated_dir = self.run_dir / "generated_tracks"
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self._child: Optional[MarineRaceGymEnv] = None
        self._episode_counter = 0
        self.current_geometry: Optional[TransitionGeometry] = None
        self.current_track: Optional[str] = None
        self.simulator_restart_count = 0

        # Build a lightweight child once to expose stable spaces to SB3. It is not
        # reset until SB3 asks for the first episode.
        self._child = self._make_child(TWO_GATE_STRAIGHT, self.base_seed)
        self.observation_space = self._child.observation_space
        self.action_space = self._child.action_space

    @property
    def episode(self):
        if self._child is None:
            raise RuntimeError("curriculum environment has no active episode")
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

    def _select_track(self, geometry: TransitionGeometry) -> str:
        stage_index = int(geometry.stage[1:])
        if geometry.source == "retention":
            return SINGLE_GATE
        if geometry.source == "straight":
            return TWO_GATE_STRAIGHT
        if stage_index == 5:
            return THREE_GATE_S
        if stage_index == 6:
            return SIX_GATE
        if stage_index >= 7:
            official = (HORSESHOE, VERTICAL, MIXED)
            return official[self._episode_counter % len(official)]
        target = self.generated_dir / "active_two_gate.json"
        generate_two_gate_track(geometry, target)
        return str(target)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ):
        if self._child is not None:
            self._child.close()
            self._child = None
        geometry = self.sampler.sample()
        self.current_geometry = geometry
        self.current_track = self._select_track(geometry)
        episode_seed = (
            int(seed)
            if seed is not None
            else self.base_seed + self._episode_counter
        )
        self._episode_counter += 1
        self._child = self._make_child(self.current_track, episode_seed)
        obs, info = self._child.reset(seed=episode_seed)
        info.update(
            {
                "curriculum_stage": self.sampler.current_stage,
                "curriculum_source": geometry.source,
                "transition_direction": geometry.direction,
                "transition_angle_deg": geometry.signed_turn_deg,
                "geometry_group": geometry.geometry_group,
                "track": self.current_track,
            }
        )
        self._append_geometry(episode_seed, geometry)
        return obs, info

    def step(self, action):
        if self._child is None:
            raise RuntimeError("reset must be called before step")
        obs, reward, terminated, truncated, info = self._child.step(action)
        geometry = self.current_geometry
        if geometry is not None:
            info.update(
                {
                    "curriculum_stage": self.sampler.current_stage,
                    "curriculum_source": geometry.source,
                    "transition_direction": geometry.direction,
                    "transition_angle_deg": geometry.signed_turn_deg,
                    "geometry_group": geometry.geometry_group,
                    "track": self.current_track,
                }
            )
        if terminated or truncated:
            progress = self._child.episode.referee_progress()
            state = self._child.episode.context.referee.states[
                self._child.episode.participant_id
            ]
            info.update(
                {
                    "episode_finished": progress["status"] == "FINISHED",
                    "episode_status": progress["status"],
                    "episode_gate_count": int(progress["valid_gate_crossings"]),
                    "episode_collisions": int(state.collision_events),
                    "episode_out_of_bounds": int(state.out_of_bounds_events),
                    "episode_wrong_direction": int(state.wrong_direction_crossings),
                }
            )
        return obs, reward, terminated, truncated, info

    def _append_geometry(self, episode_seed: int, geometry: TransitionGeometry) -> None:
        path = self.run_dir / "logs" / "sampled_geometries.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "episode": self._episode_counter,
            "seed": int(episode_seed),
            **geometry.__dict__,
            "geometry_group": geometry.geometry_group,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    def close(self) -> None:
        if self._child is not None:
            self._child.close()
            self._child = None
