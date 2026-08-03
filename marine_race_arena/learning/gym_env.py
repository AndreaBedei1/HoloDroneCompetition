"""Gymnasium-compatible, step-wise Marine Race environment.

Wraps :class:`~marine_race_arena.learning.episode.RaceEpisode` (which reuses the
benchmark runner internals) with the Gymnasium API. The policy sees only the
encoded onboard observation; the reward may use privileged simulator/referee
state and is therefore kept in a separate, swappable ``reward_fn``.

Observation: fixed ``float32`` vector of size ``OBS_DIM`` (see
:mod:`observation_encoder`). Action: ``float32`` vector ``[surge, sway, heave,
yaw]`` in ``[-1, 1]`` mapped straight to the normalized body-frame command; the
adapter clamps to the vehicle's control limits.

Gymnasium (and, for the default reward, nothing else) is imported lazily so the
numpy-only learning modules stay importable without it.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np

try:  # Gymnasium is an RL-only dependency (requirements-rl.txt).
    import gymnasium as gym
    from gymnasium import spaces

    _GYM_BASE = gym.Env
except Exception as exc:  # pragma: no cover - exercised only without gymnasium
    gym = None
    spaces = None
    _GYM_BASE = object
    _GYM_IMPORT_ERROR = exc
else:
    _GYM_IMPORT_ERROR = None

from marine_race_arena.learning.config import (
    ACTION_AXES,
    ACTION_DIM,
    FEATURE_BOUNDS,
    OBS_DIM,
    OBS_ENCODING_VERSION,
)
from marine_race_arena.learning.episode import EpisodeStep, RaceEpisode
from marine_race_arena.learning.observation_encoder import encode_observation
from marine_race_arena.learning.reward import TrainingReward
from marine_race_arena.learning.tracker_context import OnboardContextTracker

# Reward callable: (env, step, gate_delta, action) -> (reward, components_dict).
# A reward object may also expose ``reset(env)`` to restart per-episode state.
RewardFn = Callable[["MarineRaceGymEnv", EpisodeStep, int, np.ndarray], Tuple[float, Dict[str, float]]]


class MarineRaceGymEnv(_GYM_BASE):
    """Single-vehicle, onboard-only Gymnasium environment over a marine race."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        track: str,
        *,
        seed: int = 0,
        dt: float = 0.1,
        adapter: str = "fallback",
        allow_fallback: bool = True,
        max_steps: int = 2000,
        official: bool = True,
        duration_s: Optional[float] = None,
        benchmark_task: Optional[str] = None,
        current_profile: Optional[str] = None,
        obstacles: Optional[str] = None,
        obstacle_density: Optional[str] = None,
        reward_fn: Optional[RewardFn] = None,
        start_randomization=None,
        episode_seed_stream: Optional[int] = None,
        observation_encoding_version: str = OBS_ENCODING_VERSION,
    ) -> None:
        if gym is None:  # pragma: no cover - only without gymnasium installed
            raise ImportError(
                "gymnasium is required for MarineRaceGymEnv; install requirements-rl.txt"
            ) from _GYM_IMPORT_ERROR
        super().__init__()
        self._episode = RaceEpisode(
            track,
            seed=seed,
            dt=dt,
            adapter=adapter,
            allow_fallback=allow_fallback,
            max_steps=max_steps,
            official=official,
            duration_s=duration_s,
            benchmark_task=benchmark_task,
            current_profile=current_profile,
            obstacles=obstacles,
            obstacle_density=obstacle_density,
            start_randomization=start_randomization,
        )
        self.observation_encoding_version = str(observation_encoding_version)
        if self.observation_encoding_version == OBS_ENCODING_VERSION:
            self._feature_bounds = FEATURE_BOUNDS
            self._obs_dim = OBS_DIM
            self._context_type = OnboardContextTracker
            self._encoder = encode_observation
            default_reward = TrainingReward()
        else:
            from marine_race_arena.learning.config_v3 import (
                FEATURE_BOUNDS_V3,
                OBS_DIM_V3,
                OBS_ENCODING_VERSION_V3,
            )

            if self.observation_encoding_version not in {
                OBS_ENCODING_VERSION_V3,
                "onboard_ppo_sequence_v4",
                "onboard_local_transition_v1",
            }:
                raise ValueError(
                    f"unsupported observation encoding {self.observation_encoding_version!r}"
                )
            if self.observation_encoding_version == "onboard_local_transition_v1":
                from marine_race_arena.learning.config_local_transition import (
                    FEATURE_BOUNDS_LOCAL_TRANSITION,
                    OBS_DIM_LOCAL_TRANSITION,
                )
                from marine_race_arena.learning.observation_encoder_local_transition import (
                    encode_observation_local_transition,
                )
                from marine_race_arena.learning.reward_local_transition import (
                    LocalTransitionTrainingReward,
                )
                from marine_race_arena.learning.tracker_context_local_transition import (
                    OnboardLocalTransitionContextTracker,
                )

                self._feature_bounds = FEATURE_BOUNDS_LOCAL_TRANSITION
                self._obs_dim = OBS_DIM_LOCAL_TRANSITION
                self._context_type = OnboardLocalTransitionContextTracker
                self._encoder = encode_observation_local_transition
                default_reward = LocalTransitionTrainingReward()
            elif self.observation_encoding_version == "onboard_ppo_sequence_v4":
                from marine_race_arena.learning.config_sequence import (
                    FEATURE_BOUNDS_SEQUENCE,
                    OBS_DIM_SEQUENCE,
                )
                from marine_race_arena.learning.observation_encoder_sequence import (
                    encode_observation_sequence,
                )
                from marine_race_arena.learning.reward_sequence import SequenceTrainingReward
                from marine_race_arena.learning.tracker_context_sequence import (
                    OnboardSequenceContextTracker,
                )

                self._feature_bounds = FEATURE_BOUNDS_SEQUENCE
                self._obs_dim = OBS_DIM_SEQUENCE
                self._context_type = OnboardSequenceContextTracker
                self._encoder = encode_observation_sequence
                default_reward = SequenceTrainingReward()
            else:
                from marine_race_arena.learning.observation_encoder_v3 import (
                    encode_observation_v3,
                )
                from marine_race_arena.learning.reward_v3 import MultiGateTrainingReward
                from marine_race_arena.learning.tracker_context_v3 import (
                    OnboardMultiGateContextTracker,
                )

                self._feature_bounds = FEATURE_BOUNDS_V3
                self._obs_dim = OBS_DIM_V3
                self._context_type = OnboardMultiGateContextTracker
                self._encoder = encode_observation_v3
                default_reward = MultiGateTrainingReward()
        self._reward_fn: RewardFn = reward_fn or default_reward
        self._ctx_source = None
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._last_gates = 0
        # Training-only: when set and reset() is called without an explicit seed, vary the
        # randomization seed each episode so training sees diverse randomized starts. Eval
        # always passes an explicit seed, so it is unaffected.
        self._episode_seed_stream = episode_seed_stream
        self._episode_counter = 0

        low = np.array([b[0] for b in self._feature_bounds], dtype=np.float32)
        high = np.array([b[1] for b in self._feature_bounds], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low, high=high, shape=(self._obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

    # ------------------------------------------------------------------ props
    @property
    def episode(self) -> RaceEpisode:
        return self._episode

    @property
    def tracker(self):
        return self._ctx_source.tracker if self._ctx_source is not None else None

    # ------------------------------------------------------------------ api
    def reset(self, *, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None):
        if gym is not None:
            super().reset(seed=seed)
        effective_seed = seed
        if (effective_seed is None and self._episode_seed_stream is not None
                and self._episode.start_randomization is not None):
            effective_seed = int(self._episode_seed_stream) + self._episode_counter
            self._episode_counter += 1
        obs_dict = self._episode.reset(seed=effective_seed)
        ctx_cfg = self._episode.context.config
        total_beacons = max(1, len(ctx_cfg.track.gate_sequence))
        laps = max(1, int(ctx_cfg.race.laps))
        self._ctx_source = self._context_type(total_beacons=total_beacons, laps=laps)
        self._ctx_source.reset(obs_dict)
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._last_gates = self._episode.referee_progress()["valid_gate_crossings"]
        if hasattr(self._reward_fn, "reset"):
            self._reward_fn.reset(self)
        encoded = self._encode(obs_dict)
        return encoded, self._info(terminated=False, truncated=False, components={})

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != ACTION_DIM:
            raise ValueError(f"action must have {ACTION_DIM} elements, got {action.shape[0]}")
        action = np.clip(np.nan_to_num(action, nan=0.0), -1.0, 1.0)
        command = {axis: float(action[i]) for i, axis in enumerate(ACTION_AXES)}

        step = self._episode.step(command)
        gates_now = self._episode.referee_progress()["valid_gate_crossings"]
        gate_delta = max(0, gates_now - self._last_gates)
        self._last_gates = gates_now

        reward, components = self._reward_fn(self, step, gate_delta, action)
        # Temporal convention: observation o_(t+1) carries the action a_t that was
        # actually applied this step, so update prev_action BEFORE encoding. This
        # matches TrajectoryRecorder and RLGateController (train/inference parity).
        self._prev_action = action
        encoded = self._encode(step.observation)
        info = self._info(step.terminated, step.truncated, components)
        info["gate_crossings"] = gates_now
        bounds = self._episode.context.arena.bounds
        position = step.current_state.position
        out_of_bounds_frame = bounds.violation_reason(position) is not None
        x, y, z = position
        boundary_margin = min(
            x - bounds.x_min,
            bounds.x_max - x,
            y - bounds.y_min,
            bounds.y_max - y,
            z - bounds.z_min,
            bounds.z_max - z,
        )
        info.update(
            {
                "collision_contact_frame": bool(step.collision),
                "obstacle_collision_frame": bool(step.obstacle_collisions),
                "out_of_bounds_frame": bool(out_of_bounds_frame),
                "safety_warning_frame": bool(
                    not out_of_bounds_frame and boundary_margin <= 0.5
                ),
            }
        )
        return encoded, float(reward), bool(step.terminated), bool(step.truncated), info

    def _encode(self, obs_dict: Mapping[str, Any]) -> np.ndarray:
        assert self._ctx_source is not None
        context = self._ctx_source.context(
            obs_dict, dt=self._episode.dt, prev_action=self._prev_action.tolist()
        )
        return self._encoder(obs_dict, context)

    def _info(self, terminated: bool, truncated: bool, components: Mapping[str, float]) -> Dict[str, Any]:
        return {
            "reward_components": dict(components),
            "step_count": self._episode.step_count,
            "expected_gate_id": self._episode.expected_gate_id(),
        }

    def close(self):
        self._episode.close()
