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

from marine_race_arena.controllers.local_course_tracker import LocalCourseTracker
from marine_race_arena.learning.config import (
    ACTION_AXES,
    ACTION_DIM,
    FEATURE_BOUNDS,
    OBS_DIM,
    LearningContext,
)
from marine_race_arena.learning.episode import EpisodeStep, RaceEpisode
from marine_race_arena.learning.observation_encoder import _depth_m, encode_observation
from marine_race_arena.learning.reward import TrainingReward

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
        current_profile: Optional[str] = None,
        obstacles: Optional[str] = None,
        obstacle_density: Optional[str] = None,
        reward_fn: Optional[RewardFn] = None,
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
            current_profile=current_profile,
            obstacles=obstacles,
            obstacle_density=obstacle_density,
        )
        self._reward_fn: RewardFn = reward_fn or TrainingReward()
        self._tracker: Optional[LocalCourseTracker] = None
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._depth_ref: Optional[float] = None
        self._last_gates = 0

        low = np.array([b[0] for b in FEATURE_BOUNDS], dtype=np.float32)
        high = np.array([b[1] for b in FEATURE_BOUNDS], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

    # ------------------------------------------------------------------ props
    @property
    def episode(self) -> RaceEpisode:
        return self._episode

    @property
    def tracker(self) -> Optional[LocalCourseTracker]:
        return self._tracker

    # ------------------------------------------------------------------ api
    def reset(self, *, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None):
        if gym is not None:
            super().reset(seed=seed)
        obs_dict = self._episode.reset(seed=seed)
        ctx_cfg = self._episode.context.config
        total_beacons = max(1, len(ctx_cfg.track.gate_sequence))
        laps = max(1, int(ctx_cfg.race.laps))
        self._tracker = LocalCourseTracker(
            initial_beacon_id="B01", total_beacons=total_beacons, laps=laps
        )
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        sensors = obs_dict.get("sensors") or {}
        self._depth_ref = _depth_m(sensors)
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
        encoded = self._encode(step.observation)
        self._prev_action = action
        info = self._info(step.terminated, step.truncated, components)
        info["gate_crossings"] = gates_now
        return encoded, float(reward), bool(step.terminated), bool(step.truncated), info

    def _encode(self, obs_dict: Mapping[str, Any]) -> np.ndarray:
        context = self._build_context(obs_dict)
        return encode_observation(obs_dict, context)

    def _build_context(self, obs_dict: Mapping[str, Any]) -> LearningContext:
        sensors = obs_dict.get("sensors") or {}
        assert self._tracker is not None
        self._tracker.update(
            local_time_s=float(obs_dict.get("local_time_s", 0.0)),
            beacons=obs_dict.get("beacons") or [],
            camera_image=sensors.get("FrontCamera"),
            dvl_velocity=sensors.get("DVLSensor"),
            dt=self._episode.dt,
        )
        visual_lock = getattr(self._tracker, "_latest_visual_target", None) is not None
        return LearningContext(
            expected_beacon_id=self._tracker.expected_beacon_id,
            tracker_phase=self._tracker.phase,
            local_beacon_index=self._tracker.local_beacon_index,
            local_lap=self._tracker.local_lap,
            total_beacons=self._tracker.total_beacons,
            laps=self._tracker.laps,
            depth_reference_m=self._depth_ref,
            visual_lock=visual_lock,
            prev_action=self._prev_action.tolist(),
        )

    def _info(self, terminated: bool, truncated: bool, components: Mapping[str, float]) -> Dict[str, Any]:
        return {
            "reward_components": dict(components),
            "step_count": self._episode.step_count,
            "expected_gate_id": self._episode.expected_gate_id(),
        }

    def close(self):
        self._episode.close()
