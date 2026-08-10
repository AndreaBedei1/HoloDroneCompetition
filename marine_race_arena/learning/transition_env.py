"""Worker-isolated environment for universal transition PPO."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np

try:
    import gymnasium as gym
except Exception:  # pragma: no cover
    gym = None

from marine_race_arena.config.benchmark_tasks import BENCHMARK_TASK_CLEAN_GATE
from marine_race_arena.learning.config_local_transition import (
    OBS_ENCODING_VERSION_LOCAL_TRANSITION,
)
from marine_race_arena.learning.gym_env import MarineRaceGymEnv
from marine_race_arena.learning.longrun_checkpoint import atomic_append_jsonl
from marine_race_arena.learning.reward_local_transition import (
    LocalTransitionRewardConfig,
    LocalTransitionTrainingReward,
)
from marine_race_arena.learning.provenance import now_utc
from marine_race_arena.learning.episode_composition_log import (
    append_episode,
    episode_row,
)
from marine_race_arena.learning.generic_sequence_curriculum import (
    GenericSequenceCurriculum,
)
from marine_race_arena.learning.transition_curriculum import (
    TransitionGeometry,
    TransitionGeometrySampler,
    generate_transition_track,
)


_BASE = gym.Env if gym is not None else object


def _build_sequence_curriculum(spec: Any) -> Optional[GenericSequenceCurriculum]:
    """Construct the live curriculum from config, or return ``None``.

    An empty mapping must NOT silently mean "disabled": that is precisely how
    the curriculum stayed dead code while the logs showed the old coin flip.
    Enabling is explicit, and unknown keys are rejected rather than ignored.
    """

    if spec is None or isinstance(spec, GenericSequenceCurriculum):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError(f"unsupported sequence_curriculum {type(spec).__name__}")
    settings = dict(spec)
    if not settings.pop("enabled", False):
        return None
    allowed = {"stages", "promotion", "demotion"}
    unknown = sorted(set(settings) - allowed)
    if unknown:
        raise ValueError(f"unknown sequence_curriculum keys {unknown}")
    kwargs = {k: v for k, v in settings.items() if v}
    return GenericSequenceCurriculum(**kwargs)


class UniversalTransitionEnv(_BASE):
    """Exactly one simulator and one sampler, owned by one worker process."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        run_dir: str | Path,
        worker_id: int,
        sampler_seed: int,
        difficulty: str,
        transition_focus_fraction: float = 0.70,
        adapter: str = "holoocean",
        allow_fallback: bool = False,
        frames_per_sec: bool | int = False,
        max_episode_steps: int = 3600,
        reward_config: Optional[Mapping[str, Any]] = None,
        sequence_curriculum: Optional[Mapping[str, Any]] = None,
        dataset_split: str = "train",
        algorithm: str = "unknown",
        log_episode_composition: bool = True,
    ) -> None:
        if gym is None:  # pragma: no cover
            raise ImportError("gymnasium is required")
        super().__init__()
        self.run_dir = Path(run_dir)
        self.worker_id = int(worker_id)
        self.sampler_seed = int(sampler_seed)
        self.worker_dir = self.run_dir / "workers" / f"worker_{self.worker_id:02d}"
        self.generated_dir = self.worker_dir / "generated_tracks"
        self.logs_dir = self.worker_dir / "logs"
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        # A rollout worker may only ever draw TRAIN episodes.  Asserting it in
        # the constructor makes validation/test leakage impossible rather than
        # merely discouraged.
        if str(dataset_split) != "train":
            raise ValueError(
                "rollout workers must sample from the TRAIN split, got "
                f"{dataset_split!r}"
            )
        self.dataset_split = str(dataset_split)
        self.algorithm = str(algorithm)
        self.log_episode_composition = bool(log_episode_composition)
        # The generic sequence curriculum, when configured, replaces the binary
        # focus/full-sequence coin flip with a real length mixture.
        self.sequence_curriculum = _build_sequence_curriculum(sequence_curriculum)
        self.sampler = TransitionGeometrySampler(
            seed=self.sampler_seed,
            difficulty=difficulty,
            transition_focus_fraction=transition_focus_fraction,
            sequence_curriculum=self.sequence_curriculum,
            dataset_split=self.dataset_split,
        )
        self.adapter_name = str(adapter)
        self.allow_fallback = bool(allow_fallback)
        self.frames_per_sec = frames_per_sec
        self.max_episode_steps = int(max_episode_steps)
        self.reward_config = LocalTransitionRewardConfig(**dict(reward_config or {}))
        self._efficiency_unlocked = bool(self.reward_config.efficiency_unlocked)
        self._child: Optional[MarineRaceGymEnv] = None
        self._episode_counter = 0
        self.current_geometry: Optional[TransitionGeometry] = None
        self.current_track: Optional[str] = None
        self._first_gate_crossing_step: Optional[int] = None
        self._previous_action = np.zeros(4, dtype=np.float32)
        self._previous_action_delta = np.zeros(4, dtype=np.float32)
        self._jerk_sum = 0.0
        self._jerk_samples = 0
        self._sensor_health_enabled = False
        self._sensor_frames = 0
        self._sensor_missing = {
            name: 0 for name in ("FrontCamera", "DVLSensor", "IMUSensor", "DepthSensor")
        }
        self._sensor_repeated = dict(self._sensor_missing)
        self._sensor_previous: Dict[str, Any] = {}

        bootstrap = self.sampler.sample(force_episode_type="transition_focus")
        path = generate_transition_track(
            bootstrap,
            self.generated_dir / "bootstrap.json",
            frames_per_sec=self.frames_per_sec,
        )
        self._child = self._make_child(str(path), self.sampler_seed)
        self.observation_space = self._child.observation_space
        self.action_space = self._child.action_space

    @property
    def episode(self):
        if self._child is None:
            raise RuntimeError("transition environment has no active child")
        return self._child.episode

    @property
    def tracker(self):
        return None if self._child is None else self._child.tracker

    def _new_reward(self) -> LocalTransitionTrainingReward:
        values = dict(self.reward_config.__dict__)
        values["efficiency_unlocked"] = self._efficiency_unlocked
        return LocalTransitionTrainingReward(LocalTransitionRewardConfig(**values))

    def _make_child(self, track: str, seed: int) -> MarineRaceGymEnv:
        return MarineRaceGymEnv(
            track,
            seed=int(seed),
            reward_fn=self._new_reward(),
            adapter=self.adapter_name,
            allow_fallback=self.allow_fallback,
            current_profile="none",
            benchmark_task=BENCHMARK_TASK_CLEAN_GATE,
            max_steps=self.max_episode_steps,
            observation_encoding_version=OBS_ENCODING_VERSION_LOCAL_TRANSITION,
        )

    def reset(self, *, seed: Optional[int] = None, options=None):
        if self._child is not None:
            self._child.close()
        geometry = self.sampler.sample()
        self.current_geometry = geometry
        path = generate_transition_track(
            geometry,
            self.generated_dir / "active_episode.json",
            frames_per_sec=self.frames_per_sec,
        )
        self.current_track = str(path)
        episode_seed = int(seed) if seed is not None else geometry.seed
        self._child = self._make_child(self.current_track, episode_seed)
        observation, info = self._child.reset(seed=episode_seed)
        self._apply_privileged_reset_velocity(geometry)
        self._record_sensor_health()
        self._episode_counter += 1
        self._first_gate_crossing_step = None
        self._previous_action.fill(0.0)
        self._previous_action_delta.fill(0.0)
        self._jerk_sum = 0.0
        self._jerk_samples = 0
        info.update(self._context_info())
        atomic_append_jsonl(
            self.logs_dir / "sampled_geometries.jsonl",
            {
                "episode": self._episode_counter,
                "worker_id": self.worker_id,
                "episode_seed": episode_seed,
                **geometry.__dict__,
                "geometry_group": geometry.geometry_group,
            },
        )
        return observation, info

    def _apply_privileged_reset_velocity(self, geometry: TransitionGeometry) -> None:
        if self._child is None or self.adapter_name != "holoocean":
            return
        adapter = self._child.episode.context.adapter
        setter = getattr(adapter, "set_participant_body_velocity", None)
        if callable(setter):
            setter(
                self._child.episode.participant_id,
                geometry.initial_body_velocity_m_s,
            )

    def step(self, action):
        if self._child is None:
            raise RuntimeError("reset must be called before step")
        action_array = np.asarray(action, dtype=np.float32).reshape(4)
        observation, reward, terminated, truncated, info = self._child.step(action_array)
        self._record_sensor_health()
        delta = action_array - self._previous_action
        self._jerk_sum += float(np.linalg.norm(delta - self._previous_action_delta))
        self._jerk_samples += 1
        self._previous_action_delta = delta
        self._previous_action = action_array.copy()
        gates = int(info.get("gate_crossings", 0))
        if gates >= 1 and self._first_gate_crossing_step is None:
            self._first_gate_crossing_step = int(info["step_count"])
        focus_window_complete = False
        if (
            self.current_geometry is not None
            and self.current_geometry.episode_type == "transition_focus"
            and self._first_gate_crossing_step is not None
            and int(info["step_count"]) - self._first_gate_crossing_step >= 30
            and not terminated
        ):
            truncated = True
            focus_window_complete = True
        info.update(self._context_info())
        info["transition_focus_window_complete"] = focus_window_complete
        if terminated or truncated:
            info.update(self._terminal_metrics())
            # active_episode.json is overwritten every episode, so this
            # append-only row is the only durable record of what the policy
            # actually trained on.  Written at episode end so the geometry and
            # its outcome always describe the same episode.
            if self.log_episode_composition and self.current_geometry is not None:
                append_episode(self.run_dir, episode_row(
                    utc=now_utc(),
                    algorithm=self.algorithm,
                    run=self.run_dir.name,
                    worker_id=self.worker_id,
                    geometry=self.current_geometry,
                    outcome={
                        "gates_completed": info.get("completed_gate_count"),
                        "collision": bool(info.get("collision_episode", False)),
                        "missed_gate": bool(info.get("missed_gate_dnf", False)),
                        "wrong_direction": int(info.get("wrong_direction_events", 0) or 0),
                        "out_of_bounds": bool(info.get("out_of_bounds", False)),
                        "timeout": bool(truncated and not terminated),
                        "steps": int(self._jerk_samples),
                    },
                ))
        return observation, reward, terminated, truncated, info

    def _terminal_metrics(self) -> Dict[str, Any]:
        referee = self.episode.context.referee.states[self.episode.participant_id]
        reward = self._child._reward_fn
        expected = len(self.episode.context.config.track.gate_sequence)
        status = getattr(referee.status, "value", str(referee.status))
        return {
            "episode_status": status,
            "episode_gates_completed": int(referee.valid_gate_crossings),
            "episode_expected_gates": expected,
            "episode_full_sequence_completion": (
                status == "FINISHED" and int(referee.valid_gate_crossings) == expected
            ),
            "episode_first_gate_crossed": int(referee.valid_gate_crossings) >= 1,
            "episode_missed_gate_dnf": int(referee.missed_gate_attempts),
            "episode_collisions": int(referee.collision_events) + int(referee.obstacle_collision_events),
            # Entries and sustained contact frames are reported separately so a
            # single long contact is never read as many distinct collisions.
            "episode_collision_entries": int(
                getattr(reward, "collision_entries", 0)
            ),
            "episode_collision_contact_frames": int(
                getattr(reward, "collision_contact_frames", 0)
            ),
            "episode_collision": bool(getattr(reward, "collision_episode", False)),
            "episode_out_of_bounds": int(referee.out_of_bounds_events),
            "episode_wrong_direction": int(referee.wrong_direction_crossings),
            "episode_previous_gate_return": bool(
                getattr(reward, "previous_gate_return_triggered", False)
            ),
            "episode_acquisition_timeout": bool(
                getattr(reward, "acquisition_timeout_triggered", False)
            ),
            "episode_action_jerk": round(
                self._jerk_sum / max(1, self._jerk_samples), 6
            ),
        }

    def _context_info(self) -> Dict[str, Any]:
        geometry = self.current_geometry
        episode_context = (
            None if self._child is None else getattr(self._child.episode, "_ctx", None)
        )
        adapter = None if episode_context is None else episode_context.adapter
        return {
            "worker_id": self.worker_id,
            "worker_pid": os.getpid(),
            "worker_sampler_seed": self.sampler_seed,
            "worker_directory": str(self.worker_dir),
            "active_track": self.current_track,
            "holoocean_uuid": getattr(adapter, "environment_uuid", None),
            "difficulty": self.sampler.difficulty,
            "episode_type": None if geometry is None else geometry.episode_type,
            "sequence_gate_count_training_only": None if geometry is None else geometry.gate_count,
        }

    def worker_identity(self) -> Dict[str, Any]:
        return self._context_info()

    def worker_state(self) -> Dict[str, Any]:
        return {
            "schema_version": "universal_transition_worker_v1",
            "worker_id": self.worker_id,
            "sampler_seed": self.sampler_seed,
            "episode_counter": self._episode_counter,
            "sampler": self.sampler.state_dict(),
        }

    @staticmethod
    def _sensor_fingerprint(value: Any) -> tuple:
        """Cheap freshness fingerprint; benchmark-only and never policy-visible."""

        array = np.asarray(value)
        flat = array.reshape(-1)
        if flat.size == 0:
            sample = ()
        else:
            indices = np.linspace(0, flat.size - 1, min(64, flat.size), dtype=int)
            sampled = np.nan_to_num(
                flat[indices].astype(np.float64, copy=False),
                nan=0.0,
                posinf=1e30,
                neginf=-1e30,
            )
            sample = tuple(np.round(sampled, 6).tolist())
        return tuple(array.shape), str(array.dtype), sample

    def _record_sensor_health(self) -> None:
        if not self._sensor_health_enabled or self._child is None:
            return
        context = getattr(self._child.episode, "_ctx", None)
        if context is None:
            return
        sensors = context.adapter.get_allowed_sensor_data(
            self._child.episode.participant_id,
            context.participant.config.sensors,
        )
        self._sensor_frames += 1
        for name in self._sensor_missing:
            value = sensors.get(name)
            if value is None:
                self._sensor_missing[name] += 1
                self._sensor_previous.pop(name, None)
                continue
            fingerprint = self._sensor_fingerprint(value)
            if self._sensor_previous.get(name) == fingerprint:
                self._sensor_repeated[name] += 1
            self._sensor_previous[name] = fingerprint

    def enable_sensor_health(self, enabled: bool = True) -> None:
        self._sensor_health_enabled = bool(enabled)
        self._sensor_frames = 0
        self._sensor_missing = {name: 0 for name in self._sensor_missing}
        self._sensor_repeated = {name: 0 for name in self._sensor_repeated}
        self._sensor_previous = {}

    def sensor_health(self) -> Dict[str, Any]:
        frames = int(self._sensor_frames)
        expected_presence_rates = {
            "FrontCamera": 1.0,
            # The generated HoloOcean scenario runs at 30 Hz while DVL is
            # intentionally configured at 15 Hz.
            "DVLSensor": 0.5,
            "IMUSensor": 1.0,
            "DepthSensor": 1.0,
        }
        observed_presence_rates = {
            name: 1.0 - count / max(1, frames)
            for name, count in self._sensor_missing.items()
        }
        cadence_ok = {
            name: observed_presence_rates[name] >= expected - 0.08
            for name, expected in expected_presence_rates.items()
        }
        return {
            "enabled": self._sensor_health_enabled,
            "frames_checked": frames,
            "missing_frames": dict(self._sensor_missing),
            "repeated_fingerprints": dict(self._sensor_repeated),
            "missing_rates": {
                name: count / max(1, frames)
                for name, count in self._sensor_missing.items()
            },
            "repeated_rates": {
                name: count / max(1, frames - 1)
                for name, count in self._sensor_repeated.items()
            },
            "expected_presence_rates": expected_presence_rates,
            "observed_presence_rates": observed_presence_rates,
            "cadence_ok": cadence_ok,
            "sensor_contract_valid": (
                all(cadence_ok.values())
                and all(count == 0 for count in self._sensor_repeated.values())
            ),
        }

    def load_worker_state(self, value: Mapping[str, Any]) -> None:
        if value.get("schema_version") != "universal_transition_worker_v1":
            raise ValueError("unsupported transition worker state")
        if int(value["worker_id"]) != self.worker_id or int(value["sampler_seed"]) != self.sampler_seed:
            raise ValueError("transition worker identity changed")
        self._episode_counter = int(value["episode_counter"])
        self.sampler.load_state_dict(value["sampler"])

    def set_difficulty(self, difficulty: str) -> None:
        self.sampler.set_difficulty(difficulty)

    def set_total_environment_transitions(self, value: int) -> None:
        self.sampler.set_total_environment_transitions(value)

    def set_efficiency_unlocked(self, unlocked: bool) -> None:
        self._efficiency_unlocked = bool(unlocked)
        if self._child is not None and hasattr(self._child._reward_fn, "set_efficiency_unlocked"):
            self._child._reward_fn.set_efficiency_unlocked(unlocked)

    def close(self) -> None:
        if self._child is not None:
            self._child.close()
            self._child = None
