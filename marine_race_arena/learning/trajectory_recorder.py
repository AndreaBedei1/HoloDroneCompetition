"""Full-rate trajectory recording for behavioral cloning.

Runs an expert controller (an official rule-based baseline) through a
:class:`RaceEpisode` and records, at every control step, the *encoded* onboard
observation and the expert's action — the exact ``(observation, action)`` pairs a
behavioral-cloning policy learns from. The encoding uses the same
:class:`OnboardContextTracker` as the Gym env and the deployable RL controller,
so training and inference see identical features.

Privileged, training-only diagnostics (vehicle position, referee gate count) are
recorded in a clearly separated ``diagnostics`` block and are excluded by the BC
dataset loader; they never enter the policy input.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from marine_race_arena.learning.config import ACTION_AXES, ACTION_DIM, OBS_DIM, TRACKER_PHASES
from marine_race_arena.learning.episode import RaceEpisode
from marine_race_arena.learning.observation_encoder import encode_observation
from marine_race_arena.learning.tracker_context import OnboardContextTracker
from marine_race_arena.participants.controller_loader import ControllerLoader
from marine_race_arena.scripts.run_marine_race import _mission_info

_PHASE_INDEX = {phase: i for i, phase in enumerate(TRACKER_PHASES)}


@dataclass
class EpisodeRecord:
    """One recorded expert episode. Policy inputs and diagnostics are separated."""

    episode_id: int
    seed: int
    track: str
    controller: str
    observations: np.ndarray            # (T, OBS_DIM) float32 -- policy input
    expert_actions_raw: np.ndarray      # (T, 4) float32 -- pre-clamp expert command
    actions: np.ndarray                 # (T, 4) float32 -- applied (clipped) target
    dones: np.ndarray                   # (T,) bool -- terminated
    truncated: np.ndarray               # (T,) bool
    step_ids: np.ndarray                # (T,) int64
    phase_ids: np.ndarray               # (T,) int64
    final_status: str
    gate_crossings: int
    diagnostics: Dict[str, np.ndarray] = field(default_factory=dict)  # training-only, excluded from BC

    @property
    def length(self) -> int:
        return int(self.observations.shape[0])

    def save_npz(self, path) -> None:
        """Save this episode to a single compressed .npz (atomic temp-then-rename)."""
        import json as _json
        from pathlib import Path as _Path

        path = _Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "episode_id": int(self.episode_id),
            "seed": int(self.seed),
            "track": str(self.track),
            "controller": str(self.controller),
            "final_status": str(self.final_status),
            "gate_crossings": int(self.gate_crossings),
        }
        # Temp name ends in .npz so np.savez_compressed writes exactly that file
        # (it appends .npz otherwise); then atomically rename over the target.
        tmp = path.with_name("_tmp_" + path.name)
        np.savez_compressed(
            tmp,
            observations=self.observations,
            expert_actions_raw=self.expert_actions_raw,
            actions=self.actions,
            dones=self.dones,
            truncated=self.truncated,
            step_ids=self.step_ids,
            phase_ids=self.phase_ids,
            diag_positions=self.diagnostics.get("positions", np.zeros((self.length, 3), dtype=np.float32)),
            diag_gate_crossings=self.diagnostics.get("gate_crossings", np.zeros(self.length, dtype=np.int64)),
            meta=np.array(_json.dumps(meta)),
        )
        tmp.replace(path)

    @classmethod
    def load_npz(cls, path) -> "EpisodeRecord":
        import json as _json

        data = np.load(path, allow_pickle=False)
        meta = _json.loads(str(data["meta"]))
        return cls(
            episode_id=int(meta["episode_id"]),
            seed=int(meta["seed"]),
            track=str(meta["track"]),
            controller=str(meta["controller"]),
            observations=data["observations"],
            expert_actions_raw=data["expert_actions_raw"],
            actions=data["actions"],
            dones=data["dones"],
            truncated=data["truncated"],
            step_ids=data["step_ids"],
            phase_ids=data["phase_ids"],
            final_status=str(meta["final_status"]),
            gate_crossings=int(meta["gate_crossings"]),
            diagnostics={"positions": data["diag_positions"], "gate_crossings": data["diag_gate_crossings"]},
        )


def _command_to_vector(command: Any) -> np.ndarray:
    vec = np.zeros(ACTION_DIM, dtype=np.float32)
    if isinstance(command, dict):
        for i, axis in enumerate(ACTION_AXES):
            try:
                vec[i] = float(command.get(axis, 0.0))
            except (TypeError, ValueError):
                vec[i] = 0.0
    return vec


def record_episode(
    track: str,
    controller: str = "rule_gate_center_then_commit",
    *,
    seed: int = 0,
    dt: float = 0.1,
    adapter: str = "fallback",
    allow_fallback: bool = True,
    max_steps: int = 4000,
    official: bool = True,
    episode_id: int = 0,
    current_profile: Optional[str] = None,
    obstacles: Optional[str] = None,
    duration_s: Optional[float] = None,
    start_randomization=None,
) -> EpisodeRecord:
    """Record one expert episode. The expert sees the raw official observation."""
    episode = RaceEpisode(
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
        start_randomization=start_randomization,
    )
    obs = episode.reset()
    cfg = episode.context.config
    total_beacons = max(1, len(cfg.track.gate_sequence))
    laps = max(1, int(cfg.race.laps))

    expert = ControllerLoader().load(controller)
    expert.reset(_mission_info(cfg, episode.participant_id))
    ctx_source = OnboardContextTracker(total_beacons=total_beacons, laps=laps)
    ctx_source.reset(obs)

    observations: List[np.ndarray] = []
    expert_raw: List[np.ndarray] = []
    applied: List[np.ndarray] = []
    dones: List[bool] = []
    truncs: List[bool] = []
    step_ids: List[int] = []
    phase_ids: List[int] = []
    positions: List[List[float]] = []
    crossings: List[int] = []

    prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
    step_index = 0
    try:
        while True:
            context = ctx_source.context(obs, dt=dt, prev_action=prev_action.tolist())
            encoded = encode_observation(obs, context)
            command = expert.step(copy.deepcopy(obs))
            raw_action = _command_to_vector(command)
            applied_action = np.clip(raw_action, -1.0, 1.0)

            step = episode.step(command)

            observations.append(encoded)
            expert_raw.append(raw_action)
            applied.append(applied_action)
            dones.append(bool(step.terminated))
            truncs.append(bool(step.truncated))
            step_ids.append(step_index)
            phase_ids.append(_PHASE_INDEX.get(context.tracker_phase, -1))
            positions.append([float(v) for v in step.current_state.position])
            crossings.append(episode.referee_progress()["valid_gate_crossings"])

            prev_action = applied_action
            obs = step.observation
            step_index += 1
            if step.terminated or step.truncated:
                break
    finally:
        try:
            expert.close()
        except Exception:  # pragma: no cover
            pass
        progress = episode.referee_progress()
        episode.close()

    return EpisodeRecord(
        episode_id=episode_id,
        seed=seed,
        track=track,
        controller=controller,
        observations=np.asarray(observations, dtype=np.float32).reshape(-1, OBS_DIM),
        expert_actions_raw=np.asarray(expert_raw, dtype=np.float32).reshape(-1, ACTION_DIM),
        actions=np.asarray(applied, dtype=np.float32).reshape(-1, ACTION_DIM),
        dones=np.asarray(dones, dtype=bool),
        truncated=np.asarray(truncs, dtype=bool),
        step_ids=np.asarray(step_ids, dtype=np.int64),
        phase_ids=np.asarray(phase_ids, dtype=np.int64),
        final_status=progress["status"],
        gate_crossings=int(progress["valid_gate_crossings"]),
        diagnostics={
            "positions": np.asarray(positions, dtype=np.float32).reshape(-1, 3),
            "gate_crossings": np.asarray(crossings, dtype=np.int64),
        },
    )


def collect_dataset(
    track: str,
    controller: str = "rule_gate_center_then_commit",
    *,
    seeds,
    dt: float = 0.1,
    adapter: str = "fallback",
    allow_fallback: bool = True,
    max_steps: int = 4000,
    official: bool = True,
    current_profile: Optional[str] = None,
    obstacles: Optional[str] = None,
    start_randomization=None,
) -> List[EpisodeRecord]:
    """Record one episode per seed. Episode ids are the seed order index."""
    records: List[EpisodeRecord] = []
    for episode_id, seed in enumerate(seeds):
        records.append(
            record_episode(
                track,
                controller,
                seed=int(seed),
                dt=dt,
                adapter=adapter,
                allow_fallback=allow_fallback,
                max_steps=max_steps,
                official=official,
                episode_id=episode_id,
                current_profile=current_profile,
                obstacles=obstacles,
                start_randomization=start_randomization,
            )
        )
    return records
