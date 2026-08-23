"""Drive one Gen-2 episode and record legal observations with expert labels.

This module is the single place where the frozen rule controller is allowed to
touch a vehicle.  It exists to produce *training data*; nothing here is ever
imported by the Gen-2 inference path.

Two roles share one driver:

``mode="expert"``
    The frozen expert drives.  Every step records ``(observation_t, expert_t)``.
    This is the behaviour-cloning corpus.

``mode="dagger"``
    The *learner* drives.  The expert is stepped in shadow on the same raw
    observation and supplies the label only; its command never reaches the
    vehicle.  This is the DAgger aggregation pass.  There is no blending: the
    applied command is exactly the learner's, and ``applied_by_expert`` is
    recorded per step so any safety takeover is auditable and excludable.

Observation legality is enforced, not assumed:

* the episode runs with ``official=True`` so the adapter's sensor allowlist is
  active and ``debug_ground_truth`` cannot be attached;
* the raw observation dict handed to the expert is wrapped in a recording view,
  and every key it reads is checked against the official contract;
* the encoded policy vector is exactly ``OBS_DIM_LOCAL_TRANSITION`` features.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np

from marine_race_arena.controllers.official_baselines import RuleGateCenterThenCommitController
from marine_race_arena.config.benchmark_tasks import BENCHMARK_TASK_CLEAN_GATE
from marine_race_arena.learning.config import ACTION_AXES, ACTION_DIM
from marine_race_arena.learning.config_local_transition import (
    OBS_DIM_LOCAL_TRANSITION,
    OBS_ENCODING_VERSION_LOCAL_TRANSITION,
)
from marine_race_arena.learning.episode import RaceEpisode
from marine_race_arena.learning.gen2 import GEN2_ACTION_CONTRACT, GEN2_EXPERT_ID
from marine_race_arena.learning.gen2.course_family import Gen2CourseSpec
from marine_race_arena.learning.observation_encoder_local_transition import (
    encode_observation_local_transition,
)
from marine_race_arena.learning.tracker_context_local_transition import (
    OnboardLocalTransitionContextTracker,
)

#: Every top-level key the official observation may contain.
LEGAL_OBSERVATION_KEYS = frozenset({"local_time_s", "sensors", "beacons", "comms"})

#: Keys whose presence proves a privileged leak.
PRIVILEGED_OBSERVATION_KEYS = frozenset({
    "debug_ground_truth", "referee", "gate_geometry", "own_position",
    "own_rotation_rpy_deg", "expected_gate_id", "circuit_id", "track_id",
})


class ObservationLegalityError(RuntimeError):
    """Raised when an observation carries information outside the contract."""


class _RecordingObservation(dict):
    """A dict that remembers which top-level keys a consumer actually read."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        super().__init__(payload)
        self.accessed: set = set()

    def __getitem__(self, key):  # pragma: no cover - trivial
        self.accessed.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.accessed.add(key)
        return super().get(key, default)


def assert_observation_is_legal(observation: Mapping[str, Any]) -> None:
    """Raise if the raw observation dict carries anything privileged."""
    keys = set(observation)
    leaked = keys & PRIVILEGED_OBSERVATION_KEYS
    if leaked:
        raise ObservationLegalityError(
            f"observation carries privileged keys {sorted(leaked)}"
        )
    unexpected = keys - LEGAL_OBSERVATION_KEYS
    if unexpected:
        raise ObservationLegalityError(
            f"observation carries unexpected keys {sorted(unexpected)}; "
            f"the official contract is {sorted(LEGAL_OBSERVATION_KEYS)}"
        )


def command_to_action(command: Mapping[str, float]) -> np.ndarray:
    """Convert a controller command dict into the 4-D ``[-1, 1]`` action vector."""
    values = [float(command.get(axis, 0.0)) for axis in ACTION_AXES]
    action = np.asarray(values, dtype=np.float32)
    action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(action, -1.0, 1.0).astype(np.float32, copy=False)


def action_to_command(action: Sequence[float]) -> Dict[str, float]:
    """Inverse of :func:`command_to_action`."""
    array = np.asarray(action, dtype=np.float32).reshape(-1)
    if array.shape[0] != ACTION_DIM:
        raise ValueError(f"action must have {ACTION_DIM} elements, got {array.shape[0]}")
    array = np.clip(np.nan_to_num(array, nan=0.0), -1.0, 1.0)
    return {axis: float(array[index]) for index, axis in enumerate(ACTION_AXES)}


#: A learner is any callable ``(observation_35, first_step) -> action_4``.
#: It owns its own recurrent state; ``first_step`` marks an episode boundary.
LearnerFn = Callable[[np.ndarray, bool], np.ndarray]


@dataclass
class Gen2EpisodeRecord:
    """One recorded episode: legal observations, expert labels, diagnostics."""

    seed: int
    mode: str
    gate_count: int
    observations: np.ndarray          # (T, 35) float32 -- policy input
    expert_actions: np.ndarray        # (T, 4)  float32 -- supervision target
    applied_actions: np.ndarray       # (T, 4)  float32 -- what actually drove
    applied_by_expert: np.ndarray     # (T,)    bool
    gate_crossings: np.ndarray        # (T,)    int16 -- valid crossings after step t
    steps: int
    gates_completed: int
    completed: bool
    status: str
    collisions: int
    out_of_bounds: int
    wrong_direction: int
    missed_gate_attempts: int
    truncated: bool
    wall_time_s: float
    course: Dict[str, Any] = field(default_factory=dict)   # diagnostics only
    notes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.observations.shape[1] != OBS_DIM_LOCAL_TRANSITION:
            raise ValueError(
                f"observations must be ({OBS_DIM_LOCAL_TRANSITION},)-wide, "
                f"got {self.observations.shape}"
            )
        if self.expert_actions.shape[1] != ACTION_DIM:
            raise ValueError(f"expert actions must be {ACTION_DIM}-wide")
        lengths = {
            len(self.observations), len(self.expert_actions),
            len(self.applied_actions), len(self.applied_by_expert),
            len(self.gate_crossings),
        }
        if len(lengths) != 1:
            raise ValueError(f"ragged episode record: lengths {sorted(lengths)}")

    @property
    def survival_curve(self) -> np.ndarray:
        """Cumulative gate index reached, per step."""
        return self.gate_crossings

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "mode": self.mode,
            "gate_count": self.gate_count,
            "steps": self.steps,
            "gates_completed": self.gates_completed,
            "completed": self.completed,
            "status": self.status,
            "collisions": self.collisions,
            "out_of_bounds": self.out_of_bounds,
            "wrong_direction": self.wrong_direction,
            "missed_gate_attempts": self.missed_gate_attempts,
            "truncated": self.truncated,
            "expert_takeover_steps": int(self.applied_by_expert.sum()),
            "wall_time_s": round(self.wall_time_s, 3),
            **{f"course_{k}": v for k, v in self.course.items()},
        }


def _mission_info_for(episode: RaceEpisode) -> Dict[str, Any]:
    from marine_race_arena.scripts.run_marine_race import _mission_info

    context = episode.context
    return dict(_mission_info(context.config, context.participant.id))


def _referee_state(episode: RaceEpisode):
    return episode.context.referee.states[episode.participant_id]


def run_gen2_episode(
    track_path: str | Path,
    *,
    seed: int,
    spec: Optional[Gen2CourseSpec] = None,
    mode: str = "expert",
    learner: Optional[LearnerFn] = None,
    adapter: str = "holoocean",
    allow_fallback: bool = False,
    dt: float = 0.1,
    max_steps: int = 4000,
    safety_takeover: Optional[Callable[[Mapping[str, Any], int], bool]] = None,
    record_every: int = 1,
    unseal_record: Optional[Mapping[str, Any]] = None,
) -> Gen2EpisodeRecord:
    """Run one episode and return its recorded observations and expert labels.

    ``mode="expert"`` lets the frozen rule controller drive.  ``mode="dagger"``
    requires ``learner`` and lets the learner drive while the expert is stepped
    in shadow purely to produce labels.
    """
    if mode not in {"expert", "dagger"}:
        raise ValueError(f"unknown rollout mode {mode!r}")
    if mode == "dagger" and learner is None:
        raise ValueError("DAgger rollouts require a learner callable")

    # Default-deny: this is the ONE driver every Gen-2 episode goes through, so
    # gating it here is what makes the sealed holdout unreachable from
    # training, DAgger, validation and curriculum generation alike.
    from marine_race_arena.learning.gen2.holdout_seal import assert_course_accessible

    assert_course_accessible(
        track_path,
        context=f"Gen-2 {mode} rollout (seed {int(seed)})",
        unseal_record=unseal_record,
    )

    started = time.perf_counter()
    episode = RaceEpisode(
        str(track_path),
        seed=int(seed),
        dt=float(dt),
        adapter=adapter,
        allow_fallback=allow_fallback,
        max_steps=int(max_steps),
        official=True,
        current_profile="none",
        # Mixed Endurance declares benchmark_task current_gate. Running it
        # current-free -- the official protocol for the 0/30 vs 30/30
        # comparison, see scripts/run_official_mixed_no_current.bat -- requires
        # overriding the task too, or the loader rejects the config.
        benchmark_task=BENCHMARK_TASK_CLEAN_GATE,
    )
    expert = RuleGateCenterThenCommitController()

    observations: list = []
    expert_actions: list = []
    applied_actions: list = []
    applied_by_expert: list = []
    crossings: list = []

    try:
        raw = episode.reset(seed=int(seed))
        assert_observation_is_legal(raw)
        config = episode.context.config
        if not bool(config.race.official_mode):
            raise ObservationLegalityError(
                "Gen-2 episodes must run in official mode so the sensor allowlist applies"
            )
        gate_total = len(config.track.gate_sequence)
        context_source = OnboardLocalTransitionContextTracker(
            total_beacons=max(1, gate_total), laps=max(1, int(config.race.laps))
        )
        context_source.reset(raw)
        expert.reset(_mission_info_for(episode))

        prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        step_index = 0
        truncated = False
        terminated = False
        expert_key_reads: set = set()

        while step_index < max_steps:
            context = context_source.context(raw, dt=float(dt), prev_action=prev_action.tolist())
            encoded = encode_observation_local_transition(raw, context)

            view = _RecordingObservation(raw)
            expert_command = expert.step(view)
            expert_key_reads |= view.accessed
            expert_action = command_to_action(expert_command)

            takeover = False
            if mode == "expert":
                applied = expert_action
                takeover = True
            else:
                assert learner is not None
                applied = np.asarray(
                    learner(encoded, step_index == 0), dtype=np.float32
                ).reshape(ACTION_DIM)
                applied = np.clip(np.nan_to_num(applied, nan=0.0), -1.0, 1.0)
                if safety_takeover is not None and safety_takeover(raw, step_index):
                    applied = expert_action
                    takeover = True

            recorded = step_index % max(1, int(record_every)) == 0
            if recorded:
                observations.append(encoded)
                expert_actions.append(expert_action)
                applied_actions.append(applied.astype(np.float32, copy=False))
                applied_by_expert.append(takeover)

            step = episode.step(action_to_command(applied))
            raw = step.observation
            assert_observation_is_legal(raw)
            prev_action = applied.astype(np.float32, copy=False)
            step_index += 1

            if recorded:
                crossings.append(int(episode.referee_progress()["valid_gate_crossings"]))

            terminated = bool(step.terminated)
            truncated = bool(step.truncated)
            if terminated or truncated:
                break

        illegal = expert_key_reads - LEGAL_OBSERVATION_KEYS
        if illegal:
            raise ObservationLegalityError(
                f"the frozen expert read non-contract observation keys {sorted(illegal)}"
            )

        state = _referee_state(episode)
        status = getattr(state.status, "value", str(state.status))
        gates_completed = int(state.valid_gate_crossings)
        record = Gen2EpisodeRecord(
            seed=int(seed),
            mode=mode,
            gate_count=gate_total,
            observations=np.asarray(observations, dtype=np.float32).reshape(-1, OBS_DIM_LOCAL_TRANSITION),
            expert_actions=np.asarray(expert_actions, dtype=np.float32).reshape(-1, ACTION_DIM),
            applied_actions=np.asarray(applied_actions, dtype=np.float32).reshape(-1, ACTION_DIM),
            applied_by_expert=np.asarray(applied_by_expert, dtype=bool).reshape(-1),
            gate_crossings=np.asarray(crossings, dtype=np.int16).reshape(-1),
            steps=step_index,
            gates_completed=gates_completed,
            completed=(status == "FINISHED" and gates_completed == gate_total),
            status=status,
            collisions=int(state.collision_events) + int(state.obstacle_collision_events),
            out_of_bounds=int(state.out_of_bounds_events),
            wrong_direction=int(state.wrong_direction_crossings),
            missed_gate_attempts=int(state.missed_gate_attempts),
            truncated=truncated,
            wall_time_s=time.perf_counter() - started,
            course=(spec.summary() if spec is not None else {}),
            notes={
                "observation_contract": OBS_ENCODING_VERSION_LOCAL_TRANSITION,
                "action_contract": GEN2_ACTION_CONTRACT,
                "expert": GEN2_EXPERT_ID,
                "expert_observation_keys_read": sorted(expert_key_reads),
                "adapter": adapter,
                "terminated": terminated,
            },
        )
        return record
    finally:
        try:
            expert.close()
        finally:
            episode.close()


def iter_episode_windows(
    record: Gen2EpisodeRecord, window: int, *, stride: Optional[int] = None
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield ``(obs, target, first_step_mask)`` windows that respect the boundary.

    ``first_step_mask[0]`` is True only for the window that starts the episode,
    so a recurrent learner knows exactly where to zero its hidden state.
    """
    stride = int(stride or window)
    total = len(record.observations)
    for start in range(0, max(1, total - 1), stride):
        end = min(start + window, total)
        if end <= start:
            break
        mask = np.zeros(end - start, dtype=bool)
        if start == 0:
            mask[0] = True
        yield (
            record.observations[start:end],
            record.expert_actions[start:end],
            mask,
        )
