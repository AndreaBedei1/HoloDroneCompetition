"""Step-wise single-vehicle race engine that reuses the benchmark runner.

:class:`RaceEpisode` exposes the same per-tick operations that
``run_marine_race._run_race_loop`` performs for one vehicle — build the official
observation, apply a command, advance the simulator, update the referee — but one
step at a time, so a learning loop can own the control flow.

It does not modify the normal runner. It reuses the runner's own building blocks:
the track loader, arena builder, official observation builder, HoloOcean/fallback
adapter selection and the independent referee. A regression test drives an
identical action sequence through both this engine and the real
``_run_race_loop`` and asserts the observations, referee progress, termination and
score match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

if TYPE_CHECKING:  # pragma: no cover
    from marine_race_arena.learning.randomization import StartRandomization

from marine_race_arena.adapters import select_adapter
from marine_race_arena.adapters.base import AdapterParticipantState, BaseRaceAdapter
from marine_race_arena.arena.arena_builder import Arena, ArenaBuilder
from marine_race_arena.config.loader import load_track_config
from marine_race_arena.config.schema import TrackConfig
from marine_race_arena.participants.controller_interface import BaseController
from marine_race_arena.participants.participant import RaceParticipant
from marine_race_arena.referee.referee import Referee

# Reuse the runner's exact observation builder, mission info and helpers.
from marine_race_arena.scripts.run_marine_race import (
    _build_controller_observation,
    _mission_info,
    _vector3,
    _with_cli_overrides,
    _zero_command,
)


class _NullController(BaseController):
    """Placeholder controller; the episode applies actions directly."""

    def step(self, observation: Mapping[str, Any]) -> Dict[str, float]:
        return _zero_command()


@dataclass
class RaceContext:
    """Constructed single-vehicle race, ready to be stepped."""

    config: TrackConfig
    arena: Arena
    referee: Referee
    adapter: BaseRaceAdapter
    participant: RaceParticipant
    applied_randomization: Optional[Dict[str, float]] = None


def build_single_vehicle_race(
    track: str,
    *,
    seed: int = 0,
    adapter: str = "fallback",
    allow_fallback: bool = True,
    headless: bool = True,
    record: bool = False,
    official: bool = True,
    duration_s: Optional[float] = None,
    benchmark_task: Optional[str] = None,
    obstacles: Optional[str] = None,
    obstacle_density: Optional[str] = None,
    obstacle_physics: Optional[str] = None,
    current_profile: Optional[str] = None,
    controller: Optional[BaseController] = None,
    start_randomization: Optional["StartRandomization"] = None,
) -> RaceContext:
    """Build one single-vehicle race exactly as the runner does (no logging).

    Uses ``logger=None`` so stepping an episode performs no disk I/O. The
    participant carries a placeholder controller; the caller supplies commands.
    ``start_randomization`` applies a deterministic, seeded perturbation of the
    start pose and beacon noise (training only); the applied values are returned in
    the :class:`RaceContext`.
    """
    config = load_track_config(
        track,
        benchmark_task=benchmark_task,
        obstacles=obstacles,
        obstacle_density=obstacle_density,
        obstacle_physics=obstacle_physics,
        current_profile=current_profile,
        seed=seed,
    )
    config = _with_cli_overrides(config, duration_s=duration_s, official=official)

    participant_config = config.participants[0]
    spawn = participant_config.spawn or {}
    position = _vector3(spawn.get("position", config.start.position))
    rotation = _vector3(spawn.get("rotation_rpy_deg", config.start.rotation_rpy_deg))

    applied_randomization = None
    if start_randomization is not None and not start_randomization.is_noop():
        from marine_race_arena.learning.randomization import apply_start_randomization

        config, position, rotation, applied_randomization = apply_start_randomization(
            config, position, rotation, start_randomization, seed
        )

    arena = ArenaBuilder(config, seed=seed).build()
    referee = Referee(config, arena.gate_map, arena.bounds, logger=None)

    participant = RaceParticipant(
        config=participant_config,
        controller=controller or _NullController(),
        position=position,
        rotation_rpy_deg=rotation,
    )

    race_adapter = select_adapter(
        adapter_name=adapter,
        config=config,
        arena=arena,
        allow_fallback=allow_fallback,
        headless=headless,
        record=record,
        seed=seed,
    )
    race_adapter.spawn_participants({participant.id: participant})
    race_adapter.reset()
    race_adapter.spawn_visual_gates(arena.visual_gates)
    race_adapter.spawn_obstacles(arena.obstacles)

    referee.register_participants([participant.id])
    participant.controller.reset(_mission_info(config, participant.id))
    return RaceContext(
        config=config,
        arena=arena,
        referee=referee,
        adapter=race_adapter,
        participant=participant,
        applied_randomization=applied_randomization,
    )


class RaceEpisode:
    """One single-vehicle race, driven one control step at a time."""

    def __init__(
        self,
        track: str,
        *,
        seed: int = 0,
        dt: float = 0.1,
        adapter: str = "fallback",
        allow_fallback: bool = True,
        max_steps: Optional[int] = None,
        official: bool = True,
        duration_s: Optional[float] = None,
        benchmark_task: Optional[str] = None,
        obstacles: Optional[str] = None,
        obstacle_density: Optional[str] = None,
        obstacle_physics: Optional[str] = None,
        current_profile: Optional[str] = None,
        start_randomization: Optional["StartRandomization"] = None,
    ) -> None:
        self.track = track
        self.seed = seed
        self.dt = float(dt)
        self.adapter_name = adapter
        self.allow_fallback = allow_fallback
        self.max_steps = max_steps
        self.official = official
        self.duration_s = duration_s
        self.start_randomization = start_randomization
        self._build_kwargs = dict(
            benchmark_task=benchmark_task,
            obstacles=obstacles,
            obstacle_density=obstacle_density,
            obstacle_physics=obstacle_physics,
            current_profile=current_profile,
        )
        self._ctx: Optional[RaceContext] = None
        self._release_time_s = 0.0
        self._step_count = 0

    # ------------------------------------------------------------------ props
    @property
    def participant_id(self) -> str:
        return self._require_ctx().participant.id

    @property
    def context(self) -> RaceContext:
        return self._require_ctx()

    @property
    def step_count(self) -> int:
        return self._step_count

    def _require_ctx(self) -> RaceContext:
        if self._ctx is None:
            raise RuntimeError("RaceEpisode.reset() must be called before use.")
        return self._ctx

    # ------------------------------------------------------------------ api
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Construct a fresh race and return the initial official observation."""
        self.close()
        if seed is not None:
            self.seed = int(seed)
        self._ctx = build_single_vehicle_race(
            self.track,
            seed=self.seed,
            adapter=self.adapter_name,
            allow_fallback=self.allow_fallback,
            official=self.official,
            duration_s=self.duration_s,
            start_randomization=self.start_randomization,
            **self._build_kwargs,
        )
        ctx = self._ctx
        start_time = ctx.adapter.get_current_time()
        ctx.referee.start_race(start_time, start_delays={ctx.participant.id: 0.0})
        ctx.referee.release_participant(ctx.participant.id, start_time)
        self._release_time_s = start_time
        self._step_count = 0
        return self._build_observation()

    def _build_observation(self) -> Dict[str, Any]:
        ctx = self._require_ctx()
        participant_state = ctx.adapter.get_participant_state(ctx.participant.id)
        return _build_controller_observation(
            config=ctx.config,
            arena=ctx.arena,
            adapter=ctx.adapter,
            participant=ctx.participant,
            participant_state=participant_state,
            release_time_s=self._release_time_s,
            comms_inbox=None,
        )

    def step(self, command: Mapping[str, Any]) -> "EpisodeStep":
        """Apply a command, advance one tick, update the referee, return the step."""
        ctx = self._require_ctx()
        pid = ctx.participant.id
        control_mode = ctx.participant.config.control_mode

        previous_state = ctx.adapter.get_participant_state(pid)
        ctx.adapter.apply_command(pid, dict(command), control_mode)
        ctx.adapter.step(self.dt)
        time_s = ctx.adapter.get_current_time()
        current_state = ctx.adapter.get_participant_state(pid)

        obstacle_collisions = ctx.adapter.get_obstacle_collision_events(
            pid,
            previous_position=previous_state.position,
            current_position=current_state.position,
        )
        collision = ctx.adapter.get_collision_state(pid)
        ctx.referee.update(
            participant_id=pid,
            previous_position=previous_state.position,
            current_position=current_state.position,
            time_s=time_s,
            collision=collision and not obstacle_collisions,
            obstacle_collisions=obstacle_collisions,
        )
        self._step_count += 1

        state = ctx.referee.states[pid]
        terminated = bool(state.is_terminal)
        truncated = False
        if not terminated:
            if self.max_steps is not None and self._step_count >= self.max_steps:
                truncated = True
            elif time_s >= float(ctx.config.race.max_duration_s):
                truncated = True

        observation = self._build_observation()
        obstacle_collision_count = (
            len(obstacle_collisions)
            if isinstance(obstacle_collisions, (list, tuple))
            else int(bool(obstacle_collisions))
        )
        return EpisodeStep(
            observation=observation,
            terminated=terminated,
            truncated=truncated,
            time_s=time_s,
            previous_state=previous_state,
            current_state=current_state,
            collision=bool(collision),
            obstacle_collisions=obstacle_collision_count,
        )

    def referee_progress(self) -> Dict[str, Any]:
        ctx = self._require_ctx()
        state = ctx.referee.states[ctx.participant.id]
        return {
            "valid_gate_crossings": int(state.valid_gate_crossings),
            "status": state.status.value if hasattr(state.status, "value") else str(state.status),
            "is_terminal": bool(state.is_terminal),
        }

    def summary(self) -> Dict[str, Any]:
        return self._require_ctx().referee.summary()

    def expected_gate_id(self) -> Optional[str]:
        ctx = self._require_ctx()
        state = ctx.referee.states[ctx.participant.id]
        if state.is_terminal or state.valid_gate_crossings >= len(ctx.referee.gate_sequence):
            return None
        return ctx.referee.expected_gate_id(ctx.participant.id)

    def close(self) -> None:
        if self._ctx is not None:
            try:
                self._ctx.adapter.close()
            except Exception:  # pragma: no cover - defensive shutdown
                pass
            self._ctx = None


@dataclass
class EpisodeStep:
    observation: Dict[str, Any]
    terminated: bool
    truncated: bool
    time_s: float
    previous_state: AdapterParticipantState
    current_state: AdapterParticipantState
    collision: bool
    obstacle_collisions: int


class PersistentRaceSession:
    """EXPERIMENTAL: reuse ONE HoloOcean engine across episodes for fast resets.

    Launching HoloOcean (``holoocean.make``) dominates a fresh reset. This session
    builds the adapter/arena once, then per episode resets the running engine
    (``env.reset``), teleports the vehicle to the (possibly randomized) start pose,
    and installs a fresh referee -- avoiding a relaunch. It does not change the normal
    fresh-reset :class:`RaceEpisode`; use it only after validating equivalence (see
    ``reset_benchmark``). Intended for training throughput, not for the frozen
    correctness evaluations.

    Note: the visual gates spawned into the engine persist across ``env.reset`` for the
    validated Stage-1 case; re-seeded beacon noise is a no-op on noise-free training
    tracks. For tracks with beacon noise, prefer fresh reset until validated there.
    """

    def __init__(
        self,
        track: str,
        *,
        seed: int = 0,
        dt: float = 0.1,
        adapter: str = "holoocean",
        allow_fallback: bool = False,
        official: bool = True,
        duration_s: Optional[float] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        self.track = track
        self.dt = float(dt)
        self.max_steps = max_steps
        self._ctx = build_single_vehicle_race(
            track, seed=seed, adapter=adapter, allow_fallback=allow_fallback,
            official=official, duration_s=duration_s,
        )
        self._referee = None
        self._release_time_s = 0.0
        self._step_count = 0
        self.last_reset_applied_randomization: Optional[Dict[str, float]] = None

    @property
    def participant_id(self) -> str:
        return self._ctx.participant.id

    @property
    def adapter_name(self) -> str:
        return self._ctx.adapter.name

    def reset_episode(self, seed: int = 0, start_randomization=None) -> Dict[str, Any]:
        """Reset the running engine and start a fresh episode (no relaunch)."""
        from marine_race_arena.referee.referee import Referee

        ctx = self._ctx
        pid = ctx.participant.id
        base = ctx.participant
        position = base.position
        rotation = base.rotation_rpy_deg
        self.last_reset_applied_randomization = None
        if start_randomization is not None and not start_randomization.is_noop():
            from marine_race_arena.learning.randomization import apply_start_randomization

            _, position, rotation, self.last_reset_applied_randomization = apply_start_randomization(
                ctx.config, position, rotation, start_randomization, seed
            )

        # Reset the engine to its spawn state, then teleport to the (new) start pose.
        ctx.adapter.reset()
        ctx.adapter.teleport_participant(pid, position, rotation)
        ctx.adapter.step(self.dt)  # let the teleport settle and refresh sensors
        ctx.adapter.reset()        # clear the transient tick so time restarts at 0

        # Fresh referee for the new episode (arena geometry is reused).
        self._referee = Referee(ctx.config, ctx.arena.gate_map, ctx.arena.bounds, logger=None)
        self._referee.register_participants([pid])
        start_time = ctx.adapter.get_current_time()
        self._referee.start_race(start_time, start_delays={pid: 0.0})
        self._referee.release_participant(pid, start_time)
        self._release_time_s = start_time
        self._step_count = 0
        return self._build_observation()

    def _build_observation(self) -> Dict[str, Any]:
        ctx = self._ctx
        state = ctx.adapter.get_participant_state(ctx.participant.id)
        return _build_controller_observation(
            config=ctx.config, arena=ctx.arena, adapter=ctx.adapter, participant=ctx.participant,
            participant_state=state, release_time_s=self._release_time_s, comms_inbox=None,
        )

    def step(self, command: Mapping[str, Any]) -> EpisodeStep:
        ctx = self._ctx
        pid = ctx.participant.id
        previous_state = ctx.adapter.get_participant_state(pid)
        ctx.adapter.apply_command(pid, dict(command), ctx.participant.config.control_mode)
        ctx.adapter.step(self.dt)
        time_s = ctx.adapter.get_current_time()
        current_state = ctx.adapter.get_participant_state(pid)
        obstacle_collisions = ctx.adapter.get_obstacle_collision_events(
            pid, previous_position=previous_state.position, current_position=current_state.position)
        collision = ctx.adapter.get_collision_state(pid)
        self._referee.update(
            participant_id=pid, previous_position=previous_state.position,
            current_position=current_state.position, time_s=time_s,
            collision=collision and not obstacle_collisions, obstacle_collisions=obstacle_collisions)
        self._step_count += 1
        state = self._referee.states[pid]
        terminated = bool(state.is_terminal)
        truncated = (not terminated) and (
            (self.max_steps is not None and self._step_count >= self.max_steps)
            or time_s >= float(ctx.config.race.max_duration_s))
        count = len(obstacle_collisions) if isinstance(obstacle_collisions, (list, tuple)) else int(bool(obstacle_collisions))
        return EpisodeStep(self._build_observation(), terminated, truncated, time_s,
                           previous_state, current_state, bool(collision), count)

    def referee_progress(self) -> Dict[str, Any]:
        state = self._referee.states[self._ctx.participant.id]
        return {"valid_gate_crossings": int(state.valid_gate_crossings),
                "status": state.status.value if hasattr(state.status, "value") else str(state.status),
                "is_terminal": bool(state.is_terminal)}

    def dvl_speed(self) -> float:
        """Body-frame speed magnitude from the DVL (for residual-velocity checks)."""
        sensors = self._ctx.adapter.get_allowed_sensor_data(self._ctx.participant.id, self._ctx.participant.config.sensors)
        dvl = sensors.get("DVLSensor")
        if dvl is None:
            return 0.0
        try:
            values = list(dvl.tolist()) if hasattr(dvl, "tolist") else list(dvl)
            return float(sum(v * v for v in values[:3]) ** 0.5)
        except Exception:
            return 0.0

    def close(self) -> None:
        if self._ctx is not None:
            try:
                self._ctx.adapter.close()
            except Exception:  # pragma: no cover
                pass
            self._ctx = None
