"""Held-out evaluation of any controller under the unchanged referee.

Runs a controller (rule-based, BC or PPO) through the real race runner and the
independent referee on a set of held-out seeds, and reports the benchmark-facing
metrics: completion rate, gates, referee/penalized time, collisions, out-of-bounds
and stuck events. This is the same scoring the benchmark uses; the learned policy
adapts to it rather than the reverse.

The evaluation never uses privileged state for control and never modifies the
official tracks or the frozen 78-run results; callers evaluating on official tracks
should write any artifacts to a fresh directory.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from marine_race_arena.learning.episode import build_single_vehicle_race
from marine_race_arena.participants.controller_interface import BaseController
from marine_race_arena.scripts.run_marine_race import _mission_info, _run_race_loop

ControllerFactory = Callable[[], BaseController]


class _StepTimer:
    """Wrap a controller's ``step`` to accumulate inference time, unchanged behavior."""

    def __init__(self, controller: BaseController) -> None:
        self._total_s = 0.0
        self._count = 0
        self._original = controller.step

        def timed_step(observation):
            start = time.perf_counter()
            command = self._original(observation)
            self._total_s += time.perf_counter() - start
            self._count += 1
            return command

        controller.step = timed_step  # type: ignore[method-assign]

    def mean_ms(self) -> Optional[float]:
        if self._count == 0:
            return None
        return round(1000.0 * self._total_s / self._count, 4)


@dataclass
class EvalResult:
    seed: int
    status: str
    finished: bool
    completed_gates: int
    expected_gates: int
    official_time_s: Optional[float]
    penalized_time_s: Optional[float]
    collision_events: int
    obstacle_collision_events: int
    out_of_bounds_events: int
    stuck_events: int
    missed_gate_attempts: int
    wrong_direction_crossings: int = 0
    inference_time_ms: Optional[float] = None
    wall_s: Optional[float] = None
    adapter_used: Optional[str] = None
    applied_randomization: Optional[dict] = None


@dataclass
class EvalReport:
    track: str
    label: str
    results: List[EvalResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def completion_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.finished) / len(self.results)

    def _mean(self, attr: str, finished_only: bool = False) -> float:
        rows = [r for r in self.results if (r.finished or not finished_only)]
        vals = [getattr(r, attr) for r in rows if getattr(r, attr) is not None]
        return float(sum(vals) / len(vals)) if vals else 0.0

    @property
    def mean_gates(self) -> float:
        return self._mean("completed_gates")

    @property
    def mean_official_time_finished(self) -> float:
        return self._mean("official_time_s", finished_only=True)

    @property
    def mean_collisions(self) -> float:
        return self._mean("collision_events")

    def wilson_interval(self, z: float = 1.96) -> Dict[str, float]:
        """Wilson score 95% confidence interval for the completion rate."""
        n = self.n
        if n == 0:
            return {"low": 0.0, "high": 0.0}
        p = self.completion_rate
        denom = 1.0 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = (z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5) / denom
        return {"low": max(0.0, centre - half), "high": min(1.0, centre + half)}

    def summary(self) -> Dict[str, float]:
        ci = self.wilson_interval()
        return {
            "track": self.track,
            "label": self.label,
            "episodes": self.n,
            "completion_rate": self.completion_rate,
            "completion_rate_wilson95_low": round(ci["low"], 4),
            "completion_rate_wilson95_high": round(ci["high"], 4),
            "mean_gates": self.mean_gates,
            "mean_official_time_finished": self.mean_official_time_finished,
            "mean_collisions": self.mean_collisions,
            "mean_obstacle_collisions": self._mean("obstacle_collision_events"),
            "mean_out_of_bounds": self._mean("out_of_bounds_events"),
            "mean_stuck": self._mean("stuck_events"),
            "mean_missed_gate_attempts": self._mean("missed_gate_attempts"),
            "mean_wrong_direction_crossings": self._mean("wrong_direction_crossings"),
            "mean_inference_time_ms": self._mean("inference_time_ms"),
        }


def evaluate_controller(
    track: str,
    controller_factory: ControllerFactory,
    *,
    seeds: Sequence[int],
    label: str = "controller",
    adapter: str = "fallback",
    allow_fallback: bool = True,
    official: bool = True,
    duration_s: Optional[float] = None,
    dt: float = 0.1,
    current_profile: Optional[str] = None,
    obstacles: Optional[str] = None,
    start_randomization=None,
) -> EvalReport:
    """Evaluate a controller over held-out seeds through the unchanged runner."""
    report = EvalReport(track=track, label=label)
    for seed in seeds:
        controller = controller_factory()
        timer = _StepTimer(controller)  # time controller.step() without changing behavior
        ctx = build_single_vehicle_race(
            track,
            seed=int(seed),
            adapter=adapter,
            allow_fallback=allow_fallback,
            official=official,
            duration_s=duration_s,
            current_profile=current_profile,
            obstacles=obstacles,
            controller=controller,
            start_randomization=start_randomization,
        )
        pid = ctx.participant.id
        wall_start = time.time()
        try:
            controller.reset(_mission_info(ctx.config, pid))
            _run_race_loop(
                config=ctx.config,
                arena=ctx.arena,
                referee=ctx.referee,
                adapter=ctx.adapter,
                participants={pid: ctx.participant},
                dt=dt,
            )
            state = ctx.referee.states[pid]
            status = state.status.value if hasattr(state.status, "value") else str(state.status)
            report.results.append(
                EvalResult(
                    seed=int(seed),
                    status=status,
                    finished=(status == "FINISHED"),
                    completed_gates=int(state.valid_gate_crossings),
                    expected_gates=len(ctx.referee.gate_sequence) * int(ctx.config.race.laps),
                    official_time_s=_referee_time(ctx.referee, pid, "official"),
                    penalized_time_s=_referee_time(ctx.referee, pid, "penalized"),
                    collision_events=int(state.collision_events),
                    obstacle_collision_events=int(state.obstacle_collision_events),
                    out_of_bounds_events=int(state.out_of_bounds_events),
                    stuck_events=int(state.stuck_events),
                    missed_gate_attempts=int(state.missed_gate_attempts),
                    wrong_direction_crossings=int(state.wrong_direction_crossings),
                    inference_time_ms=timer.mean_ms(),
                    wall_s=round(time.time() - wall_start, 3),
                    adapter_used=ctx.adapter.name,
                    applied_randomization=ctx.applied_randomization,
                )
            )
        finally:
            try:
                controller.close()
            except Exception:  # pragma: no cover
                pass
            ctx.adapter.close()
    return report


def _referee_time(referee, pid, kind: str) -> Optional[float]:
    """Best-effort extraction of official/penalized time from the referee summary."""
    try:
        summary = referee.summary()
    except Exception:  # pragma: no cover
        return None
    participants = summary.get("participants") if isinstance(summary, dict) else None
    if not isinstance(participants, list):
        return None
    for entry in participants:
        if isinstance(entry, dict) and entry.get("participant_id") == pid:
            key = "official_time_s" if kind == "official" else "penalized_time_s"
            value = entry.get(key)
            return float(value) if isinstance(value, (int, float)) else None
    return None
