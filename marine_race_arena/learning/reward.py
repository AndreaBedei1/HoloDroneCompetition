"""Training-only reward for learning controllers.

This reward exists purely to *train* policies; it is never part of the benchmark
score, and its inputs (gate geometry, referee event counters, vehicle position)
are privileged simulator/referee state that MUST NOT be encoded into the policy
observation. The official benchmark score remains the unchanged referee output.

Design against reward hacking:
  * gate approach is rewarded through a *ratchet*: only a new closest distance to
    the current target gate earns reward, so oscillating in front of a gate cannot
    farm progress;
  * the gate-crossing bonus is driven by the referee's own crossing count, so it
    is applied exactly once per gate;
  * terminal bonuses/penalties are applied once;
  * penalties are driven by per-step deltas of the referee's authoritative event
    counters.

The pure function :func:`score_step` takes plain scalars and is fully unit
testable; :class:`TrainingReward` wires it to a running :class:`RaceEpisode`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from marine_race_arena.learning.config import ACTION_DIM

# Referee statuses (mirror of ParticipantStatus values used here).
_FINISHED = "FINISHED"
_TERMINAL_PENALIZED = {"DNF": "dnf_penalty", "DSQ": "dnf_penalty", "TIMEOUT": "timeout_penalty", "STUCK": "stuck_terminal_penalty"}


@dataclass
class RewardConfig:
    """Documented, tunable reward weights (all non-negative magnitudes)."""

    progress_scale: float = 1.0        # reward per metre of *new* approach to the gate
    alignment_scale: float = 0.5       # reward per metre of *new* lateral-offset reduction
    gate_bonus: float = 10.0           # per newly crossed gate
    completion_bonus: float = 50.0     # on FINISHED
    time_cost: float = 0.02            # per step
    collision_penalty: float = 5.0     # per new gate/world collision event
    obstacle_penalty: float = 5.0      # per new obstacle collision event
    out_of_bounds_penalty: float = 10.0
    missed_gate_penalty: float = 5.0   # per new missed-gate attempt
    stuck_penalty: float = 15.0        # per new stuck event
    dnf_penalty: float = 20.0          # terminal DNF/DSQ
    timeout_penalty: float = 10.0      # terminal TIMEOUT or truncation
    stuck_terminal_penalty: float = 15.0
    action_change_penalty: float = 0.05  # per unit of ||a_t - a_{t-1}||
    action_magnitude_penalty: float = 0.0  # per unit of ||a_t||


@dataclass
class RewardState:
    """Mutable per-episode ratchet / history."""

    best_dist: Optional[float] = None
    best_lateral: Optional[float] = None
    prev_action: np.ndarray = field(default_factory=lambda: np.zeros(ACTION_DIM, dtype=np.float32))
    terminal_awarded: bool = False
    current_target: Optional[str] = None

    def reset_gate_ratchet(self) -> None:
        self.best_dist = None
        self.best_lateral = None


def score_step(
    state: RewardState,
    config: RewardConfig,
    *,
    has_target: bool,
    dist_to_gate: float,
    lateral_offset: float,
    gate_delta: int,
    d_collision: int,
    d_obstacle: int,
    d_out_of_bounds: int,
    d_stuck: int,
    d_missed: int,
    terminated: bool,
    truncated: bool,
    terminal_status: Optional[str],
    action: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    """Compute the reward and its signed components for one step.

    ``state`` is mutated (ratchets, prev action, terminal latch). All geometry
    inputs are precomputed scalars, so this function is pure w.r.t. the simulator.
    """
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    components: Dict[str, float] = {
        "progress": 0.0,
        "alignment": 0.0,
        "gate_bonus": 0.0,
        "completion_bonus": 0.0,
        "time_cost": -abs(config.time_cost),
        "collision_penalty": 0.0,
        "obstacle_penalty": 0.0,
        "out_of_bounds_penalty": 0.0,
        "missed_gate_penalty": 0.0,
        "stuck_penalty": 0.0,
        "dnf_penalty": 0.0,
        "timeout_penalty": 0.0,
        "stuck_terminal_penalty": 0.0,
        "action_change_penalty": 0.0,
        "action_magnitude_penalty": 0.0,
    }

    # A newly crossed gate: bonus (once, via referee delta) and restart the ratchet
    # for the next target so approach reward is available again.
    if gate_delta > 0:
        components["gate_bonus"] = config.gate_bonus * float(gate_delta)
        state.reset_gate_ratchet()

    # Ratcheted approach + alignment toward the current target gate.
    if has_target and math.isfinite(dist_to_gate):
        if state.best_dist is None:
            state.best_dist = dist_to_gate
        else:
            gain = state.best_dist - dist_to_gate
            if gain > 0.0:
                components["progress"] = config.progress_scale * gain
                state.best_dist = dist_to_gate
        if math.isfinite(lateral_offset):
            if state.best_lateral is None:
                state.best_lateral = lateral_offset
            else:
                lat_gain = state.best_lateral - lateral_offset
                if lat_gain > 0.0:
                    components["alignment"] = config.alignment_scale * lat_gain
                    state.best_lateral = lateral_offset

    # Penalties from authoritative referee event deltas.
    components["collision_penalty"] = -config.collision_penalty * max(0, d_collision)
    components["obstacle_penalty"] = -config.obstacle_penalty * max(0, d_obstacle)
    components["out_of_bounds_penalty"] = -config.out_of_bounds_penalty * max(0, d_out_of_bounds)
    components["stuck_penalty"] = -config.stuck_penalty * max(0, d_stuck)
    components["missed_gate_penalty"] = -config.missed_gate_penalty * max(0, d_missed)

    # Terminal / truncation bonuses and penalties, applied once.
    if (terminated or truncated) and not state.terminal_awarded:
        state.terminal_awarded = True
        if terminated and terminal_status == _FINISHED:
            components["completion_bonus"] = config.completion_bonus
        elif terminated and terminal_status in _TERMINAL_PENALIZED:
            key = _TERMINAL_PENALIZED[terminal_status]
            components[key] = -getattr(config, key)
        elif truncated:
            components["timeout_penalty"] = -config.timeout_penalty

    # Smoothness / effort.
    change = float(np.linalg.norm(action - state.prev_action))
    components["action_change_penalty"] = -config.action_change_penalty * change
    components["action_magnitude_penalty"] = -config.action_magnitude_penalty * float(np.linalg.norm(action))
    state.prev_action = action.copy()

    reward = float(sum(components.values()))
    return reward, components


class TrainingReward:
    """Stateful reward callable wired to a running :class:`RaceEpisode`.

    Matches the Gym env ``reward_fn`` contract ``(env, step, gate_delta, action)``
    and exposes ``reset(env)`` so the env can restart it per episode.
    """

    _COUNTER_FIELDS = (
        "collision_events",
        "obstacle_collision_events",
        "out_of_bounds_events",
        "stuck_events",
        "missed_gate_attempts",
    )

    def __init__(self, config: Optional[RewardConfig] = None) -> None:
        self.config = config or RewardConfig()
        self._state = RewardState()
        self._prev_counters = {name: 0 for name in self._COUNTER_FIELDS}

    def reset(self, env: Any) -> None:
        self._state = RewardState()
        self._prev_counters = self._read_counters(env.episode)

    def __call__(self, env: Any, step: Any, gate_delta: int, action: Any = None) -> Tuple[float, Dict[str, float]]:
        episode = env.episode
        ctx = episode.context
        pid = episode.participant_id
        referee = ctx.referee
        state = referee.states[pid]

        target_gate_id = episode.expected_gate_id()
        has_target = target_gate_id is not None
        dist_to_gate = math.inf
        lateral_offset = math.inf
        if has_target:
            gate = ctx.arena.gate_map[target_gate_id]
            position = step.current_state.position
            dist_to_gate = _euclidean(position, gate.center)
            right, up = gate.local_aperture_coordinates(position)
            lateral_offset = math.hypot(right, up)

        counters = self._read_counters(episode)
        deltas = {name: counters[name] - self._prev_counters.get(name, 0) for name in self._COUNTER_FIELDS}
        self._prev_counters = counters

        if action is None:
            action = np.zeros(ACTION_DIM, dtype=np.float32)

        terminal_status = None
        if step.terminated:
            terminal_status = state.status.value if hasattr(state.status, "value") else str(state.status)

        return score_step(
            self._state,
            self.config,
            has_target=has_target,
            dist_to_gate=dist_to_gate,
            lateral_offset=lateral_offset,
            gate_delta=gate_delta,
            d_collision=deltas["collision_events"],
            d_obstacle=deltas["obstacle_collision_events"],
            d_out_of_bounds=deltas["out_of_bounds_events"],
            d_stuck=deltas["stuck_events"],
            d_missed=deltas["missed_gate_attempts"],
            terminated=step.terminated,
            truncated=step.truncated,
            terminal_status=terminal_status,
            action=np.asarray(action, dtype=np.float32),
        )

    def _read_counters(self, episode: Any) -> Dict[str, int]:
        state = episode.context.referee.states[episode.participant_id]
        return {name: int(getattr(state, name, 0)) for name in self._COUNTER_FIELDS}


def _euclidean(a, b) -> float:
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))
