"""Training-only reward for learning controllers (direction-aware).

This reward exists purely to *train* policies; it is never part of the benchmark
score, and its inputs (gate geometry, referee event counters, vehicle position)
are privileged simulator/referee state that MUST NOT be encoded into the policy
observation. The official benchmark score remains the unchanged referee output.

Directionality: progress is measured as the *signed distance to the target gate
plane* along the gate's passage direction. Reward is granted only for approaching
the plane from the legal entry side; approaching (or crossing) from the wrong side
earns no positive progress and a wrong-direction crossing is penalized. The
gate-crossing bonus is driven by the referee's own valid-crossing count (which
excludes wrong-direction crossings), so it is applied exactly once and only for a
legal crossing.

Anti reward-hacking:
  * approach and alignment use a *ratchet* (only a new best value earns reward), so
    oscillating in front of a gate cannot farm progress;
  * progress is gated to the entry side, so wrong-side motion earns nothing;
  * heading alignment is a small *signed* term (backward motion is negative), so it
    nets out under oscillation;
  * terminal bonuses/penalties are applied once; the gate bonus exactly once.

The pure function :func:`score_step` takes plain scalars and is fully unit
testable; :class:`TrainingReward` wires it to a running :class:`RaceEpisode`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from marine_race_arena.learning.config import ACTION_DIM

_FINISHED = "FINISHED"
_TERMINAL_PENALIZED = {
    "DNF": "dnf_penalty",
    "DSQ": "dnf_penalty",
    "TIMEOUT": "timeout_penalty",
    "STUCK": "stuck_terminal_penalty",
}

# Vehicle counts as "on the entry side" while its signed distance to the gate plane
# is at or below this small positive tolerance (metres): at/just before the plane.
ENTRY_SIDE_TOL_M = 0.05


@dataclass
class RewardConfig:
    """Documented, tunable reward weights (all non-negative magnitudes)."""

    progress_scale: float = 1.0        # reward per metre of *new* approach to the gate plane
    alignment_scale: float = 0.5       # reward per metre of *new* lateral-offset reduction
    heading_scale: float = 0.1         # signed reward for heading alignment with the passage direction
    gate_bonus: float = 10.0           # per newly (legally) crossed gate
    completion_bonus: float = 50.0     # on FINISHED
    time_cost: float = 0.02            # per step
    collision_penalty: float = 5.0     # per new gate/world collision event
    obstacle_penalty: float = 5.0      # per new obstacle collision event
    out_of_bounds_penalty: float = 10.0
    missed_gate_penalty: float = 5.0   # per new missed-gate attempt
    wrong_direction_penalty: float = 5.0  # per new wrong-direction crossing
    stuck_penalty: float = 15.0        # per new stuck event
    dnf_penalty: float = 20.0          # terminal DNF/DSQ
    timeout_penalty: float = 10.0      # terminal TIMEOUT or truncation
    stuck_terminal_penalty: float = 15.0
    action_change_penalty: float = 0.05  # per unit of ||a_t - a_{t-1}||
    action_magnitude_penalty: float = 0.0  # per unit of ||a_t||


@dataclass
class RewardState:
    """Mutable per-episode ratchet / history."""

    best_approach: Optional[float] = None   # smallest entry-side distance to the plane so far
    best_lateral: Optional[float] = None
    prev_action: np.ndarray = field(default_factory=lambda: np.zeros(ACTION_DIM, dtype=np.float32))
    terminal_awarded: bool = False

    def reset_gate_ratchet(self) -> None:
        self.best_approach = None
        self.best_lateral = None


def score_step(
    state: RewardState,
    config: RewardConfig,
    *,
    has_target: bool,
    signed_distance_plane: float,
    lateral_offset: float,
    heading_alignment: float,
    gate_delta: int,
    d_collision: int,
    d_obstacle: int,
    d_out_of_bounds: int,
    d_stuck: int,
    d_missed: int,
    d_wrong_direction: int,
    terminated: bool,
    truncated: bool,
    terminal_status: Optional[str],
    action: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    """Compute the reward and its signed components for one step.

    ``signed_distance_plane`` is ``(position - gate_center) . passage_normal``:
    negative on the legal entry side, zero at the plane, positive on the exit side.
    ``heading_alignment`` is the cosine of the vehicle displacement against the
    passage direction, in ``[-1, 1]`` (0 if unavailable). ``state`` is mutated.
    """
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    components: Dict[str, float] = {
        "progress": 0.0,
        "alignment": 0.0,
        "heading_alignment": 0.0,
        "gate_bonus": 0.0,
        "completion_bonus": 0.0,
        "time_cost": -abs(config.time_cost),
        "collision_penalty": 0.0,
        "obstacle_penalty": 0.0,
        "out_of_bounds_penalty": 0.0,
        "missed_gate_penalty": 0.0,
        "wrong_direction_penalty": 0.0,
        "stuck_penalty": 0.0,
        "dnf_penalty": 0.0,
        "timeout_penalty": 0.0,
        "stuck_terminal_penalty": 0.0,
        "action_change_penalty": 0.0,
        "action_magnitude_penalty": 0.0,
    }

    # A newly crossed (legal) gate: bonus once (referee count), restart the ratchet.
    if gate_delta > 0:
        components["gate_bonus"] = config.gate_bonus * float(gate_delta)
        state.reset_gate_ratchet()

    # Direction-aware approach + alignment, only from the legal entry side.
    on_entry_side = has_target and math.isfinite(signed_distance_plane) and signed_distance_plane <= ENTRY_SIDE_TOL_M
    if on_entry_side:
        approach = max(0.0, -float(signed_distance_plane))  # entry-side distance to the plane
        if state.best_approach is None:
            state.best_approach = approach
        else:
            gain = state.best_approach - approach
            if gain > 0.0:
                components["progress"] = config.progress_scale * gain
                state.best_approach = approach
        if math.isfinite(lateral_offset):
            if state.best_lateral is None:
                state.best_lateral = lateral_offset
            else:
                lat_gain = state.best_lateral - lateral_offset
                if lat_gain > 0.0:
                    components["alignment"] = config.alignment_scale * lat_gain
                    state.best_lateral = lateral_offset
        # Signed heading alignment (backward motion is penalized -> nets out under oscillation).
        if math.isfinite(heading_alignment):
            components["heading_alignment"] = config.heading_scale * float(np.clip(heading_alignment, -1.0, 1.0))

    # Penalties from authoritative referee event deltas.
    components["collision_penalty"] = -config.collision_penalty * max(0, d_collision)
    components["obstacle_penalty"] = -config.obstacle_penalty * max(0, d_obstacle)
    components["out_of_bounds_penalty"] = -config.out_of_bounds_penalty * max(0, d_out_of_bounds)
    components["stuck_penalty"] = -config.stuck_penalty * max(0, d_stuck)
    components["missed_gate_penalty"] = -config.missed_gate_penalty * max(0, d_missed)
    components["wrong_direction_penalty"] = -config.wrong_direction_penalty * max(0, d_wrong_direction)

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
        "wrong_direction_crossings",
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
        signed_distance = math.inf
        lateral_offset = math.inf
        heading_alignment = 0.0
        if has_target:
            gate = ctx.arena.gate_map[target_gate_id]
            position = step.current_state.position
            signed_distance = float(gate.signed_distance_to_plane(position))
            right, up = gate.local_aperture_coordinates(position)
            lateral_offset = math.hypot(right, up)
            heading_alignment = _heading_alignment(step.previous_state.position, position, gate.normal_vector)

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
            signed_distance_plane=signed_distance,
            lateral_offset=lateral_offset,
            heading_alignment=heading_alignment,
            gate_delta=gate_delta,
            d_collision=deltas["collision_events"],
            d_obstacle=deltas["obstacle_collision_events"],
            d_out_of_bounds=deltas["out_of_bounds_events"],
            d_stuck=deltas["stuck_events"],
            d_missed=deltas["missed_gate_attempts"],
            d_wrong_direction=deltas["wrong_direction_crossings"],
            terminated=step.terminated,
            truncated=step.truncated,
            terminal_status=terminal_status,
            action=np.asarray(action, dtype=np.float32),
        )

    def _read_counters(self, episode: Any) -> Dict[str, int]:
        state = episode.context.referee.states[episode.participant_id]
        return {name: int(getattr(state, name, 0)) for name in self._COUNTER_FIELDS}


def _heading_alignment(prev_position, position, normal) -> float:
    """Cosine of the vehicle displacement against the gate passage direction."""
    dx = float(position[0]) - float(prev_position[0])
    dy = float(position[1]) - float(prev_position[1])
    dz = float(position[2]) - float(prev_position[2])
    speed = math.sqrt(dx * dx + dy * dy + dz * dz)
    if speed < 1e-9:
        return 0.0
    dot = dx * float(normal[0]) + dy * float(normal[1]) + dz * float(normal[2])
    return dot / speed
