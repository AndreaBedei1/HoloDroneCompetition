"""Reliability-first reward for a repeatable local gate transition.

The reward may inspect simulator/referee state, but none of that state is
returned by the observation encoder.  All dense terms are bounded signed
deltas, so oscillating between two states cannot create net progress reward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from marine_race_arena.controllers.vision import (
    select_default_visual_target,
    select_visual_target_for_beacon,
    vision_targets_from_camera,
)
from marine_race_arena.learning.config import ACTION_DIM
from marine_race_arena.learning.observation_encoder import (
    _finite,
    _select_beacon_packet,
)


@dataclass
class LocalTransitionRewardConfig:
    bearing_delta_scale: float = 2.0
    elevation_delta_scale: float = 1.5
    range_delta_scale: float = 2.0
    visual_center_delta_scale: float = 2.0
    visual_area_delta_scale: float = 8.0
    gate_crossing_bonus: float = 25.0
    post_cross_alignment_delta_scale: float = 3.0
    post_cross_range_delta_scale: float = 3.0
    completion_bonus: float = 2.0
    collision_penalty: float = 50.0
    out_of_bounds_penalty: float = 60.0
    wrong_direction_penalty: float = 60.0
    missed_gate_penalty: float = 60.0
    previous_gate_return_penalty: float = 50.0
    acquisition_timeout_penalty: float = 45.0
    post_cross_orbit_penalty: float = 12.0
    moving_away_penalty: float = 12.0
    terminal_failure_penalty: float = 50.0
    post_cross_window_steps: int = 30
    acquisition_bearing_deg: float = 20.0
    acquisition_elevation_deg: float = 15.0
    persistent_frames: int = 5
    reliability_time_cost: float = 0.0005
    efficiency_time_cost: float = 0.01
    reliability_action_change_penalty: float = 0.0
    efficiency_action_change_penalty: float = 0.01
    reliability_jerk_penalty: float = 0.0
    efficiency_jerk_penalty: float = 0.005
    reliability_energy_penalty: float = 0.0
    efficiency_energy_penalty: float = 0.002
    efficiency_unlocked: bool = False
    component_abs_bound: float = 60.0
    total_abs_bound: float = 100.0

    def set_efficiency_unlocked(self, unlocked: bool) -> None:
        self.efficiency_unlocked = bool(unlocked)


@dataclass
class _State:
    expected_beacon_id: Optional[str] = None
    previous_abs_bearing: Optional[float] = None
    previous_abs_elevation: Optional[float] = None
    previous_range_m: Optional[float] = None
    previous_visual_error: Optional[float] = None
    previous_visual_area: Optional[float] = None
    previous_action: np.ndarray = field(
        default_factory=lambda: np.zeros(ACTION_DIM, dtype=np.float32)
    )
    previous_action_delta: np.ndarray = field(
        default_factory=lambda: np.zeros(ACTION_DIM, dtype=np.float32)
    )
    post_cross_steps: Optional[int] = None
    previous_gate_index: Optional[int] = None
    previous_gate_clearance_seen: bool = False
    previous_gate_return_paid: bool = False
    acquisition_timeout_paid: bool = False
    new_target_aligned: bool = False
    new_target_range_decreased: bool = False
    moving_away_streak: int = 0
    moving_away_paid: bool = False
    orbit_streak: int = 0
    orbit_paid: bool = False
    previous_gate_distance_m: Optional[float] = None
    safety_paid: Dict[str, bool] = field(default_factory=dict)
    terminal_paid: bool = False


class LocalTransitionTrainingReward:
    """Bounded potential reward with one-time transition failures."""

    def __init__(self, config: Optional[LocalTransitionRewardConfig] = None) -> None:
        self.config = config or LocalTransitionRewardConfig()
        self._state = _State()
        self.acquisition_timeout_triggered = False
        self.previous_gate_return_triggered = False

    def set_efficiency_unlocked(self, unlocked: bool) -> None:
        self.config.set_efficiency_unlocked(unlocked)

    def reset(self, env: Any) -> None:
        self._state = _State()
        self.acquisition_timeout_triggered = False
        self.previous_gate_return_triggered = False

    @staticmethod
    def _bounded_delta(previous: Optional[float], current: float, scale: float) -> float:
        if previous is None:
            return 0.0
        return float(scale * np.clip(previous - current, -1.0, 1.0))

    def __call__(
        self,
        env: Any,
        step: Any,
        gate_delta: int,
        action: Any = None,
    ) -> Tuple[float, Dict[str, float]]:
        cfg = self.config
        state = self._state
        action_array = np.asarray(
            np.zeros(ACTION_DIM) if action is None else action,
            dtype=np.float32,
        ).reshape(-1)
        if action_array.shape != (ACTION_DIM,):
            raise ValueError(f"reward action must have shape {(ACTION_DIM,)}")

        names = (
            "bearing_delta", "elevation_delta", "range_delta",
            "visual_center_delta", "visual_area_delta", "gate_crossing",
            "post_cross_alignment_delta", "post_cross_range_delta",
            "previous_gate_return_penalty", "post_cross_orbit_penalty",
            "moving_away_penalty", "acquisition_timeout_penalty",
            "collision_penalty", "out_of_bounds_penalty",
            "wrong_direction_penalty", "missed_gate_penalty",
            "action_change_penalty", "jerk_penalty", "energy_penalty",
            "time_cost", "completion", "terminal_failure_penalty",
        )
        components = {name: 0.0 for name in names}

        tracker = env.tracker
        expected = getattr(tracker, "expected_beacon_id", None)
        if state.expected_beacon_id != expected:
            state.expected_beacon_id = expected
            state.previous_abs_bearing = None
            state.previous_abs_elevation = None
            state.previous_range_m = None
            state.previous_visual_error = None
            state.previous_visual_area = None

        observation = step.observation if isinstance(step.observation, Mapping) else {}
        sensors = observation.get("sensors")
        sensors = sensors if isinstance(sensors, Mapping) else {}
        beacons = observation.get("beacons")
        beacons = beacons if isinstance(beacons, (list, tuple)) else []
        packet = _select_beacon_packet(beacons, expected)
        bearing = elevation = current_range = None
        if packet is not None:
            bearing = _finite(packet.get("bearing_deg"), 0.0)
            elevation = _finite(packet.get("elevation_deg"), 0.0)
            current_range = max(0.0, _finite(packet.get("range_m"), 0.0))
            abs_bearing = abs(bearing)
            abs_elevation = abs(elevation)
            components["bearing_delta"] = self._bounded_delta(
                state.previous_abs_bearing, abs_bearing, cfg.bearing_delta_scale / 45.0
            )
            components["elevation_delta"] = self._bounded_delta(
                state.previous_abs_elevation,
                abs_elevation,
                cfg.elevation_delta_scale / 30.0,
            )
            range_progress = self._bounded_delta(
                state.previous_range_m, current_range, cfg.range_delta_scale
            )
            components["range_delta"] = range_progress
            if state.post_cross_steps is not None:
                components["post_cross_alignment_delta"] = (
                    components["bearing_delta"] + components["elevation_delta"]
                ) * cfg.post_cross_alignment_delta_scale
                components["post_cross_range_delta"] = (
                    range_progress * cfg.post_cross_range_delta_scale
                )
                if state.previous_range_m is not None:
                    change = current_range - state.previous_range_m
                    state.new_target_range_decreased |= change < -0.01
                    state.moving_away_streak = (
                        state.moving_away_streak + 1 if change > 0.03 else 0
                    )
                state.new_target_aligned |= (
                    abs_bearing <= cfg.acquisition_bearing_deg
                    and abs_elevation <= cfg.acquisition_elevation_deg
                )
            state.previous_abs_bearing = abs_bearing
            state.previous_abs_elevation = abs_elevation
            state.previous_range_m = current_range

        target = None
        image = sensors.get("FrontCamera")
        if image is not None:
            try:
                targets = vision_targets_from_camera(image)
                target = (
                    select_visual_target_for_beacon(targets, bearing, current_range)
                    if bearing is not None else select_default_visual_target(targets)
                )
            except Exception:
                target = None
        if target is None:
            state.previous_visual_error = None
            state.previous_visual_area = None
        else:
            center_error = math.hypot(
                _finite(target.center_x), _finite(target.center_y)
            ) / math.sqrt(2.0)
            area = max(0.0, _finite(target.area_fraction))
            components["visual_center_delta"] = self._bounded_delta(
                state.previous_visual_error,
                center_error,
                cfg.visual_center_delta_scale,
            )
            if center_error <= 0.35 and state.previous_visual_area is not None:
                components["visual_area_delta"] = float(
                    cfg.visual_area_delta_scale
                    * np.clip(area - state.previous_visual_area, -0.1, 0.1)
                )
            state.previous_visual_error = center_error
            state.previous_visual_area = area

        referee = env.episode.context.referee.states[env.episode.participant_id]
        gates_now = int(referee.valid_gate_crossings)
        if gate_delta > 0:
            components["gate_crossing"] = cfg.gate_crossing_bonus * min(1, gate_delta)
            state.post_cross_steps = 0
            state.previous_gate_index = max(0, gates_now - 1)
            state.previous_gate_clearance_seen = False
            state.previous_gate_return_paid = False
            state.acquisition_timeout_paid = False
            state.new_target_aligned = False
            state.new_target_range_decreased = False
            state.moving_away_streak = 0
            state.moving_away_paid = False
            state.orbit_streak = 0
            state.orbit_paid = False
            state.previous_gate_distance_m = None
        elif state.post_cross_steps is not None:
            state.post_cross_steps += 1

        if state.post_cross_steps is not None:
            self._post_cross_privileged_terms(env, step, components)
            if (
                state.moving_away_streak >= cfg.persistent_frames
                and not state.moving_away_paid
            ):
                components["moving_away_penalty"] = -cfg.moving_away_penalty
                state.moving_away_paid = True
            acquisition_ok = state.new_target_aligned and state.new_target_range_decreased
            if (
                state.post_cross_steps >= cfg.post_cross_window_steps
                and not acquisition_ok
                and not state.acquisition_timeout_paid
            ):
                components["acquisition_timeout_penalty"] = (
                    -cfg.acquisition_timeout_penalty
                )
                state.acquisition_timeout_paid = True
                self.acquisition_timeout_triggered = True

        counter_penalties = (
            ("collision", "collision_events", cfg.collision_penalty, "collision_penalty"),
            ("out_of_bounds", "out_of_bounds_events", cfg.out_of_bounds_penalty, "out_of_bounds_penalty"),
            ("wrong_direction", "wrong_direction_crossings", cfg.wrong_direction_penalty, "wrong_direction_penalty"),
            ("missed_gate", "missed_gate_attempts", cfg.missed_gate_penalty, "missed_gate_penalty"),
        )
        for latch, counter, penalty, component in counter_penalties:
            if int(getattr(referee, counter, 0)) > 0 and not state.safety_paid.get(latch):
                components[component] = -penalty
                state.safety_paid[latch] = True

        action_delta = action_array - state.previous_action
        jerk = action_delta - state.previous_action_delta
        if cfg.efficiency_unlocked:
            components["action_change_penalty"] = -cfg.efficiency_action_change_penalty * float(
                np.linalg.norm(action_delta)
            )
            components["jerk_penalty"] = -cfg.efficiency_jerk_penalty * float(
                np.linalg.norm(jerk)
            )
            components["energy_penalty"] = -cfg.efficiency_energy_penalty * float(
                np.mean(np.square(action_array))
            )
            components["time_cost"] = -cfg.efficiency_time_cost
        else:
            components["action_change_penalty"] = -cfg.reliability_action_change_penalty * float(
                np.linalg.norm(action_delta)
            )
            components["jerk_penalty"] = -cfg.reliability_jerk_penalty * float(
                np.linalg.norm(jerk)
            )
            components["energy_penalty"] = -cfg.reliability_energy_penalty * float(
                np.mean(np.square(action_array))
            )
            components["time_cost"] = -cfg.reliability_time_cost
        state.previous_action = action_array.copy()
        state.previous_action_delta = action_delta

        status = getattr(referee.status, "value", str(referee.status))
        if (step.terminated or step.truncated) and not state.terminal_paid:
            state.terminal_paid = True
            if status == "FINISHED":
                # Deliberately constant: independent of sequence length.
                components["completion"] = cfg.completion_bonus
            else:
                components["terminal_failure_penalty"] = -cfg.terminal_failure_penalty

        for key, value in tuple(components.items()):
            finite = float(value) if math.isfinite(float(value)) else 0.0
            components[key] = float(np.clip(
                finite, -cfg.component_abs_bound, cfg.component_abs_bound
            ))
        reward = float(np.clip(
            sum(components.values()), -cfg.total_abs_bound, cfg.total_abs_bound
        ))
        return reward, components

    def _post_cross_privileged_terms(
        self, env: Any, step: Any, components: Dict[str, float]
    ) -> None:
        state = self._state
        cfg = self.config
        if state.previous_gate_index is None:
            return
        gates = env.episode.context.config.gates
        if state.previous_gate_index >= len(gates):
            return
        gate = gates[state.previous_gate_index]
        position = np.asarray(step.current_state.position, dtype=np.float64)
        center = np.asarray(gate.position, dtype=np.float64)
        normal = np.asarray(gate.passage_direction, dtype=np.float64)
        signed = float(np.dot(position - center, normal))
        distance = float(np.linalg.norm(position - center))
        if signed >= 0.75:
            state.previous_gate_clearance_seen = True
        if (
            state.previous_gate_clearance_seen
            and signed <= -0.25
            and not state.previous_gate_return_paid
        ):
            components["previous_gate_return_penalty"] = (
                -cfg.previous_gate_return_penalty
            )
            state.previous_gate_return_paid = True
            self.previous_gate_return_triggered = True
        if state.previous_gate_distance_m is not None:
            distance_change = distance - state.previous_gate_distance_m
            no_new_progress = not state.new_target_range_decreased
            state.orbit_streak = (
                state.orbit_streak + 1
                if no_new_progress and distance < 3.0 and abs(distance_change) < 0.03
                else 0
            )
        state.previous_gate_distance_m = distance
        if state.orbit_streak >= cfg.persistent_frames and not state.orbit_paid:
            components["post_cross_orbit_penalty"] = -cfg.post_cross_orbit_penalty
            state.orbit_paid = True

