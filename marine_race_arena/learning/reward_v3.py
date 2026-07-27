"""Bounded training-only reward for gate-to-gate learned control.

Policy observations remain onboard-only.  Referee counters and terminal status
are used here only for correct-crossing, safety, and completion rewards; they
are never returned in the observation vector.
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
    _dvl_body_velocity,
    _finite,
    _select_beacon_packet,
)


@dataclass
class MultiGateRewardConfig:
    beacon_progress_scale: float = 1.0
    bearing_penalty_scale: float = 0.05
    visual_acquisition_bonus: float = 0.35
    visual_centering_scale: float = 0.8
    visual_area_scale: float = 0.4
    offcenter_speed_penalty: float = 0.25
    yaw_sway_coupling_penalty: float = 0.03
    action_change_penalty: float = 0.04
    action_saturation_penalty: float = 0.10
    gate_crossing_bonus: float = 20.0
    post_gate_forward_scale: float = 0.3
    previous_gate_return_penalty: float = 0.5
    next_beacon_acquisition_bonus: float = 1.0
    next_gate_visual_acquisition_bonus: float = 2.0
    completion_bonus: float = 50.0
    time_cost: float = 0.01
    collision_penalty: float = 8.0
    obstacle_penalty: float = 8.0
    out_of_bounds_penalty: float = 15.0
    wrong_direction_penalty: float = 12.0
    missed_gate_penalty: float = 8.0
    timeout_penalty: float = 10.0
    post_gate_window_steps: int = 20
    component_abs_bound: float = 50.0
    total_abs_bound: float = 100.0


@dataclass
class _RewardState:
    expected_beacon_id: Optional[str] = None
    previous_beacon_id: Optional[str] = None
    previous_range_m: Optional[float] = None
    previous_visual_present: bool = False
    previous_center_error: Optional[float] = None
    previous_area: Optional[float] = None
    previous_action: np.ndarray = field(
        default_factory=lambda: np.zeros(ACTION_DIM, dtype=np.float32)
    )
    steps_since_referee_crossing: Optional[int] = None
    next_beacon_bonus_paid: bool = False
    next_gate_visual_bonus_paid: bool = False
    terminal_paid: bool = False


_COUNTERS = (
    "collision_events",
    "obstacle_collision_events",
    "out_of_bounds_events",
    "wrong_direction_crossings",
    "missed_gate_attempts",
)


class MultiGateTrainingReward:
    """Stateful gate-to-gate reward with per-component logging."""

    def __init__(self, config: Optional[MultiGateRewardConfig] = None) -> None:
        self.config = config or MultiGateRewardConfig()
        self._state = _RewardState()
        self._previous_counters = {name: 0 for name in _COUNTERS}

    def reset(self, env: Any) -> None:
        self._state = _RewardState()
        self._previous_counters = self._read_counters(env)

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

        components = {
            "beacon_progress": 0.0,
            "bearing_penalty": 0.0,
            "visual_acquisition": 0.0,
            "visual_centering": 0.0,
            "visual_area_progress": 0.0,
            "offcenter_speed_penalty": 0.0,
            "yaw_sway_coupling_penalty": 0.0,
            "action_change_penalty": 0.0,
            "action_saturation_penalty": 0.0,
            "gate_crossing": 0.0,
            "post_gate_forward": 0.0,
            "previous_gate_return_penalty": 0.0,
            "next_beacon_acquisition": 0.0,
            "next_gate_visual_acquisition": 0.0,
            "completion": 0.0,
            "time_cost": -abs(cfg.time_cost),
            "collision_penalty": 0.0,
            "obstacle_penalty": 0.0,
            "out_of_bounds_penalty": 0.0,
            "wrong_direction_penalty": 0.0,
            "missed_gate_penalty": 0.0,
            "timeout_penalty": 0.0,
        }

        tracker = env.tracker
        expected = getattr(tracker, "expected_beacon_id", None)
        if state.expected_beacon_id is None:
            state.expected_beacon_id = expected
        elif expected != state.expected_beacon_id:
            state.previous_beacon_id = state.expected_beacon_id
            state.expected_beacon_id = expected
            state.previous_range_m = None
            state.previous_visual_present = False
            state.previous_center_error = None
            state.previous_area = None
            state.next_beacon_bonus_paid = False
            state.next_gate_visual_bonus_paid = False

        observation = step.observation if isinstance(step.observation, Mapping) else {}
        sensors = observation.get("sensors")
        sensors = sensors if isinstance(sensors, Mapping) else {}
        beacons = observation.get("beacons")
        beacons = beacons if isinstance(beacons, (list, tuple)) else []
        packet = _select_beacon_packet(beacons, expected)
        bearing = None
        current_range = None
        if packet is not None:
            bearing = _finite(packet.get("bearing_deg"), 0.0)
            current_range = max(0.0, _finite(packet.get("range_m"), 0.0))
            in_post_gate_window = (
                state.steps_since_referee_crossing is not None
                and state.steps_since_referee_crossing <= cfg.post_gate_window_steps
            )
            if state.previous_range_m is not None and not in_post_gate_window:
                decrease = state.previous_range_m - current_range
                components["beacon_progress"] = cfg.beacon_progress_scale * float(
                    np.clip(decrease, -1.0, 1.0)
                )
            components["bearing_penalty"] = -cfg.bearing_penalty_scale * min(
                1.0, abs(bearing) / 90.0
            )
            state.previous_range_m = current_range
            if (
                state.previous_beacon_id is not None
                and not state.next_beacon_bonus_paid
            ):
                components["next_beacon_acquisition"] = (
                    cfg.next_beacon_acquisition_bonus
                )
                state.next_beacon_bonus_paid = True

        target = None
        image = sensors.get("FrontCamera")
        if image is not None:
            try:
                targets = vision_targets_from_camera(image)
                target = (
                    select_visual_target_for_beacon(targets, bearing, current_range)
                    if bearing is not None
                    else select_default_visual_target(targets)
                )
            except Exception:
                target = None
        if target is not None:
            center_error = math.hypot(
                _finite(target.center_x), _finite(target.center_y)
            ) / math.sqrt(2.0)
            area = max(0.0, _finite(target.area_fraction))
            if not state.previous_visual_present:
                components["visual_acquisition"] = cfg.visual_acquisition_bonus
            if state.previous_center_error is not None:
                components["visual_centering"] = cfg.visual_centering_scale * max(
                    0.0, state.previous_center_error - center_error
                )
            if state.previous_area is not None and center_error <= 0.45:
                components["visual_area_progress"] = cfg.visual_area_scale * max(
                    0.0, min(0.25, area - state.previous_area)
                )
            if center_error > 0.45 and action_array[0] > 0.35:
                components["offcenter_speed_penalty"] = -cfg.offcenter_speed_penalty * float(
                    action_array[0] - 0.35
                )
            if (
                state.previous_beacon_id is not None
                and not state.next_gate_visual_bonus_paid
            ):
                components["next_gate_visual_acquisition"] = (
                    cfg.next_gate_visual_acquisition_bonus
                )
                state.next_gate_visual_bonus_paid = True
            state.previous_center_error = center_error
            state.previous_area = area
        state.previous_visual_present = target is not None

        if gate_delta > 0:
            components["gate_crossing"] = cfg.gate_crossing_bonus * float(gate_delta)
            state.steps_since_referee_crossing = 0
        elif state.steps_since_referee_crossing is not None:
            state.steps_since_referee_crossing += 1

        if (
            state.steps_since_referee_crossing is not None
            and state.steps_since_referee_crossing <= cfg.post_gate_window_steps
        ):
            dvl = _dvl_body_velocity(sensors)
            if dvl is not None:
                components["post_gate_forward"] = cfg.post_gate_forward_scale * float(
                    np.clip(dvl[0], -1.0, 1.0)
                )
            if state.previous_beacon_id:
                previous_packet = _select_beacon_packet(
                    beacons, state.previous_beacon_id
                )
                if previous_packet is not None:
                    previous_bearing = abs(
                        _finite(previous_packet.get("bearing_deg"), 0.0)
                    )
                    if previous_bearing < 80.0:
                        components["previous_gate_return_penalty"] = (
                            -cfg.previous_gate_return_penalty
                            * (1.0 - previous_bearing / 80.0)
                        )

        components["yaw_sway_coupling_penalty"] = (
            -cfg.yaw_sway_coupling_penalty
            * abs(float(action_array[1] * action_array[3]))
        )
        components["action_change_penalty"] = (
            -cfg.action_change_penalty
            * float(np.linalg.norm(action_array - state.previous_action))
        )
        saturation = float(np.mean(np.abs(action_array) >= 0.98))
        components["action_saturation_penalty"] = (
            -cfg.action_saturation_penalty * saturation
        )
        state.previous_action = action_array.copy()

        counters = self._read_counters(env)
        deltas = {
            name: max(0, counters[name] - self._previous_counters.get(name, 0))
            for name in _COUNTERS
        }
        self._previous_counters = counters
        components["collision_penalty"] = -cfg.collision_penalty * deltas[
            "collision_events"
        ]
        components["obstacle_penalty"] = -cfg.obstacle_penalty * deltas[
            "obstacle_collision_events"
        ]
        components["out_of_bounds_penalty"] = -cfg.out_of_bounds_penalty * deltas[
            "out_of_bounds_events"
        ]
        components["wrong_direction_penalty"] = (
            -cfg.wrong_direction_penalty * deltas["wrong_direction_crossings"]
        )
        components["missed_gate_penalty"] = (
            -cfg.missed_gate_penalty * deltas["missed_gate_attempts"]
        )

        referee_state = env.episode.context.referee.states[env.episode.participant_id]
        terminal_status = getattr(referee_state.status, "value", str(referee_state.status))
        if (step.terminated or step.truncated) and not state.terminal_paid:
            state.terminal_paid = True
            if step.terminated and terminal_status == "FINISHED":
                components["completion"] = cfg.completion_bonus
            elif step.truncated or terminal_status == "TIMEOUT":
                components["timeout_penalty"] = -cfg.timeout_penalty

        for key, value in tuple(components.items()):
            finite = float(value) if math.isfinite(float(value)) else 0.0
            components[key] = float(
                np.clip(finite, -cfg.component_abs_bound, cfg.component_abs_bound)
            )
        reward = float(
            np.clip(
                sum(components.values()),
                -cfg.total_abs_bound,
                cfg.total_abs_bound,
            )
        )
        return reward, components

    @staticmethod
    def _read_counters(env: Any) -> Dict[str, int]:
        state = env.episode.context.referee.states[env.episode.participant_id]
        return {name: int(getattr(state, name, 0)) for name in _COUNTERS}
