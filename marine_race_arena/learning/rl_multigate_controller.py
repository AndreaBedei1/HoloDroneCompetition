"""Deployable mostly learned multi-gate controller.

All four runtime commands come directly from a v3 BC/PPO policy.  Deterministic
logic is limited to onboard observation preprocessing, expected-beacon
progression, temporal feature history, action clipping, and the final stop.
There is no rule-controller instance, action blending, beacon-homing command,
commit command, or post-gate action sequence.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from marine_race_arena.learning.bc_v3_transfer import load_v3_policy
from marine_race_arena.learning.config import (
    ACTION_AXES,
    ACTION_DIM,
    DEFAULT_LAPS,
    DEFAULT_TOTAL_BEACONS,
)
from marine_race_arena.learning.config_v3 import OBS_DIM_V3, OBS_ENCODING_VERSION_V3
from marine_race_arena.learning.observation_encoder_v3 import encode_observation_v3
from marine_race_arena.learning.tracker_context_v3 import OnboardMultiGateContextTracker
from marine_race_arena.participants.controller_interface import BaseController

_MODEL_ENV_VAR = "MARINE_RACE_RL_MULTIGATE_MODEL"


class _Inference:
    def __init__(self, kind: str, model: Any) -> None:
        self.kind = kind
        self.model = model

    def act(self, observation: np.ndarray) -> np.ndarray:
        if self.kind == "ppo":
            action, _ = self.model.predict(
                np.asarray(observation, dtype=np.float32), deterministic=True
            )
            return np.asarray(action, dtype=np.float32).reshape(-1)
        return np.asarray(self.model.act(observation), dtype=np.float32).reshape(-1)


def _load_v3_inference(model_path: str) -> _Inference:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"RL multi-gate model path does not exist: {model_path}")
    if path.suffix.lower() == ".zip":
        from stable_baselines3 import PPO

        model = PPO.load(str(path), device="cpu")
        version = getattr(model, "obs_encoding_version", None)
        shape = tuple(getattr(model.observation_space, "shape", ()) or ())
        if version != OBS_ENCODING_VERSION_V3:
            raise ValueError(
                f"incompatible observation encoding: model={version!r}, "
                f"controller={OBS_ENCODING_VERSION_V3!r}"
            )
        if shape != (OBS_DIM_V3,):
            raise ValueError(
                f"incompatible PPO observation shape: model={shape}, "
                f"controller={(OBS_DIM_V3,)}"
            )
        return _Inference("ppo", model)
    return _Inference("bc", load_v3_policy(path))


class RLMultigateController(BaseController):
    """Policy-only four-axis control with minimal local gate progression."""

    debug_only = False
    uses_ground_truth = False
    observation_encoding_version = OBS_ENCODING_VERSION_V3
    rule_action_weight = 0.0
    hybrid_blending = False
    rule_controller_instantiated = False

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._model_path = model_path or os.environ.get(_MODEL_ENV_VAR)
        self._inference: Optional[_Inference] = None
        self._context_source: Optional[OnboardMultiGateContextTracker] = None
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._pending_first_observation = True
        self._finished = False
        self._last_encoded_observation = None
        self.deterministic_runtime_intervention_count = 0
        self.previous_gate_return_count = 0
        self._previous_gate_forward_streak = 0
        self._previous_gate_return_latched = False

    @property
    def tracker(self):
        return self._context_source.tracker if self._context_source is not None else None

    @property
    def last_encoded_observation(self):
        return self._last_encoded_observation

    def reset(self, mission_info: Mapping[str, Any]) -> None:
        if not self._model_path:
            raise ValueError(
                "RLMultigateController needs model_path=... or "
                f"${_MODEL_ENV_VAR}."
            )
        if self._inference is None:
            self._inference = _load_v3_inference(self._model_path)
        mission = mission_info or {}
        total = int(mission.get("total_beacons", DEFAULT_TOTAL_BEACONS))
        laps = int(mission.get("laps", DEFAULT_LAPS))
        initial = str(mission.get("initial_beacon_id", "B01"))
        self._context_source = OnboardMultiGateContextTracker(
            total_beacons=total or DEFAULT_TOTAL_BEACONS,
            laps=laps or DEFAULT_LAPS,
            initial_beacon_id=initial,
        )
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._pending_first_observation = True
        self._finished = False
        self._last_encoded_observation = None
        self.deterministic_runtime_intervention_count = 0
        self.previous_gate_return_count = 0
        self._previous_gate_forward_streak = 0
        self._previous_gate_return_latched = False

    def step(self, observation: Mapping[str, Any]) -> dict:
        if self._context_source is None or self._inference is None:
            raise RuntimeError("RLMultigateController.reset() must be called before step().")
        if self._pending_first_observation:
            self._context_source.reset(observation)
            self._pending_first_observation = False

        context = self._context_source.context(
            observation,
            dt=None,
            prev_action=self._prev_action.tolist(),
        )
        if (
            context.previous_gate_bearing_present
            and not context.previous_gate_in_rear_sector
        ):
            self._previous_gate_forward_streak += 1
            if (
                self._previous_gate_forward_streak >= 3
                and not self._previous_gate_return_latched
            ):
                self.previous_gate_return_count += 1
                self._previous_gate_return_latched = True
        else:
            self._previous_gate_forward_streak = 0
            if context.previous_gate_in_rear_sector:
                self._previous_gate_return_latched = False
        if self._finished or self._context_source.tracker.finished:
            self._finished = True
            self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
            return _zero_command()

        encoded = encode_observation_v3(observation, context)
        self._last_encoded_observation = encoded
        action = np.asarray(self._inference.act(encoded), dtype=np.float32).reshape(-1)
        if action.shape != (ACTION_DIM,):
            raise ValueError(
                f"policy returned action shape {action.shape}; expected {(ACTION_DIM,)}"
            )
        action = np.clip(
            np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0),
            -1.0,
            1.0,
        ).astype(np.float32)
        self._prev_action = action
        return {axis: float(action[index]) for index, axis in enumerate(ACTION_AXES)}

    def close(self) -> None:
        self._inference = None
        self._context_source = None


def _zero_command() -> dict:
    return {axis: 0.0 for axis in ACTION_AXES}
