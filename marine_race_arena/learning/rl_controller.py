"""Deployable learned controller for Marine Race Arena.

``RLGateController`` is a normal :class:`BaseController` (``reset``/``step``/
``close``) that wraps a trained behavioral-cloning or PPO policy. It:

  * reconstructs the *exact* learning observation encoding used during training
    (via the shared :class:`OnboardContextTracker`), so training and deployment
    see identical features;
  * consumes only legal onboard information (received beacons, FrontCamera, depth,
    IMU, DVL, controller-local state) and returns the normalized four-axis command;
  * needs neither the referee, the Gym environment nor Gymnasium training objects
    at inference time (PyTorch is required; Stable-Baselines3 only if a PPO ``.zip``
    is loaded);
  * commands zero once its local course tracker reports completion;
  * resolves the model path from a constructor argument, the ``MARINE_RACE_RL_MODEL``
    environment variable, or a controller-config attribute — never a hard-coded path.

It is registered as the built-in alias ``rl_gate_controller`` and can also be
loaded by file path or ``module:Class``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from marine_race_arena.learning.config import (
    ACTION_AXES,
    ACTION_DIM,
    DEFAULT_LAPS,
    DEFAULT_TOTAL_BEACONS,
)
from marine_race_arena.learning.observation_encoder import encode_observation
from marine_race_arena.learning.tracker_context import OnboardContextTracker
from marine_race_arena.participants.controller_interface import BaseController

_MODEL_ENV_VAR = "MARINE_RACE_RL_MODEL"


class _Inference:
    """Uniform ``act(obs) -> action`` wrapper over BC and PPO policies."""

    def __init__(self, kind: str, model: Any) -> None:
        self.kind = kind
        self._model = model

    def act(self, observation: np.ndarray) -> np.ndarray:
        if self.kind == "ppo":
            action, _ = self._model.predict(np.asarray(observation, dtype=np.float32), deterministic=True)
            return np.asarray(action, dtype=np.float32).reshape(-1)
        return self._model.act(observation)


def _load_inference(model_path: str) -> _Inference:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"RL model path does not exist: {model_path}")
    if path.suffix == ".zip":
        from stable_baselines3 import PPO  # lazy: only when a PPO model is used

        return _Inference("ppo", PPO.load(str(path), device="cpu"))
    from marine_race_arena.learning.bc_train import load_policy  # lazy: torch only

    return _Inference("bc", load_policy(path))


class RLGateController(BaseController):
    """A learned gate-racing controller integrated through the standard API."""

    debug_only = False
    uses_ground_truth = False

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._model_path = model_path or os.environ.get(_MODEL_ENV_VAR)
        self._inference: Optional[_Inference] = None
        self._ctx_source: Optional[OnboardContextTracker] = None
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._pending_first_obs = True
        self._finished = False
        self._total_beacons = DEFAULT_TOTAL_BEACONS
        self._laps = DEFAULT_LAPS
        self._initial_beacon_id = "B01"

    # ------------------------------------------------------------------ api
    def reset(self, mission_info: Mapping[str, Any]) -> None:
        if not self._model_path:
            raise ValueError(
                "RLGateController needs a model path: pass model_path=..., set "
                f"${_MODEL_ENV_VAR}, or configure controller.model_path."
            )
        if self._inference is None:
            self._inference = _load_inference(self._model_path)

        mission = mission_info or {}
        self._total_beacons = int(mission.get("total_beacons", DEFAULT_TOTAL_BEACONS)) or DEFAULT_TOTAL_BEACONS
        self._laps = int(mission.get("laps", DEFAULT_LAPS)) or DEFAULT_LAPS
        self._initial_beacon_id = str(mission.get("initial_beacon_id", "B01"))
        self._ctx_source = OnboardContextTracker(
            total_beacons=self._total_beacons, laps=self._laps, initial_beacon_id=self._initial_beacon_id
        )
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._pending_first_obs = True
        self._finished = False

    def step(self, observation: Mapping[str, Any]) -> dict:
        if self._ctx_source is None or self._inference is None:
            raise RuntimeError("RLGateController.reset() must be called before step().")

        if self._pending_first_obs:
            self._ctx_source.reset(observation)
            self._pending_first_obs = False

        context = self._ctx_source.context(observation, dt=None, prev_action=self._prev_action.tolist())

        # Command zero once the local course tracker reports completion.
        tracker = self._ctx_source.tracker
        if self._finished or getattr(tracker, "finished", False) or context.tracker_phase == "FINISHED":
            self._finished = True
            self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
            return _zero_command()

        encoded = encode_observation(observation, context)
        action = np.clip(np.nan_to_num(self._inference.act(encoded), nan=0.0), -1.0, 1.0).astype(np.float32)
        self._prev_action = action
        return {axis: float(action[i]) for i, axis in enumerate(ACTION_AXES)}

    def close(self) -> None:
        self._inference = None
        self._ctx_source = None


def _zero_command() -> dict:
    return {axis: 0.0 for axis in ACTION_AXES}
