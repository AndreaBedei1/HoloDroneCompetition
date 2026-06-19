"""Controller interface for marine race participants."""

from __future__ import annotations

from typing import Any, Dict


class ManualStopRequested(RuntimeError):
    """Raised by manual controllers when the user asks to end the run."""


class BaseController:
    """Base interface for external race controllers.

    Controllers may return either a high-level command:
        {"surge": float, "sway": float, "heave": float, "yaw": float}

    or a thruster command:
        {"thrusters": [float, ...]}
    """

    debug_only = False
    uses_ground_truth = False

    def reset(self, race_info: Dict[str, Any]) -> None:
        pass

    def step(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        pass


def validate_controller_instance(controller: object) -> None:
    missing = [
        method_name
        for method_name in ("reset", "step", "close")
        if not callable(getattr(controller, method_name, None))
    ]
    if missing:
        raise TypeError(f"Controller is missing required methods: {', '.join(missing)}")
