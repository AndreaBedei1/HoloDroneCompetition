"""Manual keyboard controller for local HoloOcean testing.

This controller is intentionally simple and competition-safe: it does not use
ground truth, does not read gate geometry, and never commands yaw.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterable, List, Optional

from marine_race_arena.participants.controller_interface import BaseController


class KeyboardManualController(BaseController):
    """Non-blocking WASD plus arrow-key controller.

    Controls:
    - W/S: forward/backward on the horizontal plane.
    - A/D: left/right sway on the horizontal plane.
    - Up/Down arrows: raise/lower the rover.
    - Space: stop.

    The terminal window must have keyboard focus. HoloOcean viewport key events
    are not exposed through the installed Python API.
    """

    debug_only = False
    uses_ground_truth = False

    def reset(self, race_info: Dict[str, Any]) -> None:
        max_command = float(race_info.get("max_command", 0.95))
        self.linear_command = min(max_command, 0.65)
        self.vertical_command = min(max_command, 0.50)
        self.hold_s = 0.20
        self._last_input_time = 0.0
        self._command = _zero_command()
        self._reader = _KeyboardReader()
        print(
            "Manual keyboard controller active. Focus this terminal: "
            "W/S forward/back, A/D left/right, Up/Down raise/lower, Space stop."
        )

    def step(self, observation: Dict[str, Any]) -> Dict[str, float]:
        del observation
        keys = self._reader.read_keys()
        now = time.monotonic()
        if keys:
            self._command = self._command_from_keys(keys)
            self._last_input_time = now
        if now - self._last_input_time > self.hold_s:
            return _zero_command()
        command = dict(self._command)
        command["yaw"] = 0.0
        return command

    def close(self) -> None:
        pass

    def _command_from_keys(self, keys: Iterable[str]) -> Dict[str, float]:
        command = _zero_command()
        for key in keys:
            if key == "space":
                return _zero_command()
            if key == "w":
                command["surge"] = self.linear_command
            elif key == "s":
                command["surge"] = -self.linear_command
            elif key == "a":
                command["sway"] = self.linear_command
            elif key == "d":
                command["sway"] = -self.linear_command
            elif key == "up":
                command["heave"] = self.vertical_command
            elif key == "down":
                command["heave"] = -self.vertical_command
        command["yaw"] = 0.0
        return command


class _KeyboardReader:
    """Read pending keyboard events without blocking the race loop."""

    def __init__(self) -> None:
        self._msvcrt: Optional[Any] = None
        if os.name == "nt":
            try:
                import msvcrt

                self._msvcrt = msvcrt
            except ImportError:
                self._msvcrt = None

    def read_keys(self) -> List[str]:
        if self._msvcrt is None:
            return []
        keys: List[str] = []
        while self._msvcrt.kbhit():
            char = self._msvcrt.getwch()
            if char in ("\x00", "\xe0"):
                keys.extend(self._read_windows_arrow_key())
                continue
            normalized = char.lower()
            if normalized == " ":
                keys.append("space")
            elif normalized in {"w", "a", "s", "d"}:
                keys.append(normalized)
        return keys

    def _read_windows_arrow_key(self) -> List[str]:
        code = self._msvcrt.getwch()
        if code == "H":
            return ["up"]
        if code == "P":
            return ["down"]
        return []


def _zero_command() -> Dict[str, float]:
    return {"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}
