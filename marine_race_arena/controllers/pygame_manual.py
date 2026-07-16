"""Pygame manual controller for local HoloOcean testing.

The controller opens a small Pygame window and reads movement keys from it.
It does not use ground truth.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Set

from marine_race_arena.participants.controller_interface import BaseController, ManualStopRequested


def _max_command_magnitude(mission_info: Dict[str, Any]) -> float:
    limits = mission_info.get("command_limits")
    if isinstance(limits, dict):
        bounds = limits.get("surge")
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            try:
                return min(abs(float(bounds[0])), abs(float(bounds[1])))
            except (TypeError, ValueError):
                pass
    return 0.95


class PygameManualController(BaseController):
    """Manual WASD, Q/E yaw, and arrow-key controller backed by Pygame."""

    debug_only = False
    uses_ground_truth = False

    def reset(self, mission_info: Dict[str, Any]) -> None:
        max_command = _max_command_magnitude(mission_info)
        self.linear_command = min(max_command, 0.65)
        self.vertical_command = min(max_command, 0.50)
        self.yaw_command = min(max_command, 0.45)
        self._pygame: Optional[Any] = None
        self._screen: Optional[Any] = None
        self._font: Optional[Any] = None
        self._closed = False
        self._init_pygame()

    def step(self, observation: Dict[str, Any]) -> Dict[str, float]:
        del observation
        if self._pygame is None or self._closed:
            return _zero_command()

        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                self._closed = True
                raise ManualStopRequested("Pygame control window closed.")
            if event.type == self._pygame.KEYDOWN and event.key == self._pygame.K_ESCAPE:
                self._closed = True
                raise ManualStopRequested("Escape pressed in Pygame control window.")

        pressed = self._pygame.key.get_pressed()
        keys = set()
        if pressed[self._pygame.K_w]:
            keys.add("w")
        if pressed[self._pygame.K_s]:
            keys.add("s")
        if pressed[self._pygame.K_a]:
            keys.add("a")
        if pressed[self._pygame.K_d]:
            keys.add("d")
        if pressed[self._pygame.K_q]:
            keys.add("q")
        if pressed[self._pygame.K_e]:
            keys.add("e")
        if pressed[self._pygame.K_UP]:
            keys.add("up")
        if pressed[self._pygame.K_DOWN]:
            keys.add("down")
        if pressed[self._pygame.K_ESCAPE]:
            self._closed = True
            raise ManualStopRequested("Escape pressed in Pygame control window.")
        if pressed[self._pygame.K_SPACE]:
            keys.add("stop")

        command = self._command_from_keys(keys)
        self._draw_status(command)
        return command

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()
        self._pygame = None
        self._screen = None
        self._font = None

    def _init_pygame(self) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "The pygame controller requires pygame in the active Python environment."
            ) from exc

        pygame.init()
        self._pygame = pygame
        self._screen = pygame.display.set_mode((520, 180))
        pygame.display.set_caption("Marine Race Manual Controller")
        self._font = pygame.font.Font(None, 24)
        self._draw_status(_zero_command())

    def _command_from_keys(self, keys: Iterable[str]) -> Dict[str, float]:
        key_set: Set[str] = set(keys)
        if "stop" in key_set:
            return _zero_command()

        command = _zero_command()
        if "w" in key_set and "s" not in key_set:
            command["surge"] = self.linear_command
        elif "s" in key_set and "w" not in key_set:
            command["surge"] = -self.linear_command

        if "a" in key_set and "d" not in key_set:
            command["sway"] = self.linear_command
        elif "d" in key_set and "a" not in key_set:
            command["sway"] = -self.linear_command

        if "up" in key_set and "down" not in key_set:
            command["heave"] = self.vertical_command
        elif "down" in key_set and "up" not in key_set:
            command["heave"] = -self.vertical_command

        if "q" in key_set and "e" not in key_set:
            command["yaw"] = self.yaw_command
        elif "e" in key_set and "q" not in key_set:
            command["yaw"] = -self.yaw_command

        return command

    def _draw_status(self, command: Dict[str, float]) -> None:
        if self._pygame is None or self._screen is None or self._font is None:
            return
        self._screen.fill((16, 24, 32))
        lines = [
            "Focus this window. W/S forward/back, A/D left/right.",
            "Q/E yaw left/right. Arrow Up/Down raises/lowers.",
            "Space stops motion. Esc quits the race.",
            (
                f"surge={command['surge']:.2f}  sway={command['sway']:.2f}  "
                f"heave={command['heave']:.2f}  yaw={command['yaw']:.2f}"
            ),
        ]
        for index, line in enumerate(lines):
            surface = self._font.render(line, True, (230, 238, 245))
            self._screen.blit(surface, (18, 16 + index * 38))
        self._pygame.display.flip()


def _zero_command() -> Dict[str, float]:
    return {"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}
