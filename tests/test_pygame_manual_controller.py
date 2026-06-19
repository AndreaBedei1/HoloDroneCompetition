from __future__ import annotations

import pytest

from marine_race_arena.controllers.pygame_manual import PygameManualController
from marine_race_arena.participants.controller_interface import ManualStopRequested


def _controller_without_window() -> PygameManualController:
    controller = PygameManualController.__new__(PygameManualController)
    controller.linear_command = 0.65
    controller.vertical_command = 0.50
    controller.yaw_command = 0.45
    controller._screen = None
    controller._font = None
    controller._closed = False
    return controller


def test_pygame_wasd_maps_to_planar_motion() -> None:
    controller = _controller_without_window()

    assert controller._command_from_keys(["w"]) == {
        "surge": 0.65,
        "sway": 0.0,
        "heave": 0.0,
        "yaw": 0.0,
    }
    assert controller._command_from_keys(["s"]) == {
        "surge": -0.65,
        "sway": 0.0,
        "heave": 0.0,
        "yaw": 0.0,
    }
    assert controller._command_from_keys(["a"]) == {
        "surge": 0.0,
        "sway": 0.65,
        "heave": 0.0,
        "yaw": 0.0,
    }
    assert controller._command_from_keys(["d"]) == {
        "surge": 0.0,
        "sway": -0.65,
        "heave": 0.0,
        "yaw": 0.0,
    }


def test_pygame_arrows_map_to_vertical_motion_without_yaw() -> None:
    controller = _controller_without_window()

    assert controller._command_from_keys(["up"]) == {
        "surge": 0.0,
        "sway": 0.0,
        "heave": 0.50,
        "yaw": 0.0,
    }
    assert controller._command_from_keys(["down"]) == {
        "surge": 0.0,
        "sway": 0.0,
        "heave": -0.50,
        "yaw": 0.0,
    }


def test_pygame_qe_maps_to_yaw() -> None:
    controller = _controller_without_window()

    assert controller._command_from_keys(["q"]) == {
        "surge": 0.0,
        "sway": 0.0,
        "heave": 0.0,
        "yaw": 0.45,
    }
    assert controller._command_from_keys(["e"]) == {
        "surge": 0.0,
        "sway": 0.0,
        "heave": 0.0,
        "yaw": -0.45,
    }
    assert controller._command_from_keys(["q", "e"]) == {
        "surge": 0.0,
        "sway": 0.0,
        "heave": 0.0,
        "yaw": 0.0,
    }


def test_pygame_stop_key_zeros_command() -> None:
    controller = _controller_without_window()

    assert controller._command_from_keys(["w", "d", "stop"]) == {
        "surge": 0.0,
        "sway": 0.0,
        "heave": 0.0,
        "yaw": 0.0,
    }


def test_pygame_escape_requests_manual_stop() -> None:
    controller = _controller_with_fake_pygame([_FakeEvent(_FakePygame.KEYDOWN, _FakePygame.K_ESCAPE)])

    with pytest.raises(ManualStopRequested):
        controller.step({})

    assert controller._closed is True


def test_pygame_window_close_requests_manual_stop() -> None:
    controller = _controller_with_fake_pygame([_FakeEvent(_FakePygame.QUIT)])

    with pytest.raises(ManualStopRequested):
        controller.step({})

    assert controller._closed is True


def _controller_with_fake_pygame(events: list["_FakeEvent"]) -> PygameManualController:
    controller = _controller_without_window()
    controller._pygame = _FakePygame(events)
    return controller


class _FakeEvent:
    def __init__(self, event_type: int, key: int | None = None) -> None:
        self.type = event_type
        self.key = key


class _FakeEventQueue:
    def __init__(self, events: list[_FakeEvent]) -> None:
        self._events = events

    def get(self) -> list[_FakeEvent]:
        return list(self._events)


class _FakePressed:
    def __getitem__(self, key: int) -> bool:
        del key
        return False


class _FakeKeyState:
    def get_pressed(self) -> _FakePressed:
        return _FakePressed()


class _FakePygame:
    QUIT = 1
    KEYDOWN = 2
    K_w = 10
    K_s = 11
    K_a = 12
    K_d = 13
    K_q = 14
    K_e = 15
    K_UP = 16
    K_DOWN = 17
    K_ESCAPE = 18
    K_SPACE = 19

    def __init__(self, events: list[_FakeEvent]) -> None:
        self.event = _FakeEventQueue(events)
        self.key = _FakeKeyState()
