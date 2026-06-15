from __future__ import annotations

from marine_race_arena.controllers.pygame_manual import PygameManualController


def _controller_without_window() -> PygameManualController:
    controller = PygameManualController.__new__(PygameManualController)
    controller.linear_command = 0.65
    controller.vertical_command = 0.50
    controller.yaw_command = 0.45
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
