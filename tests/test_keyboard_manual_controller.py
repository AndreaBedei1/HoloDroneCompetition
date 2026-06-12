from __future__ import annotations

from marine_race_arena.controllers.keyboard_manual import KeyboardManualController


def _controller() -> KeyboardManualController:
    controller = KeyboardManualController()
    controller.reset({"max_command": 0.95})
    return controller


def test_wasd_commands_move_on_horizontal_plane_without_yaw() -> None:
    controller = _controller()

    forward = controller._command_from_keys(["w"])
    backward = controller._command_from_keys(["s"])
    left = controller._command_from_keys(["a"])
    right = controller._command_from_keys(["d"])

    assert forward == {"surge": controller.linear_command, "sway": 0.0, "heave": 0.0, "yaw": 0.0}
    assert backward == {"surge": -controller.linear_command, "sway": 0.0, "heave": 0.0, "yaw": 0.0}
    assert left == {"surge": 0.0, "sway": controller.linear_command, "heave": 0.0, "yaw": 0.0}
    assert right == {"surge": 0.0, "sway": -controller.linear_command, "heave": 0.0, "yaw": 0.0}


def test_arrow_commands_raise_and_lower_without_yaw() -> None:
    controller = _controller()

    up = controller._command_from_keys(["up"])
    down = controller._command_from_keys(["down"])

    assert up == {"surge": 0.0, "sway": 0.0, "heave": controller.vertical_command, "yaw": 0.0}
    assert down == {"surge": 0.0, "sway": 0.0, "heave": -controller.vertical_command, "yaw": 0.0}


def test_space_stops_all_motion() -> None:
    controller = _controller()

    assert controller._command_from_keys(["w", "space"]) == {
        "surge": 0.0,
        "sway": 0.0,
        "heave": 0.0,
        "yaw": 0.0,
    }
