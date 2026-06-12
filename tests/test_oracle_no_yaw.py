from __future__ import annotations

from marine_race_arena.controllers.oracle_gate_follower import OracleGateFollowerController


def test_oracle_translates_without_commanding_yaw() -> None:
    controller = OracleGateFollowerController()
    controller.reset({"bounds": {"z_min": -8.0, "z_max": -1.0}, "max_command": 0.95})

    command = controller.step(
        {
            "race": {"target_gate_id": "G01"},
            "debug_ground_truth": {
                "own_position": (0.0, -2.0, -4.0),
                "own_rotation_rpy_deg": (0.0, 0.0, 0.0),
                "target_gate_center": (0.0, 0.0, -4.0),
                "target_gate_normal": (1.0, 0.0, 0.0),
                "target_gate_right_axis": (0.0, 1.0, 0.0),
                "target_gate_up_axis": (0.0, 0.0, 1.0),
            },
        }
    )

    assert command["yaw"] == 0.0
    assert command["surge"] > 0.0
    assert command["sway"] > 0.0


def test_oracle_keeps_yaw_zero_even_with_rover_yaw_offset() -> None:
    controller = OracleGateFollowerController()
    controller.reset({"bounds": {"z_min": -8.0, "z_max": -1.0}, "max_command": 0.95})

    command = controller.step(
        {
            "race": {"target_gate_id": "G01"},
            "debug_ground_truth": {
                "own_position": (-2.0, 0.0, -4.0),
                "own_rotation_rpy_deg": (0.0, 0.0, 45.0),
                "target_gate_center": (0.0, 0.0, -4.0),
                "target_gate_normal": (1.0, 0.0, 0.0),
                "target_gate_right_axis": (0.0, 1.0, 0.0),
                "target_gate_up_axis": (0.0, 0.0, 1.0),
            },
        }
    )

    assert command["yaw"] == 0.0
