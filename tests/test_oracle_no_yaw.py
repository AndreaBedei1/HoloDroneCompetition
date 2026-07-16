from __future__ import annotations

from marine_race_arena.controllers.oracle_gate_follower import OracleGateFollowerController


MISSION_INFO = {
    "participant_id": "bluerov2_01",
    "initial_beacon_id": "B01",
    "total_beacons": 1,
    "laps": 1,
    "command_limits": {
        axis: [-0.95, 0.95] for axis in ("surge", "sway", "heave", "yaw")
    },
}


def _observation(
    own_position: tuple[float, float, float],
    yaw_deg: float,
) -> dict:
    return {
        "local_time_s": 0.0,
        "sensors": {
            "DepthSensor": [own_position[2]],
            "DVLSensor": [0.0, 0.0, 0.0],
        },
        "beacons": [],
        "debug_ground_truth": {
            "own_position": own_position,
            "own_rotation_rpy_deg": (0.0, 0.0, yaw_deg),
            "bounds": {"z_min": -8.0, "z_max": -1.0},
            "gates": [
                {
                    "gate_id": "G01",
                    "center": (0.0, 0.0, -4.0),
                    "normal": (1.0, 0.0, 0.0),
                    "right_axis": (0.0, 1.0, 0.0),
                    "up_axis": (0.0, 0.0, 1.0),
                }
            ],
        },
    }


def test_oracle_translates_without_commanding_yaw() -> None:
    controller = OracleGateFollowerController()
    controller.reset(MISSION_INFO)

    command = controller.step(_observation((0.0, -2.0, -4.0), yaw_deg=0.0))

    assert command["yaw"] == 0.0
    assert command["surge"] > 0.0
    assert command["sway"] > 0.0


def test_oracle_keeps_yaw_zero_even_with_rover_yaw_offset() -> None:
    controller = OracleGateFollowerController()
    controller.reset(MISSION_INFO)

    command = controller.step(_observation((-2.0, 0.0, -4.0), yaw_deg=45.0))

    assert command["yaw"] == 0.0


def test_oracle_stops_safely_without_its_explicit_debug_channel() -> None:
    controller = OracleGateFollowerController()
    controller.reset(MISSION_INFO)

    command = controller.step(
        {
            "local_time_s": 0.0,
            "sensors": {"DepthSensor": [-4.0], "DVLSensor": [0.0, 0.0, 0.0]},
            "beacons": [],
        }
    )

    assert command == {"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}
