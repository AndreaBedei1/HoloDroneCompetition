from __future__ import annotations

import pytest

from marine_race_arena.controllers.motion_compensation import (
    DVLCurrentCompensator,
    NoMotionCompensator,
    extract_body_velocity_xy,
)


def test_no_compensation_keeps_command_unchanged() -> None:
    compensator = NoMotionCompensator()
    command = {"surge": 0.2, "sway": -0.1, "heave": 0.05, "yaw": 0.03}

    compensated = compensator.compensate(command, {"sensors": {"DVLSensor": [0.4, 0.2, 0.0]}}, dt=0.1)

    assert compensated == command


def test_missing_dvl_keeps_command_unchanged() -> None:
    compensator = DVLCurrentCompensator()
    command = {"surge": 0.2, "sway": -0.1, "heave": 0.05, "yaw": 0.03}

    compensated = compensator.compensate(command, {"sensors": {}}, dt=0.1)

    assert compensated == pytest.approx(command)
    assert compensator.diagnostics().active is False


def test_lateral_drift_creates_opposite_sway_correction() -> None:
    compensator = DVLCurrentCompensator(kp=0.5, ki=0.0)
    command = {"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}

    compensated = compensator.compensate(command, {"sensors": {"DVLSensor": [0.0, 0.35, 0.0]}}, dt=0.1)

    assert compensated["sway"] < 0.0
    assert compensated["surge"] == pytest.approx(0.0)
    assert compensated["yaw"] == pytest.approx(0.0)


def test_surge_velocity_error_creates_surge_correction() -> None:
    compensator = DVLCurrentCompensator(kp=0.5, ki=0.0)
    command = {"surge": 0.3, "sway": 0.0, "heave": 0.0, "yaw": 0.0}

    compensated = compensator.compensate(command, {"sensors": {"DVLSensor": [0.05, 0.0, 0.0]}}, dt=0.1)

    assert compensated["surge"] > command["surge"]
    assert compensated["sway"] == pytest.approx(0.0)


def test_anti_windup_clamps_integral() -> None:
    compensator = DVLCurrentCompensator(kp=0.0, ki=1.0, integral_limit_m_s=0.2)
    command = {"surge": 1.0, "sway": 1.0, "heave": 0.0, "yaw": 0.0}

    for _ in range(20):
        compensator.compensate(command, {"sensors": {"DVLSensor": [-1.0, -1.0, 0.0]}}, dt=0.5)

    assert compensator.integral_error == pytest.approx([0.2, 0.2])
    assert compensator.diagnostics().integral_error == pytest.approx((0.2, 0.2))


def test_command_limits_are_respected() -> None:
    compensator = DVLCurrentCompensator(kp=5.0, ki=5.0, max_correction=10.0, command_limit=0.7)
    command = {"surge": 0.65, "sway": -0.65, "heave": 0.4, "yaw": -0.2}

    compensated = compensator.compensate(command, {"sensors": {"DVLSensor": [-4.0, 4.0, 0.0]}}, dt=1.0)

    assert -0.7 <= compensated["surge"] <= 0.7
    assert -0.7 <= compensated["sway"] <= 0.7
    assert compensated["heave"] == pytest.approx(0.4)
    assert compensated["yaw"] == pytest.approx(-0.2)


def test_yawing_command_reduces_lateral_compensation() -> None:
    straight = DVLCurrentCompensator(kp=0.5, ki=0.0)
    turning = DVLCurrentCompensator(kp=0.5, ki=0.0)
    observation = {"sensors": {"DVLSensor": [0.0, 0.35, 0.0]}}

    straight_command = straight.compensate(
        {"surge": 0.1, "sway": 0.0, "heave": 0.0, "yaw": 0.0},
        observation,
        dt=0.1,
    )
    turning_command = turning.compensate(
        {"surge": 0.1, "sway": 0.0, "heave": 0.0, "yaw": 0.14},
        observation,
        dt=0.1,
    )

    assert abs(turning_command["sway"]) < abs(straight_command["sway"])
    assert abs(turning_command["sway"]) > 0.0


def test_extract_body_velocity_accepts_mapping_formats() -> None:
    assert extract_body_velocity_xy({"DVLSensor": {"velocity": {"x": 0.1, "y": -0.2}}}) == pytest.approx(
        (0.1, -0.2)
    )
    assert extract_body_velocity_xy({"VelocitySensor": {"vx": 0.3, "vy": 0.4}}) == pytest.approx((0.3, 0.4))
