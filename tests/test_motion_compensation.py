from __future__ import annotations

import pytest

from marine_race_arena.controllers.motion_compensation import (
    MOTION_COMPENSATION_NONE,
    NoMotionCompensator,
    make_motion_compensator,
    normalize_motion_compensation_mode,
)


def test_no_compensation_keeps_command_unchanged() -> None:
    compensator = NoMotionCompensator()
    command = {"surge": 0.2, "sway": -0.1, "heave": 0.05, "yaw": 0.03}

    compensated = compensator.compensate(command, {"sensors": {"DVLSensor": [0.4, 0.2, 0.0]}}, dt=0.1)

    assert compensated == command
    assert compensator.diagnostics().active is False


def test_make_motion_compensator_returns_pass_through() -> None:
    compensator = make_motion_compensator(MOTION_COMPENSATION_NONE)

    assert isinstance(compensator, NoMotionCompensator)
    assert compensator.mode == MOTION_COMPENSATION_NONE


def test_normalize_accepts_none_and_rejects_unknown_modes() -> None:
    assert normalize_motion_compensation_mode(None) == MOTION_COMPENSATION_NONE
    assert normalize_motion_compensation_mode("none") == MOTION_COMPENSATION_NONE

    # The experimental dvl_pi compensator was removed; current compensation is
    # future work, so any non-"none" mode is rejected.
    with pytest.raises(ValueError):
        normalize_motion_compensation_mode("dvl_pi")
    with pytest.raises(ValueError):
        make_motion_compensator("dvl_pi")
