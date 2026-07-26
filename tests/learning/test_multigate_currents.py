"""Tests for current-free evaluation wiring and multi-gate failure classification."""

import pytest

from marine_race_arena.learning.closed_loop_eval import _IDENTITY_FIELDS, currents_summary
from marine_race_arena.learning.multigate_diagnostic import _classify_first_failure

MIXED = "marine_race_arena/tracks/marine_race_mixed_endurance.json"
HORSESHOE = "marine_race_arena/tracks/marine_race_horseshoe_bay.json"
YAW25 = "marine_race_arena/tracks/tests/single_gate_yaw_25.json"


# --------------------------------------------------------------- current-free mode
def test_current_profile_none_zeroes_currents_with_clean_gate_override():
    # mixed_endurance is a current_gate task; running it current-free needs the clean_gate
    # validation override, and must then prove currents_actual == [0,0,0].
    s = currents_summary(MIXED, "none", "clean_gate")
    assert s["currents_are_zero"] is True
    assert s["currents_actual"] == [0.0, 0.0, 0.0]
    assert s["n_currents"] == 0
    assert s["selected_current_profile"] == "none"


def test_default_profile_preserves_configured_currents():
    s = currents_summary(MIXED, None)
    assert s["n_currents"] >= 1
    assert s["currents_are_zero"] is False


def test_clean_gate_circuit_is_already_current_free():
    s = currents_summary(HORSESHOE, "none")
    assert s["currents_are_zero"] is True and s["n_currents"] == 0


def test_current_profile_is_a_resume_identity_field():
    # Two runs that differ only in current profile must NOT be merged on resume.
    assert "current_profile" in _IDENTITY_FIELDS


# ------------------------------------------------------ first-failure classification
def _ev(**kw):
    base = {"phase": "SEARCH", "gates": 0, "out_of_bounds": 0,
            "wrong_dir_delta": False, "collision_delta": False}
    base.update(kw)
    return base


def test_classify_finished():
    c = _classify_first_failure(finished=True, end_reason="FINISHED", gates=2,
                                expected_gates=2, events=[], tracker_completed=2)
    assert c["failure"] == "FINISHED"


def test_classify_collision_precedence():
    events = [_ev(collision_delta=True, gates=1, out_of_bounds=3)]
    c = _classify_first_failure(finished=False, end_reason="REFEREE_TERMINAL", gates=1,
                                expected_gates=2, events=events, tracker_completed=1)
    assert c["failure"] == "COLLISION"


def test_classify_out_of_bounds():
    events = [_ev(out_of_bounds=5, gates=1)]
    c = _classify_first_failure(finished=False, end_reason="TIME_LIMIT", gates=1,
                                expected_gates=2, events=events, tracker_completed=1)
    assert c["failure"] == "OUT_OF_BOUNDS"


def test_classify_tracker_false_advance():
    c = _classify_first_failure(finished=False, end_reason="TIME_LIMIT", gates=0,
                                expected_gates=2, events=[_ev()], tracker_completed=1)
    assert c["failure"] == "TRACKER_FALSE_ADVANCE"


def test_classify_return_to_previous_gate():
    events = [_ev(gates=1, wrong_dir_delta=True)]
    c = _classify_first_failure(finished=False, end_reason="TIME_LIMIT", gates=1,
                                expected_gates=2, events=events, tracker_completed=1)
    assert c["failure"] == "RETURN_TO_PREVIOUS_GATE"


def test_classify_never_reached_gate_one():
    c = _classify_first_failure(finished=False, end_reason="TIME_LIMIT", gates=0,
                                expected_gates=2, events=[_ev()], tracker_completed=0)
    assert c["failure"] == "FAILED_GATE_ALIGNMENT"


def test_classify_next_gate_turn_failed():
    events = [_ev(phase="SEARCH", gates=1) for _ in range(10)]
    c = _classify_first_failure(finished=False, end_reason="TIME_LIMIT", gates=1,
                                expected_gates=2, events=events, tracker_completed=1)
    assert c["failure"] == "NEXT_GATE_TURN_FAILED"


def test_classify_alignment_when_engaged_next_gate():
    events = [_ev(phase="VISUAL_ALIGN", gates=1) for _ in range(10)]
    c = _classify_first_failure(finished=False, end_reason="TIME_LIMIT", gates=1,
                                expected_gates=2, events=events, tracker_completed=1)
    assert c["failure"] == "FAILED_GATE_ALIGNMENT"


# --------------------------------------------------------- per-track aperture size
def test_vision_capture_reads_public_aperture_size():
    from marine_race_arena.learning.vision_pose_capture import _gate_aperture_m

    assert _gate_aperture_m(YAW25) == (2.0, 2.0)          # yaw validation tracks are 2.0 m
    assert _gate_aperture_m(HORSESHOE) == (1.5, 1.5)      # official gates are 1.5 m
