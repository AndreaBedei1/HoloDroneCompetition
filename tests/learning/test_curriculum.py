"""Tests for the curriculum definitions and training tracks."""

import json
from pathlib import Path

import pytest

from marine_race_arena.config.loader import load_track_config
from marine_race_arena.learning import curriculum as cur

OFFICIAL_TRACK = "marine_race_arena/tracks/marine_race_horseshoe_bay.json"
TRAINING_TRACKS = [
    "marine_race_arena/tracks/training/stage1_single_gate.json",
    "marine_race_arena/tracks/training/stage3_three_gates.json",
    "marine_race_arena/tracks/training/stage4_six_gates.json",
]


def _raw(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_stage_keys_unique_and_lookup():
    keys = [s.key for s in cur.STAGES]
    assert len(keys) == len(set(keys))
    for key in keys:
        assert cur.stage(key).key == key
    with pytest.raises(KeyError):
        cur.stage("does_not_exist")


def test_next_stage_chain():
    assert cur.next_stage("stage0").key == "stage1"
    assert cur.next_stage(cur.STAGES[-1].key) is None


def test_meets_criterion():
    assert cur.meets_criterion("stage1", 0.95) is True
    assert cur.meets_criterion("stage1", 0.80) is False
    assert cur.meets_criterion("stage3", 0.80) is True


def test_completion_criteria_are_monotone_sane():
    for s in cur.STAGES:
        assert 0.0 <= s.min_completion_rate <= 1.0
        assert s.eval_episodes >= 0
    # early single-gate stages demand higher completion than multi-gate ones
    assert cur.stage("stage1").min_completion_rate >= cur.stage("stage3").min_completion_rate


def test_training_tracks_exist_and_load():
    for key in ("stage1", "stage3", "stage4"):
        track = cur.stage(key).track
        assert Path(track).exists(), f"missing training track {track}"
        config = load_track_config(track)
        assert config.race.official_mode is True
        # official gate aperture preserved
        assert tuple(config.track.gate_inner_size_m) == (1.5, 1.5)


def test_stage4_is_five_or_six_gates():
    config = load_track_config(cur.stage("stage4").track)
    assert 5 <= len(config.track.gate_sequence) <= 6


def test_stage2_has_real_randomization():
    s = cur.stage("stage2")
    assert s.randomization is not None and not s.randomization.is_noop()
    assert s.randomization.lateral_offset_m > 0.0
    assert s.randomization.beacon_angular_noise_std_deg is not None


def test_stage6_evaluates_both_official_tracks():
    s = cur.stage("stage6_generalization")
    assert "vertical_serpent" in s.tracks[0] or "vertical_serpent" in s.tracks[1]
    assert any("mixed_endurance" in t for t in s.tracks)
    assert len(s.tracks) == 2


def test_stage7_is_a_real_current_progression():
    s = cur.stage("stage7_disturbance")
    assert s.current_progression == ("none", "medium", "strong")  # not medium-only


def test_seed_split_is_disjoint():
    split = cur.seed_split(n_train=50, n_val=10, n_eval=10)
    train, val, ev = set(split["train"]), set(split["val"]), set(split["eval"])
    assert not (train & val) and not (train & ev) and not (val & ev)
    assert len(train) == 50 and len(val) == 10 and len(ev) == 10


def test_advance_decision_logic():
    nxt, decision = cur.advance_decision("stage1", 0.95)
    assert nxt == "stage2" and "advance" in decision
    nxt2, decision2 = cur.advance_decision("stage1", 0.50)
    assert nxt2 is None and "HOLD" in decision2
    # last stage advances to None even when passed
    assert cur.advance_decision("stage7_disturbance", 0.95)[0] is None


def test_evaluate_stage_runner_refuses_advance_on_failure():
    from marine_race_arena.participants.controller_loader import ControllerLoader

    factory = lambda: ControllerLoader().load("rule_gate_center_then_commit")
    result = cur.evaluate_stage("stage1", factory, seeds=[100, 101], adapter="fallback", allow_fallback=True, duration_s=3.0)
    assert result.stage_key == "stage1"
    assert result.n_episodes == 2
    assert len(result.per_evaluation) == 1
    assert 0.0 <= result.completion_rate <= 1.0
    # The engine-free fallback stalls camera-gated controllers; the runner must not advance.
    if not result.passed:
        nxt, decision = cur.advance_decision("stage1", result.completion_rate)
        assert nxt is None and "HOLD" in decision


def test_official_stages_reference_unchanged_official_tracks():
    for key in ("stage5_horseshoe", "stage6_generalization", "stage7_disturbance"):
        s = cur.stage(key)
        assert s.official_track is True
        assert "tracks/marine_race_" in s.track


@pytest.mark.parametrize("track", TRAINING_TRACKS)
def test_training_tracks_preserve_official_difficulty(track):
    """Training tracks must not relax the benchmark's gate geometry or referee."""
    official = _raw(OFFICIAL_TRACK)
    t = _raw(track)

    # Official 1.5 x 1.5 m gate aperture, normal thickness and depth.
    assert t["track"]["gate_inner_size_m"] == [1.5, 1.5]
    assert t["track"]["gate_bar_thickness_m"] == official["track"]["gate_bar_thickness_m"]
    assert t["track"]["gate_depth_m"] == official["track"]["gate_depth_m"]

    # Official onboard observation profile and action mapping.
    part = t["participants"][0]
    off_part = official["participants"][0]
    assert part["sensors"]["profile"] == off_part["sensors"]["profile"]
    assert set(part["sensors"]["allowed_sensors"]) == set(off_part["sensors"]["allowed_sensors"])
    assert part["control_mode"] == off_part["control_mode"] == "high_level"
    assert t["race"]["official_mode"] is True

    # Independent referee with the OFFICIAL clearance margin and no relaxed validation.
    gv = t["referee"]["gate_validation"]
    off_gv = official["referee"]["gate_validation"]
    assert gv["vehicle_clearance_margin_m"] == off_gv["vehicle_clearance_margin_m"] == 0.1
    assert gv["vehicle_model"] == off_gv["vehicle_model"] == "center_point"
    assert gv["timeout_enabled"] == off_gv["timeout_enabled"]
    # Penalties/scoring present (independent referee is fully configured).
    assert "penalties" in t["referee"] and "scoring" in t["referee"]
