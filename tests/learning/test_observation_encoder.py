"""Tests for the legal fixed-size learning observation encoder."""

import math

import numpy as np
import pytest

from marine_race_arena.learning import config as lc
from marine_race_arena.learning.config import FEATURE_BOUNDS, FEATURE_NAMES, OBS_DIM, LearningContext
from marine_race_arena.learning import observation_encoder as enc
from marine_race_arena.learning.observation_encoder import encode_observation


def _full_observation():
    """A representative official observation with all sensors present."""
    return {
        "local_time_s": 12.5,
        "sensors": {
            "DepthSensor": [-4.0],
            "DVLSensor": [0.6, -0.1, 0.05],
            "IMUSensor": [[0.0, 0.0, -9.81], [0.0, 0.0, 0.3], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            # No FrontCamera key -> vision masked (image construction is covered by monkeypatch test).
        },
        "beacons": [
            {
                "beacon_id": "B02",
                "bearing_deg": 10.0,
                "elevation_deg": -5.0,
                "range_m": 6.0,
                "signal_strength": 0.8,
                "received_at_s": 12.0,
            },
            {
                "beacon_id": "B05",
                "bearing_deg": 40.0,
                "elevation_deg": 0.0,
                "range_m": 20.0,
                "signal_strength": 0.5,
                "received_at_s": 11.0,
            },
        ],
    }


def _context():
    return LearningContext(
        expected_beacon_id="B02",
        tracker_phase="APPROACH",
        local_beacon_index=1,
        local_lap=0,
        total_beacons=12,
        laps=1,
        depth_reference_m=4.0,
        visual_lock=True,
        prev_action=(0.4, -0.2, 0.0, 0.1),
    )


def test_layout_consistency():
    assert len(FEATURE_NAMES) == OBS_DIM
    assert len(FEATURE_BOUNDS) == OBS_DIM
    assert len(set(FEATURE_NAMES)) == OBS_DIM  # unique names


def test_shape_and_dtype():
    vec = encode_observation(_full_observation(), _context())
    assert vec.shape == (OBS_DIM,)
    assert vec.dtype == np.float32
    assert np.all(np.isfinite(vec))


def test_bounds_respected():
    vec = encode_observation(_full_observation(), _context())
    for name, value, (low, high) in zip(FEATURE_NAMES, vec, FEATURE_BOUNDS):
        assert low - 1e-6 <= value <= high + 1e-6, f"{name}={value} out of [{low},{high}]"


def test_determinism():
    obs, ctx = _full_observation(), _context()
    a = encode_observation(obs, ctx)
    b = encode_observation(obs, ctx)
    assert np.array_equal(a, b)


def test_extreme_values_are_clipped_and_finite():
    obs = _full_observation()
    obs["sensors"]["DVLSensor"] = [1e9, -1e9, float("inf")]
    obs["sensors"]["DepthSensor"] = [float("nan")]
    obs["beacons"][0]["range_m"] = 1e12
    vec = encode_observation(obs, _context())
    assert np.all(np.isfinite(vec))
    for value, (low, high) in zip(vec, FEATURE_BOUNDS):
        assert low - 1e-6 <= value <= high + 1e-6


def _feat(vec, name):
    return float(vec[FEATURE_NAMES.index(name)])


def test_missing_beacon_is_masked():
    obs = _full_observation()
    obs["beacons"] = []
    vec = encode_observation(obs, _context())
    assert _feat(vec, "beacon_present") == 0.0
    for name in (
        "beacon_bearing_sin",
        "beacon_bearing_cos",
        "beacon_elevation_norm",
        "beacon_range_norm",
        "beacon_signal_strength",
        "beacon_age_norm",
    ):
        assert _feat(vec, name) == 0.0


def test_expected_beacon_not_received_is_masked():
    obs = _full_observation()
    ctx = _context()
    ctx.expected_beacon_id = "B09"  # not in the received packets
    vec = encode_observation(obs, ctx)
    assert _feat(vec, "beacon_present") == 0.0


def test_expected_beacon_selected_over_stronger_other():
    obs = _full_observation()
    vec = encode_observation(obs, _context())  # expects B02
    assert _feat(vec, "beacon_present") == 1.0
    # B02 bearing 10deg -> sin positive small, cos near 1
    assert _feat(vec, "beacon_bearing_cos") > 0.9
    assert _feat(vec, "beacon_range_norm") == pytest.approx(6.0 / lc.RANGE_SCALE_M, rel=1e-5)


def test_missing_camera_masks_vision():
    vec = encode_observation(_full_observation(), _context())
    assert _feat(vec, "vision_present") == 0.0
    for name in ("vision_center_x", "vision_center_y", "vision_area_fraction", "vision_confidence"):
        assert _feat(vec, name) == 0.0


def test_vision_features_from_detection(monkeypatch):
    from marine_race_arena.controllers.vision import VisionTarget

    target = VisionTarget(center_x=0.25, center_y=-0.1, confidence=0.9, area_fraction=0.05)
    monkeypatch.setattr(enc, "vision_targets_from_camera", lambda image: [target])
    monkeypatch.setattr(enc, "select_visual_target_for_beacon", lambda t, b, r: target)
    obs = _full_observation()
    obs["sensors"]["FrontCamera"] = np.zeros((16, 16, 3), dtype=np.uint8)
    vec = encode_observation(obs, _context())
    assert _feat(vec, "vision_present") == 1.0
    assert _feat(vec, "vision_center_x") == pytest.approx(0.25, rel=1e-5)
    assert _feat(vec, "vision_center_y") == pytest.approx(-0.1, rel=1e-5)
    assert _feat(vec, "vision_confidence") == pytest.approx(0.9, rel=1e-5)


def test_sensor_dropout_masks_present_flags():
    obs = _full_observation()
    obs["sensors"] = {}
    vec = encode_observation(obs, _context())
    assert _feat(vec, "depth_present") == 0.0
    assert _feat(vec, "dvl_present") == 0.0
    assert _feat(vec, "imu_present") == 0.0
    assert _feat(vec, "depth_ref_present") == 0.0


def test_depth_error_masked_without_reference():
    obs = _full_observation()
    ctx = _context()
    ctx.depth_reference_m = None
    vec = encode_observation(obs, ctx)
    assert _feat(vec, "depth_present") == 1.0
    assert _feat(vec, "depth_ref_present") == 0.0
    assert _feat(vec, "depth_error_norm") == 0.0


def test_phase_one_hot():
    obs = _full_observation()
    for phase in lc.TRACKER_PHASES:
        ctx = _context()
        ctx.tracker_phase = phase
        vec = encode_observation(obs, ctx)
        hot = [_feat(vec, f"phase_{p}") for p in lc.TRACKER_PHASES]
        assert sum(hot) == 1.0
        assert _feat(vec, f"phase_{phase}") == 1.0


def test_phase_none_is_all_zero():
    ctx = _context()
    ctx.tracker_phase = None
    vec = encode_observation(_full_observation(), ctx)
    assert sum(_feat(vec, f"phase_{p}") for p in lc.TRACKER_PHASES) == 0.0


def test_prev_action_passthrough_and_clip():
    ctx = _context()
    ctx.prev_action = (2.0, -2.0, 0.3, -0.4)  # first two exceed [-1,1]
    vec = encode_observation(_full_observation(), ctx)
    assert _feat(vec, "prev_surge") == 1.0
    assert _feat(vec, "prev_sway") == -1.0
    assert _feat(vec, "prev_heave") == pytest.approx(0.3, rel=1e-5)
    assert _feat(vec, "prev_yaw") == pytest.approx(-0.4, rel=1e-5)


def test_no_privileged_leakage():
    """Injecting privileged keys must not change the encoding."""
    obs = _full_observation()
    baseline = encode_observation(obs, _context())
    poisoned = _full_observation()
    poisoned["referee"] = {"target_gate": "B03", "valid_gate_crossings": 5, "status": "RUNNING"}
    poisoned["ground_truth"] = {"position": [1, 2, 3], "velocity": [4, 5, 6]}
    poisoned["own_position"] = [10.0, 20.0, -3.0]
    poisoned["true_current"] = [0.5, 0.5, 0.0]
    poisoned["sensors"]["GroundTruthPose"] = [1.0, 2.0, 3.0]
    out = encode_observation(poisoned, _context())
    assert np.array_equal(baseline, out)


def test_empty_observation_is_safe():
    vec = encode_observation({}, None)
    assert vec.shape == (OBS_DIM,)
    assert np.all(np.isfinite(vec))
    assert _feat(vec, "beacon_present") == 0.0
    assert _feat(vec, "vision_present") == 0.0


# --- Drift guards: keep the encoder consistent with benchmark internals -------

def test_phase_constants_match_tracker():
    from marine_race_arena.controllers import local_course_tracker as t

    expected = (
        t.PHASE_SEARCH,
        t.PHASE_APPROACH,
        t.PHASE_VISUAL_ALIGN,
        t.PHASE_COMMIT,
        t.PHASE_VERIFY_EXIT,
        t.PHASE_ADVANCE,
        t.PHASE_FINISHED,
    )
    assert lc.TRACKER_PHASES == expected


def test_sensor_parsing_matches_official_helpers():
    from marine_race_arena.controllers import official_baselines as ob

    sensors = _full_observation()["sensors"]
    assert enc._depth_m(sensors) == pytest.approx(ob._depth_m_from_sensors(sensors))
    assert enc._imu_yaw_rate(sensors) == pytest.approx(ob._yaw_rate_from_sensors(sensors))
    assert enc._dvl_body_velocity(sensors)[2] == pytest.approx(ob._vertical_velocity_from_sensors(sensors))
