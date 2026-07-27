"""Observation-v3 and minimal temporal-progress context tests."""

from __future__ import annotations

import numpy as np
import pytest

from marine_race_arena.controllers.vision import VisionTarget
from marine_race_arena.learning.config import FEATURE_NAMES as V1_FEATURE_NAMES
from marine_race_arena.learning.config_v3 import (
    FEATURE_BOUNDS_V3,
    FEATURE_NAMES_V3,
    OBS_DIM_V3,
    OBS_ENCODING_VERSION_V3,
    MultiGateLearningContext,
)
from marine_race_arena.learning.observation_encoder_v3 import encode_observation_v3
from marine_race_arena.learning.tracker_context_v3 import OnboardMultiGateContextTracker


def _packet(range_m: float, *, bearing: float = 0.0, beacon_id: str = "B01", t: float = 0.0):
    return {
        "beacon_id": beacon_id,
        "bearing_deg": bearing,
        "elevation_deg": 0.0,
        "range_m": range_m,
        "signal_strength": 0.8,
        "received_at_s": t,
    }


def _obs(
    step: int,
    *,
    range_m: float = 5.0,
    camera=True,
    visible=True,
    dvl=(0.0, 0.0, 0.0),
    privileged=False,
):
    sensors = {
        "DepthSensor": [-3.0],
        "DVLSensor": list(dvl),
        "IMUSensor": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    }
    if camera:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frame[0, 0, 0] = 1 if visible else 0
        sensors["FrontCamera"] = frame
    out = {
        "local_time_s": step * 0.1,
        "sensors": sensors,
        "beacons": [_packet(range_m, t=step * 0.1)],
    }
    if privileged:
        out["referee"] = {"expected_gate": "G99", "valid_gate_crossings": 99}
        out["ground_truth"] = {"pose": [99.0] * 6}
        out["own_position"] = [99.0, 99.0, 99.0]
    return out


@pytest.fixture
def patched_vision(monkeypatch):
    import marine_race_arena.controllers.local_course_tracker as tracker_module

    target = VisionTarget(
        center_x=0.10,
        center_y=-0.08,
        confidence=0.85,
        area_fraction=0.12,
        width_fraction=0.3,
        height_fraction=0.4,
    )

    def detect(image):
        return [target] if int(np.asarray(image)[0, 0, 0]) == 1 else []

    monkeypatch.setattr(tracker_module, "vision_targets_from_camera", detect)
    monkeypatch.setattr(
        tracker_module,
        "select_visual_target_for_beacon",
        lambda targets, bearing, range_m: targets[0] if targets else None,
    )
    return target


def _feature(vector, name):
    return float(vector[FEATURE_NAMES_V3.index(name)])


def test_v3_layout_appends_to_frozen_v1():
    assert OBS_ENCODING_VERSION_V3 == "onboard_multigate_rl_v3"
    assert FEATURE_NAMES_V3[: len(V1_FEATURE_NAMES)] == V1_FEATURE_NAMES
    assert len(FEATURE_NAMES_V3) == len(FEATURE_BOUNDS_V3) == OBS_DIM_V3
    assert len(set(FEATURE_NAMES_V3)) == OBS_DIM_V3


def test_v3_bounds_masks_and_base_prefix():
    context = MultiGateLearningContext(
        expected_beacon_id="B01",
        tracker_phase="APPROACH",
        last_seen_center_x=4.0,
        last_seen_center_y=-4.0,
        last_seen_area_fraction=2.0,
        last_seen_confidence=2.0,
        last_seen_present=True,
        beacon_range_delta_m=-9.0,
        beacon_range_delta_present=True,
        forward_displacement_since_visual_loss_m=99.0,
        forward_displacement_since_visual_loss_present=True,
    )
    vector = encode_observation_v3(_obs(0, camera=False), context)
    assert vector.shape == (OBS_DIM_V3,)
    assert vector.dtype == np.float32
    assert np.all(np.isfinite(vector))
    for name, value, (low, high) in zip(FEATURE_NAMES_V3, vector, FEATURE_BOUNDS_V3):
        assert low - 1e-6 <= value <= high + 1e-6, name


def test_recent_seen_lost_and_dvl_displacement(patched_vision):
    source = OnboardMultiGateContextTracker(total_beacons=2)
    source.reset(_obs(0, visible=True))
    seen = source.context(_obs(0, visible=True), dt=0.1)
    assert seen.vision_recently_seen
    assert not seen.vision_recently_lost
    assert seen.gate_was_recently_centered
    assert seen.gate_was_recently_large

    lost = source.context(_obs(1, visible=False, dvl=(0.5, 0.0, 0.0)), dt=0.1)
    assert lost.vision_recently_seen
    assert lost.vision_recently_lost
    assert lost.steps_since_gate_seen == 1
    assert lost.forward_displacement_since_visual_loss_present
    assert lost.forward_displacement_since_visual_loss_m == pytest.approx(0.05)


def test_camera_dropout_does_not_become_visual_loss(patched_vision):
    source = OnboardMultiGateContextTracker(total_beacons=2)
    source.reset(_obs(0, visible=True))
    source.context(_obs(0, visible=True), dt=0.1)
    dropped = source.context(_obs(1, camera=False, dvl=(0.5, 0.0, 0.0)), dt=0.1)
    assert not dropped.camera_present
    assert not dropped.vision_recently_lost
    assert not dropped.forward_displacement_since_visual_loss_present
    assert source.tracker.expected_beacon_id == "B01"


def test_range_delta_minimum_and_receding(patched_vision):
    source = OnboardMultiGateContextTracker(total_beacons=2)
    source.reset(_obs(0, range_m=5.0))
    first = source.context(_obs(0, range_m=5.0), dt=0.1)
    approaching = source.context(_obs(1, range_m=4.7), dt=0.1)
    receding = source.context(_obs(2, range_m=5.0), dt=0.1)

    assert not first.beacon_range_delta_present
    assert approaching.beacon_range_delta_m == pytest.approx(-0.3)
    assert not approaching.beacon_now_receding
    assert receding.beacon_range_delta_m == pytest.approx(0.3)
    assert receding.beacon_now_receding
    assert receding.beacon_range_min_recent_m == pytest.approx(4.7)
    assert receding.steps_since_beacon_change == 0


def test_temporal_feature_masks_zero_absent_history():
    vector = encode_observation_v3({}, MultiGateLearningContext())
    assert _feature(vector, "last_seen_present") == 0.0
    assert _feature(vector, "last_seen_center_x") == 0.0
    assert _feature(vector, "beacon_range_delta_present") == 0.0
    assert _feature(vector, "beacon_range_delta_norm") == 0.0
    assert _feature(vector, "forward_displacement_since_visual_loss_present") == 0.0


def test_v3_encoder_ignores_privileged_observation_keys():
    context = MultiGateLearningContext(expected_beacon_id="B01", tracker_phase="APPROACH")
    clean = encode_observation_v3(_obs(2, camera=False), context)
    poisoned = encode_observation_v3(_obs(2, camera=False, privileged=True), context)
    assert np.array_equal(clean, poisoned)


def test_v3_dataset_preserves_version_and_dimension(tmp_path):
    from marine_race_arena.learning.dataset import BCDataset
    from marine_race_arena.learning.trajectory_recorder import EpisodeRecord

    record = EpisodeRecord(
        episode_id=0,
        seed=20000,
        track="two_gate",
        controller="rule_gate_center_then_commit",
        observations=np.zeros((2, OBS_DIM_V3), dtype=np.float32),
        expert_actions_raw=np.zeros((2, 4), dtype=np.float32),
        actions=np.zeros((2, 4), dtype=np.float32),
        dones=np.array([False, True]),
        truncated=np.array([False, False]),
        step_ids=np.array([0, 1]),
        phase_ids=np.array([0, 1]),
        final_status="FINISHED",
        gate_crossings=2,
        metadata={"obs_encoding_version": OBS_ENCODING_VERSION_V3},
    )
    dataset = BCDataset.from_records([record])
    dataset.check_integrity()
    assert dataset.expected_obs_dim == OBS_DIM_V3
    path = tmp_path / "v3.npz"
    dataset.save(path)
    loaded = BCDataset.load(path)
    loaded.check_integrity()
    assert loaded.observation_encoding_version == OBS_ENCODING_VERSION_V3
