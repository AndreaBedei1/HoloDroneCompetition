"""Comprehensive tests for the Gen-2 27-D observation contract and pipeline."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from marine_race_arena.learning.config_local_transition import (
    FEATURE_NAMES_LOCAL_TRANSITION,
    OBS_DIM_LOCAL_TRANSITION,
)
from marine_race_arena.learning.config_local_transition_27d import (
    FEATURE_BOUNDS_LOCAL_TRANSITION_27D,
    FEATURE_NAMES_LOCAL_TRANSITION_27D,
    NEW_ORIENTATION_FEATURES_27D,
    OBS_DIM_LOCAL_TRANSITION_27D,
    OBS_ENCODING_VERSION_LOCAL_TRANSITION_27D,
    REMOVED_FEATURES_FROM_35D,
    LocalTransition27dLearningContext,
)
from marine_race_arena.learning.gen2 import track_fragments as tf
from marine_race_arena.learning.gen2.fog_contract import (
    APPROVED_WATER_FOG,
    assert_approved_water_fog,
    assert_approved_water_fog_dict,
    verify_fog_sources,
)
from marine_race_arena.learning.gen2.monitor_snapshot import (
    render_monitor_overlay,
)
from marine_race_arena.learning.gen2.prepare_dataset_27d import (
    collect_expert_27d_rollout,
)
from marine_race_arena.learning.gen2.recurrent_policy import (
    GEN2_27D_ARCH,
    Gen2RecurrentController,
    build_gen2_policy_for_training,
)
from marine_race_arena.learning.gen2.transfer_27d import (
    get_feature_index_mapping,
    sha256_file,
    transfer_parent_35d_to_27d,
    verify_transfer_27d_tensors,
)
from marine_race_arena.learning.observation_encoder_local_transition_27d import (
    encode_observation_local_transition_27d,
)


EXPECTED_27_FEATURES = (
    "beacon_present",
    "beacon_bearing_sin",
    "beacon_bearing_cos",
    "beacon_elevation_norm",
    "beacon_range_norm",
    "vision_center_x",
    "vision_center_y",
    "vision_area_fraction",
    "depth_norm",
    "dvl_surge_norm",
    "dvl_sway_norm",
    "dvl_heave_norm",
    "dvl_present",
    "imu_yaw_rate_norm",
    "previous_surge",
    "previous_sway",
    "previous_heave",
    "previous_yaw",
    "vision_center_x_rate",
    "vision_center_y_rate",
    "beacon_bearing_rate",
    "beacon_elevation_rate",
    "beacon_range_rate",
    "target_changed_recently",
    "gate_orientation_present",
    "gate_yaw_sin",
    "gate_yaw_cos",
)


def test_feature_names_and_exact_order():
    """1. Verify exact 27 features in exact order."""
    assert FEATURE_NAMES_LOCAL_TRANSITION_27D == EXPECTED_27_FEATURES
    assert len(FEATURE_NAMES_LOCAL_TRANSITION_27D) == 27
    assert OBS_DIM_LOCAL_TRANSITION_27D == 27
    assert len(set(FEATURE_NAMES_LOCAL_TRANSITION_27D)) == 27

    # Verify deprecation of all 11 features
    for removed in REMOVED_FEATURES_FROM_35D:
        assert removed not in FEATURE_NAMES_LOCAL_TRANSITION_27D

    # Verify exact 3 new features
    for new_feat in NEW_ORIENTATION_FEATURES_27D:
        assert new_feat in FEATURE_NAMES_LOCAL_TRANSITION_27D


def test_feature_bounds_and_finite():
    """2. Bounds and finite checks under extreme / corrupted inputs."""
    assert len(FEATURE_BOUNDS_LOCAL_TRANSITION_27D) == 27

    corrupted_obs = {
        "local_time_s": float("nan"),
        "sensors": {
            "FrontCamera": None,
            "DepthSensor": [float("inf")],
            "DVLSensor": [float("-inf"), float("nan"), 1e6],
            "IMUSensor": [float("nan"), [0.0, 0.0, float("inf")]],
        },
        "beacons": [
            {
                "beacon_id": "B01",
                "bearing_deg": float("nan"),
                "elevation_deg": float("inf"),
                "range_m": -50.0,
            }
        ],
    }
    context = LocalTransition27dLearningContext(
        expected_beacon_id="B01",
        vision_center_x_rate_s=float("nan"),
        vision_center_y_rate_s=float("inf"),
        vision_rate_valid=True,
        beacon_bearing_rate_deg_s=float("-inf"),
        beacon_rate_valid=True,
        gate_orientation_present=True,
        gate_yaw_deg=float("nan"),
    )

    encoded = encode_observation_local_transition_27d(corrupted_obs, context)
    assert encoded.shape == (27,)
    assert encoded.dtype == np.float32
    assert np.isfinite(encoded).all()

    for idx, (low, high) in enumerate(FEATURE_BOUNDS_LOCAL_TRANSITION_27D):
        val = encoded[idx]
        assert low <= val <= high, f"Feature {idx} ({FEATURE_NAMES_LOCAL_TRANSITION_27D[idx]}): {val} outside [{low}, {high}]"


def test_missing_data_rules():
    """3. Verify missing-data rules."""
    obs_empty = {"sensors": {}, "beacons": []}

    # Case A: Visual absence
    ctx_no_vis = LocalTransition27dLearningContext(
        visual_target=None,
        use_tracked_visual_target=True,
    )
    enc = encode_observation_local_transition_27d(obs_empty, ctx_no_vis)
    # Index 5 (vision_center_x), 6 (vision_center_y), 7 (vision_area_fraction)
    assert enc[5] == 0.0
    assert enc[6] == 0.0
    assert enc[7] == 0.0

    # Case B: Rate non valido / non calcolabile -> rate = 0
    ctx_invalid_rates = LocalTransition27dLearningContext(
        vision_center_x_rate_s=10.0,
        vision_center_y_rate_s=-10.0,
        vision_rate_valid=False,
        beacon_bearing_rate_deg_s=45.0,
        beacon_rate_valid=False,
    )
    enc_rates = encode_observation_local_transition_27d(obs_empty, ctx_invalid_rates)
    # Index 18 (vision_center_x_rate), 19 (vision_center_y_rate)
    assert enc_rates[18] == 0.0
    assert enc_rates[19] == 0.0
    # Index 20 (beacon_bearing_rate), 21 (elevation_rate), 22 (range_rate)
    assert enc_rates[20] == 0.0
    assert enc_rates[21] == 0.0
    assert enc_rates[22] == 0.0

    # Case C: Orientation unavailable -> present=0, sin=0, cos=0
    ctx_no_orient = LocalTransition27dLearningContext(
        gate_orientation_present=False,
        gate_yaw_deg=45.0,  # Should be masked to 0
    )
    enc_no_orient = encode_observation_local_transition_27d(obs_empty, ctx_no_orient)
    # Index 24, 25, 26
    assert enc_no_orient[24] == 0.0
    assert enc_no_orient[25] == 0.0
    assert enc_no_orient[26] == 0.0

    # Case D: Frontal valid gate -> present=1, sin≈0, cos≈1
    ctx_frontal = LocalTransition27dLearningContext(
        gate_orientation_present=True,
        gate_yaw_deg=0.0,
    )
    enc_frontal = encode_observation_local_transition_27d(obs_empty, ctx_frontal)
    assert enc_frontal[24] == 1.0
    assert abs(enc_frontal[25] - 0.0) < 1e-6
    assert abs(enc_frontal[26] - 1.0) < 1e-6

    # Case E: DVL present/absent
    assert enc[12] == 0.0  # dvl_present without DVL
    obs_dvl = {"sensors": {"DVLSensor": [0.5, -0.2, 0.1]}}
    enc_dvl = encode_observation_local_transition_27d(obs_dvl, ctx_no_vis)
    assert enc_dvl[12] == 1.0  # dvl_present with DVL
    assert abs(enc_dvl[9] - (0.5 / 1.5)) < 1e-4


def test_onboard_only_gt_isolation():
    """4. Injected GT and referee keys must not affect encoded vector."""
    clean_obs = {
        "local_time_s": 12.5,
        "sensors": {
            "DepthSensor": [-4.2],
            "DVLSensor": [0.3, 0.0, -0.1],
            "IMUSensor": [0.0, [0.0, 0.0, 0.05]],
        },
        "beacons": [
            {"beacon_id": "B01", "bearing_deg": 10.0, "elevation_deg": -5.0, "range_m": 8.0}
        ],
    }
    context = LocalTransition27dLearningContext(
        expected_beacon_id="B01",
        gate_orientation_present=True,
        gate_yaw_deg=15.0,
    )

    encoded_clean = encode_observation_local_transition_27d(clean_obs, context)

    # Injected observation with simulator / referee state
    polluted_obs = dict(clean_obs)
    polluted_obs["debug_ground_truth"] = {"position": [100.0, 200.0, -5.0]}
    polluted_obs["referee"] = {"status": "ACTIVE", "valid_gate_crossings": 5}
    polluted_obs["world_pose"] = [1.0, 2.0, 3.0]
    polluted_obs["gate_coordinates"] = [[1, 2, 3], [4, 5, 6]]
    polluted_obs["track_id"] = "horseshoe_bay"

    encoded_polluted = encode_observation_local_transition_27d(polluted_obs, context)
    assert np.array_equal(encoded_clean, encoded_polluted)


def test_fog_preflight_on_three_circuits():
    """5. Verify approved fog on the three official tracks."""
    for track in tf.OFFICIAL_TRACKS:
        path = tf.track_path(track)
        report = assert_approved_water_fog(path)
        assert report["valid"] is True
        assert report["water_fog"] == APPROVED_WATER_FOG
        assert report["water_fog"]["density"] == 5.0
        assert report["water_fog"]["start_distance_m"] == 1.0
        assert report["water_fog"]["color_rgb"] == [0.4, 0.6, 1.0]
        assert report["water_fog"]["enabled"] is True


def test_fog_preflight_on_fragments():
    """6. Verify approved fog is applied automatically to fragments."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for track in tf.OFFICIAL_TRACKS:
            frags = tf.enumerate_fragments(track, lengths=(3,))
            assert len(frags) > 0
            frag = frags[0]
            out_file = tmp_path / f"{frag.name}.json"
            tf.materialize_fragment(frag, out_file)
            report = assert_approved_water_fog(out_file)
            assert report["valid"] is True
            assert report["water_fog"] == APPROVED_WATER_FOG


def test_transfer_35_to_27_load_and_verification():
    """7. Semantic warm start 35-D -> 27-D verification."""
    parent_path = Path("results/rl/gen2/best/best_completion_policy.zip")
    assert parent_path.exists()
    hash_before = sha256_file(parent_path)
    assert hash_before == "2546ab2fc8257b674501302b3da72486f314cf26664986fe5544c90b82ba9f89"

    child = transfer_parent_35d_to_27d(parent_path)
    assert child.observation_space.shape == (27,)

    report = verify_transfer_27d_tensors(parent_path, child)
    assert report["verified"] is True
    assert report["mismatches"] == []
    assert report["retained_features_mapped"] == 24
    assert report["new_features_neutral"] == 3
    assert report["removed_features_count"] == 11
    assert report["all_parameters_trainable"] is True
    assert report["parent_sha256"] == hash_before

    # Verify parent was not altered
    hash_after = sha256_file(parent_path)
    assert hash_after == hash_before


def test_recurrent_inference_smoke():
    """8. Test recurrent controller inference on 27-D model."""
    parent_path = "results/rl/gen2/best/best_completion_policy.zip"
    model = transfer_parent_35d_to_27d(parent_path)

    controller = Gen2RecurrentController(model, deterministic=True)
    obs = np.zeros(27, dtype=np.float32)
    obs[0] = 1.0  # beacon present
    obs[2] = 1.0  # beacon bearing cos
    obs[4] = 0.5  # beacon range norm
    obs[8] = 0.4  # depth norm
    obs[12] = 1.0 # dvl present

    action1 = controller.act(obs, first_step=True)
    assert action1.shape == (4,)
    assert np.all(np.abs(action1) <= 1.0)

    action2 = controller.act(obs, first_step=False)
    assert action2.shape == (4,)
    assert np.all(np.abs(action2) <= 1.0)


def test_monitor_snapshot_rendering_smoke():
    """9. Smoke test for monitor overlay rendering and snapshot files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)

        # Synthetic 640x480 frame
        img = np.full((480, 640, 3), 120, dtype=np.uint8)
        rendered = render_monitor_overlay(
            img,
            track="horseshoe_bay",
            step=150,
            episode=50,
            target_gate="B02",
            detected_gate="YES",
            center=(0.1, -0.05),
            corners_norm=[(-0.2, -0.2), (0.2, -0.2), (0.2, 0.2), (-0.2, 0.2)],
            yaw_deg=-8.5,
            orientation_class="RIGHT",
            orientation_present=True,
            beacon_bearing_deg=5.2,
            beacon_range_m=6.8,
            fog_config=APPROVED_WATER_FOG,
            action=[0.6, 0.0, 0.1, -0.2],
        )

        assert rendered.shape == (480, 640, 3)
        assert rendered.dtype == np.uint8

        # Test writing snapshot image and json
        import cv2
        img_path = out_dir / "latest_snapshot.png"
        cv2.imwrite(str(img_path), rendered)
        assert img_path.exists()
        assert img_path.stat().st_size > 1000

        json_meta = {
            "step": 150,
            "track": "horseshoe_bay",
            "episode": 50,
            "gate": "B02",
            "fog_config": dict(APPROVED_WATER_FOG),
            "observation_validity": {"beacon_present": True, "vision_present": True},
            "yaw": -8.5,
            "orientation_availability": True,
            "gate_progress": {"gate_crossings": 1, "expected_gate_id": "G02"},
            "safety_events": {"collision": False, "out_of_bounds": False},
        }
        json_path = out_dir / "latest_snapshot.json"
        json_path.write_text(json.dumps(json_meta, indent=2), encoding="utf-8")
        assert json_path.exists()
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert loaded["gate"] == "B02"
        assert loaded["yaw"] == -8.5


def test_dataset_rollout_micro_smoke():
    """10. Micro-smoke (2 steps) of expert dataset collection with 27-D contract."""
    track_path = tf.track_path("horseshoe_bay")
    report = collect_expert_27d_rollout(
        track_path,
        seed=0,
        adapter="fallback",
        allow_fallback=True,
        max_steps=2,
    )
    assert report["fog_verified"] is True
    assert report["steps"] == 2
    assert report["observations_shape"] == (2, 27)
    assert report["actions_shape"] == (2, 4)
    assert report["contract"] == OBS_ENCODING_VERSION_LOCAL_TRANSITION_27D
