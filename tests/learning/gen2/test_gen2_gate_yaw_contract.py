"""Minimal, onboard-only gate-yaw observation and exact warm-start tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from marine_race_arena.learning.config_local_transition import (
    FEATURE_NAMES_LOCAL_TRANSITION,
    LocalTransitionLearningContext,
)
from marine_race_arena.learning.config_local_transition_gate_yaw import (
    FEATURE_NAMES_LOCAL_TRANSITION_GATE_YAW,
    OBS_DIM_LOCAL_TRANSITION_GATE_YAW,
    OBS_ENCODING_VERSION_LOCAL_TRANSITION_GATE_YAW,
    LocalTransitionGateYawLearningContext,
)
from marine_race_arena.learning.observation_encoder_local_transition import (
    encode_observation_local_transition,
)
from marine_race_arena.learning.observation_encoder_local_transition_gate_yaw import (
    encode_observation_local_transition_gate_yaw,
)


def _onboard_observation():
    return {
        "local_time_s": 1.2,
        "beacons": [
            {
                "id": "B01",
                "bearing_deg": 12.0,
                "elevation_deg": -4.0,
                "range_m": 7.0,
                "signal_strength": 0.8,
                "timestamp_s": 1.2,
            }
        ],
        "sensors": {
            "DepthSensor": 3.0,
            "DVLSensor": [0.2, -0.1, 0.0],
            "IMUSensor": [0.0, 0.0, 0.1],
        },
    }


def test_contract_is_exactly_35_plus_the_three_requested_features():
    assert OBS_ENCODING_VERSION_LOCAL_TRANSITION_GATE_YAW == (
        "onboard_local_transition_gate_yaw_v2"
    )
    assert OBS_DIM_LOCAL_TRANSITION_GATE_YAW == 38
    assert FEATURE_NAMES_LOCAL_TRANSITION_GATE_YAW[:35] == FEATURE_NAMES_LOCAL_TRANSITION
    assert FEATURE_NAMES_LOCAL_TRANSITION_GATE_YAW[35:] == (
        "gate_orientation_present",
        "gate_yaw_sin",
        "gate_yaw_cos",
    )


def test_v2_prefix_is_byte_identical_to_v1():
    raw = _onboard_observation()
    base = LocalTransitionLearningContext(expected_beacon_id="B01")
    extended = LocalTransitionGateYawLearningContext(
        expected_beacon_id="B01",
        gate_orientation_present=True,
        gate_yaw_deg=30.0,
    )
    old = encode_observation_local_transition(raw, base)
    new = encode_observation_local_transition_gate_yaw(raw, extended)
    assert np.array_equal(new[:35], old)
    assert new[35] == 1.0
    assert new[36] == pytest.approx(0.5)
    assert new[37] == pytest.approx(np.sqrt(3.0) / 2.0)


@pytest.mark.parametrize("yaw", [None, float("nan"), float("inf")])
def test_unavailable_or_invalid_orientation_is_all_zero(yaw):
    vector = encode_observation_local_transition_gate_yaw(
        _onboard_observation(),
        LocalTransitionGateYawLearningContext(
            gate_orientation_present=True, gate_yaw_deg=yaw
        ),
    )
    assert np.array_equal(vector[35:], np.zeros(3, np.float32))
    assert np.isfinite(vector).all()


def test_privileged_keys_cannot_change_any_of_the_38_features():
    raw = _onboard_observation()
    context = LocalTransitionGateYawLearningContext(
        expected_beacon_id="B01",
        gate_orientation_present=True,
        gate_yaw_deg=-24.0,
    )
    expected = encode_observation_local_transition_gate_yaw(raw, context)
    poisoned = copy.deepcopy(raw)
    poisoned.update(
        {
            "world_position": [999.0, -999.0, 42.0],
            "gate_coordinates": [[1.0, 2.0, 3.0]],
            "map": {"future_gate": "G22"},
            "referee_state": {"expected_gate": "G22", "crossings": 21},
            "simulator_ground_truth": {"gate_yaw_deg": 89.0},
        }
    )
    actual = encode_observation_local_transition_gate_yaw(poisoned, context)
    assert np.array_equal(actual, expected)


def test_every_official_track_and_materialized_fragment_keeps_exact_fog(tmp_path):
    from marine_race_arena.learning.gen2 import track_fragments as tf
    from marine_race_arena.learning.gen2.fog_contract import (
        APPROVED_WATER_FOG,
        assert_approved_water_fog,
    )

    for track in tf.OFFICIAL_TRACKS:
        source = tf.track_path(track)
        assert assert_approved_water_fog(source)["water_fog"] == APPROVED_WATER_FOG
        fragment = tf.enumerate_fragments(track, lengths=(2,))[0]
        made = tf.materialize_fragment(fragment, tmp_path / f"{track}.json")
        assert assert_approved_water_fog(made)["water_fog"] == APPROVED_WATER_FOG


def test_fog_contract_fails_fast_on_any_mismatch(tmp_path):
    from marine_race_arena.learning.gen2.fog_contract import assert_approved_water_fog

    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "water_fog": {
                    "enabled": True,
                    "density": 4.9,
                    "start_distance_m": 1.0,
                    "color_rgb": [0.4, 0.6, 1.0],
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fog mismatch"):
        assert_approved_water_fog(path)


def test_gym_environment_accepts_only_the_explicit_38d_version():
    pytest.importorskip("gymnasium")
    from marine_race_arena.learning.gym_env import MarineRaceGymEnv

    env = MarineRaceGymEnv(
        "marine_race_arena/tracks/marine_race_horseshoe_bay.json",
        adapter="fallback",
        observation_encoding_version=OBS_ENCODING_VERSION_LOCAL_TRANSITION_GATE_YAW,
        max_steps=2,
    )
    try:
        observation, _ = env.reset(seed=1)
        assert observation.shape == (38,)
        assert np.isfinite(observation).all()
    finally:
        env.close()


def test_exact_recurrent_transfer_and_perturbation_guard(tmp_path):
    pytest.importorskip("sb3_contrib")
    import torch as th

    from marine_race_arena.learning.gen2.gate_yaw_transfer import (
        transfer_parent_to_gate_yaw,
        verify_expanded_recurrent_parity,
        verify_transfer_tensors,
    )
    from marine_race_arena.learning.gen2.recurrent_policy import (
        build_gen2_policy_for_training,
    )

    parent = build_gen2_policy_for_training(seed=7)
    parent_path = tmp_path / "parent.zip"
    parent.save(parent_path)
    child = transfer_parent_to_gate_yaw(parent_path, seed=9)
    tensors = verify_transfer_tensors(parent_path, child)
    parity = verify_expanded_recurrent_parity(parent_path, child)
    assert tensors["exact"], tensors
    assert tensors["new_input_columns_zero"]
    assert parity["bit_exact"], parity
    assert parity["max_abs_deviation"] == 0.0

    # Prove the parity instrument is live rather than tautological.
    with th.no_grad():
        child.policy.action_net.bias.add_(0.05)
    perturbed = verify_expanded_recurrent_parity(parent_path, child)
    assert not perturbed["parity"]
    assert perturbed["max_abs_deviation"] > 0.0


def test_smoke_sources_are_three_full_tracks_plus_three_exact_fragments(tmp_path):
    from marine_race_arena.learning.gen2.train_gate_yaw_ppo import (
        materialize_training_sources,
    )

    sources, audit = materialize_training_sources(tmp_path)
    assert len(sources) == 6
    assert {source.track for source in sources} == {
        "horseshoe_bay",
        "vertical_serpent",
        "mixed_endurance",
    }
    assert sum(source.kind == "full_circuit" for source in sources) == 3
    assert sum(source.kind == "exact_fragment" for source in sources) == 3
    assert audit["fog"]["all_valid"]
    assert all(row["identical"] for row in audit["geometry"])


def test_long_ppo_requires_an_explicit_second_authorization(tmp_path):
    from marine_race_arena.learning.gen2.train_gate_yaw_ppo import run

    with pytest.raises(RuntimeError, match="--authorize-long"):
        run(
            parent_path="unused.zip",
            preflight_path="unused.json",
            run_dir=tmp_path / "must_not_start",
            total_timesteps=10_001,
        )
    assert not (tmp_path / "must_not_start").exists()
