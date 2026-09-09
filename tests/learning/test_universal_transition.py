"""Contract, reward, transfer, worker, and resume tests for local transition PPO."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# The RL stack (gymnasium/torch/SB3) lives in requirements-rl.txt and is not
# installed in the benchmark environment; skip rather than fail collection.
pytest.importorskip("gymnasium")

from marine_race_arena.controllers.vision import VisionTarget
from marine_race_arena.learning.config_local_transition import (
    FEATURE_BOUNDS_LOCAL_TRANSITION,
    FEATURE_NAMES_LOCAL_TRANSITION,
    LOCAL_TO_SEQUENCE_FEATURE_MAP,
    OBS_DIM_LOCAL_TRANSITION,
    OBS_ENCODING_VERSION_LOCAL_TRANSITION,
)
from marine_race_arena.learning.config_sequence import (
    FEATURE_NAMES_SEQUENCE,
    OBS_DIM_SEQUENCE,
)
from marine_race_arena.learning.longrun_checkpoint import (
    atomic_save_checkpoint,
    latest_valid_checkpoint,
    load_checkpoint_state,
)
from marine_race_arena.learning.observation_encoder_local_transition import (
    encode_observation_local_transition,
)
from marine_race_arena.learning.reward_local_transition import (
    LocalTransitionTrainingReward,
)
from marine_race_arena.learning.tracker_context_local_transition import (
    OnboardLocalTransitionContextTracker,
    _wrapped_angle_delta_deg,
)
from marine_race_arena.learning.transition_curriculum import (
    FULL_SEQUENCE_LENGTHS,
    TransitionCurriculumController,
    TransitionGeometrySampler,
)
from marine_race_arena.learning.transition_env import UniversalTransitionEnv
from marine_race_arena.learning.transition_evaluation import (
    aggregate_transition_benchmark,
)
from marine_race_arena.learning.transition_policy import (
    build_local_transition_ppo,
    selective_warm_start_from_sequence,
)


EXPECTED_FEATURES = (
    "beacon_present", "beacon_bearing_sin", "beacon_bearing_cos",
    "beacon_elevation_norm", "beacon_range_norm", "beacon_signal_strength",
    "beacon_age_norm", "vision_present", "vision_center_x", "vision_center_y",
    "vision_area_fraction", "vision_confidence", "depth_norm", "depth_present",
    "depth_error_norm", "depth_reference_present", "dvl_surge_norm",
    "dvl_sway_norm", "dvl_heave_norm", "dvl_present", "imu_yaw_rate_norm",
    "imu_present", "previous_surge", "previous_sway", "previous_heave",
    "previous_yaw", "vision_center_x_rate", "vision_center_y_rate",
    "vision_rate_valid", "beacon_bearing_rate", "beacon_elevation_rate",
    "beacon_range_rate", "beacon_rate_valid", "target_changed_recently",
    "time_since_target_change_norm",
)


def _packet(beacon_id="B01", bearing=0.0, elevation=0.0, range_m=5.0, time_s=0.0):
    return {
        "beacon_id": beacon_id,
        "bearing_deg": bearing,
        "elevation_deg": elevation,
        "range_m": range_m,
        "signal_strength": 0.8,
        "received_at_s": time_s,
    }


def _observation(time_s=0.0, *, beacons=None, camera=None, privileged=False):
    sensors = {
        "DepthSensor": [-4.0],
        "DVLSensor": [0.2, -0.1, 0.05],
        "IMUSensor": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]],
    }
    if camera is not None:
        sensors["FrontCamera"] = camera
    value = {
        "local_time_s": time_s,
        "sensors": sensors,
        "beacons": list(beacons or []),
    }
    if privileged:
        value.update({
            "ground_truth": {"pose": [99.0] * 6},
            "referee": {"gate_index": 21, "remaining": 1},
            "own_position": [99.0, 99.0, 99.0],
            "future_gates": [[99.0, 99.0, 99.0]],
        })
    return value


class _FakeTracker:
    def __init__(self):
        self.expected_beacon_id = "B01"
        self._latest_visual_target = None
        self.finished = False

    def update(self, *, beacons, camera_image, **kwargs):
        if beacons and beacons[0].get("beacon_id") == "B02":
            self.expected_beacon_id = "B02"
        self._latest_visual_target = (
            camera_image if isinstance(camera_image, VisionTarget) else None
        )


def _source_with_fake_tracker():
    source = OnboardLocalTransitionContextTracker(total_beacons=22)
    source.reset(_observation())
    source._tracker = _FakeTracker()
    return source


def test_exact_35_feature_order_and_bounds():
    assert OBS_ENCODING_VERSION_LOCAL_TRANSITION == "onboard_local_transition_v1"
    assert FEATURE_NAMES_LOCAL_TRANSITION == EXPECTED_FEATURES
    assert len(FEATURE_NAMES_LOCAL_TRANSITION) == len(FEATURE_BOUNDS_LOCAL_TRANSITION) == 35
    vector = encode_observation_local_transition(
        _observation(beacons=[_packet()]),
    )
    assert vector.shape == (35,)
    assert vector.dtype == np.float32
    assert np.all(np.isfinite(vector))
    for value, (low, high) in zip(vector, FEATURE_BOUNDS_LOCAL_TRANSITION):
        assert low <= value <= high


def test_shortcut_and_privileged_features_are_absent_and_ignored():
    forbidden_fragments = (
        "gate_index", "remaining", "total_gate", "sequence_progress",
        "lap", "phase_", "pose", "future", "referee",
    )
    assert not any(
        fragment in name
        for name in FEATURE_NAMES_LOCAL_TRANSITION
        for fragment in forbidden_fragments
    )
    clean = _observation(beacons=[_packet()])
    poisoned = _observation(beacons=[_packet()], privileged=True)
    assert np.array_equal(
        encode_observation_local_transition(clean),
        encode_observation_local_transition(poisoned),
    )


def test_beacon_angle_wrapping_and_rate_validity():
    assert _wrapped_angle_delta_deg(-179.0, 179.0) == pytest.approx(2.0)
    assert _wrapped_angle_delta_deg(179.0, -179.0) == pytest.approx(-2.0)
    source = _source_with_fake_tracker()
    first = source.context(
        _observation(0.0, beacons=[_packet(bearing=179.0)]), dt=0.1
    )
    wrapped = source.context(
        _observation(0.1, beacons=[_packet(bearing=-179.0, time_s=0.1)]), dt=0.1
    )
    assert not first.beacon_rate_valid
    assert wrapped.beacon_rate_valid
    assert wrapped.beacon_bearing_rate_deg_s == pytest.approx(20.0)


def test_derivative_validity_resets_on_missing_packets_reacquisition_and_target_change():
    source = _source_with_fake_tracker()
    source.context(_observation(0.0, beacons=[_packet()]), dt=0.1)
    valid = source.context(
        _observation(0.1, beacons=[_packet(range_m=4.9, time_s=0.1)]), dt=0.1
    )
    missing = source.context(_observation(0.2), dt=0.1)
    reacquired = source.context(
        _observation(0.3, beacons=[_packet(range_m=4.8, time_s=0.3)]), dt=0.1
    )
    switched = source.context(
        _observation(0.4, beacons=[_packet("B02", range_m=6.0, time_s=0.4)]),
        dt=0.1,
    )
    assert valid.beacon_rate_valid
    assert not missing.beacon_rate_valid
    assert not reacquired.beacon_rate_valid
    assert not switched.beacon_rate_valid
    assert switched.target_changed_recently
    assert switched.time_since_target_change_s == 0.0


def test_visual_derivative_validity_resets_after_loss_and_reacquisition():
    source = _source_with_fake_tracker()
    first_target = VisionTarget(0.1, -0.1, 0.9, 0.1, 0.2, 0.2)
    second_target = VisionTarget(0.2, -0.3, 0.9, 0.1, 0.2, 0.2)
    source.context(_observation(0.0, camera=first_target), dt=0.1)
    valid = source.context(_observation(0.1, camera=second_target), dt=0.1)
    lost = source.context(_observation(0.2), dt=0.1)
    reacquired = source.context(_observation(0.3, camera=first_target), dt=0.1)
    assert valid.vision_rate_valid
    assert valid.vision_center_x_rate_s == pytest.approx(1.0)
    assert valid.vision_center_y_rate_s == pytest.approx(-2.0)
    assert not lost.vision_rate_valid
    assert not reacquired.vision_rate_valid


def test_train_and_inference_dt_paths_encode_identically():
    train = _source_with_fake_tracker()
    inference = _source_with_fake_tracker()
    observations = [
        _observation(0.0, beacons=[_packet(bearing=10.0)]),
        _observation(0.1, beacons=[_packet(bearing=8.0, range_m=4.8, time_s=0.1)]),
    ]
    train_vectors = []
    inference_vectors = []
    for obs in observations:
        train_context = train.context(obs, dt=0.1, prev_action=[0.1, 0.2, 0.3, 0.4])
        inference_context = inference.context(obs, dt=None, prev_action=[0.1, 0.2, 0.3, 0.4])
        train_vectors.append(encode_observation_local_transition(obs, train_context))
        inference_vectors.append(encode_observation_local_transition(obs, inference_context))
    # First inference observation has no preceding clock delta; both masks are
    # invalid there. The second observation must be exactly equal.
    assert np.array_equal(train_vectors[1], inference_vectors[1])


import gymnasium as gym


class _TinyEnv(gym.Env):
    def __init__(self, dim):
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, True, {}


def test_selective_warm_start_maps_only_retained_columns_and_keeps_all_trainable():
    from marine_race_arena.learning.sequence_policy import build_sequence_ppo

    source = build_sequence_ppo(
        _TinyEnv(OBS_DIM_SEQUENCE), architecture="feedforward_ppo", seed=1,
        learning_rate=1e-5, hidden_sizes=(16, 16), n_steps=8, batch_size=8,
        n_epochs=1,
    )
    target = build_local_transition_ppo(
        _TinyEnv(OBS_DIM_LOCAL_TRANSITION), seed=1, learning_rate=1e-5,
        hidden_sizes=(16, 16), n_steps=8, batch_size=8, n_epochs=1,
    )
    source_state = source.policy.state_dict()
    for layer in (
        "mlp_extractor.policy_net.0.weight",
        "mlp_extractor.value_net.0.weight",
    ):
        value = source_state[layer]
        for column in range(value.shape[1]):
            value[:, column] = float(column + 1)
    source.policy.load_state_dict(source_state)
    report = selective_warm_start_from_sequence(source, target)
    target_state = target.policy.state_dict()
    old_index = {name: index for index, name in enumerate(FEATURE_NAMES_SEQUENCE)}
    new_index = {name: index for index, name in enumerate(FEATURE_NAMES_LOCAL_TRANSITION)}
    for local, sequence in LOCAL_TO_SEQUENCE_FEATURE_MAP.items():
        actual = target_state["mlp_extractor.policy_net.0.weight"][:, new_index[local]]
        assert np.allclose(actual.numpy(), old_index[sequence] + 1)
    for name in report["new_zero_initialized_features"]:
        for layer in (
            "mlp_extractor.policy_net.0.weight",
            "mlp_extractor.value_net.0.weight",
        ):
            assert np.allclose(target_state[layer][:, new_index[name]].numpy(), 0.0)
    assert all(parameter.requires_grad for parameter in target.policy.parameters())
    assert report["optimizer_state_copied"] is False


def test_potential_delta_is_non_farmable_and_completion_is_length_independent():
    reward = LocalTransitionTrainingReward()
    forward = reward._bounded_delta(5.0, 4.0, 2.0)
    backward = reward._bounded_delta(4.0, 5.0, 2.0)
    assert forward + backward == pytest.approx(0.0)
    assert reward.config.completion_bonus == pytest.approx(2.0)
    assert not hasattr(reward.config, "completion_bonus_per_gate")


def test_geometry_difficulty_never_controls_sequence_length():
    for difficulty in ("G1", "G3", "G6"):
        sampler = TransitionGeometrySampler(
            seed=1234, difficulty=difficulty, transition_focus_fraction=0.0
        )
        lengths = {sampler.sample(force_episode_type="full_sequence").gate_count for _ in range(100)}
        assert lengths.issubset(set(FULL_SEQUENCE_LENGTHS))
        assert len(lengths) >= 4


def test_worker_paths_and_seeds_are_isolated(tmp_path):
    first = UniversalTransitionEnv(
        run_dir=tmp_path, worker_id=0, sampler_seed=100, difficulty="G1",
        adapter="fallback", allow_fallback=True, max_episode_steps=5,
    )
    second = UniversalTransitionEnv(
        run_dir=tmp_path, worker_id=1, sampler_seed=200, difficulty="G1",
        adapter="fallback", allow_fallback=True, max_episode_steps=5,
    )
    try:
        assert first.worker_identity()["holoocean_uuid"] is None
        assert second.worker_identity()["holoocean_uuid"] is None
        first.reset(seed=1)
        second.reset(seed=1)
        one = first.worker_identity()
        two = second.worker_identity()
        assert one["worker_directory"] != two["worker_directory"]
        assert one["active_track"] != two["active_track"]
        assert one["worker_sampler_seed"] == 100
        assert two["worker_sampler_seed"] == 200
        assert first.sampler.state_dict()["rng_state"] != second.sampler.state_dict()["rng_state"]
    finally:
        first.close()
        second.close()


@pytest.mark.skipif(
    os.environ.get("RUN_HOLOOCEAN_TESTS") != "1",
    reason="real HoloOcean concurrency test is opt-in",
)
def test_two_simultaneous_holoocean_instances_have_distinct_uuids(tmp_path):
    first = UniversalTransitionEnv(
        run_dir=tmp_path, worker_id=0, sampler_seed=100, difficulty="G1",
        adapter="holoocean", allow_fallback=False, max_episode_steps=5,
    )
    second = UniversalTransitionEnv(
        run_dir=tmp_path, worker_id=1, sampler_seed=200, difficulty="G1",
        adapter="holoocean", allow_fallback=False, max_episode_steps=5,
    )
    try:
        first.reset(seed=1)
        second.reset(seed=2)
        assert first.worker_identity()["holoocean_uuid"]
        assert second.worker_identity()["holoocean_uuid"]
        assert first.worker_identity()["holoocean_uuid"] != second.worker_identity()["holoocean_uuid"]
    finally:
        first.close()
        second.close()


class _CheckpointSpace:
    shape = (OBS_DIM_LOCAL_TRANSITION,)


class _FakeCheckpointModel:
    num_timesteps = 4096
    _n_updates = 6
    _current_progress_remaining = 0.5
    observation_space = _CheckpointSpace()
    longrun_observation_version = OBS_ENCODING_VERSION_LOCAL_TRANSITION
    longrun_architecture = "feedforward_ppo"
    longrun_policy_mode = "feedforward"
    longrun_frame_stack = 1
    policy = None

    def save(self, path):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("policy.txt", "two worker optimizer state")


def test_atomic_resume_preserves_multiple_worker_states(tmp_path):
    controller = TransitionCurriculumController()
    workers = [
        {"schema_version": "universal_transition_worker_v1", "worker_id": index,
         "sampler_seed": 100 + index, "episode_counter": 5,
         "sampler": {"rng": f"worker-{index}"}}
        for index in range(2)
    ]
    atomic_save_checkpoint(
        _FakeCheckpointModel(), tmp_path, total_timesteps=4096,
        config_contract_hash="d" * 64,
        curriculum_state=controller.state_dict(workers),
        evaluation_state={"history": [], "last": None, "best": None},
        status="unverified", reason="vector_resume_test",
        extra_state={"architecture": "feedforward_ppo", "n_envs": 2,
                     "checkpoint_aliases": {}, "checkpoint_selection_state": {}},
    )
    checkpoint = latest_valid_checkpoint(
        tmp_path, expected_contract_hash="d" * 64, safe_only=False
    )
    assert checkpoint is not None
    state = load_checkpoint_state(checkpoint)
    restored = TransitionCurriculumController()
    restored_workers = restored.load_state_dict(state["curriculum"])
    assert [row["worker_id"] for row in restored_workers] == [0, 1]
    assert state["model_training"]["num_timesteps"] == 4096


def test_transition_benchmark_aggregation_and_sampler_are_deterministic():
    first = TransitionGeometrySampler(seed=987, difficulty="G6")
    second = TransitionGeometrySampler(seed=987, difficulty="G6")
    assert [first.sample() for _ in range(20)] == [second.sample() for _ in range(20)]
    rows = [{
        "episode_type": "transition_focus",
        "universal_transition_success": True,
        "first_gate_crossed": True,
        "correct_target_switch": True,
        "new_target_acquired_and_aligned": True,
        "new_target_range_decreasing": True,
        "previous_gate_return": False,
        "missed_gate_dnf": 0,
        "collision_events": 0,
        "out_of_bounds_events": 0,
        "wrong_direction_events": 0,
        "acquisition_timeout": False,
        "acquisition_time_s": 0.4,
        "action_jerk": 0.1,
        "action_source": "ppo_policy",
    }]
    assert aggregate_transition_benchmark(rows) == aggregate_transition_benchmark(list(rows))
    assert aggregate_transition_benchmark(rows)["universal_transition_success_rate"] == 1.0


def test_checkpoint_benchmark_parallel_cases_are_atomic_and_resumable(tmp_path):
    from marine_race_arena.learning.transition_evaluation import (
        evaluate_checkpoint_universal_transition_benchmark,
    )

    model = build_local_transition_ppo(
        _TinyEnv(OBS_DIM_LOCAL_TRANSITION), seed=9, learning_rate=1e-5,
        hidden_sizes=(16, 16), n_steps=8, batch_size=8, n_epochs=1,
    )
    checkpoint = tmp_path / "model.zip"
    model.save(checkpoint)
    output = tmp_path / "benchmark"
    first = evaluate_checkpoint_universal_transition_benchmark(
        checkpoint,
        output_dir=output,
        seed=8123,
        difficulty="G1",
        transition_cases=2,
        full_cases_per_length=0,
        adapter="fallback",
        max_steps=2,
        parallel_workers=2,
    )
    assert first["metrics"]["n_eval"] == 2
    case_files = sorted((output / "transitions").glob("case_*/episode.json"))
    assert len(case_files) == 2

    # A complete rerun loads atomic per-case rows and must not rewrite them.
    mtimes = [path.stat().st_mtime_ns for path in case_files]
    second = evaluate_checkpoint_universal_transition_benchmark(
        checkpoint,
        output_dir=output,
        seed=8123,
        difficulty="G1",
        transition_cases=2,
        full_cases_per_length=0,
        adapter="fallback",
        max_steps=2,
        parallel_workers=2,
    )
    assert first["episodes"] == second["episodes"]
    assert mtimes == [path.stat().st_mtime_ns for path in case_files]


def test_track_atomic_write_retries_transient_windows_permission_error(
    monkeypatch, tmp_path
):
    from marine_race_arena.learning import sequence_curriculum

    target = tmp_path / "track.json"
    original = Path.replace
    attempts = 0

    def flaky_replace(self, destination):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("transient Windows sharing violation")
        return original(self, destination)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    sequence_curriculum._atomic_json(target, {"complete": True})
    assert attempts == 2
    assert json.loads(target.read_text(encoding="utf-8")) == {"complete": True}


def test_vectorized_config_counts_total_rollout_across_workers():
    from marine_race_arena.learning.train_ppo_transition import _load_config

    config = _load_config("configs/rl/ppo_universal_transition_ab_warm.json")
    assert config["n_envs"] == 2
    assert config["ppo"]["n_steps"] == 1024
    assert config["n_envs"] * config["ppo"]["n_steps"] == 2048


def test_training_workers_are_recreated_with_state_before_learning_resumes(
    monkeypatch, tmp_path
):
    import marine_race_arena.learning.train_ppo_transition as training

    calls = []

    class FakeVecEnv:
        num_envs = 2

        def env_method(self, name, *args, **kwargs):
            calls.append((name, args, kwargs))
            return [None, None]

        def reset(self):
            calls.append(("reset", (), {}))
            return np.zeros((2, OBS_DIM_LOCAL_TRANSITION), dtype=np.float32)

        def close(self):
            calls.append(("close", (), {}))

    replacement = FakeVecEnv()
    monkeypatch.setattr(training, "_make_vec_env", lambda *args: replacement)

    class FakeModel:
        num_timesteps = 8192
        _last_obs = "stale"

        def set_env(self, env):
            calls.append(("set_env", (env,), {}))

    model = FakeModel()
    curriculum = TransitionCurriculumController()
    config = {
        "n_envs": 2,
    }
    states = [
        {"worker_id": 0, "sampler_seed": 10},
        {"worker_id": 1, "sampler_seed": 20},
    ]

    actual = training._recreate_training_env(
        model, config, tmp_path, curriculum, states
    )

    assert actual is replacement
    load_calls = [call for call in calls if call[0] == "load_worker_state"]
    assert [call[2]["indices"] for call in load_calls] == [0, 1]
    assert calls.index(("reset", (), {})) < next(
        index for index, call in enumerate(calls) if call[0] == "set_env"
    )
    assert model._last_obs is None
    assert any(
        name == "set_total_environment_transitions" and args == (8192,)
        for name, args, _ in calls
    )


def test_target_boundary_checkpoint_resume_keeps_pending_evaluation():
    from marine_race_arena.learning.train_ppo_transition import (
        _next_evaluation_transition,
        _training_work_pending,
    )

    frequency = 40_960
    pending = {"history": [], "last": None, "best": None}
    next_evaluation = _next_evaluation_transition(pending, frequency)
    assert next_evaluation == frequency
    assert _training_work_pending(frequency, frequency, next_evaluation)

    completed = {"history": [{"timesteps": frequency}]}
    next_evaluation = _next_evaluation_transition(completed, frequency)
    assert next_evaluation == 2 * frequency
    assert not _training_work_pending(frequency, frequency, next_evaluation)


def test_ab_configs_differ_only_by_initialization_and_use_1000_unseen_cases():
    from marine_race_arena.learning.compare_transition_initializations import (
        validate_pair,
    )

    warm = json.loads(Path(
        "configs/rl/ppo_universal_transition_ab_warm.json"
    ).read_text(encoding="utf-8"))
    scratch = json.loads(Path(
        "configs/rl/ppo_universal_transition_ab_scratch.json"
    ).read_text(encoding="utf-8"))
    validate_pair(warm, scratch)
    assert warm["evaluation"]["dedicated_transition_cases"] >= 1000


def test_parallel_benchmark_summary_understands_15hz_dvl_cadence():
    from marine_race_arena.learning.benchmark_transition_parallelism import (
        _summarize,
    )

    cases = []
    for frames in (False, True):
        for workers, rate in ((1, 5.0), (2, 8.0)):
            for repeat in range(2):
                health = [{
                    "frames_checked": 101,
                    "missing_frames": {
                        "FrontCamera": 0, "DVLSensor": 50,
                        "IMUSensor": 0, "DepthSensor": 0,
                    },
                    "repeated_fingerprints": {
                        "FrontCamera": 0, "DVLSensor": 0,
                        "IMUSensor": 0, "DepthSensor": 0,
                    },
                } for _ in range(workers)]
                cases.append({
                    "ok": True,
                    "n_envs": workers,
                    "frames_per_sec": frames,
                    "environment_transitions_per_second": rate,
                    "rollout_wall_time_s": 10.0,
                    "engine_launch_failures": 0,
                    "observation_trace_sha256": f"{frames}-{workers}-{repeat}",
                    "sensor_health": health,
                })
    summary = _summarize(cases)
    assert summary["two_workers_meaningful"]
    assert summary["deterministic_case_generation_reproducible"]
    assert all(row["sensor_contract_valid"] for row in summary["groups"].values())
