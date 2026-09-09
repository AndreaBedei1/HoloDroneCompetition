"""Regression tests for PPO-only ordered-sequence training and resume."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from marine_race_arena.config.benchmark_tasks import BENCHMARK_TASK_CLEAN_GATE
from marine_race_arena.config.loader import load_track_config
from marine_race_arena.config.validation import validate_track_config
from marine_race_arena.learning.config_sequence import (
    FEATURE_NAMES_SEQUENCE,
    OBS_DIM_SEQUENCE,
    OBS_ENCODING_VERSION_SEQUENCE,
    SequenceLearningContext,
)
from marine_race_arena.learning.config_v3 import FEATURE_NAMES_V3, OBS_DIM_V3
from marine_race_arena.learning.longrun_checkpoint import (
    atomic_save_checkpoint,
    latest_valid_checkpoint,
    load_checkpoint_state,
    restore_model_training_state,
)
from marine_race_arena.learning.observation_encoder_sequence import encode_observation_sequence
from marine_race_arena.learning.observation_encoder_v3 import encode_observation_v3
from marine_race_arena.learning.seed_registry import (
    PPO_SEQUENCE_CHECKPOINT_SELECTION_SEEDS,
    PPO_SEQUENCE_CURRICULUM_EVAL_SEEDS,
    PPO_SEQUENCE_OFFICIAL_HOLDOUT_SEEDS,
    PPO_SEQUENCE_TRAINING_SEEDS,
    PPO_SEQUENCE_UNSEEN_HOLDOUT_SEEDS,
    assert_pairwise_disjoint,
)
from marine_race_arena.learning.sequence_curriculum import (
    SEQUENCE_STAGES,
    SequenceCurriculumSampler,
    generate_sequence_track,
)
from marine_race_arena.learning.sequence_evaluation import (
    OFFICIAL_TRACKS,
    aggregate_sequence_evaluation,
    checkpoint_rank_key,
    evaluate_episode,
)
from marine_race_arena.learning.tracker_context_sequence import OnboardSequenceContextTracker
from marine_race_arena.learning.sequence_policy import build_sequence_ppo
from marine_race_arena.learning.train_multigate_longrun import AbsoluteLearningRateSchedule


def _observation(beacon_id="B01"):
    return {
        "local_time_s": 0.1,
        "sensors": {
            "DepthSensor": [-4.0],
            "DVLSensor": [0.2, 0.0, 0.0],
            "IMUSensor": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        },
        "beacons": [{
            "beacon_id": beacon_id, "bearing_deg": 0.0, "elevation_deg": 0.0,
            "range_m": 5.0, "signal_strength": 1.0, "received_at_s": 0.1,
        }],
    }


def test_sequence_observation_preserves_v3_prefix_and_ignores_privileged_keys():
    context = SequenceLearningContext(
        expected_beacon_id="B01", total_beacons=5, current_gate_index=2,
        remaining_gate_count=3, sequence_progress=0.4,
        previous_gate_cleared=True, new_target_acquired_since_crossing=True,
        time_since_confirmed_crossing_steps=7,
    )
    clean = _observation()
    poisoned = {**clean, "ground_truth": {"full_gate_list": [[99] * 6]},
                "referee": {"valid_gate_crossings": 99}, "own_position": [99] * 3}
    v4 = encode_observation_sequence(clean, context)
    assert v4.shape == (OBS_DIM_SEQUENCE,)
    assert np.array_equal(v4[:OBS_DIM_V3], encode_observation_v3(clean, context))
    assert np.array_equal(v4, encode_observation_sequence(poisoned, context))
    assert FEATURE_NAMES_SEQUENCE[:OBS_DIM_V3] == FEATURE_NAMES_V3


class _AdvancingTracker:
    expected_beacon_id = "B01"
    local_completed = 0
    local_beacon_index = 1
    local_lap = 0
    total_beacons = 3
    laps = 1
    phase = "ADVANCE"
    finished = False
    _latest_visual_target = None

    def update(self, **kwargs):
        self.expected_beacon_id = "B02"
        self.local_completed = 1
        self.local_beacon_index = 2


def test_target_switch_is_visible_in_the_same_observation_step():
    source = OnboardSequenceContextTracker(total_beacons=3)
    source.reset(_observation())
    source._tracker = _AdvancingTracker()
    context = source.context(_observation("B02"), dt=0.1)
    encoded = encode_observation_sequence(_observation("B02"), context)
    assert context.expected_beacon_changed is True
    assert context.current_gate_index == 1
    assert context.remaining_gate_count == 2
    assert context.previous_gate_cleared is True
    assert encoded[FEATURE_NAMES_SEQUENCE.index("expected_beacon_changed")] == 1.0
    assert encoded[FEATURE_NAMES_SEQUENCE.index("current_gate_index_norm")] == pytest.approx(0.5)
    assert encoded[FEATURE_NAMES_SEQUENCE.index("time_since_confirmed_crossing_norm")] == 0.0


def test_all_sequence_seed_roles_are_disjoint():
    roles = [
        PPO_SEQUENCE_TRAINING_SEEDS, PPO_SEQUENCE_CURRICULUM_EVAL_SEEDS,
        PPO_SEQUENCE_CHECKPOINT_SELECTION_SEEDS, PPO_SEQUENCE_UNSEEN_HOLDOUT_SEEDS,
        PPO_SEQUENCE_OFFICIAL_HOLDOUT_SEEDS,
    ]
    for index, role in enumerate(roles):
        for other in roles[index + 1:]:
            assert set(role).isdisjoint(other)
    assert_pairwise_disjoint()


@pytest.mark.parametrize("stage", SEQUENCE_STAGES)
def test_procedural_curriculum_gate_counts_and_no_currents(tmp_path, stage):
    mixture = {"current_stage": 1.0, "three_gate": 0.0, "previous_stage": 0.0,
               "short_retention": 0.0, "targeted_failure": 0.0}
    sampler = SequenceCurriculumSampler(
        seed=30000 + int(stage[1:]), initial_stage=stage,
        maximum_stage=stage, replay_mixture=mixture,
    )
    # Exercise a deterministic production-sized matrix for every stage, not
    # merely a single favourable sample.  Track validation is pure/config-only
    # and therefore does not launch HoloOcean.
    for sample_index in range(64):
        geometry = sampler.sample()
        data_path = generate_sequence_track(
            geometry, tmp_path / f"{stage}_{sample_index:02d}.json"
        )
        data = json.loads(data_path.read_text(encoding="utf-8"))
        assert len(data["gates"]) == geometry.gate_count
        assert data["currents"] == []
        assert data["benchmark_task"]["mode"] == BENCHMARK_TASK_CLEAN_GATE
        assert data["race"]["official_mode"] is False
        assert data["participants"][0]["controller"] == "rl_sequence_ppo"
        assert 1 <= geometry.gate_count <= 22

        config = load_track_config(data_path, current_profile="none")
        result = validate_track_config(config)
        assert result.errors == []
        assert config.selected_current_profile == "none"
        assert config.benchmark_task.mode == BENCHMARK_TASK_CLEAN_GATE


def test_all_official_sequence_holdouts_validate_current_free():
    """Regression for Mixed Endurance seed 33502 at the 100352 holdout."""
    for track in OFFICIAL_TRACKS:
        config = load_track_config(
            track,
            benchmark_task=BENCHMARK_TASK_CLEAN_GATE,
            current_profile="none",
        )
        assert config.selected_current_profile == "none"
        assert config.benchmark_task.mode == BENCHMARK_TASK_CLEAN_GATE
        assert not config.currents
        assert validate_track_config(config).errors == []


def test_procedural_generator_replaces_current_gate_from_alternate_base(tmp_path):
    mixture = {"current_stage": 1.0, "three_gate": 0.0, "previous_stage": 0.0,
               "short_retention": 0.0, "targeted_failure": 0.0}
    sampler = SequenceCurriculumSampler(
        seed=33502, initial_stage="S2", maximum_stage="S2", replay_mixture=mixture,
    )
    path = generate_sequence_track(
        sampler.sample(), tmp_path / "s2_from_current_gate_base.json",
        base_track=OFFICIAL_TRACKS[2],
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["benchmark_task"]["mode"] == BENCHMARK_TASK_CLEAN_GATE
    assert raw["currents"] == []
    config = load_track_config(path, current_profile="none")
    assert config.benchmark_task.mode == BENCHMARK_TASK_CLEAN_GATE
    assert validate_track_config(config).errors == []


def test_sequence_episode_pairs_current_free_profile_with_clean_gate(monkeypatch):
    captured = {}

    class ProbeReached(RuntimeError):
        pass

    def probe_env(track, **kwargs):
        captured.update({"track": track, **kwargs})
        raise ProbeReached

    monkeypatch.setattr(
        "marine_race_arena.learning.sequence_evaluation.MarineRaceGymEnv",
        probe_env,
    )
    with pytest.raises(ProbeReached):
        evaluate_episode(
            object(), architecture="feedforward_ppo",
            track=OFFICIAL_TRACKS[2], seed=33502, category="official",
            adapter="fallback", max_steps=1,
        )
    assert captured["track"] == OFFICIAL_TRACKS[2]
    assert captured["current_profile"] == "none"
    assert captured["benchmark_task"] == BENCHMARK_TASK_CLEAN_GATE


def _clean_metrics(rate=0.95, longest=3):
    return {
        "full_sequence_completion_rate": rate,
        "longest_sequence_reliably_completed": longest,
        "collision_episodes": 0, "out_of_bounds_episodes": 0,
        "wrong_direction_events": 0, "previous_gate_returns": 0,
        "missed_gate_dnf": 0, "mean_completion_time_s": 20.0,
        "mean_action_jerk": 0.1,
    }


def test_promotion_needs_duration_and_two_consecutive_clean_full_evaluations():
    sampler = SequenceCurriculumSampler(seed=30000)
    assert not sampler.observe_full_evaluation(_clean_metrics(), 49_999)
    assert sampler.state.consecutive_qualifying_evaluations == 1
    assert sampler.observe_full_evaluation(_clean_metrics(), 50_000)
    assert sampler.current_stage == "S2"
    assert sampler.state.stage_entry_timesteps == 50_000
    assert sampler.state.efficiency_reward_active is False
    assert sampler.state.curriculum_stage_history[-1]["from"] == "S1"


def test_unsafe_evaluation_resets_streak_and_never_unlocks_efficiency():
    sampler = SequenceCurriculumSampler(seed=30000)
    sampler.observe_full_evaluation(_clean_metrics(), 25_000)
    unsafe = _clean_metrics(); unsafe["previous_gate_returns"] = 1
    sampler.observe_full_evaluation(unsafe, 50_000)
    assert sampler.current_stage == "S1"
    assert sampler.state.consecutive_qualifying_evaluations == 0
    assert sampler.state.efficiency_reward_active is False


def test_unevaluated_categories_are_none_not_zero():
    metrics = aggregate_sequence_evaluation([
        {"category": "three_gate", "full_sequence_completion": False,
         "gates_completed": 2, "expected_gates": 3, "missed_gate_dnf": 1,
         "previous_gate_returns": 1, "collisions": 0, "out_of_bounds": 0,
         "wrong_direction": 0, "completion_time_s": 20.0,
         "time_per_gate_s": 10.0, "path_length_m": 10.0, "action_jerk": 0.2,
         "next_gate_acquisition_times_s": [], "any_safety": False,
         "action_source": "ppo_policy"}
    ])
    assert metrics["three_gate_completion_rate"] == 0.0
    assert metrics["official_completion_rate"] is None
    assert metrics["official_n"] == 0


def test_longer_reliable_sequence_outranks_faster_short_sequence():
    short = _clean_metrics(rate=1.0, longest=5)
    short["mean_completion_time_s"] = 5.0
    long = _clean_metrics(rate=0.90, longest=12)
    long["mean_completion_time_s"] = 60.0
    assert checkpoint_rank_key(long) > checkpoint_rank_key(short)


class _Space:
    shape = (OBS_DIM_SEQUENCE,)


class _FakeSequenceModel:
    num_timesteps = 12_288
    _n_updates = 6
    _current_progress_remaining = 0.8
    observation_space = _Space()
    longrun_observation_version = OBS_ENCODING_VERSION_SEQUENCE
    longrun_architecture = "recurrent_ppo_lstm"
    longrun_policy_mode = "recurrent"
    longrun_frame_stack = 1
    policy = None

    def save(self, path):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("policy.txt", "ppo optimizer recurrent state")


def test_atomic_resume_preserves_all_sequence_state_non_null(tmp_path):
    sampler = SequenceCurriculumSampler(seed=30000)
    sampler.state.training_timesteps = 12_288
    sampler.state.stage_entry_timesteps = 4_096
    sampler.state.consecutive_qualifying_evaluations = 1
    sampler.state.efficiency_reward_active = True
    sampler.state.evaluation_history = [{"timesteps": 10_000, "qualifies": True}]
    sampler.state.curriculum_stage_history.append(
        {"timesteps": 4_096, "from": "S1", "to": "S2", "reason": "test"}
    )
    selection = {"best_metrics": _clean_metrics(), "best_timestep": 10_000,
                 "last_full_metrics": _clean_metrics()}
    aliases = {"last": "checkpoint.zip", "latest_safe": "safe.zip",
               "best_sequence": "best.zip"}
    atomic_save_checkpoint(
        _FakeSequenceModel(), tmp_path, total_timesteps=12_288,
        config_contract_hash="b" * 64, curriculum_state=sampler.state_dict(),
        evaluation_state={"history": [{"metrics": _clean_metrics()}],
                          "last": {"metrics": _clean_metrics()},
                          "best": {"metrics": _clean_metrics()}},
        status="safe", reason="full_evaluation",
        extra_state={"architecture": "recurrent_ppo_lstm",
                     "checkpoint_aliases": aliases,
                     "checkpoint_selection_state": selection,
                     "initialization": {"checkpoint": "ppo_900462_steps.zip",
                                        "sha256": "a" * 64, "policy_steps": 900462}},
    )
    checkpoint = latest_valid_checkpoint(
        tmp_path, expected_contract_hash="b" * 64, safe_only=False
    )
    assert checkpoint is not None
    state = load_checkpoint_state(checkpoint)
    recovered = SequenceCurriculumSampler(seed=30000)
    recovered.load_state_dict(state["curriculum"])
    assert recovered.state.training_timesteps == 12_288
    assert recovered.state.stage_entry_timesteps == 4_096
    assert recovered.state.consecutive_qualifying_evaluations == 1
    assert recovered.state.efficiency_reward_active is True
    assert recovered.state.evaluation_history
    assert recovered.state.curriculum_stage_history
    assert recovered.state.replay_mixture
    extra = state["extra"]
    for key in ("last", "latest_safe", "best_sequence"):
        assert extra["checkpoint_aliases"][key]
    for key in ("best_metrics", "best_timestep", "last_full_metrics"):
        assert extra["checkpoint_selection_state"][key] is not None
    assert state["rng"] and state["model_training"]


def test_recurrent_ppo_optimizer_scheduler_and_counters_round_trip(tmp_path):
    import gymnasium as gym
    from sb3_contrib import RecurrentPPO

    class TinyEnv(gym.Env):
        observation_space = gym.spaces.Box(-1.0, 1.0, (OBS_DIM_SEQUENCE,), dtype=np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            return np.zeros(OBS_DIM_SEQUENCE, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(OBS_DIM_SEQUENCE, dtype=np.float32), 0.1, False, True, {}

    schedule = AbsoluteLearningRateSchedule(1e-5, 1e-6, "linear")
    schedule.set_training_horizon(sb3_total_timesteps=16, absolute_total_timesteps=16)
    model = build_sequence_ppo(
        TinyEnv(), architecture="recurrent_ppo_lstm", seed=7,
        learning_rate=schedule, hidden_sizes=(16,), n_steps=8,
        batch_size=8, n_epochs=1,
    )
    model.learn(total_timesteps=8)
    sampler = SequenceCurriculumSampler(seed=30000)
    checkpoint = atomic_save_checkpoint(
        model, tmp_path, total_timesteps=int(model.num_timesteps),
        config_contract_hash="c" * 64, curriculum_state=sampler.state_dict(),
        evaluation_state={"history": [], "last": None, "best": None},
        status="unverified", reason="resume_test",
        extra_state={"architecture": "recurrent_ppo_lstm",
                     "checkpoint_aliases": {"last": "x", "latest_safe": "y", "best_sequence": "z"},
                     "checkpoint_selection_state": {"best_metrics": {}, "best_timestep": 8,
                                                    "last_full_metrics": {}},
                     "initialization": {"checkpoint": "ppo_900462_steps.zip",
                                        "sha256": "a" * 64, "policy_steps": 900462}},
    )
    state = load_checkpoint_state(checkpoint)
    loaded = RecurrentPPO.load(str(checkpoint.model_path), env=TinyEnv(), device="cpu")
    restore_model_training_state(loaded, state["model_training"])
    assert loaded.num_timesteps == model.num_timesteps == 8
    assert loaded._n_updates == model._n_updates
    assert loaded.policy.optimizer.param_groups[0]["lr"] == pytest.approx(
        model.policy.optimizer.param_groups[0]["lr"]
    )
    assert state["model_training"]["learning_rate_schedule"]["last_value"] > 0


def test_chunked_absolute_schedule_does_not_decay_to_final_lr_in_first_rollout():
    schedule = AbsoluteLearningRateSchedule(6e-6, 1e-6, "linear")
    schedule.set_training_horizon(sb3_total_timesteps=512, absolute_total_timesteps=2048)
    after_first = schedule(0.0)
    assert after_first == pytest.approx(4.75e-6)
    schedule.set_training_horizon(sb3_total_timesteps=1024, absolute_total_timesteps=2048)
    after_second = schedule(0.0)
    assert after_second == pytest.approx(3.5e-6)


def test_active_config_excludes_hybrid_and_bc_from_ppo_workflow():
    config = json.loads(Path("configs/rl/ppo_sequence_curriculum.json").read_text())
    assert config["active_workflow"]["controller_family"] == "ppo_only"
    assert config["active_workflow"]["expert_actions_or_losses"] is False
    assert config["initialization_checkpoint"].endswith("ppo_900462_steps.zip")
    assert "525678" not in config["initialization_checkpoint"]
    assert "1000814" not in config["initialization_checkpoint"]
