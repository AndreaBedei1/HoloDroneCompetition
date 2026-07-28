"""Bounded unit tests for the multi-day multi-gate training infrastructure."""

from __future__ import annotations

import json
import pickle
import zipfile
from pathlib import Path

import pytest

# The benchmark-only ``ocean`` environment deliberately excludes RL/Torch
# dependencies. Keep this module collectable there while exercising it fully in
# ``marine_race_rl``.
pytest.importorskip("torch")

from marine_race_arena.learning.longrun_checkpoint import (
    atomic_save_checkpoint,
    atomic_write_json,
    latest_valid_checkpoint,
    load_checkpoint_state,
)
from marine_race_arena.learning.bc_longrun_v3 import _geometry
from marine_race_arena.learning.longrun_config import LongRunConfig
from marine_race_arena.learning.longrun_env import ObservationFrameStack
from marine_race_arena.learning.longrun_evaluation import (
    checkpoint_metric_key,
    is_better,
    plateau_detected,
)
from marine_race_arena.learning.longrun_monitor import StatusStore
from marine_race_arena.learning.longrun_tools import request_stop, run_status
from marine_race_arena.learning.multigate_longrun_data import build_balanced_plan
from marine_race_arena.learning.parametric_curriculum import (
    CurriculumSampler,
    curriculum_stage_decision,
    generate_two_gate_track,
)
from marine_race_arena.learning.seed_registry import (
    MULTIGATE_LONGRUN_BC_EVAL_SEEDS,
    MULTIGATE_LONGRUN_CHECKPOINT_SELECTION_SEEDS,
    MULTIGATE_LONGRUN_DEMONSTRATION_SEEDS,
    MULTIGATE_LONGRUN_DEV_EVAL_SEEDS,
    MULTIGATE_LONGRUN_PPO_TRAINING_SEEDS,
    assert_pairwise_disjoint,
)
from marine_race_arena.learning.train_multigate_longrun import (
    AbsoluteLearningRateSchedule,
    _is_recoverable_simulator_error,
)


class _FakeModel:
    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr("model.txt", "policy optimizer scheduler")


def _checkpoint(tmp_path: Path, steps: int, sampler=None):
    sampler = sampler or CurriculumSampler(seed=22001)
    return atomic_save_checkpoint(
        _FakeModel(),
        tmp_path,
        total_timesteps=steps,
        config_contract_hash="a" * 64,
        curriculum_state=sampler.state_dict(),
        evaluation_state={"history": [], "best": {}},
    )


def test_longrun_config_validation_and_editable_smoke():
    config = LongRunConfig(total_timesteps=1024)
    config.ppo.n_steps = 256
    config.ppo.batch_size = 64
    config.evaluation.light_frequency = 256
    config.evaluation.full_frequency = 1280
    config.evaluation.checkpoint_frequency = 256
    config.evaluation.plateau_steps = 1280
    config.validate(allow_smoke=True)
    config.ppo.batch_size = 100
    with pytest.raises(ValueError, match="divide"):
        config.validate(allow_smoke=True)


def test_experimental_frame_stack_is_explicit_and_validated():
    config = LongRunConfig(policy_mode="frame_stack", frame_stack=4)
    config.validate()
    config.frame_stack = 2
    with pytest.raises(ValueError, match="3 or 4"):
        config.validate()


def test_frame_stack_resets_history_and_rolls_newest_observation():
    gym = pytest.importorskip("gymnasium")
    np = pytest.importorskip("numpy")

    class TinyEnv(gym.Env):
        observation_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            return np.asarray([0.1, 0.2], dtype=np.float32), {}

        def step(self, action):
            return np.asarray([0.3, 0.4], dtype=np.float32), 0.0, False, False, {}

    env = ObservationFrameStack(TinyEnv(), 3)
    observation, _ = env.reset()
    assert observation.tolist() == pytest.approx([0.1, 0.2] * 3)
    observation, *_ = env.step(np.zeros(1, dtype=np.float32))
    assert observation.tolist() == pytest.approx([0.1, 0.2] * 2 + [0.3, 0.4])


def test_scheduler_resume_is_picklable_and_continuous():
    schedule = AbsoluteLearningRateSchedule(3e-5, 5e-6, "linear")
    before = schedule(0.5)
    restored = pickle.loads(pickle.dumps(schedule))
    assert restored(0.5) == pytest.approx(before)
    restored.scale(0.5)
    assert 5e-6 <= restored(0.5) < before


def test_atomic_checkpoint_and_timestep_resume_continuity(tmp_path):
    checkpoint = _checkpoint(tmp_path, 2048)
    assert checkpoint.model_path.exists()
    assert checkpoint.state_path.exists()
    latest = latest_valid_checkpoint(
        tmp_path, expected_contract_hash="a" * 64
    )
    assert latest is not None and latest.timesteps == 2048
    state = load_checkpoint_state(latest)
    assert state["total_timesteps"] == 2048
    assert not list((tmp_path / "checkpoints").glob("*.tmp"))
    assert not list((tmp_path / "checkpoints").glob("*.partial.zip"))


def test_corrupted_checkpoint_falls_back_to_previous(tmp_path):
    first = _checkpoint(tmp_path, 1024)
    second = _checkpoint(tmp_path, 2048)
    second.model_path.write_bytes(b"corrupt")
    latest = latest_valid_checkpoint(
        tmp_path, expected_contract_hash="a" * 64
    )
    assert latest is not None
    assert latest.timesteps == 1024
    assert latest.model_path == first.model_path


def test_curriculum_rng_and_stage_resume_exactly():
    sampler = CurriculumSampler(seed=22001, initial_stage="C2")
    for _ in range(7):
        sampler.sample()
    state = sampler.state_dict()
    expected = sampler.sample()
    resumed = CurriculumSampler(seed=22001, initial_stage="C2")
    resumed.load_state_dict(state)
    assert resumed.sample() == expected


def test_balanced_plan_has_requested_left_right_and_geometry_splits():
    plan = build_balanced_plan(straight=30, left=50, right=50)
    counts = {
        direction: sum(row.geometry.direction == direction for row in plan)
        for direction in ("straight", "left", "right")
    }
    assert counts == {"straight": 30, "left": 50, "right": 50}
    assert {abs(row.geometry.signed_turn_deg) for row in plan} >= {
        0.0,
        5.0,
        10.0,
        15.0,
        20.0,
        30.0,
        40.0,
        45.0,
    }
    split_by_geometry = {}
    for row in plan:
        split_by_geometry.setdefault(row.geometry.geometry_group, set()).add(row.split)
    assert all(len(splits) == 1 for splits in split_by_geometry.values())


def test_fixed_bc_track_is_indivisible_across_seeds():
    _, first = _geometry("missing/fixed_track.json", 1, "single_gate")
    _, second = _geometry("missing/fixed_track.json", 999, "single_gate")
    assert first == second


def test_parametric_track_keeps_referee_and_currents_unchanged(tmp_path):
    geometry = build_balanced_plan(straight=0, left=1, right=0)[0].geometry
    output = generate_two_gate_track(geometry, tmp_path / "track.json")
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["currents"] == []
    assert data["referee"]["gate_validation"]["vehicle_model"] == "center_point"
    assert data["training_geometry"]["signed_turn_deg"] > 0
    assert data["gates"][1]["position"][1] != 0


def test_curriculum_promotion_requires_balanced_multi_episode_evidence():
    passing = {
        "n_eval": 20,
        "completions": 16,
        "left_n": 10,
        "left_successes": 7,
        "right_n": 10,
        "right_successes": 7,
        "straight_completion_rate": 1.0,
        "previous_gate_returns": 0,
        "safety_events": 0,
    }
    assert curriculum_stage_decision("C1", passing) == "C2"
    assert curriculum_stage_decision("C1", {**passing, "right_successes": 6}) is None


def test_best_checkpoint_selection_is_lexicographic_not_reward():
    safe = {
        "completion_rate": 0.8,
        "left_completion_rate": 0.8,
        "right_completion_rate": 0.8,
        "mean_gates": 1.8,
        "safety_events": 0,
        "previous_gate_returns": 0,
        "mean_penalized_time_s": 40,
        "mean_action_jerk": 0.04,
        "mean_reward": -100,
    }
    unsafe_reward = {
        **safe,
        "completion_rate": 0.7,
        "mean_reward": 10_000,
    }
    assert checkpoint_metric_key(safe) > checkpoint_metric_key(unsafe_reward)
    assert is_better(safe, unsafe_reward)


def test_plateau_detection_uses_evaluation_metrics():
    history = [
        {
            "timesteps": step,
            "completion_rate": 0.5,
            "mean_gates": 1.5,
            "approx_kl": 0.0002,
            "left_completion_rate": 0.8,
            "right_completion_rate": 0.2,
        }
        for step in (100_000, 150_000, 200_000)
    ]
    plateau = plateau_detected(
        history,
        current_timesteps=200_000,
        plateau_steps=100_000,
        min_evaluations=3,
    )
    assert plateau and plateau["detected"]
    assert plateau["low_kl"]


def test_status_file_atomic_update_and_graceful_stop(tmp_path):
    store = StatusStore(tmp_path, {"total_timesteps": 100})
    store.update(total_timesteps=200, state="RUNNING")
    data = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert data["total_timesteps"] == 200
    result = request_stop(tmp_path)
    assert result["stop_requested"]
    assert (tmp_path / "stop.requested").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_status_pid_handling_marks_missing_process_dead(tmp_path):
    atomic_write_json(
        tmp_path / "status.json",
        {"state": "RUNNING", "last_log_update_unix": 1},
    )
    (tmp_path / "pid.txt").write_text("99999999", encoding="utf-8")
    status = run_status(tmp_path)
    assert status["process_alive"] is False


def test_recoverable_holoocean_failure_classification_is_bounded():
    assert _is_recoverable_simulator_error(RuntimeError("Unreal closed"))
    assert _is_recoverable_simulator_error(ConnectionError("socket"))
    assert not _is_recoverable_simulator_error(ValueError("bad config"))


def test_no_runtime_rule_controller_in_longrun_training_source():
    source = Path(
        "marine_race_arena/learning/train_multigate_longrun.py"
    ).read_text(encoding="utf-8")
    assert "RuleGateCenterThenCommitController" not in source
    assert "ControllerLoader" not in source
    assert '"rule_gate_center_then_commit"' not in source


def test_longrun_seed_roles_remain_pairwise_disjoint():
    assert_pairwise_disjoint()
    roles = [
        set(MULTIGATE_LONGRUN_PPO_TRAINING_SEEDS),
        set(MULTIGATE_LONGRUN_DEV_EVAL_SEEDS),
        set(MULTIGATE_LONGRUN_CHECKPOINT_SELECTION_SEEDS),
        set(MULTIGATE_LONGRUN_DEMONSTRATION_SEEDS),
        set(MULTIGATE_LONGRUN_BC_EVAL_SEEDS),
    ]
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            assert left.isdisjoint(right)
