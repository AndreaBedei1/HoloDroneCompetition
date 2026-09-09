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
    capture_model_training_state,
    checkpoint_passed_safety,
    latest_valid_checkpoint,
    load_checkpoint_state,
    restore_model_training_state,
)
from marine_race_arena.learning.bc_longrun_v3 import _geometry, _split_for_group
from marine_race_arena.learning.longrun_config import LongRunConfig
from marine_race_arena.learning.longrun_env import ObservationFrameStack
from marine_race_arena.learning.longrun_evaluation import (
    bc_checkpoint_metric_key,
    checkpoint_metric_key,
    evaluate_longrun_policy,
    is_better,
    plateau_detected,
)
from marine_race_arena.learning.longrun_monitor import (
    StatusStore,
    UpdateMetricsRecorder,
)
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
    _migrate_learning_rate_schedule,
    run_contract_hash,
)


class _FakeModel:
    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr("model.txt", "policy optimizer scheduler")


def _checkpoint(tmp_path: Path, steps: int, sampler=None):
    sampler = sampler or CurriculumSampler(seed=22001)
    safe_report = {
        "mode": "full",
        "timesteps": steps,
        "n_eval": 20,
        "collision_events": 0,
        "out_of_bounds_events": 0,
        "wrong_direction_count": 0,
        "previous_gate_returns": 0,
        "all_actions_finite": True,
        "runtime_rule_controller_instantiated": False,
    }
    return atomic_save_checkpoint(
        _FakeModel(),
        tmp_path,
        total_timesteps=steps,
        config_contract_hash="a" * 64,
        curriculum_state=sampler.state_dict(),
        evaluation_state={
            "history": [safe_report],
            "best": {},
            "last": safe_report,
        },
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


def test_scheduler_uses_original_absolute_horizon_on_short_resume():
    schedule = AbsoluteLearningRateSchedule(1e-5, 2e-6, "linear")
    short_resume_target = 332_382
    current = 234_430
    schedule.set_training_horizon(
        sb3_total_timesteps=short_resume_target,
        absolute_total_timesteps=1_000_000,
    )
    sb3_progress = 1.0 - current / short_resume_target
    expected = 2e-6 + (1e-5 - 2e-6) * (1.0 - current / 1_000_000)
    assert schedule(sb3_progress) == pytest.approx(expected)
    assert schedule(sb3_progress) > 8e-6


def test_legacy_cloudpickled_schedule_is_migrated_with_state():
    class LegacySchedule:
        start = 1e-5
        end = 2e-6
        kind = "linear"
        multiplier = 0.5
        last_value = 8.19776e-6

        def __call__(self, progress_remaining):
            return 4e-6

    class Model:
        learning_rate = LegacySchedule()
        lr_schedule = LegacySchedule()

    config = LongRunConfig()
    model = Model()
    _migrate_learning_rate_schedule(model, config)
    assert isinstance(model.learning_rate, AbsoluteLearningRateSchedule)
    assert model.learning_rate.multiplier == pytest.approx(0.5)
    assert model.learning_rate.last_value == pytest.approx(8.19776e-6)
    assert model.lr_schedule.value_schedule is model.learning_rate
    assert hasattr(model.learning_rate, "set_training_horizon")


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
    assert state["model_training"]["num_timesteps"] == 0
    assert state["model_training"]["schema_version"] == (
        "multigate_model_training_state_v1"
    )
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


def test_latest_safe_requires_matching_clean_full_evaluation(tmp_path):
    verified = _checkpoint(tmp_path, 1024)
    stale_report = load_checkpoint_state(verified)["evaluation"]["last"]
    periodic = atomic_save_checkpoint(
        _FakeModel(),
        tmp_path,
        total_timesteps=2048,
        config_contract_hash="a" * 64,
        curriculum_state=CurriculumSampler(seed=22001).state_dict(),
        evaluation_state={
            "history": [stale_report],
            "best": {},
            "last": stale_report,
        },
        status="safe",
        reason="periodic",
    )
    collision_report = {
        **stale_report,
        "timesteps": 3072,
        "collision_events": 1,
    }
    collision = atomic_save_checkpoint(
        _FakeModel(),
        tmp_path,
        total_timesteps=3072,
        config_contract_hash="a" * 64,
        curriculum_state=CurriculumSampler(seed=22001).state_dict(),
        evaluation_state={
            "history": [collision_report],
            "best": {},
            "last": collision_report,
        },
        status="safe",
        reason="final_or_stop",
    )
    assert checkpoint_passed_safety(verified)
    assert not checkpoint_passed_safety(periodic)
    assert not checkpoint_passed_safety(collision)
    latest = latest_valid_checkpoint(
        tmp_path, expected_contract_hash="a" * 64
    )
    assert latest is not None
    assert latest.model_path == verified.model_path
    atomic_write_json(
        tmp_path / "status.json",
        {
            "total_timesteps": 3072,
            "checkpoint_aliases": {
                "latest_safe": str(collision.model_path)
            },
        },
    )
    status = run_status(tmp_path)
    assert status["checkpoint_aliases"]["latest_safe"] == str(
        verified.model_path
    )


def test_unsafe_periodic_resume_preserves_previous_verified_latest_safe(
    tmp_path,
):
    from marine_race_arena.learning.train_multigate_longrun import (
        _upgrade_resume_state,
    )

    verified = _checkpoint(tmp_path, 1024)
    stale_report = load_checkpoint_state(verified)["evaluation"]["last"]
    periodic = atomic_save_checkpoint(
        _FakeModel(),
        tmp_path,
        total_timesteps=2048,
        config_contract_hash="a" * 64,
        curriculum_state=CurriculumSampler(seed=22001).state_dict(),
        evaluation_state={
            "history": [stale_report],
            "best": {},
            "last": stale_report,
        },
        status="unsafe",
        reason="periodic",
        extra_state={
            "checkpoint_aliases": {
                "latest_safe": str(verified.model_path)
            }
        },
    )
    config = LongRunConfig()
    config.output_root = str(tmp_path.parent)
    config.run_name = tmp_path.name
    atomic_write_json(
        tmp_path / "run_manifest.json",
        {"config_contract_sha256": "a" * 64},
    )
    state = _upgrade_resume_state(
        config,
        periodic,
        load_checkpoint_state(periodic),
        _FakeModel(),
    )
    assert state["extra"]["checkpoint_aliases"]["latest_safe"] == str(
        verified.model_path
    )
    assert state["extra"]["checkpoint_aliases"]["last"] == str(
        periodic.model_path
    )


def test_recovery_selects_best_reliable_instead_of_last_checkpoint(tmp_path):
    from marine_race_arena.learning.prepare_c3_recovery import (
        select_best_reliable_checkpoint,
    )

    best = _checkpoint(tmp_path, 1024)
    best_report = load_checkpoint_state(best)["evaluation"]["last"]
    unsafe_report = {
        **best_report,
        "timesteps": 2048,
        "collision_events": 1,
    }
    last = atomic_save_checkpoint(
        _FakeModel(),
        tmp_path,
        total_timesteps=2048,
        config_contract_hash="a" * 64,
        curriculum_state=CurriculumSampler(seed=22001).state_dict(),
        evaluation_state={
            "history": [best_report, unsafe_report],
            "best": {
                "best_reliable": {
                    **best_report,
                    "checkpoint": str(best.model_path),
                }
            },
            "last": unsafe_report,
        },
        status="safe",
        reason="final_or_stop",
    )
    assert not checkpoint_passed_safety(last)
    selected = select_best_reliable_checkpoint(tmp_path)
    assert selected.model_path == best.model_path
    assert selected.timesteps == 1024


def test_curriculum_rng_and_stage_resume_exactly():
    sampler = CurriculumSampler(seed=22001, initial_stage="C2")
    for _ in range(7):
        sampler.sample()
    state = sampler.state_dict()
    expected = sampler.sample()
    resumed = CurriculumSampler(seed=22001, initial_stage="C2")
    resumed.load_state_dict(state)
    assert resumed.sample() == expected


def test_model_counter_schedule_and_optimizer_lr_resume_together():
    torch = pytest.importorskip("torch")

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))
            self.optimizer = torch.optim.Adam(self.parameters(), lr=8.2e-6)

    class Model:
        def __init__(self):
            self.policy = Policy()
            self.num_timesteps = 232_382
            self._n_updates = 226
            self._current_progress_remaining = 0.768576
            self.learning_rate = AbsoluteLearningRateSchedule(
                1e-5, 2e-6, "linear"
            )
            self.learning_rate.multiplier = 0.5
            self.learning_rate.last_value = 7.5e-6
            self.lr_schedule = self.learning_rate

    model = Model()
    saved = capture_model_training_state(model)
    model._n_updates = 0
    model._current_progress_remaining = 1.0
    model.policy.optimizer.param_groups[0]["lr"] = 1e-5
    model.learning_rate.multiplier = 1.0
    model.learning_rate.last_value = 1e-5
    restore_model_training_state(model, saved)
    assert model._n_updates == 226
    assert model._current_progress_remaining == pytest.approx(0.768576)
    assert model.policy.optimizer.param_groups[0]["lr"] == pytest.approx(8.2e-6)
    assert model.learning_rate.multiplier == pytest.approx(0.5)
    assert model.learning_rate.last_value == pytest.approx(7.5e-6)


def test_existing_run_contract_survives_compatible_code_fix(tmp_path):
    config = LongRunConfig(output_root=str(tmp_path), run_name="resume")
    config.run_dir.mkdir(parents=True)
    atomic_write_json(
        config.run_dir / "run_manifest.json",
        {"config_contract_sha256": "b" * 64},
    )
    assert run_contract_hash(config) == "b" * 64


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
    assert _split_for_group(first) == "train"


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


def test_bc_checkpoint_selection_prioritizes_each_retention_category():
    complete = {
        "single_gate_completion_rate": 1.0,
        "straight_completion_rate": 1.0,
        "left_completion_rate": 1.0,
        "right_completion_rate": 1.0,
        "safety_events": 0,
        "previous_gate_returns": 0,
        "mean_penalized_time_s": 40,
        "mean_action_jerk": 0.04,
    }
    faster_but_regressed = {
        **complete,
        "straight_completion_rate": 0.75,
        "mean_penalized_time_s": 1,
    }
    assert bc_checkpoint_metric_key(complete) > bc_checkpoint_metric_key(
        faster_but_regressed
    )


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
    assert (
        plateau_detected(
            history[:2],
            current_timesteps=150_000,
            plateau_steps=100_000,
            min_evaluations=2,
        )
        is None
    )


def test_evaluation_reports_case_progress(tmp_path):
    class ZeroModel:
        @staticmethod
        def predict(observation, deterministic=True):
            return [0.0, 0.0, 0.0, 0.0], None

    progress = []
    evaluate_longrun_policy(
        ZeroModel(),
        stage="C0",
        mode="light",
        seeds=[25000],
        output_dir=tmp_path,
        env_kwargs={
            "adapter": "fallback",
            "allow_fallback": True,
            "current_profile": "none",
            "max_steps": 2,
            "observation_encoding_version": "onboard_multigate_rl_v3",
        },
        progress_callback=lambda row: progress.append(dict(row)),
    )
    assert [row["completed"] for row in progress] == [0, 1]
    assert all(row["total"] == 1 for row in progress)


def test_status_file_atomic_update_and_graceful_stop(tmp_path):
    store = StatusStore(tmp_path, {"total_timesteps": 100})
    store.update(total_timesteps=200, state="RUNNING")
    data = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert data["total_timesteps"] == 200
    result = request_stop(tmp_path)
    assert result["stop_requested"]
    assert (tmp_path / "stop.requested").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_resume_quarantines_metric_rows_newer_than_selected_checkpoint(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "update_metrics.csv").write_text(
        "num_timesteps,n_updates\n232382,226\n235885,228\n",
        encoding="utf-8",
    )
    (tmp_path / "update_metrics.jsonl").write_text(
        "\n".join(
            (
                '{"num_timesteps":232382,"n_updates":226}',
                '{"num_timesteps":235885,"n_updates":228}',
            )
        )
        + "\n",
        encoding="utf-8",
    )

    class Model:
        num_timesteps = 232_382

    recorder = UpdateMetricsRecorder(Model(), tmp_path)
    assert recorder.last_timestep == 232_382
    assert [int(row["num_timesteps"]) for row in recorder.rows] == [232_382]
    assert "235885" not in (
        tmp_path / "update_metrics.jsonl"
    ).read_text(encoding="utf-8")
    quarantines = list(
        (tmp_path / "recovery").glob("update_metrics_after_232382_*.json")
    )
    assert len(quarantines) == 1
    recovered = json.loads(quarantines[0].read_text(encoding="utf-8"))
    assert recovered["rows"][0]["num_timesteps"] == 235_885


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
