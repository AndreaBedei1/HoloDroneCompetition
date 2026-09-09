"""Reliability-first PPO invariants that do not require HoloOcean."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from marine_race_arena.learning.longrun_config import LongRunConfig
from marine_race_arena.learning.longrun_checkpoint import atomic_save_checkpoint
from marine_race_arena.learning.longrun_evaluation import (
    aggregate_evaluation,
    checkpoint_metric_key,
    fast_reliable_metric_key,
)
from marine_race_arena.learning.longrun_monitor import (
    StatusStore,
    evaluation_regression_reasons,
    make_longrun_callback,
    rollback_attempt_allowed,
    rollback_target_stage,
    restore_ppo_training_state,
)
from marine_race_arena.learning.parametric_curriculum import (
    CurriculumSampler,
    reliability_requirements_met,
)
from marine_race_arena.learning.reward_v3 import MultiGateRewardConfig
from marine_race_arena.learning.seed_registry import (
    MULTIGATE_RELIABILITY_CHECKPOINT_SELECTION_SEEDS,
    MULTIGATE_RELIABILITY_DEV_EVAL_SEEDS,
    MULTIGATE_RELIABILITY_PPO_TRAINING_SEEDS,
    MULTIGATE_RELIABILITY_VALIDATION_SEEDS,
    assert_pairwise_disjoint,
)


CONFIG_PATH = Path("configs/rl_multigate_longrun_reliability_first.json")
RECOVERY_CONFIG_PATH = Path(
    "configs/rl/multigate_reliability_first_c3_recovery.json"
)


def _config() -> LongRunConfig:
    return LongRunConfig.load(CONFIG_PATH)


def _passing_report(**overrides):
    report = {
        "mode": "full",
        "n_eval": 20,
        "completions": 19,
        "completion_rate": 0.95,
        "single_gate_completion_rate": 1.0,
        "straight_completion_rate": 1.0,
        "left_completion_rate": 0.9,
        "right_completion_rate": 0.9,
        "episodes_with_collision": 0,
        "episodes_with_out_of_bounds": 0,
        "episodes_with_wrong_direction": 0,
        "episodes_with_any_safety": 0,
        "previous_gate_returns": 1,
    }
    report.update(overrides)
    return report


def _sampler(config: LongRunConfig) -> CurriculumSampler:
    curriculum = config.curriculum
    return CurriculumSampler(
        seed=config.seed,
        initial_stage=curriculum.initial_stage,
        maximum_stage=curriculum.maximum_stage,
        mixture=(
            curriculum.retention_fraction,
            curriculum.straight_fraction,
            curriculum.current_stage_fraction,
            curriculum.previous_stage_fraction,
            curriculum.failure_case_fraction,
        ),
        early_mixture=(
            curriculum.early_retention_fraction,
            curriculum.early_straight_fraction,
            curriculum.early_current_stage_fraction,
            curriculum.early_previous_stage_fraction,
            curriculum.early_failure_case_fraction,
        ),
        early_until_timesteps=curriculum.early_replay_until_timesteps,
        geometry_ramp_timesteps=curriculum.geometry_ramp_timesteps,
        promotion_config=curriculum.promotion,
    )


def test_committed_reliability_config_is_conservative_and_exact():
    config = _config()
    config.validate()
    assert config.training_profile == "reliability_first"
    assert config.total_timesteps == 1_000_000
    assert config.seed == 23001
    assert (
        config.ppo.learning_rate,
        config.ppo.final_learning_rate,
        config.ppo.n_steps,
        config.ppo.batch_size,
        config.ppo.n_epochs,
    ) == (1e-5, 2e-6, 2048, 256, 2)
    assert (
        config.ppo.clip_range,
        config.ppo.target_kl,
        config.ppo.high_kl,
        config.ppo.absolute_kl_stop,
        config.ppo.initial_action_std,
    ) == (0.05, 0.005, 0.012, 0.02, 0.08)
    assert config.initialization_sha256 == (
        "c9a889125278122b57307e21b7c2390baa61429f8952093bcc2760d9b3a49bf2"
    )
    assert config.adapter == "holoocean"
    assert not config.allow_fallback
    assert config.current_profile == "none"


def test_c3_recovery_config_is_conservative_policy_only():
    config = LongRunConfig.load(RECOVERY_CONFIG_PATH)
    config.validate()
    assert config.total_timesteps == 1_000_000
    assert config.initialization_kind == "ppo_weights"
    assert config.initialization_checkpoint.endswith(
        "ppo_476525_steps.zip"
    )
    assert config.retention.reference_checkpoint.endswith("bc_v3.pt")
    assert config.ppo.learning_rate == pytest.approx(5e-6)
    assert config.ppo.target_kl == pytest.approx(0.003)
    assert config.retention.active_through_stage == "C3"
    assert config.retention.final_weight == pytest.approx(0.06)
    assert config.curriculum.failure_case_fraction == pytest.approx(0.20)
    assert config.curriculum.geometry_ramp_timesteps == 150_000
    assert (
        config.curriculum.promotion.required_consecutive_full_evaluations
        == 2
    )
    assert (
        config.curriculum.promotion.minimum_stage_timesteps["C3"]
        == 100_000
    )
    assert config.curriculum.promotion.maximum_collision_episodes == 0
    assert config.curriculum.promotion.maximum_out_of_bounds_episodes == 0
    assert config.curriculum.promotion.maximum_wrong_direction_episodes == 0
    assert config.curriculum.promotion.maximum_previous_gate_returns == 0
    assert config.reliability.reward_phase.action_change_penalty == pytest.approx(
        0.06
    )


def test_c3_geometry_is_ramped_without_hiding_targeted_failure_replay():
    config = LongRunConfig.load(RECOVERY_CONFIG_PATH)
    sampler = _sampler(config)
    sampler.state.current_stage = "C3"
    sampler.state.stage_entry_timesteps = 451_949
    sampler.set_timesteps(525_678)
    limits = sampler._sampling_limits("C3", "current_stage")
    assert 20.0 < limits.max_abs_turn_deg < 30.0
    sampler.set_timesteps(601_949)
    assert sampler._sampling_limits(
        "C3", "current_stage"
    ).max_abs_turn_deg == pytest.approx(30.0)


def test_recovery_preserves_absolute_timestep_and_curriculum_history():
    from dataclasses import asdict

    from marine_race_arena.learning.parametric_curriculum import (
        TransitionGeometry,
    )
    from marine_race_arena.learning.prepare_c3_recovery import (
        _recovery_curriculum,
    )

    config = LongRunConfig.load(RECOVERY_CONFIG_PATH)
    source_sampler = _sampler(_config())
    source_sampler.state.current_stage = "C3"
    source_sampler.state.training_timesteps = 525_678
    source_sampler.state.stage_entry_timesteps = 451_949
    source_sampler.state.stage_changes = [
        {
            "timesteps": 451_949,
            "from": "C2",
            "to": "C3",
            "reason": "evaluation_promotion",
        }
    ]
    geometry = TransitionGeometry(
        signed_turn_deg=30.0,
        gate_separation_m=6.0,
        lateral_displacement_m=0.4,
        vertical_displacement_m=0.3,
        starting_yaw_error_deg=6.0,
        initial_lateral_offset_m=0.5,
        source="held_out_evaluation",
        stage="C3",
    )
    recovered = _recovery_curriculum(
        config,
        {
            "total_timesteps": 525_678,
            "curriculum": source_sampler.state_dict(),
        },
        asdict(geometry),
    )
    state = recovered["state"]
    assert state["training_timesteps"] == 525_678
    assert state["current_stage"] == "C3"
    assert state["stage_entry_timesteps"] == 451_949
    assert state["stage_changes"] == source_sampler.state.stage_changes
    assert state["targeted_failures"][-1]["geometry"] == asdict(geometry)
    assert recovered["geometry_ramp_timesteps"] == 150_000


def test_early_replay_mixture_switches_exactly_at_50k():
    sampler = _sampler(_config())
    assert sampler.active_mixture == pytest.approx((0.30, 0.25, 0.25, 0.15, 0.05))
    sampler.set_timesteps(49_999)
    assert sampler.active_mixture == pytest.approx((0.30, 0.25, 0.25, 0.15, 0.05))
    sampler.set_timesteps(50_000)
    assert sampler.active_mixture == pytest.approx((0.20, 0.20, 0.30, 0.20, 0.10))


def test_only_two_consecutive_full_passes_after_lock_promote():
    config = _config()
    sampler = _sampler(config)
    light = _passing_report(mode="light")
    assert not reliability_requirements_met(light, config.curriculum.promotion)
    sampler.record_evaluation(light, timesteps=25_000)
    assert sampler.current_stage == "C0"
    assert sampler.state.consecutive_full_passes == 0
    sampler.record_evaluation(_passing_report(), timesteps=25_000)
    assert sampler.current_stage == "C0"
    assert sampler.state.consecutive_full_passes == 1
    sampler.record_evaluation(_passing_report(), timesteps=50_000)
    assert sampler.current_stage == "C1"
    assert sampler.state.stage_entry_timesteps == 50_000
    assert sampler.state.consecutive_full_passes == 0


def test_c3_requires_minimum_duration_and_two_clean_full_suites():
    config = LongRunConfig.load(RECOVERY_CONFIG_PATH)
    sampler = _sampler(config)
    sampler.state.current_stage = "C3"
    sampler.state.stage_entry_timesteps = 451_949
    clean = _passing_report(
        stage="C3",
        completions=20,
        completion_rate=1.0,
        previous_gate_returns=0,
    )
    sampler.record_evaluation(clean, timesteps=525_678)
    sampler.record_evaluation(clean, timesteps=550_000)
    assert sampler.current_stage == "C3"
    assert sampler.state.consecutive_full_passes == 2
    sampler.record_evaluation(
        _passing_report(
            stage="C3",
            completions=20,
            completion_rate=1.0,
            previous_gate_returns=0,
            episodes_with_collision=1,
        ),
        timesteps=552_000,
    )
    assert sampler.current_stage == "C3"
    assert sampler.state.consecutive_full_passes == 0
    sampler.record_evaluation(clean, timesteps=575_000)
    assert sampler.current_stage == "C3"
    sampler.record_evaluation(clean, timesteps=600_000)
    assert sampler.current_stage == "C4"


def test_failed_full_suite_resets_consecutive_promotion_streak():
    sampler = _sampler(_config())
    sampler.record_evaluation(_passing_report(), timesteps=25_000)
    sampler.record_evaluation(
        _passing_report(right_completion_rate=0.8), timesteps=50_000
    )
    assert sampler.current_stage == "C0"
    assert sampler.state.consecutive_full_passes == 0


def test_later_stages_require_progressively_stronger_directional_balance():
    promotion = _config().curriculum.promotion
    assert reliability_requirements_met(
        _passing_report(stage="C1", left_completion_rate=0.85), promotion
    )
    assert not reliability_requirements_met(
        _passing_report(stage="C3", left_completion_rate=0.87), promotion
    )
    assert reliability_requirements_met(
        _passing_report(
            stage="C3",
            left_completion_rate=0.90,
            right_completion_rate=0.90,
        ),
        promotion,
    )


def test_distinct_safety_events_frames_and_affected_episodes_are_preserved():
    rows = [
        {
            "finished": False,
            "category": "left",
            "completed_gates": 1,
            "collision_events": 2,
            "collision_frames": 157,
            "out_of_bounds_events": 0,
            "out_of_bounds_frames": 0,
            "wrong_direction_crossings": 1,
            "safety_warning_events": 3,
            "safety_warning_frames": 20,
            "previous_gate_returns": 0,
            "action_jerk": 0.1,
            "mean_inference_ms": 1.0,
            "actions_finite": True,
        },
        {
            "finished": True,
            "category": "right",
            "completed_gates": 2,
            "collision_events": 0,
            "collision_frames": 0,
            "out_of_bounds_events": 1,
            "out_of_bounds_frames": 8,
            "wrong_direction_crossings": 0,
            "safety_warning_events": 1,
            "safety_warning_frames": 4,
            "previous_gate_returns": 0,
            "raw_time_s": 20.0,
            "penalized_time_s": 30.0,
            "action_jerk": 0.05,
            "mean_inference_ms": 2.0,
            "actions_finite": True,
        },
    ]
    result = aggregate_evaluation(rows)
    assert result["collision_events"] == 2
    assert result["collision_frames"] == 157
    assert result["out_of_bounds_events"] == 1
    assert result["out_of_bounds_frames"] == 8
    assert result["wrong_direction_count"] == 1
    assert result["safety_warning_events"] == 4
    assert result["safety_warning_frames"] == 24
    assert result["episodes_with_any_safety"] == 2


def test_selection_prefers_completion_then_left_right_floor_before_speed():
    reliable = {
        **_passing_report(),
        "mean_gates": 1.9,
        "mean_penalized_time_s": 40.0,
        "mean_successful_time_s": 40.0,
        "mean_action_jerk": 0.05,
        "mean_inference_ms": 2.0,
    }
    faster_but_less_complete = {
        **reliable,
        "completion_rate": 0.90,
        "mean_penalized_time_s": 10.0,
    }
    faster_but_unbalanced = {
        **reliable,
        "left_completion_rate": 0.8,
        "mean_penalized_time_s": 10.0,
    }
    assert checkpoint_metric_key(reliable) > checkpoint_metric_key(
        faster_but_less_complete
    )
    assert checkpoint_metric_key(reliable) > checkpoint_metric_key(
        faster_but_unbalanced
    )
    promotion = _config().curriculum.promotion
    assert fast_reliable_metric_key(reliable, promotion) is not None
    assert fast_reliable_metric_key(
        {**reliable, "episodes_with_collision": 1}, promotion
    ) is None
    slower_success = {**reliable, "mean_penalized_time_s": 1000.0}
    fast_failure = {
        **reliable,
        "completion_rate": 0.0,
        "mean_penalized_time_s": 1.0,
    }
    assert checkpoint_metric_key(slower_success) > checkpoint_metric_key(
        fast_failure
    )


def test_regression_gate_detects_retention_direction_safety_and_returns():
    rollback = _config().reliability.rollback
    incumbent = _passing_report(completion_rate=1.0)
    report = _passing_report(
        completion_rate=0.8,
        single_gate_completion_rate=0.5,
        straight_completion_rate=0.5,
        left_completion_rate=0.5,
        episodes_with_any_safety=1,
        previous_gate_returns=3,
    )
    reasons = evaluation_regression_reasons(report, incumbent, rollback)
    assert set(reasons) == {
        "overall_completion_drop",
        "single_gate_retention_floor",
        "straight_retention_floor",
        "directional_completion_floor",
        "safety_episode_regression",
        "previous_gate_return_regression",
    }


def test_rollback_restores_policy_optimizer_scheduler_and_keeps_timestep():
    import torch

    from marine_race_arena.learning.train_multigate_longrun import (
        AbsoluteLearningRateSchedule,
    )

    class Policy(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([float(value)]))
            self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)

    class Model:
        def __init__(self, value, steps):
            self.policy = Policy(value)
            self.num_timesteps = steps
            self.lr_schedule = AbsoluteLearningRateSchedule(1e-5, 2e-6, "linear")

    source = Model(2.0, 10)
    source.policy.optimizer.zero_grad()
    source.policy.weight.sum().backward()
    source.policy.optimizer.step()
    target = Model(-1.0, 50_000)
    restore_ppo_training_state(target, source, 0.5)
    assert target.policy.weight.detach().item() == pytest.approx(
        source.policy.weight.detach().item()
    )
    assert target.policy.optimizer.state_dict()["state"]
    assert target.num_timesteps == 50_000
    assert target.lr_schedule.multiplier == pytest.approx(0.5)
    assert rollback_target_stage("C3") == "C2"
    assert rollback_target_stage("C0") == "C0"
    assert rollback_attempt_allowed(1, 2)
    assert rollback_attempt_allowed(2, 2)
    assert not rollback_attempt_allowed(3, 2)


def test_multiple_rollbacks_atomically_record_complete_events_and_status(
    tmp_path,
):
    import json

    import numpy as np
    import torch

    from marine_race_arena.learning.longrun_rollback import (
        ROLLBACK_EVENT_SCHEMA_VERSION,
    )
    from marine_race_arena.learning.train_multigate_longrun import (
        AbsoluteLearningRateSchedule,
    )

    class Policy(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([float(value)]))
            self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)

    class Retention:
        def __init__(self):
            self.weight_multiplier = 1.0
            self.base_weight = 0.1

        def increase_weight(self, factor):
            self.weight_multiplier *= float(factor)
            return self.base_weight * self.weight_multiplier

        def state_dict(self):
            return {
                "schema_version": "offline_retention_state_v1",
                "weight_multiplier": self.weight_multiplier,
            }

    class Env:
        @staticmethod
        def reset():
            return np.zeros((1, 3), dtype=np.float32)

    class Model:
        def __init__(self, value, timesteps):
            self.policy = Policy(value)
            self.num_timesteps = timesteps
            self.learning_rate = AbsoluteLearningRateSchedule(
                1e-5, 2e-6, "linear"
            )
            self.learning_rate.last_value = 8e-6
            self.lr_schedule = self.learning_rate
            self._retention_regularizer = Retention()
            self.env = Env()
            self.n_envs = 1

        @classmethod
        def load(cls, _path, device="cpu"):
            assert device == "cpu"
            return cls(2.0, 100_000)

        def save(self, path):
            Path(path).write_bytes(b"fake-ppo")

    class Reward:
        reward_phase = "efficiency"

        def set_phase(self, phase):
            self.reward_phase = phase

    config = _config()
    config.output_root = str(tmp_path.parent)
    config.run_name = tmp_path.name
    sampler = _sampler(config)
    sampler.state.current_stage = "C3"
    model = Model(-1.0, 300_397)
    reward = Reward()
    status = StatusStore(tmp_path)
    callback = make_longrun_callback(
        run_dir=tmp_path,
        config=config,
        sampler=sampler,
        status_store=status,
        contract_hash="a" * 64,
        evaluate_fn=lambda *_: _passing_report(),
        reward_config=reward,
    )
    callback.model = model
    callback.best = {
        "best_reliable": {
            "checkpoint": str(tmp_path / "checkpoints" / "source.zip")
        }
    }

    callback._perform_rollback(["straight_retention_floor"])
    model.num_timesteps = 325_000
    callback._perform_rollback(["directional_completion_floor"])

    assert callback.rollback_count == 2
    assert len(callback.rollback_history) == 2
    first, second = callback.rollback_history
    assert first["stage_before"] == "C3"
    assert first["stage_after"] == "C2"
    assert second["stage_before"] == "C2"
    assert second["stage_after"] == "C1"
    assert first["retention_multiplier_before"] == pytest.approx(1.0)
    assert first["retention_multiplier_after"] == pytest.approx(1.25)
    assert second["retention_multiplier_before"] == pytest.approx(1.25)
    assert second["retention_multiplier_after"] == pytest.approx(1.5625)
    for attempt, event in enumerate(callback.rollback_history, start=1):
        assert event["schema_version"] == ROLLBACK_EVENT_SCHEMA_VERSION
        assert event["attempt"] == attempt
        assert event["reason"]
        assert event["source_checkpoint"].endswith("source.zip")
        assert event["learning_rate_before"]["schedule_value"] is not None
        assert event["learning_rate_after"]["schedule_value"] is not None
        assert event["outcome"] == "restored"
        assert event["curriculum_transition"] == {
            "timesteps": event["timesteps"],
            "from": event["stage_before"],
            "to": event["stage_after"],
            "reason": "automatic_full_evaluation_rollback",
        }

    journal = [
        json.loads(line)
        for line in (
            tmp_path / "logs" / "rollback_history.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert journal == callback.rollback_history
    assert status.data["rollback_count"] == 2
    assert len(status.data["rollback_history"]) == 2
    assert status.data["last_rollback_attempt"] == 2
    assert status.data["last_rollback_outcome"] == "restored"
    assert status.data["last_rollback_event"] == second
    assert status.data["curriculum_stage"] == "C1"
    assert status.data["stage_entry_timesteps"] == 325_000
    assert status.data["curriculum_stage_history"][-1] == {
        "timesteps": 325_000,
        "from": "C2",
        "to": "C1",
        "reason": "automatic_full_evaluation_rollback",
    }
    assert reward.reward_phase == "reliability"

    model.num_timesteps = 350_000
    callback._perform_rollback(["safety_episode_regression"])
    assert callback.rollback_count == 3
    assert callback.stop_reason == "REPEATED_FULL_EVALUATION_REGRESSION"
    assert callback.rollback_history[-1]["outcome"] == "stopped"
    assert callback.rollback_history[-1]["recovery_attempt"] == 3
    assert status.data["rollback_count"] == 3
    assert status.data["last_rollback_outcome"] == "stopped"
    assert len(
        (
            tmp_path / "logs" / "rollback_history.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ) == 3

    unsafe = _passing_report(
        stage="C1",
        episodes_with_collision=1,
    )
    unsafe.update(
        {
            "timesteps": 350_000,
            "collision_events": 1,
            "runtime_rule_controller_instantiated": False,
        }
    )
    callback.last_eval = unsafe
    status.data["checkpoint_aliases"] = {
        "latest_safe": str(tmp_path / "checkpoints" / "known_safe.zip")
    }
    final = callback._save_checkpoint(
        reason="final_or_stop",
        safe=True,
    )
    assert final.manifest["status"] == "unsafe"
    assert status.data["checkpoint_aliases"]["latest_safe"].endswith(
        "known_safe.zip"
    )


def test_status_and_resume_recover_second_rollback_from_journals(tmp_path):
    import json
    from types import SimpleNamespace

    from marine_race_arena.learning.longrun_tools import run_status
    from marine_race_arena.learning.train_multigate_longrun import (
        _upgrade_resume_state,
    )

    config = _config()
    config.output_root = str(tmp_path.parent)
    config.run_name = tmp_path.name
    sampler = _sampler(config)
    sampler.state.current_stage = "C1"
    sampler.state.training_timesteps = 302_445
    sampler.state.stage_entry_timesteps = 300_397
    first_transition = {
        "timesteps": 225_280,
        "from": "C2",
        "to": "C1",
        "reason": "automatic_full_evaluation_rollback",
    }
    second_transition = {
        "timesteps": 300_397,
        "from": "C2",
        "to": "C1",
        "reason": "automatic_full_evaluation_rollback",
    }
    sampler.state.stage_changes = [first_transition]
    first = {
        "timesteps": 225_280,
        "attempt": 1,
        "reasons": ["safety_episode_regression"],
        "source": "ppo_126976_steps.zip",
        "retention_weight_after": 0.125,
        "outcome": "restored",
        "stage": "C1",
    }
    second = {
        "timesteps": 300_397,
        "attempt": 2,
        "reasons": ["straight_retention_floor"],
        "source": "ppo_251245_steps.zip",
        "retention_weight_after": 0.15625,
        "outcome": "restored",
        "stage": "C1",
    }
    (tmp_path / "logs").mkdir(parents=True)
    (tmp_path / "logs" / "rollback_history.jsonl").write_text(
        "\n".join(json.dumps(row) for row in (first, second)) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "curriculum_history.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (first_transition, second_transition)
        )
        + "\n",
        encoding="utf-8",
    )
    status = StatusStore(
        tmp_path,
        {
            "total_timesteps": 302_445,
            "curriculum_stage": "C1",
            "stage_entry_timesteps": 300_397,
            "rollback_count": 2,
            "last_rollback_reason": ["straight_retention_floor"],
            "last_rollback_source": second["source"],
            "rollback_history": [first],
            "curriculum_stage_history": [first_transition],
        },
    )
    recovered_status = run_status(tmp_path)
    assert recovered_status["rollback_count"] == 2
    assert len(recovered_status["rollback_history"]) == 2
    assert recovered_status["last_rollback_attempt"] == 2
    assert recovered_status["last_rollback_reason"] == [
        "straight_retention_floor"
    ]
    assert recovered_status["rollback_history"][-1]["stage_before"] == "C2"
    assert recovered_status["rollback_history"][-1]["stage_after"] == "C1"
    assert len(recovered_status["curriculum_stage_history"]) == 2

    state = {
        "total_timesteps": 302_445,
        "curriculum": sampler.state_dict(),
        "evaluation": {"history": [], "best": {}, "last": None},
        "model_training": {
            "schema_version": "multigate_model_training_state_v1",
            "optimizer_learning_rates": [3.79e-6],
            "retention_regularizer": {"weight_multiplier": 1.5625},
        },
        "extra": {
            "rollback_count": 2,
            "rollback_attempt_base": 2,
            "last_rollback_reason": second["reasons"],
            "last_rollback_source": second["source"],
            "rollback_history": [first],
            "reward_phase": "reliability",
        },
    }
    checkpoint = SimpleNamespace(
        model_path=tmp_path / "checkpoints" / "ppo_302445_steps.zip",
        manifest={"status": "safe"},
    )
    upgraded = _upgrade_resume_state(
        config, checkpoint, state, SimpleNamespace()
    )
    sampler.load_state_dict(upgraded["curriculum"])
    reward = MultiGateRewardConfig(reward_phase="efficiency")
    callback = make_longrun_callback(
        run_dir=tmp_path,
        config=config,
        sampler=sampler,
        status_store=status,
        contract_hash="a" * 64,
        evaluate_fn=lambda *_: _passing_report(),
        reward_config=reward,
    )
    callback.restore_pipeline_state(upgraded)

    assert callback.rollback_count == 2
    assert callback.rollback_attempt_base == 2
    assert len(callback.rollback_history) == 2
    assert callback.last_rollback_reason == ["straight_retention_floor"]
    assert callback.last_rollback_source == second["source"]
    assert status.data["rollback_count"] == 2
    assert status.data["rollback_attempt_base"] == 2
    assert len(status.data["rollback_history"]) == 2
    assert len(status.data["curriculum_stage_history"]) == 2
    assert status.data["last_rollback_attempt"] == 2
    assert status.data["stage_entry_timesteps"] == 300_397
    assert reward.reward_phase == "reliability"


def test_reward_phase_time_term_is_bounded_below_failure_penalty():
    config = _config()
    phase = config.reliability.reward_phase
    reward = MultiGateRewardConfig(
        reward_phase="reliability",
        reliability_time_cost=phase.reliability_time_cost,
        efficiency_time_cost=phase.efficiency_time_cost,
        efficiency_detour_penalty=phase.efficiency_detour_penalty,
        efficiency_jerk_penalty=phase.efficiency_jerk_penalty,
        efficiency_energy_penalty=phase.efficiency_energy_penalty,
        per_step_efficiency_penalty_cap=phase.per_step_efficiency_penalty_cap,
    )
    assert reward.maximum_episode_time_term(config.max_episode_steps) == 0
    reward.set_phase("efficiency")
    assert reward.maximum_episode_time_term(config.max_episode_steps) == pytest.approx(3.6)
    assert reward.timeout_penalty > reward.maximum_episode_time_term(
        config.max_episode_steps
    )
    assert phase.per_step_efficiency_penalty_cap < reward.completion_bonus


def test_retention_weight_decays_and_is_disabled_after_c2_without_runtime_expert():
    from marine_race_arena.learning.reliability_first import (
        OfflineRetentionRegularizer,
    )

    regularizer = object.__new__(OfflineRetentionRegularizer)
    regularizer.base_weight = 0.10
    regularizer.maximum_weight = 0.20
    regularizer.final_weight = 0.0
    regularizer.decay_until_timesteps = 150_000
    regularizer.active_through_stage_index = 2
    regularizer.weight_multiplier = 1.0
    regularizer.calls = 17
    regularizer.rng = __import__("numpy").random.default_rng(42)
    regularizer.last_metrics = {
        "bc_retention_loss": 0.2,
        "initial_policy_kl": 0.1,
        "retention_weight": 0.05,
    }
    state = regularizer.state_dict()
    expected_rng_value = regularizer.rng.integers(0, 1000)
    regularizer.calls = 0
    regularizer.weight_multiplier = 2.0
    regularizer.load_state_dict(state)
    assert regularizer.calls == 17
    assert regularizer.weight_multiplier == 1.0
    assert regularizer.rng.integers(0, 1000) == expected_rng_value
    assert regularizer.effective_weight(0, "C0") == pytest.approx(0.10)
    assert regularizer.effective_weight(75_000, "C2") == pytest.approx(0.05)
    assert regularizer.effective_weight(150_000, "C2") == 0
    assert regularizer.effective_weight(1, "C3") == 0
    source = Path(
        "marine_race_arena/learning/reliability_first.py"
    ).read_text(encoding="utf-8")
    assert "RuleGateCenterThenCommitController" not in source
    assert "ControllerLoader" not in source


def test_resume_hydrates_exact_reliability_pipeline_state_without_resets(
    tmp_path,
):
    import json
    from types import SimpleNamespace

    import numpy as np

    from marine_race_arena.learning.reliability_first import (
        OfflineRetentionRegularizer,
    )
    from marine_race_arena.learning.train_multigate_longrun import (
        AbsoluteLearningRateSchedule,
        _upgrade_resume_state,
    )

    config = _config()
    config.output_root = str(tmp_path.parent)
    config.run_name = tmp_path.name
    sampler = _sampler(config)
    sampler.state.current_stage = "C1"
    sampler.state.training_timesteps = 232_382
    sampler.state.stage_entry_timesteps = 225_280
    sampler.state.consecutive_full_passes = 2
    sampler.state.reliable_full_streak = 2
    sampler.state.efficiency_phase_active = True
    sampler.state.stage_changes = [
        {"timesteps": 25_000, "from": "C0", "to": "C1"},
        {"timesteps": 126_976, "from": "C1", "to": "C2"},
        {
            "timesteps": 225_280,
            "from": "C2",
            "to": "C1",
            "reason": "automatic_full_evaluation_rollback",
        },
    ]
    report = {
        **_passing_report(
            timesteps=225_280,
            stage="C2",
            collision_events=2,
            collision_frames=9,
            out_of_bounds_events=3,
            out_of_bounds_frames=4,
            wrong_direction_count=5,
            safety_warning_events=6,
            safety_warning_frames=7,
            episodes_with_collision=1,
            episodes_with_out_of_bounds=1,
            episodes_with_wrong_direction=1,
            episodes_with_any_safety=1,
            mean_successful_time_s=15.33,
            mean_penalized_time_s=15.83,
            mean_action_jerk=0.036,
        )
    }

    class FakeModel:
        num_timesteps = 0

        def save(self, path):
            import zipfile

            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("model.txt", "policy optimizer scheduler")

    best_checkpoint = atomic_save_checkpoint(
        FakeModel(),
        tmp_path,
        total_timesteps=126_976,
        config_contract_hash="a" * 64,
        curriculum_state=sampler.state_dict(),
        evaluation_state={"history": [], "best": {}},
    )
    resume_checkpoint = atomic_save_checkpoint(
        FakeModel(),
        tmp_path,
        total_timesteps=232_382,
        config_contract_hash="a" * 64,
        curriculum_state=sampler.state_dict(),
        evaluation_state={"history": [], "best": {}},
    )
    rollback = {
        "timesteps": 225_280,
        "attempt": 1,
        "reasons": ["safety_episode_regression"],
        "source": str(best_checkpoint.model_path),
        "retention_weight_after": 0.125,
        "outcome": "restored",
        "stage": "C1",
    }
    (tmp_path / "logs").mkdir(exist_ok=True)
    (tmp_path / "logs" / "rollback_history.jsonl").write_text(
        json.dumps(rollback) + "\n", encoding="utf-8"
    )
    regularizer = object.__new__(OfflineRetentionRegularizer)
    regularizer.rng = np.random.default_rng(23001 + 9173)
    regularizer.calls = 0
    regularizer.weight_multiplier = 1.0
    regularizer.batch_size = 4
    regularizer.observations = np.zeros((10, 3), dtype=np.float32)
    regularizer.base_weight = 0.1
    regularizer.last_metrics = {
        "bc_retention_loss": None,
        "initial_policy_kl": None,
        "retention_weight": 0.1,
    }
    schedule = AbsoluteLearningRateSchedule(1e-5, 2e-6, "linear")
    schedule.last_value = 8.19776e-6
    legacy_model = SimpleNamespace(
        num_timesteps=232_382,
        _n_updates=226,
        n_epochs=2,
        _current_progress_remaining=0.768576,
        learning_rate=schedule,
        lr_schedule=schedule,
        policy=SimpleNamespace(
            optimizer=SimpleNamespace(param_groups=[{"lr": 8.148608e-6}])
        ),
        _retention_regularizer=regularizer,
    )
    # This is the exact v1 shape that triggered the bug: no model-training
    # sidecar, last evaluation, rollback detail/history, or alias snapshot.
    legacy_state = {
        "total_timesteps": 232_382,
        "curriculum": sampler.state_dict(),
        "evaluation": {
            "history": [report],
            "best": {
                "best_overall": {
                    **report,
                    "checkpoint": str(best_checkpoint.model_path),
                },
                "best_reliable": {
                    **report,
                    "checkpoint": str(best_checkpoint.model_path),
                },
                "best_fast_reliable": {
                    **report,
                    "checkpoint": str(best_checkpoint.model_path),
                },
            },
        },
        "extra": {
            "automatic_changes": [{"kind": "bounded_lr_change"}],
            "rollback_count": 1,
            "reward_phase": "efficiency",
        },
    }
    state = _upgrade_resume_state(
        config, resume_checkpoint, legacy_state, legacy_model
    )
    reward = MultiGateRewardConfig(reward_phase="reliability")
    status = StatusStore(tmp_path)
    callback = make_longrun_callback(
        run_dir=tmp_path,
        config=config,
        sampler=sampler,
        status_store=status,
        contract_hash="a" * 64,
        evaluate_fn=lambda *_: report,
        reward_config=reward,
    )
    callback.restore_pipeline_state(
        state, resume_checkpoint=resume_checkpoint
    )

    restored = status.data
    assert restored["total_timesteps"] == 232_382
    assert restored["curriculum_stage"] == "C1"
    assert len(restored["curriculum_stage_history"]) == 3
    assert restored["stage_entry_timesteps"] == 225_280
    assert restored["next_promotion_eligibility_step"] == 275_280
    assert restored["consecutive_reliable_full_evaluations"] == 2
    assert restored["evaluation_history_count"] == 1
    assert restored["latest_full_evaluation"].endswith(
        "full_000225280.json"
    )
    assert restored["rollback_count"] == 1
    assert restored["last_rollback_reason"] == [
        "safety_episode_regression"
    ]
    assert restored["last_rollback_source"] == str(
        best_checkpoint.model_path
    )
    assert len(restored["rollback_history"]) == 1
    restored_rollback = restored["rollback_history"][0]
    assert all(
        restored_rollback[key] == value
        for key, value in rollback.items()
    )
    assert restored_rollback["source_checkpoint"] == rollback["source"]
    assert restored_rollback["stage_before"] == "C2"
    assert restored_rollback["stage_after"] == "C1"
    assert restored["best_reliable_checkpoint"] == str(
        best_checkpoint.model_path
    )
    assert restored["best_fast_reliable_checkpoint"] == str(
        best_checkpoint.model_path
    )
    assert restored["last_checkpoint"] == str(resume_checkpoint.model_path)
    assert restored["checkpoint_aliases"]["best_reliable"] == str(
        best_checkpoint.model_path
    )
    assert restored["current_learning_rate"] == pytest.approx(8.148608e-6)
    assert restored["reward_phase"] == "efficiency"
    assert restored["retention_state"]["calls"] == 113
    assert restored["retention_state"]["weight_multiplier"] == pytest.approx(
        1.25
    )
    assert restored["collision_events"] == 2
    assert restored["collision_frames"] == 9
    assert restored["out_of_bounds_events"] == 3
    assert restored["wrong_direction_count"] == 5
    assert all(value is not None for value in (
        restored["last_checkpoint"],
        restored["latest_full_evaluation"],
        restored["best_reliable_checkpoint"],
        restored["best_fast_reliable_checkpoint"],
        restored["last_rollback_reason"],
        restored["last_rollback_source"],
        restored["checkpoint_aliases"],
        restored["retention_state"],
    ))
    assert reward.reward_phase == "efficiency"
    assert callback.rollback_count == 1
    assert callback.eval_history == [report]


def test_auxiliary_losses_are_logged_separately_by_reliability_ppo():
    gym = pytest.importorskip("gymnasium")
    np = pytest.importorskip("numpy")
    import torch.nn as nn

    from marine_race_arena.learning.reliability_first import ReliabilityFirstPPO

    class TinyEnv(gym.Env):
        observation_space = gym.spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            return np.zeros(3, np.float32), {}

        def step(self, action):
            return np.zeros(3, np.float32), 0.0, False, False, {}

    class FakeRegularizer:
        kl_weight = 0.02

        @staticmethod
        def apply(model, *, timesteps, stage):
            return {
                "bc_retention_loss": 0.25,
                "initial_policy_kl": 0.05,
                "retention_weight": 0.10,
            }

    model = ReliabilityFirstPPO(
        "MlpPolicy",
        TinyEnv(),
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        learning_rate=1e-5,
        policy_kwargs={
            "net_arch": {"pi": [8, 8], "vf": [8, 8]},
            "activation_fn": nn.Tanh,
        },
        verbose=0,
    )
    model.attach_retention_regularizer(FakeRegularizer(), lambda: "C0")
    model.learn(total_timesteps=8)
    assert model.logger.name_to_value["train/bc_retention_loss"] == pytest.approx(0.25)
    assert model.logger.name_to_value["train/initial_policy_kl"] == pytest.approx(0.05)
    assert "train/ppo_policy_loss" in model.logger.name_to_value
    assert "train/combined_policy_loss" in model.logger.name_to_value
    model.env.close()


def test_protected_model_hashes_are_unchanged():
    from marine_race_arena.learning.longrun_checkpoint import sha256_file

    protected = {
        "results/rl_public/stage1/bc/model/best_model.pt": (
            "fd6fc7e6fba9b88ccc84fa76275d72ce3c907c75d632367995cb5257ae71d5d3"
        ),
        "results/rl/multigate_v3/r1/ppo_multigate_v3/20260727_133031/"
        "best_model/best_model.zip": (
            "de7e835132ee57fbe94ee3a2388f0f7554fd6bf41228c8e048000c61fd5b0b6b"
        ),
        "results/rl/multigate_longrun/bc_v3_balanced_v2_20260728/bc_v3.pt": (
            "c9a889125278122b57307e21b7c2390baa61429f8952093bcc2760d9b3a49bf2"
        ),
    }
    assert {path: sha256_file(path) for path in protected} == protected


def test_reliability_seed_namespaces_are_pairwise_disjoint():
    assert_pairwise_disjoint()
    roles = [
        set(MULTIGATE_RELIABILITY_PPO_TRAINING_SEEDS),
        set(MULTIGATE_RELIABILITY_DEV_EVAL_SEEDS),
        set(MULTIGATE_RELIABILITY_CHECKPOINT_SELECTION_SEEDS),
        set(MULTIGATE_RELIABILITY_VALIDATION_SEEDS),
    ]
    assert 23001 in roles[0]
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            assert left.isdisjoint(right)
