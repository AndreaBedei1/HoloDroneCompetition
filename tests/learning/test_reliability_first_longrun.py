"""Reliability-first PPO invariants that do not require HoloOcean."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from marine_race_arena.learning.longrun_config import LongRunConfig
from marine_race_arena.learning.longrun_evaluation import (
    aggregate_evaluation,
    checkpoint_metric_key,
    fast_reliable_metric_key,
)
from marine_race_arena.learning.longrun_monitor import (
    evaluation_regression_reasons,
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


def test_failed_full_suite_resets_consecutive_promotion_streak():
    sampler = _sampler(_config())
    sampler.record_evaluation(_passing_report(), timesteps=25_000)
    sampler.record_evaluation(
        _passing_report(right_completion_rate=0.8), timesteps=50_000
    )
    assert sampler.current_stage == "C0"
    assert sampler.state.consecutive_full_passes == 0


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
    assert regularizer.effective_weight(0, "C0") == pytest.approx(0.10)
    assert regularizer.effective_weight(75_000, "C2") == pytest.approx(0.05)
    assert regularizer.effective_weight(150_000, "C2") == 0
    assert regularizer.effective_weight(1, "C3") == 0
    source = Path(
        "marine_race_arena/learning/reliability_first.py"
    ).read_text(encoding="utf-8")
    assert "RuleGateCenterThenCommitController" not in source
    assert "ControllerLoader" not in source


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
