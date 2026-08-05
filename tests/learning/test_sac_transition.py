from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from marine_race_arena.learning.config import ACTION_DIM
from marine_race_arena.learning.config_local_transition import (
    FEATURE_NAMES_LOCAL_TRANSITION,
    OBS_DIM_LOCAL_TRANSITION,
)
from marine_race_arena.learning.sac_replay_buffer import (
    DEFAULT_BATCH_COMPOSITION,
    EVENT_CATEGORIES,
    MultiWorkerNStepAccumulator,
    NStepTransition,
    StratifiedReplayBuffer,
    classify_replay_event,
    is_legal_bootstrap_truncation,
)
from marine_race_arena.learning.sac_transition_checkpoint import (
    atomic_save_sac_checkpoint,
    load_sac_checkpoint,
    validate_sac_checkpoint,
)
from marine_race_arena.learning.sac_transition_policy import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    SACTransitionAgent,
    SquashedGaussianActor,
    compute_sac_bootstrap_target,
    transfer_ppo_mean_actor,
)
from marine_race_arena.learning.train_ppo_transition import apply_evaluation_selection
from marine_race_arena.learning.transition_selection import (
    DEFAULT_COMPETENCE_THRESHOLDS,
)


def _raw(
    accumulator,
    worker,
    reward,
    *,
    terminated=False,
    truncated=False,
    bootstrap=True,
    category="general",
    value=0.0,
):
    return accumulator.append(
        worker,
        observation=np.full(OBS_DIM_LOCAL_TRANSITION, value, np.float32),
        action=np.full(ACTION_DIM, value, np.float32),
        reward=reward,
        next_observation=np.full(OBS_DIM_LOCAL_TRANSITION, value + 1, np.float32),
        terminated=terminated,
        truncated=truncated,
        bootstrap_allowed=bootstrap,
        event_category=category,
    )


def _transition(category="general", value=0.0):
    return NStepTransition(
        observation=np.full(OBS_DIM_LOCAL_TRANSITION, value, np.float32),
        action=np.full(ACTION_DIM, value, np.float32),
        reward=float(value),
        next_observation=np.full(OBS_DIM_LOCAL_TRANSITION, value + 1, np.float32),
        discount=0.9,
        event_category=category,
    )


def test_sac_observation_and_action_contract_is_exact():
    assert OBS_DIM_LOCAL_TRANSITION == 35
    forbidden = {
        "gate_index", "remaining_gates", "total_gates", "sequence_progress",
        "sequence_length", "lap", "tracker_phase", "referee_state",
        "collision_state", "true_vehicle_position", "true_gate_position",
    }
    assert forbidden.isdisjoint(FEATURE_NAMES_LOCAL_TRANSITION)
    actor = SquashedGaussianActor()
    assert actor.observation_dim == 35
    assert actor.action_dim == 4


def test_long_profile_records_capacity_selected_single_worker_layout():
    config = json.loads(Path(
        "configs/rl/sac_universal_transition_warm_long.json"
    ).read_text(encoding="utf-8"))
    assert config["n_envs"] == 1
    assert config["evaluation"]["intermediate_parallel_workers"] == 1
    assert config["evaluation"]["dedicated_parallel_workers"] == 1


def test_squashed_actor_bounds_and_conservative_state_dependent_std():
    import torch

    actor = SquashedGaussianActor(initial_std=0.075)
    observations = torch.randn(1024, 35) * 3
    actions, log_probability, means = actor.sample(observations)
    assert actions.shape == (1024, 4)
    assert log_probability.shape == (1024, 1)
    assert torch.all(actions < 1.0)
    assert torch.all(actions > -1.0)
    assert torch.all(means <= 1.0)
    assert torch.allclose(actor.log_std_head.weight, torch.zeros_like(actor.log_std_head.weight))
    assert torch.all(actor.log_std_head.bias >= LOG_STD_MIN)
    assert torch.all(actor.log_std_head.bias <= LOG_STD_MAX)
    assert np.isclose(float(actor.log_std_head.bias[0].exp().detach()), 0.075)


class _FakeObservationSpace:
    shape = (35,)


class _FakePPOPolicy:
    def __init__(self):
        import torch.nn as nn

        self.policy = nn.Sequential(nn.Linear(35, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh())
        self.action = nn.Linear(256, 4)

    def state_dict(self):
        return {
            "mlp_extractor.policy_net.0.weight": self.policy[0].weight,
            "mlp_extractor.policy_net.0.bias": self.policy[0].bias,
            "mlp_extractor.policy_net.2.weight": self.policy[2].weight,
            "mlp_extractor.policy_net.2.bias": self.policy[2].bias,
            "action_net.weight": self.action.weight,
            "action_net.bias": self.action.bias,
        }


class _FakePPO:
    observation_space = _FakeObservationSpace()

    def __init__(self):
        self.policy = _FakePPOPolicy()

    def predict(self, observations, deterministic=True):
        import torch

        with torch.no_grad():
            values = self.policy.action(
                self.policy.policy(torch.as_tensor(observations, dtype=torch.float32))
            ).clamp(-1, 1)
        return values.numpy(), None


def test_warm_actor_mean_parity_without_critic_or_optimizer_transfer(monkeypatch):
    from stable_baselines3 import PPO

    source = _FakePPO()
    monkeypatch.setattr(PPO, "load", lambda *args, **kwargs: source)
    agent = SACTransitionAgent()
    report = transfer_ppo_mean_actor(agent.actor, "fake.zip", validation_samples=512)
    assert report["passed"]
    assert report["maximum_absolute_action_error"] <= report["tolerance"]
    assert report["ppo_critic_weights_copied"] is False
    assert report["ppo_optimizer_state_copied"] is False
    assert report["ppo_rollout_data_imported"] is False


def test_twin_critic_target_uses_stored_discount_and_entropy():
    import torch

    reward = torch.tensor([[2.0]])
    discount = torch.tensor([[0.5]])
    minimum_q = torch.tensor([[4.0]])
    alpha = torch.tensor(0.2)
    log_probability = torch.tensor([[-1.5]])
    target = compute_sac_bootstrap_target(
        reward, discount, minimum_q, alpha, log_probability
    )
    assert target.item() == pytest.approx(4.15)
    terminal = compute_sac_bootstrap_target(
        reward, torch.zeros_like(discount), minimum_q, alpha, log_probability
    )
    assert terminal.item() == pytest.approx(2.0)


def test_entropy_tuning_and_losses_remain_finite():
    agent = SACTransitionAgent(hidden_sizes=(16, 16))
    before = agent.alpha
    rng = np.random.default_rng(7)
    batch = {
        "observations": rng.normal(size=(32, 35)).astype(np.float32),
        "actions": rng.uniform(-1, 1, size=(32, 4)).astype(np.float32),
        "rewards": rng.normal(size=32).astype(np.float32),
        "next_observations": rng.normal(size=(32, 35)).astype(np.float32),
        "discounts": np.full(32, 0.99, np.float32),
    }
    losses = agent.update(batch)
    assert agent.gradient_updates == 1
    assert all(np.isfinite(value) for value in losses.values())
    assert agent.alpha != before


def test_full_three_step_return_and_bootstrap_discount():
    accumulator = MultiWorkerNStepAccumulator(n_step=3, gamma=0.5, n_workers=1)
    assert _raw(accumulator, 0, 1.0) == []
    assert _raw(accumulator, 0, 2.0) == []
    rows = _raw(accumulator, 0, 4.0)
    assert len(rows) == 1
    assert rows[0].reward == pytest.approx(3.0)
    assert rows[0].discount == pytest.approx(0.125)


def test_one_and_two_step_shortened_returns_at_episode_end_without_bootstrap():
    accumulator = MultiWorkerNStepAccumulator(n_step=3, gamma=0.5, n_workers=1)
    _raw(accumulator, 0, 2.0)
    rows = _raw(accumulator, 0, 4.0, terminated=True, bootstrap=False)
    assert [row.reward for row in rows] == pytest.approx([4.0, 4.0])
    assert [row.discount for row in rows] == [0.0, 0.0]
    single = _raw(accumulator, 0, 7.0, terminated=True, bootstrap=False)
    assert single[0].reward == 7.0
    assert single[0].discount == 0.0


def test_legal_truncation_bootstraps_but_domain_failure_does_not():
    assert is_legal_bootstrap_truncation({"TimeLimit.truncated": True})
    assert not is_legal_bootstrap_truncation({
        "TimeLimit.truncated": True, "episode_acquisition_timeout": True
    })
    accumulator = MultiWorkerNStepAccumulator(n_step=3, gamma=0.5, n_workers=1)
    rows = _raw(
        accumulator, 0, 3.0, truncated=True, bootstrap=True
    )
    assert rows[0].discount == pytest.approx(0.5)


def test_worker_queues_are_independent_and_flush_loses_nothing():
    accumulator = MultiWorkerNStepAccumulator(n_step=3, gamma=0.9, n_workers=2)
    _raw(accumulator, 0, 1, value=10)
    _raw(accumulator, 1, 2, value=20)
    _raw(accumulator, 0, 3, value=11)
    rows = accumulator.flush_all(bootstrap_allowed=False)
    assert len(rows) == 3
    assert len(accumulator.queues[0]) == len(accumulator.queues[1]) == 0
    assert sorted(row.observation[0] for row in rows) == [10, 11, 20]


def test_n_step_state_restores_exactly():
    first = MultiWorkerNStepAccumulator(n_step=3, gamma=0.995, n_workers=2)
    _raw(first, 0, 1.5, category="crossing", value=3)
    _raw(first, 1, -2.0, category="collision_entry", value=4)
    second = MultiWorkerNStepAccumulator(n_step=3, gamma=0.995, n_workers=2)
    second.load_state_dict(first.state_dict())
    assert second.state_dict() == first.state_dict()


@pytest.mark.parametrize(
    "info,category",
    [
        ({"reward_components": {"collision_penalty": -50}}, "collision_entry"),
        ({"reward_components": {"gate_crossing": 25}}, "crossing"),
        ({"episode_acquisition_timeout": True}, "acquisition_timeout"),
        ({"episode_full_sequence_completion": True}, "successful_transition"),
    ],
)
def test_replay_event_classification(info, category):
    observation = np.zeros(35, np.float32)
    assert classify_replay_event(info, observation, observation) == category


def test_stratified_sampling_proportions_keep_general_component():
    replay = StratifiedReplayBuffer(capacity=1000, seed=31)
    for category in EVENT_CATEGORIES:
        for index in range(50):
            replay.add(_transition(category, index / 100))
    replay.sample(200)
    report = replay.sampling_report()
    assert report["actual_group_proportions"] == pytest.approx(DEFAULT_BATCH_COMPOSITION)
    assert report["actual_group_proportions"]["general"] >= 0.35


def test_replay_persistence_restores_data_indices_and_sampling_rng(tmp_path):
    replay = StratifiedReplayBuffer(capacity=50, seed=41)
    for index, category in enumerate(EVENT_CATEGORIES * 2):
        replay.add(_transition(category, index))
    path = tmp_path / "replay.npz"
    replay.save(path)
    restored = StratifiedReplayBuffer.load(path)
    first = replay.sample(20)
    second = restored.sample(20)
    assert np.array_equal(first["indices"], second["indices"])
    assert np.array_equal(first["event_categories"], second["event_categories"])
    assert restored.metadata_state() == replay.metadata_state()


def test_atomic_sac_checkpoint_restores_every_training_component(tmp_path):
    agent = SACTransitionAgent(hidden_sizes=(8, 8))
    replay = StratifiedReplayBuffer(capacity=100, seed=9)
    replay.add(_transition("crossing", 1))
    accumulator = MultiWorkerNStepAccumulator(n_step=3, gamma=0.995, n_workers=1)
    _raw(accumulator, 0, 2, category="post_target_switch")
    checkpoint = atomic_save_sac_checkpoint(
        agent, replay, tmp_path,
        total_environment_transitions=123,
        config_contract_sha256="a" * 64,
        curriculum_state={"schema_version": "test", "worker_states": []},
        n_step_state=accumulator.state_dict(),
        evaluation_state={"history": []},
        aliases={"last": "checkpoint", "latest_competent": None},
        selection_state={"rollback": None},
        initialization={"source_sha256": "b" * 64},
        worker_identities=[],
        reason="test",
    )
    valid = validate_sac_checkpoint(
        checkpoint.manifest_path, expected_contract_sha256="a" * 64
    )
    assert valid is not None
    loaded_agent, loaded_replay, state = load_sac_checkpoint(valid)
    assert loaded_agent.gradient_updates == 0
    assert loaded_replay.size == 1
    assert state["n_step"] == accumulator.state_dict()
    assert state["checkpoint_aliases"]["latest_competent"] is None
    assert not list((tmp_path / "checkpoints").glob("*.partial.*"))


def _competent_metrics(**overrides):
    value = {
        "n_eval": 100,
        "transition_n": 100,
        "first_gate_crossing_rate": 0.9,
        "target_switch_rate": 0.8,
        "universal_transition_success_rate": 0.5,
        "completed_gate_count": 100,
        "mean_completed_gates_per_episode": 1.0,
        "fraction_of_episodes_reaching_first_gate": 0.9,
        "mean_distance_travelled_m": 5.0,
        "mean_absolute_action": 0.1,
        "nontrivial_action_fraction": 0.9,
        "collision_episodes": 1,
        "collision_entries": 1,
        "collision_contact_frames": 100,
        "missed_gate_dnf": 0,
        "wrong_direction_events": 0,
        "out_of_bounds_episodes": 0,
        "previous_gate_returns": 0,
        "acquisition_timeouts": 0,
        "safety_clean": False,
        "long_sequence_completion_score": 0.2,
    }
    value.update(overrides)
    return value


def test_sac_aliases_require_competence_and_collapse_records_rollback():
    aliases = {name: None for name in (
        "last", "latest_competent", "latest_safe_competent",
        "best_universal_transition", "best_long_sequence",
    )}
    selection = {
        "best_metrics": None, "best_timestep": None,
        "last_evaluation_metrics": None, "last_competence_verdict": None,
        "best_long_sequence_score": None,
        "consecutive_baseline_collapses": 0,
        "dedicated": {}, "rollback": None,
    }
    inactive = _competent_metrics(
        completed_gate_count=0,
        mean_completed_gates_per_episode=0,
        fraction_of_episodes_reaching_first_gate=0,
        mean_distance_travelled_m=0,
        mean_absolute_action=0,
        nontrivial_action_fraction=0,
        first_gate_crossing_rate=0,
        target_switch_rate=0,
        universal_transition_success_rate=0,
    )
    apply_evaluation_selection(
        inactive, checkpoint_path="inactive.pt", timesteps=100,
        aliases=aliases, selection=selection,
        thresholds=DEFAULT_COMPETENCE_THRESHOLDS,
        rollback_limit=2,
        baseline_record={"baseline_checkpoint": "baseline.pt"},
    )
    assert aliases["latest_competent"] is None
    outcome = apply_evaluation_selection(
        inactive, checkpoint_path="inactive2.pt", timesteps=200,
        aliases=aliases, selection=selection,
        thresholds=DEFAULT_COMPETENCE_THRESHOLDS,
        rollback_limit=2,
        baseline_record={"baseline_checkpoint": "baseline.pt"},
    )
    assert outcome["collapsed"]
    assert selection["rollback"]["rollback_checkpoint"] == "baseline.pt"


def test_sac_training_loop_writes_atomic_checkpoint_without_ppo_data(monkeypatch, tmp_path):
    import argparse
    from marine_race_arena.learning import train_sac_transition as trainer

    class FakeVecEnv:
        num_envs = 1

        def __init__(self):
            self.steps = 0
            self.closed = False

        def reset(self):
            return np.zeros((1, 35), np.float32)

        def step(self, actions):
            self.steps += 1
            observation = np.full((1, 35), self.steps / 100, np.float32)
            done = self.steps % 3 == 0
            info = {
                "reward_components": {},
                "TimeLimit.truncated": done,
                "terminal_observation": observation[0].copy(),
            }
            return observation, np.asarray([0.1], np.float32), np.asarray([done]), [info]

        def env_method(self, name, *args, indices=None):
            if name == "worker_state":
                return [{
                    "schema_version": "universal_transition_worker_v1",
                    "worker_id": 0, "sampler_seed": 63001,
                    "episode_counter": self.steps,
                    "sampler": {},
                }]
            if name == "worker_identity":
                return [{
                    "worker_id": 0, "worker_pid": 123,
                    "worker_sampler_seed": 63001,
                    "holoocean_uuid": "fake-no-engine",
                }]
            return [None]

        def close(self):
            self.closed = True

    source_checkpoint = tmp_path / "source.zip"
    source_checkpoint.write_bytes(b"unit-test PPO actor placeholder")
    config = {
        "run_name": "unit", "output_root": str(tmp_path),
        "observation_version": "onboard_local_transition_v1",
        "action_version": "surge_sway_heave_yaw_pm1_v1",
        "initialization": {
            "mode": "ppo_actor_mean_warm_start",
            "source_checkpoint": str(source_checkpoint), "source_sha256": "a" * 64,
        },
        "seed": 1, "worker_seed_base": 63001, "torch_threads": 1,
        "new_environment_steps": 8, "n_envs": 1,
        "adapter": "holoocean", "allow_fallback": False,
        "holoocean_frames_per_sec": False, "max_episode_steps": 10,
        "sac": {
            "hidden_sizes": [8, 8], "actor_learning_rate": 3e-5,
            "critic_learning_rate": 1e-4, "entropy_learning_rate": 3e-5,
            "n_step": 3, "gamma": 0.995, "tau": 0.005,
            "batch_size": 4, "replay_capacity": 100,
            "learning_starts": 10000, "automatic_entropy": True,
            "target_entropy": -4, "initial_alpha": 0.02,
            "initial_std": 0.075, "collector_steps": 2,
            "updates_per_transition": 0.25,
            "replay_composition": DEFAULT_BATCH_COMPOSITION,
            "event_categories": list(EVENT_CATEGORIES),
        },
        "curriculum": {
            "initial_difficulty": "G1", "maximum_difficulty": "G6",
            "transition_focus_fraction": 0.7,
        },
        "competence_gate": {}, "reward": {},
        "evaluation": {
            "early_schedule": [1000], "frequency": 1000,
            "checkpoint_frequency": 4, "seed": 88001,
            "transition_cases": 2, "full_cases_per_length": 0,
            "rollback_consecutive_evaluations": 2,
            "dedicated_transition_cases": 2,
        },
    }
    monkeypatch.setattr(trainer, "_load_config", lambda path: config)
    monkeypatch.setattr(trainer, "_preflight", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(trainer, "_contract", lambda value: "c" * 64)
    monkeypatch.setattr(trainer, "_make_vec_env", lambda *args, **kwargs: FakeVecEnv())
    monkeypatch.setattr(
        trainer,
        "transfer_ppo_mean_actor",
        lambda *args, **kwargs: {"passed": True, "ppo_rollout_data_imported": False},
    )
    monkeypatch.setattr(
        trainer, "_preserve_baseline",
        lambda *args, **kwargs: {"baseline_checkpoint": "initial_actor.pt"},
    )
    args = argparse.Namespace(
        config="unused.json", run_dir=str(tmp_path / "run"), steps=None,
        n_envs=None, resume=False, allow_dirty_smoke=True,
    )
    assert trainer.run_training(args) == 0
    status = json.loads((tmp_path / "run" / "status.json").read_text())
    assert status["state"] == "completed"
    assert status["total_environment_transitions"] == 8
    assert status["replay_size"] == 8
    assert status["ppo_replay_imported"] is False
    manifests = list((tmp_path / "run" / "checkpoints").glob("*.manifest.json"))
    assert manifests
    assert all(validate_sac_checkpoint(path) is not None for path in manifests)
