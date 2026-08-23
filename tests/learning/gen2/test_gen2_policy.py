"""Recurrent-policy semantics, BC training and BC -> PPO transfer parity."""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
th = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
pytest.importorskip("sb3_contrib")

from marine_race_arena.learning.config import ACTION_DIM
from marine_race_arena.learning.config_local_transition import OBS_DIM_LOCAL_TRANSITION
from marine_race_arena.learning.gen2.bc_recurrent import (
    BCConfig,
    calibrate_action_std,
    split_episodes,
    train_recurrent_bc,
)
from marine_race_arena.learning.gen2.dataset import Gen2Episode
from marine_race_arena.learning.gen2.recurrent_policy import (
    GEN2_ARCH,
    Gen2ObsEncoder,
    Gen2RecurrentController,
    build_gen2_policy_for_training,
    policy_parameter_count,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _episode(seed: int, steps: int, rng) -> Gen2Episode:
    obs = rng.normal(0.0, 0.3, (steps, OBS_DIM_LOCAL_TRANSITION)).astype(np.float32)
    # A target that depends on history, so a memoryless policy cannot fit it.
    running = np.cumsum(obs[:, 0]) / np.arange(1, steps + 1)
    actions = np.stack(
        [np.tanh(running), np.tanh(obs[:, 1]), np.tanh(0.5 * running), np.tanh(obs[:, 2])],
        axis=1,
    ).astype(np.float32)
    return Gen2Episode(
        seed=seed, observations=obs, expert_actions=actions,
        applied_actions=actions, applied_by_expert=np.ones(steps, bool),
        gate_crossings=np.zeros(steps, np.int16),
        meta={"seed": seed, "gate_count": 3, "completed": True, "gates_completed": 3},
    )


@pytest.fixture(scope="module")
def episodes():
    rng = np.random.default_rng(7)
    return [_episode(40_000 + k, int(rng.integers(30, 80)), rng) for k in range(24)]


# ------------------------------------------------------- architecture shape

def test_architecture_is_encoder_then_lstm_then_heads():
    model = build_gen2_policy_for_training(seed=0)
    policy = model.policy
    assert isinstance(policy.features_extractor, Gen2ObsEncoder)
    # The encoder must run BEFORE the LSTM: the LSTM input width is the
    # encoder's output width, not the raw 35-feature observation.
    assert policy.lstm_actor.input_size == GEN2_ARCH.encoder_hidden[-1]
    assert policy.lstm_actor.hidden_size == GEN2_ARCH.lstm_hidden_size
    assert policy.lstm_actor.num_layers == GEN2_ARCH.n_lstm_layers
    assert policy.action_net.out_features == ACTION_DIM


def test_parameter_count_stays_modest():
    counts = policy_parameter_count(build_gen2_policy_for_training(seed=0))
    assert counts["total"] == counts["trainable"]
    assert 150_000 < counts["total"] < 1_000_000, counts


def test_encoder_rejects_a_wrong_width_contract():
    from gymnasium import spaces

    with pytest.raises(ValueError):
        Gen2ObsEncoder(spaces.Box(low=-1, high=1, shape=(34,), dtype=np.float32))


def test_normalization_travels_inside_the_checkpoint(tmp_path):
    from sb3_contrib import RecurrentPPO

    model = build_gen2_policy_for_training(seed=0)
    mean = np.linspace(-1, 1, OBS_DIM_LOCAL_TRANSITION).astype(np.float32)
    std = np.full(OBS_DIM_LOCAL_TRANSITION, 0.5, np.float32)
    model.policy.features_extractor.set_normalization(mean, std)
    path = tmp_path / "policy.zip"
    model.save(path)
    reloaded = RecurrentPPO.load(path, device="cpu")
    assert np.allclose(reloaded.policy.features_extractor.obs_mean.numpy(), mean)
    assert np.allclose(reloaded.policy.features_extractor.obs_std.numpy(), std)


# ------------------------------------------------- recurrent state semantics

def test_hidden_state_resets_at_the_episode_boundary():
    controller = Gen2RecurrentController(build_gen2_policy_for_training(seed=0))
    obs = np.random.default_rng(0).random(OBS_DIM_LOCAL_TRANSITION).astype(np.float32)
    first = controller.act(obs, first_step=True)
    controller.act(obs)
    controller.act(obs)
    controller.reset()
    again = controller.act(obs, first_step=True)
    assert np.allclose(first, again), "a reset controller must reproduce its first action"


def test_hidden_state_is_preserved_between_steps():
    controller = Gen2RecurrentController(build_gen2_policy_for_training(seed=0))
    obs = np.random.default_rng(1).random(OBS_DIM_LOCAL_TRANSITION).astype(np.float32)
    controller.act(obs, first_step=True)
    state_one = controller.recurrent_state[0].copy()
    controller.act(obs)
    state_two = controller.recurrent_state[0]
    assert not np.allclose(state_one, state_two), "the LSTM state must advance"


def test_memory_actually_changes_the_action():
    """Same observation, different history -> different action.

    If this passes trivially the policy is behaving like a feed-forward net and
    the whole Gen-2 hypothesis is untestable.
    """
    controller = Gen2RecurrentController(build_gen2_policy_for_training(seed=3))
    rng = np.random.default_rng(2)
    probe = rng.random(OBS_DIM_LOCAL_TRANSITION).astype(np.float32)
    fresh = controller.act(probe, first_step=True)
    controller.reset()
    controller.act(rng.random(OBS_DIM_LOCAL_TRANSITION).astype(np.float32), first_step=True)
    for _ in range(6):
        controller.act(rng.random(OBS_DIM_LOCAL_TRANSITION).astype(np.float32))
    with_history = controller.act(probe)
    assert not np.allclose(fresh, with_history, atol=1e-7)


def test_deterministic_replay_is_reproducible():
    rng = np.random.default_rng(5)
    stream = rng.random((12, OBS_DIM_LOCAL_TRANSITION)).astype(np.float32)
    model = build_gen2_policy_for_training(seed=11)

    def rollout():
        controller = Gen2RecurrentController(model, deterministic=True)
        return np.asarray([
            controller.act(obs, first_step=(i == 0)) for i, obs in enumerate(stream)
        ])

    assert np.allclose(rollout(), rollout())


def test_controller_rejects_a_wrong_width_observation():
    controller = Gen2RecurrentController(build_gen2_policy_for_training(seed=0))
    with pytest.raises(ValueError):
        controller.act(np.zeros(34, np.float32), first_step=True)


# ------------------------------------------------------- behaviour cloning

def test_bc_splits_by_episode_never_by_step(episodes):
    train, validation = split_episodes(episodes, 0.25, seed=0)
    assert train and validation
    assert len(train) + len(validation) == len(episodes)
    assert {e.seed for e in train}.isdisjoint({e.seed for e in validation})


def test_recurrent_bc_reduces_validation_error(episodes):
    model = build_gen2_policy_for_training(seed=0)
    result = train_recurrent_bc(
        model, episodes,
        BCConfig(epochs=8, batch_episodes=8, chunk_length=32, patience=8, seed=1),
    )
    assert result.history, "BC produced no history"
    assert result.validation["mse"] < result.history[0]["validation_mse"]
    assert result.validation["samples"] > 0


def test_bc_latches_normalization_from_the_corpus(episodes):
    model = build_gen2_policy_for_training(seed=0)
    before = model.policy.features_extractor.obs_mean.numpy().copy()
    train_recurrent_bc(
        model, episodes,
        BCConfig(epochs=1, batch_episodes=8, chunk_length=32, patience=1, seed=0),
    )
    after = model.policy.features_extractor.obs_mean.numpy()
    assert not np.allclose(before, after), "normalization was never latched"


def test_action_std_is_sized_from_bc_residuals(episodes):
    model = build_gen2_policy_for_training(seed=0)
    train_recurrent_bc(
        model, episodes,
        BCConfig(epochs=4, batch_episodes=8, chunk_length=32, patience=4, seed=0),
    )
    std = calibrate_action_std(model, episodes)
    assert std.shape == (ACTION_DIM,)
    # Default log_std=0 means std 1.0 next to actions bounded in [-1, 1]; PPO
    # would drown the cloned behaviour on its first update.
    assert float(std.max()) < 0.5
    assert np.allclose(np.exp(model.policy.log_std.detach().numpy()), std, atol=1e-5)


# ------------------------------------------------------- BC -> PPO transfer

def test_bc_to_ppo_transfer_is_an_identity(tmp_path, episodes):
    """Gen-1 lost time to hand-folded BC weights disagreeing with the PPO net.

    Gen-2 trains the real RecurrentPPO policy, so 'transfer' is save/load and
    parity must be exact, not approximate.
    """
    from sb3_contrib import RecurrentPPO

    model = build_gen2_policy_for_training(seed=0)
    train_recurrent_bc(
        model, episodes,
        BCConfig(epochs=3, batch_episodes=8, chunk_length=32, patience=3, seed=0),
    )
    stream = np.random.default_rng(9).random((10, OBS_DIM_LOCAL_TRANSITION)).astype(np.float32)

    def rollout(m):
        controller = Gen2RecurrentController(m, deterministic=True)
        return np.asarray([
            controller.act(obs, first_step=(i == 0)) for i, obs in enumerate(stream)
        ])

    before = rollout(model)
    path = tmp_path / "bc_policy.zip"
    model.save(path)
    reloaded = RecurrentPPO.load(path, device="cpu")
    after = rollout(reloaded)
    assert np.allclose(before, after, atol=1e-6), "BC -> PPO transfer is not parity"


def test_ppo_setup_does_not_reset_the_transferred_policy(tmp_path, episodes):
    """``_setup_model`` re-initialises the network; loading must not call it."""
    from sb3_contrib import RecurrentPPO

    model = build_gen2_policy_for_training(seed=0)
    train_recurrent_bc(
        model, episodes,
        BCConfig(epochs=3, batch_episodes=8, chunk_length=32, patience=3, seed=0),
    )
    reference = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
    path = tmp_path / "policy.zip"
    model.save(path)
    reloaded = RecurrentPPO.load(path, device="cpu")
    for key, value in reloaded.policy.state_dict().items():
        assert th.allclose(value, reference[key], atol=1e-7), f"parameter {key} changed on load"


def test_learning_rate_survives_a_resume(tmp_path):
    from sb3_contrib import RecurrentPPO

    model = build_gen2_policy_for_training(seed=0, learning_rate=7.5e-5)
    path = tmp_path / "policy.zip"
    model.save(path)
    reloaded = RecurrentPPO.load(path, device="cpu")
    assert reloaded.lr_schedule(1.0) == pytest.approx(7.5e-5)
    for group in reloaded.policy.optimizer.param_groups:
        assert group["lr"] == pytest.approx(7.5e-5)


# ------------------------------------------------- fine-tuning must not harm

def test_a_warm_start_never_returns_a_policy_worse_than_its_parent(episodes):
    """The run keeps the parent's weights unless an epoch actually beats them.

    A from-scratch learning rate destroyed a converged policy in one epoch --
    8.3e-05 to 2.65e-03 -- and eight more epochs never recovered. Seeding the
    best-state with the parent means the worst case is 'no change', not
    'catastrophe'.
    """
    model = build_gen2_policy_for_training(seed=0)
    result = train_recurrent_bc(
        model, episodes,
        BCConfig(epochs=2, batch_episodes=8, chunk_length=32, patience=2,
                 seed=0, learning_rate=5.0),  # absurd rate, guaranteed to diverge
    )
    assert result.history[0]["epoch"] == -1, "the parent fit must be recorded first"
    assert result.improved_on_parent is not None
    # Whatever happened during training, the returned policy is never worse
    # than what it started from.
    assert result.validation["mse"] <= result.initial_validation_mse + 1e-9


def test_the_parent_fit_is_recorded_before_any_update(episodes):
    model = build_gen2_policy_for_training(seed=0)
    result = train_recurrent_bc(
        model, episodes,
        BCConfig(epochs=1, batch_episodes=8, chunk_length=32, patience=1, seed=0),
    )
    assert result.history[0]["epoch"] == -1
    assert result.history[0]["note"].startswith("parent policy")
    assert result.initial_train_mse > 0
    assert result.initial_validation_mse > 0


def test_a_warm_start_drops_to_the_finetune_learning_rate(tmp_path, episodes):
    """A warm start must not silently keep the from-scratch rate."""
    from marine_race_arena.learning.gen2.train_bc import FINETUNE_LEARNING_RATE, train

    model = build_gen2_policy_for_training(seed=0)
    parent = tmp_path / "parent.zip"
    model.save(parent)

    corpus = tmp_path / "corpus"
    from marine_race_arena.learning.gen2.dataset import save_episode, write_manifest

    for episode in episodes[:12]:
        save_episode(episode, corpus)
    write_manifest(corpus)

    summary = train(
        [corpus], tmp_path / "out",
        config=BCConfig(epochs=1, batch_episodes=6, chunk_length=32, patience=1, seed=0),
        init_from=parent, evaluate=False,
    )
    assert summary["bc"]["config"]["learning_rate"] == pytest.approx(FINETUNE_LEARNING_RATE)
    assert FINETUNE_LEARNING_RATE < BCConfig().learning_rate
