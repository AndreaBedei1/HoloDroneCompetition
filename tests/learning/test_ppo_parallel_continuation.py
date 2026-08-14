"""Worker resharding, continuation parity and preserved learning state."""

from __future__ import annotations

import json

import gymnasium as gym
import numpy as np
import pytest

from marine_race_arena.learning.config_local_transition import (
    OBS_DIM_LOCAL_TRANSITION,
)
from marine_race_arena.learning.longrun_checkpoint import (
    atomic_save_checkpoint, sha256_file,
)
from marine_race_arena.learning.train_ppo_transition import (
    BASELINE_ROLLOUT_SIZE,
    MINIBATCH_ALIGNMENT,
    ROLLOUT_SIZE_TOLERANCE,
    _continue_from_parent,
    _inherit_parent_history,
    _load_config,
    _parent_checkpoint,
    reshard_workers,
    supported_worker_shardings,
)
from marine_race_arena.learning.train_multigate_longrun import (
    AbsoluteLearningRateSchedule,
)
from marine_race_arena.learning.transition_curriculum import (
    TransitionCurriculumController,
)
from marine_race_arena.learning.transition_policy import build_local_transition_ppo


class _TinyEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(
            -1.0, 1.0, (OBS_DIM_LOCAL_TRANSITION,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return (
            np.zeros(self.observation_space.shape, dtype=np.float32),
            0.0, False, True, {},
        )


def _vec(n_envs):
    from stable_baselines3.common.vec_env import DummyVecEnv

    return DummyVecEnv([_TinyEnv for _ in range(n_envs)])


# ------------------------------------------------------------------ resharding


def test_every_supported_sharding_keeps_the_effective_rollout():
    shardings = supported_worker_shardings()
    assert set(shardings) >= {2, 4, 6, 8, 10}
    low = BASELINE_ROLLOUT_SIZE * (1 - ROLLOUT_SIZE_TOLERANCE)
    high = BASELINE_ROLLOUT_SIZE * (1 + ROLLOUT_SIZE_TOLERANCE)
    for workers, steps in shardings.items():
        rollout = workers * steps
        assert low <= rollout <= high, (workers, steps, rollout)
        assert steps % MINIBATCH_ALIGNMENT == 0
        # Same minibatch count per rollout as the parent run.
        assert rollout % 256 == 0


def test_documented_shardings_are_the_expected_ones():
    shardings = supported_worker_shardings()
    assert shardings[2] == 1024 and 2 * shardings[2] == 2048
    assert shardings[4] == 512 and 4 * shardings[4] == 2048
    assert shardings[6] == 384 and 6 * shardings[6] == 2304
    assert shardings[8] == 256 and 8 * shardings[8] == 2048
    assert shardings[10] == 256 and 10 * shardings[10] == 2560


def test_reshard_rejects_unsupported_worker_counts():
    config = {"n_envs": 2, "ppo": {"n_steps": 1024}}
    reshard_workers(config, 8)
    assert config["n_envs"] == 8 and config["ppo"]["n_steps"] == 256
    with pytest.raises(ValueError, match="no sharding"):
        reshard_workers(config, 3)


def test_config_rejects_a_rollout_far_from_the_baseline(tmp_path):
    base = json.loads(
        open("configs/rl/ppo_universal_transition_warm_reliability.json").read()
    )
    base["n_envs"] = 8
    base["ppo"]["n_steps"] = 1024  # 8192 transitions -> 4x the baseline
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ValueError, match="rollout size"):
        _load_config(path)


# --------------------------------------------------------------- continuation


def _parent_run(tmp_path, timesteps=8192, n_envs=2, n_steps=1024):
    env = _vec(n_envs)
    schedule = AbsoluteLearningRateSchedule(6e-6, 1e-6, "linear")
    model = build_local_transition_ppo(
        env, seed=11, learning_rate=schedule, hidden_sizes=(32, 32),
        n_steps=n_steps, batch_size=256, n_epochs=3,
    )
    model.num_timesteps = timesteps
    model._n_updates = 42
    model._current_progress_remaining = 0.37
    for group in model.policy.optimizer.param_groups:
        group["lr"] = 4.2e-6
    curriculum = TransitionCurriculumController()
    run = tmp_path / "parent"
    atomic_save_checkpoint(
        model, run, total_timesteps=timesteps, config_contract_hash="a" * 64,
        curriculum_state=curriculum.state_dict([
            {"schema_version": "universal_transition_worker_v1", "worker_id": i,
             "sampler_seed": 100 + i, "episode_counter": 3,
             "sampler": {"rng": f"w{i}"}} for i in range(n_envs)
        ]),
        evaluation_state={
            "history": [{"timesteps": timesteps, "metrics": {"universal_transition_success_rate": 0.84}}],
            "last": {"timesteps": timesteps}, "best": {"timesteps": timesteps},
        },
        status="unverified", reason="pre_evaluation_atomic_boundary",
        extra_state={
            "checkpoint_aliases": {
                "last": "parent_last.zip",
                "latest_competent": "parent_competent.zip",
                "best_universal_transition": "parent_best.zip",
                "best_long_sequence": "parent_long.zip",
            },
            "checkpoint_selection_state": {
                "best_metrics": {"universal_transition_success_rate": 0.84},
                "best_timestep": timesteps,
                "best_long_sequence_score": 0.5,
                "consecutive_baseline_collapses": 0,
                "dedicated": {"last_timesteps": 5000, "best_success_rate": 0.81,
                              "history": [{"timesteps": 5000}]},
            },
        },
    )
    env.close()
    return run, model


def _continuation_config(run, n_envs, n_steps, timesteps):
    return {
        "n_envs": n_envs,
        "new_environment_steps": 1_000_000,
        "ppo": {"n_steps": n_steps, "batch_size": 256, "n_epochs": 3,
                "hidden_sizes": [32, 32], "learning_rate": 6e-6,
                "final_learning_rate": 1e-6, "learning_rate_schedule": "linear",
                "gamma": 0.995, "gae_lambda": 0.95, "clip_range": 0.05,
                "target_kl": 0.004, "ent_coef": 0.0002},
        "initialization": {
            "mode": "parallel_continuation",
            "parent_run_dir": str(run),
            "parent_checkpoint": f"ppo_{timesteps}_steps.zip",
            "parent_n_envs": 2, "parent_n_steps": 1024,
            "reshard_reason": "measured throughput",
        },
    }


def test_continuation_preserves_weights_optimizer_scheduler_and_counter(tmp_path):
    run, parent = _parent_run(tmp_path)
    config = _continuation_config(run, 8, 256, 8192)
    env = _vec(8)
    try:
        model, report, state = _continue_from_parent(config, env, tmp_path / "child")
    finally:
        pass
    assert report["mode"] == "parallel_continuation"
    assert report["total_environment_transitions_preserved"] == 8192
    assert model.num_timesteps == 8192
    assert model._n_updates == 42
    # A continuation is NOT an extension: it gets its own learning-rate policy
    # and its own horizon.  The parent's rate (4.2e-6 here) must not survive,
    # otherwise a run configured to re-warm silently trains at the old rate --
    # PPO v4 advertised 7e-06 and actually ran at 3.489754098360656e-06.
    # Optimizer *moments* and the transition counter are still inherited.
    assert model._current_progress_remaining == pytest.approx(1.0)
    configured = float(config["ppo"]["learning_rate"])
    assert model.policy.optimizer.param_groups[0]["lr"] == pytest.approx(configured)
    assert all(
        g["lr"] == pytest.approx(configured)
        for g in model.policy.optimizer.param_groups
    )
    assert report["restored_parent_lr"] == pytest.approx(4.2e-6)
    assert report["effective_optimizer_lr"] == pytest.approx(configured)
    assert report["lr_rewarm_verified"] is True
    assert report["optimizer_state_preserved"] is True
    assert report["rollout_buffer_reset"] is True
    assert report["previous_n_envs"] == 2 and report["n_envs"] == 8
    assert report["rollout_size"] == 2048
    for name, value in parent.policy.state_dict().items():
        assert np.allclose(value.numpy(), model.policy.state_dict()[name].numpy())
    env.close()


def test_continuation_action_and_value_parity_with_the_parent(tmp_path):
    import torch

    run, parent = _parent_run(tmp_path)
    config = _continuation_config(run, 8, 256, 8192)
    env = _vec(8)
    model, _, _ = _continue_from_parent(config, env, tmp_path / "child")
    rng = np.random.default_rng(7)
    obs = rng.uniform(-1, 1, size=(512, OBS_DIM_LOCAL_TRANSITION)).astype(np.float32)
    parent_actions, _ = parent.predict(obs, deterministic=True)
    child_actions, _ = model.predict(obs, deterministic=True)
    assert np.array_equal(parent_actions, child_actions)
    with torch.no_grad():
        tensor = torch.as_tensor(obs)
        assert torch.allclose(
            parent.policy.predict_values(tensor), model.policy.predict_values(tensor)
        )
    env.close()


def test_continuation_rebuilds_the_rollout_buffer_for_the_new_sharding(tmp_path):
    run, _ = _parent_run(tmp_path)
    config = _continuation_config(run, 8, 256, 8192)
    env = _vec(8)
    model, _, _ = _continue_from_parent(config, env, tmp_path / "child")
    assert model.n_steps == 256
    assert model.rollout_buffer.n_envs == 8
    assert model.rollout_buffer.buffer_size == 256
    assert model.rollout_buffer.buffer_size * model.rollout_buffer.n_envs == 2048
    assert model.rollout_buffer.pos == 0 and not model.rollout_buffer.full
    env.close()


def test_continuation_inherits_ranking_curriculum_and_evaluation_history(tmp_path):
    run, _ = _parent_run(tmp_path)
    config = _continuation_config(run, 4, 512, 8192)
    env = _vec(4)
    _, _, state = _continue_from_parent(config, env, tmp_path / "child")
    curriculum = TransitionCurriculumController()
    aliases = {k: None for k in (
        "last", "latest_competent", "latest_safe_competent",
        "best_universal_transition", "best_long_sequence")}
    selection = {"best_metrics": None, "best_timestep": None,
                 "last_evaluation_metrics": None, "last_competence_verdict": None,
                 "best_long_sequence_score": None,
                 "consecutive_baseline_collapses": 0,
                 "dedicated": {"last_timesteps": 0, "best_success_rate": None,
                               "history": []},
                 "rollback": None}
    evaluation = {"history": [], "last": None, "best": None}
    _inherit_parent_history(state, curriculum=curriculum, aliases=aliases,
                            selection=selection, evaluation_state=evaluation)
    assert aliases["best_universal_transition"] == "parent_best.zip"
    assert aliases["best_long_sequence"] == "parent_long.zip"
    assert selection["best_timestep"] == 8192
    assert selection["dedicated"]["best_success_rate"] == pytest.approx(0.81)
    assert len(evaluation["history"]) == 1
    assert evaluation["best"]["timesteps"] == 8192
    env.close()


def test_parent_checkpoint_hash_is_verified(tmp_path):
    run, _ = _parent_run(tmp_path)
    config = _continuation_config(run, 4, 512, 8192)
    path, manifest = _parent_checkpoint(config)
    assert sha256_file(path) == manifest["model_sha256"]
    config["initialization"]["parent_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash does not match the config"):
        _parent_checkpoint(config)


def test_continuation_refuses_a_missing_parent(tmp_path):
    config = _continuation_config(tmp_path / "nope", 4, 512, 8192)
    with pytest.raises(FileNotFoundError):
        _parent_checkpoint(config)


# --------------------------------------------------------- worker-death safety


class _DeadWorkerVecEnv:
    """SubprocVecEnv-like object whose close() never returns, as when a worker dies."""

    processes = ()

    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        import time as _time

        _time.sleep(30)


class _RaisingVecEnv:
    processes = ()

    def close(self):
        raise EOFError("pipe terminated")


def test_close_is_bounded_when_a_worker_pipe_is_dead():
    from marine_race_arena.learning.train_ppo_transition import close_vec_env_safely
    import time as _time

    env = _DeadWorkerVecEnv()
    started = _time.perf_counter()
    outcome = close_vec_env_safely(env, timeout=1.0)
    elapsed = _time.perf_counter() - started
    assert elapsed < 15.0, "a dead worker must not block the trainer for minutes"
    assert outcome["method"] == "forced"


def test_close_never_masks_the_original_failure():
    from marine_race_arena.learning.train_ppo_transition import close_vec_env_safely

    outcome = close_vec_env_safely(_RaisingVecEnv(), timeout=5.0)
    assert outcome["method"] == "forced"
    assert "EOFError" in outcome["error"]


def test_close_of_a_healthy_env_is_graceful():
    from marine_race_arena.learning.train_ppo_transition import close_vec_env_safely

    class _Healthy:
        processes = ()

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    env = _Healthy()
    outcome = close_vec_env_safely(env, timeout=5.0)
    assert env.closed and outcome["closed"] and outcome["method"] == "graceful"


def test_supported_worker_counts_are_selectable_on_the_command_line():
    from marine_race_arena.learning.train_ppo_transition import main, supported_worker_shardings

    with pytest.raises(SystemExit):
        main(["train", "--config", "x.json", "--n-envs", "3"])
    assert 8 in supported_worker_shardings()
