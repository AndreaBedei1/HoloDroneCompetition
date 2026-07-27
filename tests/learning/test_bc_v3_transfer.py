"""Safe v1-to-v3 transfer and learned-only runtime controller tests."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from marine_race_arena.learning.bc_train import BCPolicy, save_policy
from marine_race_arena.learning.bc_v3_transfer import (
    expand_bc_v1_to_v3,
    load_v3_policy,
    save_v3_policy,
)
from marine_race_arena.learning.config import ACTION_AXES, OBS_DIM
from marine_race_arena.learning.config_v3 import OBS_DIM_V3
from marine_race_arena.learning.rl_multigate_controller import RLMultigateController
from marine_race_arena.participants.controller_loader import ControllerLoader


def _v1_policy(seed=11):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean = rng.normal(0.0, 0.2, OBS_DIM).astype(np.float32)
    std = rng.uniform(0.3, 1.5, OBS_DIM).astype(np.float32)
    return BCPolicy(hidden_sizes=(32, 32), obs_mean=mean, obs_std=std)


def test_v1_to_v3_neutral_and_nonneutral_feature_output_parity():
    v1 = _v1_policy()
    v3 = expand_bc_v1_to_v3(v1)
    rng = np.random.default_rng(4)
    for _ in range(20):
        base = rng.uniform(-1.0, 1.0, OBS_DIM).astype(np.float32)
        temporal = rng.uniform(-1.0, 1.0, OBS_DIM_V3 - OBS_DIM).astype(np.float32)
        expected = v1.act(base)
        actual = v3.act(np.concatenate((base, temporal)))
        np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_transfer_copies_first_36_columns_and_zeros_new_columns():
    v1 = _v1_policy()
    v3 = expand_bc_v1_to_v3(v1)
    source = next(module for module in v1.extractor if isinstance(module, torch.nn.Linear))
    target = next(module for module in v3.extractor if isinstance(module, torch.nn.Linear))
    torch.testing.assert_close(target.weight[:, :OBS_DIM], source.weight)
    assert torch.count_nonzero(target.weight[:, OBS_DIM:]).item() == 0
    torch.testing.assert_close(target.bias, source.bias)


def test_versioned_v3_save_load_and_v1_rejection(tmp_path):
    v3_path = tmp_path / "v3.pt"
    save_v3_policy(expand_bc_v1_to_v3(_v1_policy()), v3_path)
    loaded = load_v3_policy(v3_path)
    assert loaded.obs_dim == OBS_DIM_V3

    v1_path = tmp_path / "v1.pt"
    save_policy(_v1_policy(), v1_path)
    with pytest.raises(ValueError, match="incompatible observation"):
        load_v3_policy(v1_path)


def test_multigate_controller_outputs_all_axes_from_v3_policy(tmp_path):
    path = tmp_path / "v3.pt"
    save_v3_policy(expand_bc_v1_to_v3(_v1_policy()), path)
    controller = RLMultigateController(model_path=str(path))
    controller.reset(
        {
            "participant_id": "bluerov2_01",
            "initial_beacon_id": "B01",
            "total_beacons": 2,
            "laps": 1,
        }
    )
    command = controller.step(
        {
            "local_time_s": 0.0,
            "sensors": {
                "DepthSensor": [-3.0],
                "DVLSensor": [0.0, 0.0, 0.0],
                "IMUSensor": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            },
            "beacons": [],
        }
    )
    assert set(command) == set(ACTION_AXES)
    assert all(np.isfinite(value) and -1.0 <= value <= 1.0 for value in command.values())
    assert controller.last_encoded_observation.shape == (OBS_DIM_V3,)


def test_controller_contract_has_no_rule_actions_or_hybrid_blending():
    controller = RLMultigateController(model_path="unused")
    assert controller.rule_action_weight == 0.0
    assert controller.hybrid_blending is False
    assert controller.rule_controller_instantiated is False
    assert controller.deterministic_runtime_intervention_count == 0
    source = inspect.getsource(inspect.getmodule(RLMultigateController))
    assert "official_baselines" not in source
    assert "hybrid_gate_controller" not in source


def test_controller_loader_alias():
    assert isinstance(
        ControllerLoader().load("rl_multigate_controller"),
        RLMultigateController,
    )


def test_dagger_expert_is_explicitly_training_only():
    from marine_race_arena.learning.multigate_dagger_expert import (
        MultigateDAggerExpertController,
    )

    assert MultigateDAggerExpertController.training_only is True
    assert MultigateDAggerExpertController.debug_only is True
    assert "multigate_dagger_expert" in ControllerLoader.BUILT_INS


def test_ppo_v3_transfer_and_compatibility_stamp():
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("stable_baselines3")
    from gymnasium import spaces

    from marine_race_arena.learning.bc_v3_transfer import transfer_bc_v1_to_v3_ppo
    from marine_race_arena.learning.config import ACTION_DIM
    from marine_race_arena.learning.rl_train import build_ppo

    class DummyEnv(gym.Env):
        observation_space = spaces.Box(-1.0, 1.0, (OBS_DIM_V3,), np.float32)
        action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(OBS_DIM_V3, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(OBS_DIM_V3, dtype=np.float32), 0.0, False, True, {}

    v1 = _v1_policy()
    ppo = build_ppo(DummyEnv(), hidden_sizes=(32, 32), seed=5)
    transfer_bc_v1_to_v3_ppo(v1, ppo)
    assert ppo.obs_encoding_version == "onboard_multigate_rl_v3"

    rng = np.random.default_rng(9)
    for _ in range(5):
        base = rng.uniform(-1.0, 1.0, OBS_DIM).astype(np.float32)
        v3_obs = np.concatenate(
            (base, rng.uniform(-1.0, 1.0, OBS_DIM_V3 - OBS_DIM).astype(np.float32))
        )
        ppo_action, _ = ppo.predict(v3_obs, deterministic=True)
        np.testing.assert_allclose(ppo_action, v1.act(base), atol=1e-4)


def test_v3_zero_step_workflow_records_contract(tmp_path):
    pytest.importorskip("gymnasium")
    pytest.importorskip("stable_baselines3")
    import json

    from marine_race_arena.learning.config_v3 import OBS_ENCODING_VERSION_V3
    from marine_race_arena.learning.reward_v3 import MultiGateRewardConfig
    from marine_race_arena.learning.train_workflow import run_ppo_training

    source = tmp_path / "v1.pt"
    save_policy(_v1_policy(), source)
    run_dir = tmp_path / "run"
    path, model = run_ppo_training(
        "marine_race_arena/tracks/tests/two_gate_straight.json",
        total_timesteps=0,
        train_seed=20100,
        eval_seeds=[21000],
        run_dir=str(run_dir),
        bc_model_path=str(source),
        action_std_strategy="fixed",
        action_std_value=0.10,
        max_acceptable_kl=0.02,
        reward_config=MultiGateRewardConfig(),
        hidden_sizes=(32, 32),
        env_kwargs={
            "adapter": "fallback",
            "allow_fallback": True,
            "max_steps": 2,
            "observation_encoding_version": OBS_ENCODING_VERSION_V3,
        },
        initial_eval=False,
        ppo_kwargs={
            "n_steps": 10,
            "batch_size": 5,
            "n_epochs": 1,
            "learning_rate": 1e-5,
            "clip_range": 0.05,
            "target_kl": 0.01,
        },
    )
    config = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
    assert config["obs_encoding_version"] == OBS_ENCODING_VERSION_V3
    assert config["obs_dim"] == OBS_DIM_V3
    assert model.obs_encoding_version == OBS_ENCODING_VERSION_V3
