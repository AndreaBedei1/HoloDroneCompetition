"""Warm initialization, evaluation schedule, aliases and rollback contracts."""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from marine_race_arena.learning.config import ACTION_CONTRACT_VERSION
from marine_race_arena.learning.config_local_transition import (
    FEATURE_NAMES_LOCAL_TRANSITION,
    OBS_DIM_LOCAL_TRANSITION,
    OBS_ENCODING_VERSION_LOCAL_TRANSITION,
)
from marine_race_arena.learning.longrun_checkpoint import sha256_file
from marine_race_arena.learning.train_ppo_transition import (
    _dedicated_benchmark_due,
    _load_config,
    _next_evaluation_transition,
    _preserve_initialization_baseline,
    apply_evaluation_selection,
)
from marine_race_arena.learning.transition_policy import (
    build_local_transition_ppo,
    initialize_local_transition_policy,
    warm_start_from_transition_checkpoint,
)
from marine_race_arena.learning.transition_selection import (
    DEFAULT_COMPETENCE_THRESHOLDS,
)

CONFIG_PATH = "configs/rl/ppo_universal_transition_warm_reliability.json"


class _TinyEnv(gym.Env):
    def __init__(self, dim=OBS_DIM_LOCAL_TRANSITION):
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return (
            np.zeros(self.observation_space.shape, dtype=np.float32),
            0.0, False, True, {},
        )


def _model(seed=1):
    return build_local_transition_ppo(
        _TinyEnv(), seed=seed, learning_rate=1e-5, hidden_sizes=(16, 16),
        n_steps=8, batch_size=8, n_epochs=1,
    )


# ------------------------------------------------------------ initialization


def test_warm_start_transfers_policy_and_value_weights_verbatim():
    source = _model(seed=1)
    target = _model(seed=2)
    source_state = {
        name: value.clone() for name, value in source.policy.state_dict().items()
    }
    assert any(
        not np.allclose(source_state[name].numpy(), value.numpy())
        for name, value in target.policy.state_dict().items()
        if value.numel()
    )
    report = warm_start_from_transition_checkpoint(source, target)
    for name, value in target.policy.state_dict().items():
        assert np.allclose(value.numpy(), source_state[name].numpy()), name
    assert report["policy_and_value_weights_transferred"] is True
    assert any("value_net" in name for name in report["copied_tensors"])
    assert any("policy_net" in name for name in report["copied_tensors"])


def test_warm_start_uses_a_fresh_optimizer_schedule_and_counter():
    source = _model(seed=1)
    source.num_timesteps = 40_960
    source._n_updates = 60
    source._current_progress_remaining = 0.1
    for group in source.policy.optimizer.param_groups:
        group["lr"] = 9.9e-7
    target = _model(seed=2)
    report = warm_start_from_transition_checkpoint(source, target)
    assert report["optimizer_state_copied"] is False
    assert report["learning_rate_schedule_reset"] is True
    assert report["rollout_state_reset"] is True
    assert report["total_environment_transitions_reset_to_zero"] is True
    assert target.num_timesteps == 0
    assert target._n_updates == 0
    assert target._current_progress_remaining == pytest.approx(1.0)
    assert all(
        not state for state in target.policy.optimizer.state_dict()["state"].values()
    ) or not target.policy.optimizer.state_dict()["state"]


def test_all_transferred_parameters_remain_trainable():
    source, target = _model(seed=1), _model(seed=2)
    warm_start_from_transition_checkpoint(source, target)
    assert all(p.requires_grad for p in target.policy.parameters())


def test_warm_start_rejects_a_foreign_observation_contract():
    source = build_local_transition_ppo(
        _TinyEnv(dim=OBS_DIM_LOCAL_TRANSITION + 1), seed=1, learning_rate=1e-5,
        hidden_sizes=(16, 16), n_steps=8, batch_size=8, n_epochs=1,
    )
    with pytest.raises(ValueError, match="observation shape"):
        warm_start_from_transition_checkpoint(source, _model(seed=2))


def test_initialize_round_trips_through_a_saved_checkpoint(tmp_path):
    source = _model(seed=7)
    checkpoint = tmp_path / "warm.zip"
    source.save(checkpoint)
    target = _model(seed=8)
    report = initialize_local_transition_policy(
        target, mode="warm_start_transition_checkpoint",
        source_checkpoint=str(checkpoint),
    )
    assert report["mode"] == "warm_start_transition_checkpoint"
    assert report["all_parameters_trainable"] is True
    for name, value in target.policy.state_dict().items():
        assert np.allclose(value.numpy(), source.policy.state_dict()[name].numpy())


def test_unknown_initialization_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown initialization mode"):
        initialize_local_transition_policy(
            _model(), mode="behaviour_cloning", source_checkpoint="x.zip"
        )


# ------------------------------------------------------------------- config


def test_warm_reliability_config_matches_the_required_run_contract():
    config = _load_config(CONFIG_PATH)
    assert config["initialization"]["mode"] == "warm_start_transition_checkpoint"
    assert config["n_envs"] == 2
    assert config["ppo"]["n_steps"] == 1024
    assert config["n_envs"] * config["ppo"]["n_steps"] == 2048
    assert config["ppo"]["batch_size"] == 256
    assert config["adapter"] == "holoocean"
    assert config["allow_fallback"] is False
    assert config["observation_version"] == OBS_ENCODING_VERSION_LOCAL_TRANSITION
    assert config["action_version"] == ACTION_CONTRACT_VERSION
    assert config["output_root"].endswith("longrun")
    assert config["run_name"] == "universal_transition_warm_reliability_seed23001"
    assert config["run_name"] != "universal_transition_seed23001"


def test_warm_reliability_config_points_at_the_validated_checkpoint():
    config = _load_config(CONFIG_PATH)
    source = Path(config["initialization"]["source_checkpoint"])
    if not source.exists():
        pytest.skip("A/B warm checkpoint is not present in this checkout")
    assert sha256_file(source) == config["initialization"]["source_sha256"]
    manifest = source.parent / f"{source.stem}.manifest.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    assert value["model_sha256"] == config["initialization"]["source_sha256"]
    assert value["observation_version"] == OBS_ENCODING_VERSION_LOCAL_TRANSITION
    assert value["policy_observation_dim"] == OBS_DIM_LOCAL_TRANSITION == 35
    assert value["action_version"] == ACTION_CONTRACT_VERSION
    assert value["total_timesteps"] == 40960


def test_warm_reliability_config_keeps_efficiency_penalties_disabled():
    config = _load_config(CONFIG_PATH)
    assert config["reward"] == {}
    from marine_race_arena.learning.reward_local_transition import (
        LocalTransitionRewardConfig,
    )

    reward = LocalTransitionRewardConfig(**config["reward"])
    assert reward.efficiency_unlocked is False
    assert reward.reliability_jerk_penalty == 0.0
    assert reward.reliability_energy_penalty == 0.0
    assert reward.reliability_action_change_penalty == 0.0
    assert reward.reliability_time_cost <= 0.001


def test_observation_contract_is_unchanged_and_shortcut_free():
    assert len(FEATURE_NAMES_LOCAL_TRANSITION) == OBS_DIM_LOCAL_TRANSITION == 35
    forbidden = (
        "gate_index", "remaining", "total_gate", "sequence_progress", "lap",
        "phase_", "pose", "future", "referee", "distance_travelled",
        "collision", "nontrivial", "gates_completed", "jerk",
    )
    assert not any(
        fragment in name
        for name in FEATURE_NAMES_LOCAL_TRANSITION
        for fragment in forbidden
    )


# --------------------------------------------------------- evaluation schedule


def test_early_schedule_then_regular_frequency():
    early = (25_000, 50_000, 75_000, 100_000, 150_000)
    state = {"history": []}
    assert _next_evaluation_transition(state, 50_000, early) == 25_000
    for completed, expected in (
        (25_000, 50_000), (50_000, 75_000), (75_000, 100_000),
        (100_000, 150_000), (150_000, 200_000), (200_000, 250_000),
    ):
        state = {"history": [{"timesteps": completed}]}
        assert _next_evaluation_transition(state, 50_000, early) == expected


def test_schedule_without_early_points_matches_previous_behaviour():
    assert _next_evaluation_transition({"history": []}, 50_000) == 50_000
    assert _next_evaluation_transition(
        {"history": [{"timesteps": 50_000}]}, 50_000
    ) == 100_000


def test_final_model_is_evaluated_at_a_target_off_the_schedule():
    early = (25_000,)
    target = 999_424  # 1,000,000 rounded down to whole 2,048-transition rollouts
    state = {"history": [{"timesteps": 950_272}]}
    # Without the target the run would end at 999,424 with no final evaluation.
    assert _next_evaluation_transition(state, 50_000, early) == 1_000_000
    assert _next_evaluation_transition(state, 50_000, early, target) == target
    # Once the target has been evaluated the schedule must move past it so the
    # loop terminates instead of re-evaluating forever.
    state["history"].append({"timesteps": target})
    assert _next_evaluation_transition(state, 50_000, early, target) == 1_000_000


def test_dedicated_benchmark_only_runs_when_earned():
    config = _load_config(CONFIG_PATH)
    metrics = {"universal_transition_success_rate": 0.31}
    fresh = {"last_timesteps": 0, "best_success_rate": None, "history": []}
    assert _dedicated_benchmark_due(
        metrics=metrics, competent=False, candidate_best=True,
        dedicated_state=fresh, timesteps=25_000, target=1_000_000, config=config,
    ) is None
    assert _dedicated_benchmark_due(
        metrics=metrics, competent=True, candidate_best=True,
        dedicated_state=fresh, timesteps=25_000, target=1_000_000, config=config,
    ) == "first_competent_checkpoint"
    settled = {
        "last_timesteps": 25_000, "best_success_rate": 0.31, "history": [],
    }
    assert _dedicated_benchmark_due(
        metrics=metrics, competent=True, candidate_best=True,
        dedicated_state=settled, timesteps=50_000, target=1_000_000, config=config,
    ) is None
    assert _dedicated_benchmark_due(
        metrics={"universal_transition_success_rate": 0.40}, competent=True,
        candidate_best=True, dedicated_state=settled, timesteps=150_000,
        target=1_000_000, config=config,
    ) == "material_competence_improvement"
    assert _dedicated_benchmark_due(
        metrics=metrics, competent=False, candidate_best=False,
        dedicated_state=settled, timesteps=1_000_000, target=1_000_000,
        config=config,
    ) == "final_model_validation"


# ----------------------------------------------------------- aliases/rollback


def _fresh_state():
    aliases = {
        "last": None, "latest_competent": None, "latest_safe_competent": None,
        "best_universal_transition": None, "best_long_sequence": None,
    }
    selection = {
        "best_metrics": None, "best_timestep": None,
        "last_evaluation_metrics": None, "last_competence_verdict": None,
        "best_long_sequence_score": None, "consecutive_baseline_collapses": 0,
        "dedicated": {"last_timesteps": 0, "best_success_rate": None, "history": []},
        "rollback": None,
    }
    return aliases, selection


_INACTIVE = {
    "n_eval": 1012, "transition_n": 1000,
    "universal_transition_success_rate": 0.0, "first_gate_crossing_rate": 0.015,
    "target_switch_rate": 0.002, "completed_gate_count": 15,
    "fraction_of_episodes_reaching_first_gate": 0.015,
    "mean_absolute_action": 0.0034, "nontrivial_action_fraction": 0.0,
    "mean_distance_travelled_m": 0.4, "long_sequence_completion_score": 0.0,
    "collision_episodes": 0, "missed_gate_dnf": 0, "wrong_direction_events": 0,
    "previous_gate_returns": 0, "acquisition_timeouts": 0,
    "mean_action_jerk": 0.005, "safety_clean": True,
}
_COMPETENT = {
    "n_eval": 1012, "transition_n": 1000,
    "universal_transition_success_rate": 0.314, "first_gate_crossing_rate": 0.912,
    "target_switch_rate": 0.819, "completed_gate_count": 946,
    "fraction_of_episodes_reaching_first_gate": 0.913,
    "mean_absolute_action": 0.0848, "nontrivial_action_fraction": 0.979,
    "mean_distance_travelled_m": 9.2, "long_sequence_completion_score": 0.25,
    "collision_episodes": 417, "collision_entries": 500, "missed_gate_dnf": 78,
    "wrong_direction_events": 51, "previous_gate_returns": 1,
    "acquisition_timeouts": 436, "mean_action_jerk": 0.11, "safety_clean": False,
}


def _apply(metrics, aliases, selection, *, timesteps, path, rollback_limit=2):
    return apply_evaluation_selection(
        metrics, checkpoint_path=path, timesteps=timesteps, aliases=aliases,
        selection=selection, thresholds=DEFAULT_COMPETENCE_THRESHOLDS,
        rollback_limit=rollback_limit,
    )


def test_inactive_policy_never_receives_competent_or_best_aliases():
    aliases, selection = _fresh_state()
    outcome = _apply(_INACTIVE, aliases, selection, timesteps=25_000, path="a.zip")
    assert outcome["competent"] is False
    assert outcome["safe"] is False
    assert aliases["latest_competent"] is None
    assert aliases["latest_safe_competent"] is None
    assert aliases["best_universal_transition"] is None
    assert aliases["best_long_sequence"] is None


def test_competent_policy_receives_competent_and_best_aliases():
    aliases, selection = _fresh_state()
    outcome = _apply(_COMPETENT, aliases, selection, timesteps=25_000, path="b.zip")
    assert outcome["competent"] is True
    # Competent, not yet reliable: safety-clean at 99% is still required.
    assert outcome["safe"] is False
    assert aliases["latest_competent"] == "b.zip"
    assert aliases["latest_safe_competent"] is None
    assert aliases["best_universal_transition"] == "b.zip"
    assert aliases["best_long_sequence"] == "b.zip"


def test_latest_safe_competent_requires_both_safety_and_competence():
    aliases, selection = _fresh_state()
    reliable = dict(_COMPETENT)
    reliable.update({
        "universal_transition_success_rate": 0.995, "safety_clean": True,
        "collision_episodes": 0, "missed_gate_dnf": 0,
        "wrong_direction_events": 0, "previous_gate_returns": 0,
        "acquisition_timeouts": 0,
    })
    outcome = _apply(reliable, aliases, selection, timesteps=500_000, path="c.zip")
    assert outcome["safe"] is True
    assert aliases["latest_safe_competent"] == "c.zip"

    idle_but_clean = dict(_INACTIVE)
    idle_but_clean["universal_transition_success_rate"] = 0.0
    aliases2, selection2 = _fresh_state()
    _apply(idle_but_clean, aliases2, selection2, timesteps=1000, path="d.zip")
    assert aliases2["latest_safe_competent"] is None


def test_repeated_sub_baseline_evaluations_trigger_a_controlled_rollback():
    aliases, selection = _fresh_state()
    _apply(_COMPETENT, aliases, selection, timesteps=25_000, path="good.zip")
    collapsed = dict(_COMPETENT)
    collapsed.update({
        "first_gate_crossing_rate": 0.4, "target_switch_rate": 0.3,
        "universal_transition_success_rate": 0.02,
    })
    first = _apply(collapsed, aliases, selection, timesteps=50_000, path="bad1.zip")
    assert first["collapsed"] is False
    second = _apply(collapsed, aliases, selection, timesteps=75_000, path="bad2.zip")
    assert second["collapsed"] is True
    rollback = selection["rollback"]
    assert rollback["rollback_checkpoint"] == "good.zip"
    assert rollback["consecutive_evaluations_below_baseline"] == 2
    assert set(rollback["failed_criteria"]) == {
        "first_gate_crossing", "target_switch", "universal_transition_success",
    }


def test_a_recovered_evaluation_clears_the_collapse_counter():
    aliases, selection = _fresh_state()
    collapsed = dict(_COMPETENT)
    collapsed["universal_transition_success_rate"] = 0.02
    _apply(collapsed, aliases, selection, timesteps=50_000, path="bad.zip")
    assert selection["consecutive_baseline_collapses"] == 1
    _apply(_COMPETENT, aliases, selection, timesteps=75_000, path="good.zip")
    assert selection["consecutive_baseline_collapses"] == 0
    assert selection["rollback"] is None


def test_initialization_baseline_is_written_once_and_left_immutable(tmp_path):
    source = tmp_path / "source.zip"
    source.write_bytes(b"policy-weights")
    config = {
        "initialization": {
            "mode": "warm_start_transition_checkpoint",
            "source_checkpoint": str(source),
            "source_sha256": sha256_file(source),
        }
    }
    run_dir = tmp_path / "run"
    first = _preserve_initialization_baseline(run_dir, config)
    assert first["immutable"] is True
    assert first["baseline_sha256"] == config["initialization"]["source_sha256"]
    baseline = Path(first["baseline_checkpoint"])
    assert baseline.read_bytes() == b"policy-weights"
    stamp = baseline.stat().st_mtime_ns
    # A later source edit must not alter the preserved baseline.
    source.write_bytes(b"different-weights")
    second = _preserve_initialization_baseline(run_dir, config)
    assert second == first
    assert baseline.stat().st_mtime_ns == stamp
    assert baseline.read_bytes() == b"policy-weights"


def test_scratch_initialization_has_no_baseline_to_preserve(tmp_path):
    config = {"initialization": {"mode": "scratch", "source_checkpoint": None}}
    assert _preserve_initialization_baseline(tmp_path / "run", config) is None
