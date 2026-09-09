"""Expert-corpus legality, DAgger semantics, and the no-expert inference path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from marine_race_arena.learning.config import ACTION_DIM
from marine_race_arena.learning.config_local_transition import OBS_DIM_LOCAL_TRANSITION
from marine_race_arena.learning.gen2 import dagger as gen2_dagger
from marine_race_arena.learning.gen2 import dataset as gen2_dataset
from marine_race_arena.learning.gen2 import expert_rollout as rollout
from marine_race_arena.learning.gen2 import seeds as gen2_seeds

REPO_ROOT = Path(__file__).resolve().parents[3]
GEN2_DIR = REPO_ROOT / "marine_race_arena" / "learning" / "gen2"


def _episode(seed: int, steps: int = 20, takeover: bool = False) -> gen2_dataset.Gen2Episode:
    rng = np.random.default_rng(seed)
    obs = rng.normal(0, 0.2, (steps, OBS_DIM_LOCAL_TRANSITION)).astype(np.float32)
    expert = np.tanh(rng.normal(0, 0.4, (steps, ACTION_DIM))).astype(np.float32)
    applied = expert.copy() if takeover else np.tanh(
        rng.normal(0, 0.4, (steps, ACTION_DIM))).astype(np.float32)
    marks = np.zeros(steps, bool)
    if takeover:
        marks[:] = True
    return gen2_dataset.Gen2Episode(
        seed=seed, observations=obs, expert_actions=expert,
        applied_actions=applied, applied_by_expert=marks,
        gate_crossings=np.zeros(steps, np.int16),
        meta={"seed": seed, "gate_count": 3, "completed": True, "gates_completed": 3,
              "collisions": 0, "out_of_bounds": 0, "wrong_direction": 0,
              "course_pattern": "mixed"},
    )


# ------------------------------------------------ observation legality

def test_the_legal_observation_key_set_is_the_official_contract():
    assert rollout.LEGAL_OBSERVATION_KEYS == {"local_time_s", "sensors", "beacons", "comms"}


def test_a_privileged_observation_is_rejected():
    for leak in ("debug_ground_truth", "referee", "expected_gate_id", "circuit_id"):
        with pytest.raises(rollout.ObservationLegalityError):
            rollout.assert_observation_is_legal(
                {"local_time_s": 0.0, "sensors": {}, "beacons": [], leak: {"x": 1}}
            )


def test_a_legal_observation_passes():
    rollout.assert_observation_is_legal(
        {"local_time_s": 1.0, "sensors": {"DepthSensor": [1.0]}, "beacons": []}
    )


def test_actions_round_trip_through_the_command_contract():
    action = np.asarray([0.3, -0.2, 0.5, -0.9], np.float32)
    command = rollout.action_to_command(action)
    assert sorted(command) == ["heave", "surge", "sway", "yaw"]
    assert np.allclose(rollout.command_to_action(command), action)


def test_out_of_contract_commands_are_clipped_not_accepted():
    action = rollout.command_to_action(
        {"surge": 5.0, "sway": float("nan"), "heave": -3.0, "yaw": 0.1}
    )
    assert np.all(np.abs(action) <= 1.0)
    assert np.isfinite(action).all()


# --------------------------------------------------------- corpus schema

def test_a_shard_round_trips_with_only_legal_arrays(tmp_path):
    episode = _episode(40_000)
    path = gen2_dataset.save_episode(episode, tmp_path)
    with np.load(path, allow_pickle=False) as payload:
        stored = set(payload.files)
    assert stored == {
        "observations", "expert_actions", "applied_actions",
        "applied_by_expert", "gate_crossings", "meta",
    }
    assert not (stored & gen2_dataset.FORBIDDEN_ARRAY_NAMES)
    reloaded = gen2_dataset.load_episode(path)
    assert np.allclose(reloaded.observations, episode.observations)
    assert np.allclose(reloaded.expert_actions, episode.expert_actions)


def test_corpus_observations_are_exactly_the_35_feature_contract(tmp_path):
    gen2_dataset.save_episode(_episode(40_001), tmp_path)
    for episode in gen2_dataset.load_corpus(tmp_path):
        assert episode.observations.shape[1] == OBS_DIM_LOCAL_TRANSITION
        assert episode.expert_actions.shape[1] == ACTION_DIM


def test_labels_stay_inside_the_action_contract(tmp_path):
    bad = _episode(40_002)
    bad.expert_actions = bad.expert_actions * 3.0
    with pytest.raises(ValueError):
        gen2_dataset.save_episode(bad, tmp_path)


def test_manifest_pins_every_shard(tmp_path):
    for seed in (40_003, 40_004):
        gen2_dataset.save_episode(_episode(seed), tmp_path)
    gen2_dataset.write_manifest(tmp_path)
    report = gen2_dataset.verify_manifest(tmp_path)
    assert report["intact"] and report["recorded_shards"] == 2
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["observation_contract"] == "onboard_local_transition_v1"
    assert manifest["action_contract"] == "surge_sway_heave_yaw_pm1_v1"
    assert manifest["expert"] == "rule_gate_center_then_commit"


def test_manifest_detects_a_modified_shard(tmp_path):
    gen2_dataset.save_episode(_episode(40_005), tmp_path)
    gen2_dataset.write_manifest(tmp_path)
    shard = next(gen2_dataset.iter_shards(tmp_path))
    gen2_dataset.save_episode(_episode(40_005, steps=25), tmp_path)
    assert shard.name in gen2_dataset.verify_manifest(tmp_path)["changed"]


def test_takeover_episodes_can_be_excluded(tmp_path):
    gen2_dataset.save_episode(_episode(40_006, takeover=False), tmp_path)
    gen2_dataset.save_episode(_episode(40_007, takeover=True), tmp_path)
    assert len(gen2_dataset.load_corpus(tmp_path)) == 2
    clean = gen2_dataset.load_corpus(tmp_path, exclude_expert_takeover=True)
    assert [e.seed for e in clean] == [40_006]


def test_observation_statistics_floor_constant_features(tmp_path):
    episode = _episode(40_008)
    episode.observations[:, 5] = 0.7  # a constant presence-style flag
    mean, std = gen2_dataset.observation_statistics([episode])
    assert mean.shape == (OBS_DIM_LOCAL_TRANSITION,)
    assert std[5] == pytest.approx(1.0), "a constant feature must not divide by ~0"
    assert np.all(std > 0)


# --------------------------------------------------------- DAgger semantics

def test_dagger_rollouts_are_pure_learner_control(tmp_path):
    episodes = [_episode(44_000, takeover=False), _episode(44_001, takeover=False)]
    report = gen2_dagger.verify_no_blending(episodes)
    assert report["pure_learner_rollout"]
    assert report["takeover_steps"] == 0


def test_a_takeover_is_visible_in_the_audit(tmp_path):
    report = gen2_dagger.verify_no_blending([_episode(44_002, takeover=True)])
    assert not report["pure_learner_rollout"]
    assert report["takeover_fraction"] == pytest.approx(1.0)


def test_dagger_curriculum_retains_shorter_sequences():
    for entry in gen2_dagger.DAGGER_CURRICULUM:
        counts = gen2_dagger.round_gate_counts(entry["round"], 40)
        assert set(counts) <= set(entry["lengths"]) | set(entry["retention"])
        retained = sum(1 for c in counts if c in entry["retention"])
        assert retained > 0, f"round {entry['round']} forgot its retention lengths"
        assert any(c in entry["lengths"] for c in counts)


def test_dagger_curriculum_lengths_increase():
    peaks = [max(entry["lengths"]) for entry in gen2_dagger.DAGGER_CURRICULUM]
    assert peaks == sorted(peaks) and len(set(peaks)) == len(peaks)


def test_dagger_aggregates_previous_rounds(tmp_path):
    (tmp_path / "expert_corpus").mkdir()
    for index in (1, 2, 3):
        (tmp_path / f"dagger_round_{index:02d}").mkdir()
    roots = gen2_dagger.aggregated_corpus_roots(tmp_path, through_round=3)
    assert [p.name for p in roots] == [
        "expert_corpus", "dagger_round_01", "dagger_round_02", "dagger_round_03",
    ]


def test_dagger_only_queries_the_expert_on_train_seeds():
    for index in range(1, 6):
        for seed in gen2_dagger.gen2_seeds.dagger_round_seeds(index)[:3]:
            assert gen2_seeds.assert_expert_labelling_seed(seed) == seed


# ------------------------------------------- no expert at inference time

def test_the_inference_module_never_references_the_expert():
    """The guarantee is a property of the import graph, not a comment."""
    source = (GEN2_DIR / "evaluation.py").read_text(encoding="utf-8")
    for forbidden in (
        "RuleGateCenterThenCommit",
        "official_baselines",
        "expert_rollout",
        "hybrid",
    ):
        assert forbidden not in source, f"evaluation.py references {forbidden!r}"


def test_the_recurrent_policy_module_never_references_the_expert():
    source = (GEN2_DIR / "recurrent_policy.py").read_text(encoding="utf-8")
    for forbidden in ("RuleGateCenterThenCommit", "official_baselines", "LocalCourseTracker"):
        assert forbidden not in source, f"recurrent_policy.py references {forbidden!r}"


def test_the_inference_controller_has_no_fallback_branch():
    """No conditional in ``act`` may choose a non-policy action.

    Checked on the parsed code with docstrings stripped, so the prose that
    *describes* the prohibition cannot be mistaken for a violation of it.
    """
    import ast

    tree = ast.parse((GEN2_DIR / "recurrent_policy.py").read_text(encoding="utf-8"))
    node = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "Gen2RecurrentController"
    )
    def _strip_docstrings(scope):
        scope.body = [
            stmt for stmt in scope.body
            if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str))
        ]
        for stmt in scope.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                _strip_docstrings(stmt)

    _strip_docstrings(node)
    code = ast.unparse(node).lower()
    for forbidden in ("fallback", "rule", "blend", "expert", "hybrid"):
        assert forbidden not in code, (
            f"the inference controller's code mentions {forbidden!r}; the action "
            f"must be exactly learned_policy(observation, recurrent_state)"
        )
    # The single action source is the model itself.
    assert code.count("self.model.predict(") == 1


def test_the_expert_is_never_stepped_during_evaluation(tmp_path):
    """The behavioural guarantee, not the textual one.

    ``official_baselines`` is reachable from the shared episode/runner plumbing
    that every controller uses, so its mere presence in ``sys.modules`` proves
    nothing.  What must hold is that no Gen-2 evaluation episode ever *calls*
    the expert.  This booby-traps the expert and runs a real episode.
    """
    pytest.importorskip("torch")
    pytest.importorskip("sb3_contrib")

    from marine_race_arena.controllers import official_baselines
    from marine_race_arena.learning.gen2 import course_family as cf
    from marine_race_arena.learning.gen2.evaluation import run_policy_episode
    from marine_race_arena.learning.gen2.recurrent_policy import (
        Gen2RecurrentController,
        build_gen2_policy_for_training,
    )

    calls: list = []

    def _trap(self, *args, **kwargs):
        calls.append(1)
        raise AssertionError("the expert was stepped during Gen-2 evaluation")

    original_step = official_baselines.RuleGateBaselineController.step
    original_reset = official_baselines.RuleGateBaselineController.reset
    official_baselines.RuleGateBaselineController.step = _trap
    official_baselines.RuleGateBaselineController.reset = _trap
    try:
        spec = cf.sample_course(50_000, gate_count=2)
        track = cf.materialize_course(spec, tmp_path / "eval.json")
        controller = Gen2RecurrentController(build_gen2_policy_for_training(seed=0))
        episode = run_policy_episode(
            controller, track, seed=50_000, spec=spec,
            adapter="fallback", allow_fallback=True, max_steps=15,
        )
    finally:
        official_baselines.RuleGateBaselineController.step = original_step
        official_baselines.RuleGateBaselineController.reset = original_reset

    assert not calls, "the expert was invoked on the inference path"
    assert episode.steps > 0


# --------------------------------------------------------------- corpus cache

def test_cached_corpus_is_identical_to_the_uncached_one(tmp_path):
    for seed in (40_010, 40_011, 40_012):
        gen2_dataset.save_episode(_episode(seed, steps=17), tmp_path)
    gen2_dataset.write_manifest(tmp_path)

    direct = gen2_dataset.load_corpus(tmp_path)
    built = gen2_dataset.load_corpus_cached(tmp_path)   # builds the cache
    reread = gen2_dataset.load_corpus_cached(tmp_path)  # reads it back

    assert [e.seed for e in direct] == [e.seed for e in built] == [e.seed for e in reread]
    for a, b in zip(direct, reread):
        assert np.array_equal(a.observations, b.observations)
        assert np.array_equal(a.expert_actions, b.expert_actions)
        assert np.array_equal(a.applied_actions, b.applied_actions)
        assert np.array_equal(a.applied_by_expert, b.applied_by_expert)
        assert np.array_equal(a.gate_crossings, b.gate_crossings)
        assert a.meta == b.meta


def test_cache_filters_are_applied_after_loading(tmp_path):
    """One cache must serve callers that want different subsets."""
    gen2_dataset.save_episode(_episode(40_020, takeover=False), tmp_path)
    gen2_dataset.save_episode(_episode(40_021, takeover=True), tmp_path)
    gen2_dataset.load_corpus_cached(tmp_path)  # build once
    assert len(gen2_dataset.load_corpus_cached(tmp_path)) == 2
    clean = gen2_dataset.load_corpus_cached(tmp_path, exclude_expert_takeover=True)
    assert [e.seed for e in clean] == [40_020]


def test_a_new_shard_invalidates_the_cache(tmp_path):
    gen2_dataset.save_episode(_episode(40_030), tmp_path)
    gen2_dataset.write_manifest(tmp_path)
    assert len(gen2_dataset.load_corpus_cached(tmp_path)) == 1
    gen2_dataset.save_episode(_episode(40_031), tmp_path)
    gen2_dataset.write_manifest(tmp_path)
    # The key is content-derived, so the stale cache cannot be served.
    assert len(gen2_dataset.load_corpus_cached(tmp_path)) == 2
