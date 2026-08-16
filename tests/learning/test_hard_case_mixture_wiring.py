"""The hard-case mixture as it is actually wired into a rollout worker.

``rl_hard_families`` was tested as a generator but nothing constructed it, so
the realised mixture had never been observed on an env.  These tests pin the
wiring itself: that the default path is byte-identical to the sampler stream it
replaces, that the requested split really materialises, that hard episodes stay
inside the TRAIN band, and that the mixture survives a resume instead of
silently restarting.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from marine_race_arena.learning.episode_composition_log import read_episodes
from marine_race_arena.learning.rl_hard_families import (
    HARD_FAMILY_NAMES,
    MAX_HARD_FRACTION,
    HardCaseMixer,
    family_of,
)
from marine_race_arena.learning.rl_holdout_policy import role_of_seed, seed_group
from marine_race_arena.learning.train_ppo_transition import (
    _hard_case_composition,
    _load_config,
    _make_vec_env,
    _make_worker,
    _status,
)
from marine_race_arena.learning.transition_curriculum import (
    TransitionCurriculumController,
    TransitionGeometrySampler,
)
from marine_race_arena.learning.transition_env import (
    UniversalTransitionEnv,
    _build_hard_case_mixer,
)

REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO / "configs" / "rl" / "ppo_universal_transition_final_generic_hardcase.json"
)

SAMPLER_SEED = 43001
FOCUS_FRACTION = 0.70
MIXTURE = {"enabled": True, "base_fraction": 0.65}


def _env(tmp_path, *, mixture=None, worker_id=0, seed=SAMPLER_SEED):
    return UniversalTransitionEnv(
        run_dir=tmp_path,
        worker_id=worker_id,
        sampler_seed=seed,
        difficulty="G1",
        transition_focus_fraction=FOCUS_FRACTION,
        adapter="fallback",
        allow_fallback=True,
        max_episode_steps=5,
        hard_case_mixture=mixture,
    )


def _bare_sampler(seed=SAMPLER_SEED):
    """The sampler stream a worker consumes, bootstrap draw already spent."""

    sampler = TransitionGeometrySampler(
        seed=seed,
        difficulty="G1",
        transition_focus_fraction=FOCUS_FRACTION,
        dataset_split="train",
    )
    # The env constructor spends exactly one forced draw on its bootstrap track.
    sampler.sample(force_episode_type="transition_focus")
    return sampler


# ------------------------------------------------- the default path is inert


def test_without_a_mixture_the_geometry_stream_is_unchanged(tmp_path):
    env = _env(tmp_path)
    try:
        assert env.hard_case_mixer is None
        reference = _bare_sampler()
        ours = [env._sample_episode_geometry() for _ in range(200)]
        theirs = [reference.sample() for _ in range(200)]
        assert [asdict(g) for g in ours] == [asdict(g) for g in theirs]
        assert env.hard_case_composition() is None
    finally:
        env.close()


def test_the_mixture_is_opt_in_and_off_by_default():
    default = inspect.signature(
        UniversalTransitionEnv.__init__
    ).parameters["hard_case_mixture"].default
    assert default is None
    # An absent config reaches the worker as {}, which must not mean "on".
    assert _build_hard_case_mixer({}, sampler_seed=1) is None
    assert _build_hard_case_mixer(None, sampler_seed=1) is None
    assert _build_hard_case_mixer({"enabled": False, "base_fraction": 0.6},
                                  sampler_seed=1) is None
    with pytest.raises(ValueError, match="unknown hard_case_mixture keys"):
        _build_hard_case_mixer(
            {"enabled": True, "base_franction": 0.65}, sampler_seed=1
        )


# --------------------------------------------------- the mixture materialises


def test_a_mixture_realises_roughly_the_requested_split(tmp_path):
    env = _env(tmp_path, mixture=MIXTURE)
    try:
        geometries = [env._sample_episode_geometry() for _ in range(600)]
        families = [family_of(g) for g in geometries]
        hard = [name for name in families if name]
        assert len(hard) / len(geometries) == pytest.approx(0.35, abs=0.05)

        composition = env.hard_case_composition()
        assert composition["episodes"] == 600
        assert composition["hard_episodes"] == len(hard)
        assert composition["base_episodes"] == len(geometries) - len(hard)
        assert composition["realised_hard_fraction"] <= MAX_HARD_FRACTION
        assert composition["hard_fraction_within_cap"]
        # A mixture that only ever drew one family would still hit the split.
        assert len({name for name in hard}) >= 8
        assert sum(composition["family_counts"].values()) == len(hard)

        # Hard episodes are drawn at the full G6 envelope, which G1 cannot
        # express at all: this is the point of the whole intervention.
        hard_turns = [
            max(abs(v) for v in g.turn_deltas_deg)
            for g in geometries if family_of(g) and g.turn_deltas_deg
        ]
        assert max(hard_turns) > 30.0
        base_turns = [
            max(abs(v) for v in g.turn_deltas_deg)
            for g in geometries if not family_of(g) and g.turn_deltas_deg
        ]
        assert max(base_turns) <= 8.0
    finally:
        env.close()


def test_mixing_does_not_perturb_the_base_distribution(tmp_path):
    """Base episodes must be the same stream, in order, that a bare run sees."""

    env = _env(tmp_path, mixture=MIXTURE)
    try:
        drawn = [env._sample_episode_geometry() for _ in range(300)]
        base = [g for g in drawn if family_of(g) is None]
        reference = _bare_sampler()
        expected = [reference.sample() for _ in range(len(base))]
        assert [asdict(g) for g in base] == [asdict(g) for g in expected]
    finally:
        env.close()


def test_every_episode_seed_under_the_mixture_is_a_train_seed(tmp_path):
    env = _env(tmp_path, mixture=MIXTURE)
    try:
        geometries = [env._sample_episode_geometry() for _ in range(400)]
        group = seed_group("train")
        assert all(group.contains(int(g.seed)) for g in geometries)
        assert {role_of_seed(int(g.seed)) for g in geometries} == {"train"}
        assert {g.dataset_split for g in geometries} == {"train"}
        hard = [g for g in geometries if family_of(g)]
        assert hard, "the mixture produced no hard episodes to check"
        assert all(group.contains(int(g.seed)) for g in hard)
    finally:
        env.close()


# ------------------------------------------------------------- the hard cap


@pytest.mark.parametrize("base_fraction", [0.49, 0.35, 0.1])
def test_a_mixture_above_the_cap_is_refused_at_construction(tmp_path, base_fraction):
    with pytest.raises(ValueError, match="MAX_HARD_FRACTION"):
        _env(tmp_path, mixture={"enabled": True, "base_fraction": base_fraction})
    # Refused before any track was generated, so no worker or simulator was
    # ever started for a configuration that cannot be trained.
    assert not (
        tmp_path / "workers" / "worker_00" / "generated_tracks" / "bootstrap.json"
    ).exists()


def test_the_cap_boundary_itself_is_allowed(tmp_path):
    env = _env(tmp_path, mixture={"enabled": True,
                                  "base_fraction": 1.0 - MAX_HARD_FRACTION})
    try:
        assert env.hard_case_mixer.hard_fraction == pytest.approx(MAX_HARD_FRACTION)
    finally:
        env.close()


# --------------------------------------------------------- composition log


def _run_episodes(env, count):
    for _ in range(count):
        env.reset()
        for _ in range(env.max_episode_steps + 1):
            _, _, terminated, truncated, _ = env.step([0.2, 0.0, 0.0, 0.0])
            if terminated or truncated:
                break


def test_the_composition_log_records_base_versus_hard_and_the_family(tmp_path):
    env = _env(tmp_path, mixture={"enabled": True, "base_fraction": 0.5})
    try:
        _run_episodes(env, 14)
    finally:
        env.close()
    rows = read_episodes(tmp_path)
    assert len(rows) == 14
    for row in rows:
        assert row["case_source"] in {"base", "hard"}
        expected = "hard" if str(row["pattern"]).startswith("hard_") else "base"
        assert row["case_source"] == expected
        assert row["hard_family"] == (
            str(row["pattern"])[len("hard_"):] if expected == "hard" else None
        )
        assert row["dataset_split"] == "train"
        assert json.dumps(row)
    sources = {row["case_source"] for row in rows}
    assert sources == {"base", "hard"}, "the log must distinguish both sources"
    assert {row["hard_family"] for row in rows if row["case_source"] == "hard"} <= set(
        HARD_FAMILY_NAMES
    )


def test_an_unmixed_run_keeps_the_historical_log_columns(tmp_path):
    env = _env(tmp_path)
    try:
        _run_episodes(env, 3)
    finally:
        env.close()
    rows = read_episodes(tmp_path)
    assert rows
    assert all("case_source" not in row for row in rows)
    assert all("hard_family" not in row for row in rows)


# ------------------------------------------------------------------ resume


def test_worker_state_round_trips_the_mixer(tmp_path):
    first = _env(tmp_path, mixture=MIXTURE)
    try:
        [first._sample_episode_geometry() for _ in range(60)]
        state = first.worker_state()
        assert "hard_case_mixer" in state
        assert json.dumps(state)
        expected = [
            asdict(first._sample_episode_geometry()) for _ in range(60)
        ]
    finally:
        first.close()

    resumed = _env(tmp_path / "resumed", mixture=MIXTURE)
    control = _env(tmp_path / "control", mixture=MIXTURE)
    try:
        resumed.load_worker_state(state)
        assert resumed.hard_case_composition()["episodes"] == 60
        continued = [asdict(resumed._sample_episode_geometry()) for _ in range(60)]
        assert continued == expected
        # Without the restore the same worker restarts its mixture from zero,
        # which is precisely the silent difference this guards against.
        restarted = [asdict(control._sample_episode_geometry()) for _ in range(60)]
        assert restarted != expected
    finally:
        resumed.close()
        control.close()


def test_a_resume_that_would_drop_the_mixture_is_refused(tmp_path):
    mixed = _env(tmp_path, mixture=MIXTURE)
    try:
        [mixed._sample_episode_geometry() for _ in range(5)]
        state = mixed.worker_state()
    finally:
        mixed.close()

    unmixed = _env(tmp_path / "unmixed")
    try:
        with pytest.raises(ValueError, match="hard case mixture"):
            unmixed.load_worker_state(state)
    finally:
        unmixed.close()


def test_a_continuation_may_introduce_the_mixture(tmp_path):
    """The parent of this campaign has no mixer state; that must still resume."""

    parent = _env(tmp_path)
    try:
        [parent._sample_episode_geometry() for _ in range(5)]
        state = parent.worker_state()
    finally:
        parent.close()
    assert "hard_case_mixer" not in state

    child = _env(tmp_path / "child", mixture=MIXTURE)
    try:
        child.load_worker_state(state)
        assert child.hard_case_composition()["episodes"] == 0
        drawn = [child._sample_episode_geometry() for _ in range(200)]
        assert any(family_of(g) for g in drawn)
    finally:
        child.close()


def test_the_mixer_state_survives_a_json_round_trip(tmp_path):
    """Worker state is written into a checkpoint sidecar, so it must serialise."""

    env = _env(tmp_path, mixture=MIXTURE)
    try:
        [env._sample_episode_geometry() for _ in range(30)]
        restored = json.loads(json.dumps(env.worker_state()))
        expected = [asdict(env._sample_episode_geometry()) for _ in range(30)]
    finally:
        env.close()
    resumed = _env(tmp_path / "resumed", mixture=MIXTURE)
    try:
        resumed.load_worker_state(restored)
        assert [
            asdict(resumed._sample_episode_geometry()) for _ in range(30)
        ] == expected
    finally:
        resumed.close()


# ----------------------------------------------------------- trainer wiring


class _StubWorker:
    def __init__(self, mixer):
        self._mixer = mixer

    def hard_case_composition(self):
        return None if self._mixer is None else self._mixer.realised_composition()

    def worker_identity(self):
        return {"worker_id": 0}


class _StubVecEnv:
    num_envs = 2

    def __init__(self, workers):
        self._workers = list(workers)

    def env_method(self, name, *args, **kwargs):
        return [getattr(worker, name)(*args, **kwargs) for worker in self._workers]


class _StubModel:
    num_timesteps = 528384


def test_the_trainer_forwards_the_mixture_to_its_workers(tmp_path, monkeypatch):
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "marine_race_arena.learning.train_ppo_transition._make_worker", _capture
    )
    monkeypatch.setattr(
        "stable_baselines3.common.vec_env.DummyVecEnv",
        lambda factories: [factory() for factory in factories],
    )
    config = {
        "n_envs": 1,
        "worker_seed_base": 43001,
        "curriculum": {"transition_focus_fraction": 0.7},
        "adapter": "fallback",
        "allow_fallback": True,
        "holoocean_frames_per_sec": False,
        "max_episode_steps": 5,
        "reward": {},
        "sequence_curriculum": {"enabled": True},
        "hard_case_mixture": dict(MIXTURE),
    }
    _make_vec_env(config, tmp_path, "G1")
    assert captured["hard_case_mixture"] == MIXTURE

    captured.clear()
    config.pop("hard_case_mixture")
    _make_vec_env(config, tmp_path, "G1")
    assert captured["hard_case_mixture"] == {}


def test_the_worker_factory_builds_a_mixed_environment(tmp_path):
    env = _make_worker(
        run_dir=str(tmp_path),
        worker_id=0,
        sampler_seed=SAMPLER_SEED,
        difficulty="G1",
        transition_focus_fraction=FOCUS_FRACTION,
        adapter="fallback",
        allow_fallback=True,
        frames_per_sec=False,
        max_episode_steps=5,
        reward_config={},
        sequence_curriculum={},
        hard_case_mixture=dict(MIXTURE),
    )
    try:
        assert env.hard_case_mixer is not None
        assert env.hard_case_mixer.base_fraction == pytest.approx(0.65)
        assert env.dataset_split == "train"
    finally:
        env.close()


def test_the_realised_mixture_is_aggregated_across_workers():
    mixers = [HardCaseMixer(base_fraction=0.65, rng=seed) for seed in (11, 12)]
    for mixer in mixers:
        for _ in range(300):
            mixer.next_source()
    report = _hard_case_composition(_StubVecEnv([_StubWorker(m) for m in mixers]))
    assert report["workers_reporting"] == 2
    assert report["workers_total"] == 2
    assert report["episodes"] == 600
    assert report["base_episodes"] + report["hard_episodes"] == 600
    assert report["realised_hard_fraction"] == pytest.approx(0.35, abs=0.06)
    assert report["requested_hard_fraction"] == pytest.approx(0.35)
    assert report["hard_fraction_within_cap"]
    assert sum(report["family_counts"].values()) == report["hard_episodes"]
    assert sum(report["family_share_of_hard"].values()) == pytest.approx(1.0, abs=1e-4)
    assert len(report["per_worker"]) == 2
    # An unmixed run must report nothing rather than an empty mixture.
    assert _hard_case_composition(_StubVecEnv([_StubWorker(None)] * 2)) is None


def test_status_json_surfaces_the_realised_mixture(tmp_path):
    config = _load_config(CONFIG_PATH)
    mixer = HardCaseMixer(base_fraction=0.65, rng=5)
    for _ in range(200):
        mixer.next_source()
    env = _StubVecEnv([_StubWorker(mixer), _StubWorker(None)])
    _status(
        tmp_path,
        state="running",
        model=_StubModel(),
        curriculum=TransitionCurriculumController(
            initial_difficulty="G1", maximum_difficulty="G6"
        ),
        env=env,
        aliases={},
        selection={},
        config=config,
    )
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    reported = status["hard_case_mixture"]
    assert reported["requested"]["base_fraction"] == 0.65
    assert reported["realised"]["episodes"] == 200
    assert reported["realised"]["realised_hard_fraction"] == pytest.approx(
        0.35, abs=0.08
    )
    # One of the two workers reports no mixture at all, which the status file
    # must expose rather than average away.
    assert reported["realised"]["workers_reporting"] == 1
    assert reported["realised"]["workers_total"] == 2


# ------------------------------------------------------------- the config


def test_the_final_hardcase_config_loads_and_validates():
    config = _load_config(CONFIG_PATH)
    assert config["run_name"] == (
        "universal_transition_ppo_final_generic_hardcase_seed23001"
    )
    # new_environment_steps is an ABSOLUTE cumulative target, not an increment:
    # the trainer computes (value // rollout_size) * rollout_size and trains while
    # num_timesteps < target, counting from the parent's total.  Pinning the
    # literal 200000 would describe a run that trains nothing, because the parent
    # alone is already well past it.
    parent_transitions = config["initialization"][
        "parent_total_environment_transitions"
    ]
    rollout = config["n_envs"] * config["ppo"]["n_steps"]
    target = (config["new_environment_steps"] // rollout) * rollout
    assert target > parent_transitions, (
        "target is at or below the parent count, so the run would stop immediately"
    )
    assert 195_000 <= target - parent_transitions <= 205_000, (
        f"intended ~200k new transitions, got {target - parent_transitions}"
    )
    assert config["new_environment_steps"] % rollout == 0, (
        "target should be a whole number of rollouts so none is silently dropped"
    )
    assert config["ppo"]["learning_rate"] == 4.5e-06
    assert config["hard_case_mixture"]["base_fraction"] == 0.65
    assert config["curriculum"]["initial_difficulty"] == "G1"
    assert config["seed"] == 23001
    assert config["worker_seed_base"] == 43001
    assert config["evaluation"]["frequency"] == 50000
    # Structure inherited from the v5 rewarm config it continues.
    assert config["n_envs"] == 2
    assert config["ppo"]["n_steps"] == 1024
    assert config["adapter"] == "holoocean"
    assert config["allow_fallback"] is False
    assert config["sequence_curriculum"] == {"enabled": True}


def test_the_parent_is_the_checkpoint_the_matched_benchmark_selected():
    """The parent must be real and self-consistent, not a leftover placeholder.

    This config shipped with ``PENDING_MATCHED_BENCHMARK`` placeholders while the
    selection benchmark was still running.  Now that it has finished, the danger
    reverses: an unfilled placeholder would refuse to launch (loudly, which is
    fine), but a parent whose recorded hash no longer matches its bytes would
    launch happily and continue from something other than the selected policy.
    """

    from marine_race_arena.learning.longrun_checkpoint import sha256_file

    config = _load_config(CONFIG_PATH)
    initialization = config["initialization"]
    assert initialization["mode"] == "parallel_continuation"

    for key in ("parent_run_dir", "parent_checkpoint", "parent_sha256"):
        assert "PENDING" not in str(initialization[key]), (
            f"{key} still carries the pre-benchmark placeholder"
        )

    checkpoint = (
        REPO / initialization["parent_run_dir"]
        / "checkpoints" / initialization["parent_checkpoint"]
    )
    assert checkpoint.is_file(), f"parent checkpoint missing: {checkpoint}"
    assert sha256_file(checkpoint) == initialization["parent_sha256"], (
        "recorded parent_sha256 does not match the checkpoint bytes, so the run "
        "would continue from a policy other than the one that was selected"
    )
    assert (
        initialization["parent_total_environment_transitions"]
        == int(initialization["parent_checkpoint"].split("_")[1])
    ), "recorded parent transition count disagrees with the checkpoint name"

    reason = initialization["selection_reason"].lower()
    assert "ci" in reason and "n=" in reason, (
        "the selection reason must carry the interval evidence it rests on, not "
        "just the winning name"
    )


def test_the_configured_mixture_builds_a_legal_mixer():
    config = _load_config(CONFIG_PATH)
    spec = config["hard_case_mixture"]
    assert set(spec["family_weights"]) == set(HARD_FAMILY_NAMES), (
        "every hard family must carry weight; an unattacked family is a "
        "diagnosed failure with no training data"
    )
    assert all(float(w) > 0.0 for w in spec["family_weights"].values())
    mixer = _build_hard_case_mixer(spec, sampler_seed=43001)
    assert mixer.base_fraction == pytest.approx(0.65)
    assert mixer.hard_fraction == pytest.approx(0.35)
    assert mixer.hard_fraction <= MAX_HARD_FRACTION
    assert sum(mixer.family_weights.values()) == pytest.approx(1.0)
