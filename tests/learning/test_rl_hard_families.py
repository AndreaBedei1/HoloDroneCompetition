"""Hard TRAIN families must be a distribution, TRAIN-only, and never dominate.

The two ways this component fails silently are (a) generating a small repeated
pool of "hard" layouts that the policy memorises, and (b) quietly turning the
curriculum into hard cases only.  Both are pinned here, together with the
train/validation/test separation the whole generalization claim rests on.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

import marine_race_arena
from marine_race_arena.learning.episode_composition_log import episode_row
from marine_race_arena.learning.generic_sequence_curriculum import (
    BUCKET_GATE_RANGES,
    STAGES,
    GenericSequenceCurriculum,
)
from marine_race_arena.learning.rl_hard_families import (
    DEFAULT_BASE_FRACTION,
    HARD_ENVELOPE,
    HARD_FAMILY_BY_NAME,
    HARD_FAMILY_NAMES,
    HARD_FAMILY_SPECS,
    HARD_PATTERN_PREFIX,
    MAX_HARD_FRACTION,
    MIN_BASE_FRACTION,
    STRAIGHT_TURN_RANGE_DEG,
    TAXONOMY_FAILURE_LABELS,
    DiversityCollapseError,
    HardCaseMixer,
    HardFamilySpec,
    assert_diverse,
    diversity_report,
    families_for_failure,
    family_of,
    geometry_signature,
    hard_family_manifest,
    sample_hard_episode,
    sample_hard_geometry,
    track_label,
    uncovered_failure_labels,
    validate_specs,
)
from marine_race_arena.learning.rl_holdout_policy import (
    FINAL_CIRCUIT_NAMES,
    role_of_seed,
    seed_group,
)
from marine_race_arena.learning.transition_curriculum import (
    TransitionGeometry,
    TransitionGeometrySampler,
    generate_transition_track,
)

BASE_TRACK = (
    Path(marine_race_arena.__file__).parent / "tracks" / "training"
    / "stage3_three_gates.json"
)


def _episodes(spec, count, seed=7):
    rng = np.random.default_rng(seed)
    return [sample_hard_episode(spec, rng) for _ in range(count)]


def _geometries(spec, count, seed=7):
    return [episode.geometry for episode in _episodes(spec, count, seed)]


# ------------------------------------------------------------ dataset split


@pytest.mark.parametrize("spec", HARD_FAMILY_SPECS, ids=HARD_FAMILY_NAMES)
def test_every_hard_episode_seed_is_in_the_train_band(spec):
    train = seed_group("train")
    for geometry in _geometries(spec, 120, seed=sum(map(ord, spec.name))):
        assert train.contains(geometry.seed), (spec.name, geometry.seed)
        assert role_of_seed(geometry.seed) == "train"
        assert geometry.dataset_split == "train"


@pytest.mark.parametrize("split", ["validation", "test", "final_circuits", ""])
def test_a_validation_or_test_seed_can_never_be_produced(split):
    """Hard cases are a training intervention; other splits are refused."""

    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="TRAIN-only"):
        sample_hard_geometry(HARD_FAMILY_SPECS[0], rng, dataset_split=split)


def test_no_generated_seed_ever_lands_in_a_holdout_band():
    rng = np.random.default_rng(3)
    seeds = {
        sample_hard_geometry(spec, rng).seed
        for spec in HARD_FAMILY_SPECS for _ in range(300)
    }
    for role in ("validation", "test"):
        group = seed_group(role)
        assert not any(group.contains(seed) for seed in seeds)
    assert {role_of_seed(seed) for seed in seeds} == {"train"}


# ---------------------------------------------------- families are two-sided


@pytest.mark.parametrize("spec", HARD_FAMILY_SPECS, ids=HARD_FAMILY_NAMES)
def test_each_family_turns_both_ways_and_both_climbs_and_descends(spec):
    """A one-sided family teaches a bias, not a skill."""

    geometries = _geometries(spec, 200, seed=11)
    turns = [v for g in geometries for v in g.turn_deltas_deg]
    vertical = [v for g in geometries for v in g.vertical_deltas_m]
    assert min(turns) < 0 < max(turns), f"{spec.name}: one-sided turns"
    assert min(vertical) < 0 < max(vertical), f"{spec.name}: no climb or no descent"
    yaw = [g.initial_yaw_error_deg for g in geometries]
    offsets = [g.initial_lateral_offset_m for g in geometries]
    assert min(yaw) < 0 < max(yaw), f"{spec.name}: entry yaw error is one-sided"
    assert min(offsets) < 0 < max(offsets), f"{spec.name}: entry offset is one-sided"


@pytest.mark.parametrize("spec", HARD_FAMILY_SPECS, ids=HARD_FAMILY_NAMES)
def test_spacing_and_magnitude_are_distributions_not_constants(spec):
    geometries = _geometries(spec, 200, seed=5)
    spacings = [v for g in geometries for v in g.spacings_m]
    magnitudes = [abs(v) for g in geometries for v in g.turn_deltas_deg]
    vertical = [abs(v) for g in geometries for v in g.vertical_deltas_m]
    starts = [g.start_distance_m for g in geometries]
    speeds = [g.initial_body_velocity_m_s[0] for g in geometries]
    for name, values in (
        ("spacing", spacings), ("turn", magnitudes), ("vertical", vertical),
        ("start_distance", starts), ("entry_speed", speeds),
    ):
        assert min(values) < max(values), f"{spec.name}: {name} never varies"
    low, high = spec.gate_count_range
    assert len({g.gate_count for g in geometries}) > 1, (
        f"{spec.name}: episode length is pinned, not drawn from {low}..{high}"
    )


@pytest.mark.parametrize("spec", HARD_FAMILY_SPECS, ids=HARD_FAMILY_NAMES)
def test_generation_does_not_collapse_to_a_small_pool(spec):
    episodes = _episodes(spec, 400, seed=2024)
    report = assert_diverse(episodes, 0.95)
    assert report["distinct_fraction"] > 0.95
    assert report["distinct_signatures"] > 380
    assert report["family_share"][spec.name] == pytest.approx(1.0)


def test_assert_diverse_raises_on_a_deliberately_collapsed_pool():
    """The historical bug: a handful of layouts recycled with fresh seeds."""

    spec = HARD_FAMILY_BY_NAME["sharp_turn"]
    template = sample_hard_geometry(spec, np.random.default_rng(1))
    pool = [
        dataclasses.replace(template, seed=seed_group("train").start + index)
        for index in range(200)
    ]
    # Fresh seeds every time, but only one shape: the signature must not be
    # fooled by the seed.
    assert len({g.seed for g in pool}) == 200
    assert len({geometry_signature(g) for g in pool}) == 1
    with pytest.raises(DiversityCollapseError, match="collapsed"):
        assert_diverse(pool, 0.95)


def test_assert_diverse_rejects_a_one_sided_pool():
    spec = HARD_FAMILY_BY_NAME["sharp_turn"]
    rng = np.random.default_rng(4)
    pool = []
    while len(pool) < 60:
        geometry = sample_hard_geometry(spec, rng)
        if all(value > 0 for value in geometry.turn_deltas_deg):
            pool.append(geometry)
    with pytest.raises(DiversityCollapseError, match="one-sided"):
        assert_diverse(pool, 0.95)


def test_assert_diverse_refuses_an_empty_batch():
    with pytest.raises(DiversityCollapseError):
        assert_diverse([])


def test_diversity_report_measures_the_batch_it_is_given():
    rng = np.random.default_rng(9)
    episodes = [
        sample_hard_episode(HARD_FAMILY_BY_NAME[name], rng)
        for name in ("sharp_turn", "steep_vertical", "steep_vertical")
    ]
    report = diversity_report(episodes)
    assert report["samples"] == 3
    assert report["family_share"] == pytest.approx(
        {"sharp_turn": 1 / 3, "steep_vertical": 2 / 3}, abs=1e-5
    )
    assert report["dataset_split_share"]["train"] == pytest.approx(1.0)
    assert report["spacing_m"]["min"] <= report["spacing_m"]["max"]
    assert json.loads(json.dumps(report)) == report


# ---------------------------------------------- neighbouring-transition rules


def _hard_neighbours(episode, *, role):
    """(hard index, neighbour index) pairs a family's neighbour rule must govern.

    A transition sitting between two hard ones is claimed by the preceding rule
    when that rule constrains anything, so a family whose only rule is
    ``following`` never sees a "preceding" pair and vice versa.
    """

    spec = HARD_FAMILY_BY_NAME[episode.family]
    hard = set(episode.hard_positions)
    count = episode.geometry.gate_count - 1
    for index in range(count):
        if index in hard:
            continue
        precedes = index + 1 in hard
        follows = index - 1 >= 0 and index - 1 in hard
        if precedes and follows:
            precedes = spec.preceding_policy != "any" or spec.following_policy == "any"
            follows = not precedes
        if role == "preceding" and precedes:
            yield index + 1, index
        elif role == "following" and follows:
            yield index - 1, index


def test_hairpin_reversal_really_reverses_before_the_hard_turn():
    spec = HARD_FAMILY_BY_NAME["hairpin_reversal"]
    checked = 0
    for episode in _episodes(spec, 300, seed=17):
        turns = episode.geometry.turn_deltas_deg
        for hard, neighbour in _hard_neighbours(episode, role="preceding"):
            checked += 1
            assert turns[neighbour] * turns[hard] < 0, episode.as_dict()
            assert abs(turns[neighbour]) >= 0.6 * spec.turn_magnitude_range_deg[0]
    assert checked > 50, "the reversal constraint was never exercised"


def test_sharp_turn_gives_no_run_up_before_the_hard_transition():
    spec = HARD_FAMILY_BY_NAME["sharp_turn"]
    checked = 0
    for episode in _episodes(spec, 400, seed=19):
        turns = episode.geometry.turn_deltas_deg
        for hard, neighbour in _hard_neighbours(episode, role="preceding"):
            checked += 1
            assert abs(turns[neighbour]) <= STRAIGHT_TURN_RANGE_DEG[1]
            assert abs(turns[hard]) >= spec.turn_magnitude_range_deg[0]
    assert checked > 20


def test_turn_with_vertical_keeps_turning_the_same_way_afterwards():
    spec = HARD_FAMILY_BY_NAME["turn_with_vertical"]
    checked = 0
    for episode in _episodes(spec, 300, seed=23):
        turns = episode.geometry.turn_deltas_deg
        for hard, neighbour in _hard_neighbours(episode, role="following"):
            checked += 1
            assert turns[neighbour] * turns[hard] > 0
    assert checked > 20


def test_serpentine_alternates_every_transition():
    spec = HARD_FAMILY_BY_NAME["serpentine_alternation"]
    for episode in _episodes(spec, 150, seed=29):
        turns = episode.geometry.turn_deltas_deg
        assert len(episode.hard_positions) == len(turns), "every transition is hard"
        for previous, current in zip(turns, turns[1:]):
            assert previous * current < 0, turns


def test_sustained_same_direction_spirals_one_way_per_episode():
    """One direction inside an episode, both directions across the family."""

    spec = HARD_FAMILY_BY_NAME["sustained_same_direction"]
    directions = set()
    for episode in _episodes(spec, 150, seed=37):
        turns = episode.geometry.turn_deltas_deg
        signs = {v > 0 for v in turns}
        assert len(signs) == 1, turns
        assert min(abs(v) for v in turns) >= spec.turn_magnitude_range_deg[0]
        directions |= signs
    assert directions == {True, False}


@pytest.mark.parametrize("family,policy", [
    ("sustained_same_direction", "monotone"),
    ("serpentine_alternation", "alternating"),
])
def test_filler_transitions_still_obey_the_family_sign_policy(family, policy):
    """A family must keep its identity when only part of the episode is hard.

    Both shipped non-random families happen to make every transition hard, so a
    filler that turned freely would break nothing today and everything the first
    time a hard share below one is configured.
    """

    spec = dataclasses.replace(
        HARD_FAMILY_BY_NAME[family], name=f"{family}_partial",
        hard_share_range=(0.30, 0.50),
    )
    assert spec.turn_sign_policy == policy
    rng = np.random.default_rng(3)
    filler_seen = 0
    for _ in range(200):
        episode = sample_hard_episode(spec, rng)
        turns = episode.geometry.turn_deltas_deg
        filler_seen += len(turns) - len(episode.hard_positions)
        if policy == "monotone":
            assert len({v > 0 for v in turns}) == 1, turns
        else:
            assert all(a * b < 0 for a, b in zip(turns, turns[1:])), turns
    assert filler_seen > 100, "the filler branch was never exercised"


def test_long_chain_puts_the_hard_transitions_late_but_not_in_a_fixed_block():
    """The family's mechanism is depth, so uniform scatter would not be it."""

    spec = HARD_FAMILY_BY_NAME["long_chain_accumulation"]
    uniform = HARD_FAMILY_BY_NAME["tight_spacing"]
    assert spec.hard_position_bias == "late"
    assert uniform.hard_position_bias == "uniform"

    def _mean_relative_position(target):
        rng = np.random.default_rng(1)
        relative = []
        for _ in range(400):
            episode = sample_hard_episode(target, rng)
            last = episode.geometry.gate_count - 2
            relative += [p / last for p in episode.hard_positions]
        return sum(relative) / len(relative)

    late = _mean_relative_position(spec)
    assert late > 0.55, f"hard transitions are not late (mean {late:.3f})"
    assert _mean_relative_position(uniform) == pytest.approx(0.5, abs=0.05)
    # A fixed late block would be memorisable, so early positions must survive.
    rng = np.random.default_rng(2)
    firsts = {sample_hard_episode(spec, rng).hard_positions[0] for _ in range(300)}
    assert min(firsts) <= 1, "the early part of the chain is never hard"


def test_an_unknown_hard_position_bias_is_refused():
    broken = dataclasses.replace(
        HARD_FAMILY_BY_NAME["sharp_turn"], name="broken_bias",
        hard_position_bias="early_ish",
    )
    with pytest.raises(ValueError, match="hard_position_bias"):
        validate_specs([broken])


def test_tight_spacing_chains_hard_transitions_back_to_back():
    spec = HARD_FAMILY_BY_NAME["tight_spacing"]
    for episode in _episodes(spec, 200, seed=31):
        turns = episode.geometry.turn_deltas_deg
        for role in ("preceding", "following"):
            for _, neighbour in _hard_neighbours(episode, role=role):
                assert abs(turns[neighbour]) >= spec.turn_magnitude_range_deg[0]
        assert max(episode.geometry.spacings_m) <= spec.spacing_range_m[1]


# ------------------------------------------------------------- the envelope


@pytest.mark.parametrize("spec", HARD_FAMILY_SPECS, ids=HARD_FAMILY_NAMES)
def test_hard_geometry_stays_inside_the_validated_envelope(spec):
    """Hard means an unusual combination, not an out-of-contract value."""

    for geometry in _geometries(spec, 150, seed=41):
        assert 2 <= geometry.gate_count <= 24
        assert len(geometry.spacings_m) == geometry.gate_count - 1
        assert len(geometry.turn_deltas_deg) == geometry.gate_count - 1
        assert len(geometry.vertical_deltas_m) == geometry.gate_count - 1
        assert all(
            HARD_ENVELOPE.min_spacing_m <= v <= HARD_ENVELOPE.max_spacing_m
            for v in geometry.spacings_m
        )
        assert all(abs(v) <= HARD_ENVELOPE.max_turn_deg for v in geometry.turn_deltas_deg)
        assert all(
            abs(v) <= HARD_ENVELOPE.max_vertical_step_m
            for v in geometry.vertical_deltas_m
        )
        assert abs(geometry.initial_yaw_error_deg) <= HARD_ENVELOPE.max_initial_yaw_error_deg
        assert abs(geometry.initial_lateral_offset_m) <= HARD_ENVELOPE.max_lateral_offset_m
        assert all(
            abs(v) <= HARD_ENVELOPE.max_initial_speed_m_s
            for v in geometry.initial_body_velocity_m_s
        )
        assert geometry.sequence_bucket in BUCKET_GATE_RANGES
        assert isinstance(geometry, TransitionGeometry)


def test_a_spec_outside_the_envelope_is_refused_at_definition_time():
    reference = HARD_FAMILY_BY_NAME["sharp_turn"]
    too_sharp = dataclasses.replace(
        reference, name="too_sharp",
        turn_magnitude_range_deg=(60.0, 90.0),
    )
    with pytest.raises(ValueError, match="envelope"):
        validate_specs([too_sharp])
    # A neighbour rule that cannot exist on a two-gate episode is also refused.
    impossible = dataclasses.replace(
        reference, name="impossible", gate_count_range=(2, 3),
        preceding_policy="opposite_sign",
    )
    with pytest.raises(ValueError, match="at least two"):
        validate_specs([impossible])


# --------------------------------------------------------- sealed circuits


def test_no_generated_track_references_a_final_circuit(tmp_path):
    rng = np.random.default_rng(13)
    for spec in HARD_FAMILY_SPECS:
        assert not any(name in spec.name.lower() for name in FINAL_CIRCUIT_NAMES)
        for _ in range(20):
            geometry = sample_hard_geometry(spec, rng)
            for text in (geometry.pattern, geometry.geometry_group,
                         track_label(geometry)):
                assert not any(
                    name in str(text).lower() for name in FINAL_CIRCUIT_NAMES
                ), text
    geometry = sample_hard_geometry(HARD_FAMILY_BY_NAME["sharp_turn"], rng)
    target = generate_transition_track(
        geometry, tmp_path / "hard_episode.json", base_track=BASE_TRACK
    )
    body = target.read_text(encoding="utf-8").lower()
    assert not any(name in body for name in FINAL_CIRCUIT_NAMES)


def test_hard_geometry_drives_the_real_track_generator(tmp_path):
    geometry = sample_hard_geometry(
        HARD_FAMILY_BY_NAME["hairpin_reversal"], np.random.default_rng(77)
    )
    target = generate_transition_track(
        geometry, tmp_path / "episode.json", base_track=BASE_TRACK
    )
    data = json.loads(target.read_text(encoding="utf-8"))
    assert len(data["gates"]) == geometry.gate_count
    assert data["universal_transition"]["pattern"] == geometry.pattern
    assert data["universal_transition"]["dataset_split"] == "train"
    assert data["water_fog"] == {
        "enabled": True,
        "density": 5.0,
        "start_distance_m": 1.0,
        "color_rgb": [0.4, 0.6, 1.0],
    }
    headings = [gate["rotation_rpy_deg"][2] for gate in data["gates"]]
    deltas = [round(b - a, 3) for a, b in zip(headings, headings[1:])]
    assert deltas == [round(v, 3) for v in geometry.turn_deltas_deg]


def test_the_family_survives_into_the_episode_composition_log():
    """Otherwise the realised mixture cannot be verified from logs later."""

    geometry = sample_hard_geometry(
        HARD_FAMILY_BY_NAME["tight_spacing"], np.random.default_rng(3),
        curriculum_stage=STAGES[0]["name"],
    )
    row = episode_row(utc="t", algorithm="ppo", run="r", worker_id=0,
                      geometry=geometry)
    assert row["pattern"] == f"{HARD_PATTERN_PREFIX}tight_spacing"
    assert family_of(geometry) == "tight_spacing"
    assert row["dataset_split"] == "train"
    assert row["curriculum_stage"] == STAGES[0]["name"]
    assert json.dumps(row)


# ---------------------------------------------------------------- the mixer


def test_mixer_honours_the_requested_base_hard_ratio():
    mixer = HardCaseMixer(DEFAULT_BASE_FRACTION, rng=np.random.default_rng(101))
    sources = [mixer.next_source()[0] for _ in range(6000)]
    hard = sources.count("hard") / len(sources)
    assert hard == pytest.approx(1.0 - DEFAULT_BASE_FRACTION, abs=0.02)
    assert sources.count("base") / len(sources) == pytest.approx(
        DEFAULT_BASE_FRACTION, abs=0.02
    )


@pytest.mark.parametrize("base_fraction", [0.55, 0.65, 0.80, 1.0])
def test_requested_ratios_are_reproduced_across_the_allowed_range(base_fraction):
    mixer = HardCaseMixer(base_fraction, rng=np.random.default_rng(7))
    sources = [mixer.next_source()[0] for _ in range(4000)]
    assert sources.count("hard") / len(sources) == pytest.approx(
        1.0 - base_fraction, abs=0.02
    )


@pytest.mark.parametrize("base_fraction", [0.49, 0.35, 0.0, -0.1, 1.5])
def test_the_mixer_refuses_a_hard_fraction_above_the_cap(base_fraction):
    with pytest.raises(ValueError):
        HardCaseMixer(base_fraction, rng=np.random.default_rng(0))


def test_the_cap_is_exactly_max_hard_fraction():
    assert MAX_HARD_FRACTION == 0.50
    assert MIN_BASE_FRACTION == pytest.approx(1.0 - MAX_HARD_FRACTION)
    boundary = HardCaseMixer(MIN_BASE_FRACTION, rng=np.random.default_rng(2))
    for _ in range(3000):
        boundary.next_source()
    composition = boundary.realised_composition()
    assert composition["realised_hard_fraction"] <= MAX_HARD_FRACTION
    assert composition["hard_fraction_within_cap"] is True


def test_realised_composition_reports_the_true_observed_mixture():
    mixer = HardCaseMixer(0.70, rng=np.random.default_rng(5))
    observed = [mixer.next_source() for _ in range(1500)]
    hard = [family for source, family in observed if source == "hard"]
    composition = mixer.realised_composition()
    assert composition["episodes"] == 1500
    assert composition["hard_episodes"] == len(hard)
    assert composition["base_episodes"] == 1500 - len(hard)
    assert composition["realised_hard_fraction"] == pytest.approx(len(hard) / 1500)
    assert composition["requested_hard_fraction"] == pytest.approx(0.30)
    for name in HARD_FAMILY_NAMES:
        assert composition["family_counts"][name] == hard.count(name)
    assert sum(composition["family_share_of_hard"].values()) == pytest.approx(
        1.0, abs=1e-4
    )
    assert json.loads(json.dumps(composition)) == composition


def test_family_weights_steer_which_family_is_drawn():
    mixer = HardCaseMixer(
        0.60, {"sharp_turn": 3.0, "steep_vertical": 1.0},
        rng=np.random.default_rng(6),
    )
    families = [
        family for source, family in
        (mixer.next_source() for _ in range(4000)) if source == "hard"
    ]
    assert set(families) == {"sharp_turn", "steep_vertical"}
    assert families.count("sharp_turn") / len(families) == pytest.approx(0.75, abs=0.03)


@pytest.mark.parametrize("weights", [
    {"no_such_family": 1.0},
    {"sharp_turn": -1.0},
    {"sharp_turn": 0.0},
])
def test_invalid_family_weights_are_refused(weights):
    with pytest.raises(ValueError):
        HardCaseMixer(0.65, weights, rng=np.random.default_rng(0))


def test_the_mixer_draws_base_episodes_from_the_normal_s0_distribution():
    curriculum = GenericSequenceCurriculum()
    sampler = TransitionGeometrySampler(
        seed=1_234, difficulty="G6", sequence_curriculum=curriculum,
        dataset_split="train",
    )
    mixer = HardCaseMixer(0.65, rng=np.random.default_rng(8))
    geometries = [mixer.sample(sampler) for _ in range(800)]
    hard = [g for g in geometries if family_of(g) is not None]
    base = [g for g in geometries if family_of(g) is None]
    composition = mixer.realised_composition()
    assert len(hard) == composition["hard_episodes"]
    assert len(base) == composition["base_episodes"]
    assert len(hard) / len(geometries) == pytest.approx(0.35, abs=0.05)
    # Base episodes really come from the curriculum, hard ones carry its stage.
    assert {g.pattern for g in base} & {"aligned", "yaw", "elevation", "combined"}
    assert all(g.curriculum_stage == STAGES[0]["name"] for g in geometries)
    assert all(seed_group("train").contains(g.seed) for g in geometries)
    assert max(g.gate_count for g in hard) > BUCKET_GATE_RANGES["medium"][1], (
        "hard families must reach lengths S0 alone never produces"
    )


def test_the_mixer_refuses_to_inject_hard_cases_into_a_holdout_rollout():
    class _ValidationSampler:
        """A holdout rollout worker: its base episodes are fine, hard ones are not."""

        dataset_split = "validation"
        drawn = 0

        def sample(self):
            self.drawn += 1
            return "validation_episode"

    sampler = _ValidationSampler()
    mixer = HardCaseMixer(0.50, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="TRAIN-only"):
        for _ in range(200):
            assert mixer.sample(sampler) == "validation_episode"
    # Whatever the mixer served before refusing came from the holdout sampler.
    composition = mixer.realised_composition()
    assert composition["base_episodes"] == sampler.drawn
    # A refused draw is not an episode: counting it would report hard training
    # that never happened, which is precisely what the composition log is for.
    assert composition["hard_episodes"] == 0
    assert composition["realised_hard_fraction"] == 0.0
    assert composition["episodes"] == sampler.drawn
    assert composition["refused_draws"] == 1
    assert set(composition["family_counts"].values()) == {0}


def test_a_refused_draw_is_never_counted_however_often_it_happens():
    """A caller that keeps going past the refusal must still get honest counts."""

    class _ValidationSampler:
        dataset_split = "validation"
        drawn = 0

        def sample(self):
            self.drawn += 1
            return "validation_episode"

    sampler = _ValidationSampler()
    mixer = HardCaseMixer(0.50, rng=np.random.default_rng(0))
    refused = 0
    for _ in range(400):
        try:
            mixer.sample(sampler)
        except ValueError:
            refused += 1
    composition = mixer.realised_composition()
    assert refused > 100, "the hard branch was barely exercised"
    assert composition["refused_draws"] == refused
    assert composition["hard_episodes"] == 0
    assert composition["episodes"] == sampler.drawn == 400 - refused


# ----------------------------------------------------- determinism and json


def test_generation_is_deterministic_given_the_seed():
    spec = HARD_FAMILY_BY_NAME["long_chain_accumulation"]
    first = [
        e.as_dict() for e in _episodes(spec, 25, seed=1234)
    ]
    second = [e.as_dict() for e in _episodes(spec, 25, seed=1234)]
    third = [e.as_dict() for e in _episodes(spec, 25, seed=1235)]
    assert first == second
    assert first != third


def test_mixer_state_round_trips_and_resumes_the_same_stream():
    mixer = HardCaseMixer(0.65, rng=np.random.default_rng(21))
    for _ in range(120):
        mixer.next_source()
    state = json.loads(json.dumps(mixer.state_dict()))
    resumed = HardCaseMixer(0.65, rng=np.random.default_rng(0))
    resumed.load_state_dict(state)
    assert resumed.realised_composition() == mixer.realised_composition()
    assert [resumed.next_source() for _ in range(50)] == [
        mixer.next_source() for _ in range(50)
    ]
    with pytest.raises(ValueError, match="mixture changed"):
        HardCaseMixer(0.80, rng=np.random.default_rng(0)).load_state_dict(state)


def test_resume_refuses_a_changed_family_mixture():
    """Same families at different weights changes the distribution just as much."""

    mixer = HardCaseMixer(
        0.65, {"sharp_turn": 1.0, "steep_vertical": 1.0},
        rng=np.random.default_rng(1),
    )
    for _ in range(50):
        mixer.next_source()
    state = json.loads(json.dumps(mixer.state_dict()))
    with pytest.raises(ValueError, match="weights changed"):
        HardCaseMixer(
            0.65, {"sharp_turn": 99.0, "steep_vertical": 1.0},
            rng=np.random.default_rng(2),
        ).load_state_dict(state)
    with pytest.raises(ValueError, match="family set changed"):
        HardCaseMixer(
            0.65, {"sharp_turn": 1.0, "tight_spacing": 1.0},
            rng=np.random.default_rng(2),
        ).load_state_dict(state)
    # An unchanged mixture still resumes, weights normalised the same way.
    resumed = HardCaseMixer(
        0.65, {"sharp_turn": 4.0, "steep_vertical": 4.0},
        rng=np.random.default_rng(3),
    )
    resumed.load_state_dict(state)
    assert resumed.realised_composition() == mixer.realised_composition()


def test_everything_reported_is_json_serialisable():
    rng = np.random.default_rng(99)
    episodes = [
        sample_hard_episode(HARD_FAMILY_BY_NAME[name], rng)
        for name in HARD_FAMILY_NAMES
    ]
    manifest = hard_family_manifest()
    assert json.loads(json.dumps(manifest))["max_hard_fraction"] == MAX_HARD_FRACTION
    assert len(manifest["families"]) == len(HARD_FAMILY_SPECS)
    for episode in episodes:
        assert json.loads(json.dumps(episode.as_dict()))["family"] == episode.family
    assert json.dumps(diversity_report(episodes))


# ------------------------------------------------------------- taxonomy map


def test_every_taxonomy_failure_label_has_at_least_one_family():
    """A diagnosed failure with no training distribution behind it is a dead end."""

    assert uncovered_failure_labels() == ()
    for label in TAXONOMY_FAILURE_LABELS:
        families = families_for_failure(label)
        assert families, label
        assert all(name in HARD_FAMILY_BY_NAME for name in families)


@pytest.mark.parametrize("label,expected", [
    ("alternating_s_transitions", "serpentine_alternation"),
    ("consecutive_same_direction_turns", "sustained_same_direction"),
    ("combined_yaw_vertical", "turn_with_vertical"),
    ("short_spacing", "tight_spacing"),
    ("long_spacing", "wide_spacing_drift"),
    ("first_gate_acquisition", "offset_entry"),
    ("vertical_climb", "steep_vertical"),
    ("vertical_descent", "steep_vertical"),
    ("large_left_turn", "sharp_turn"),
    ("large_right_turn", "sharp_turn"),
    ("accumulated_lateral_error", "long_chain_accumulation"),
])
def test_each_reported_family_resolves_to_the_geometry_that_trains_it(label, expected):
    assert expected in families_for_failure(label)


def test_failure_lookup_tolerates_taxonomy_label_formatting():
    assert "hairpin_reversal" in families_for_failure("Wrong-Direction Events")
    assert families_for_failure("collision") , "a bare label must still resolve"
    assert families_for_failure("no_such_failure") == ()
    # A blank or degenerate label must not substring-match its way to full
    # coverage: that would report every failure as already trained for.
    for empty in ("", "   ", "_", "no"):
        assert families_for_failure(empty) == (), empty
    assert uncovered_failure_labels(("", "no_such_failure")) == (
        "", "no_such_failure"
    )


def test_the_family_set_spans_the_geometric_failure_mechanisms():
    """Every family must be reachable and describe why it is hard."""

    assert len(HARD_FAMILY_SPECS) >= 8
    assert len(HARD_FAMILY_NAMES) == len(set(HARD_FAMILY_NAMES))
    for spec in HARD_FAMILY_SPECS:
        assert isinstance(spec, HardFamilySpec)
        assert spec.description.strip()
        assert spec.failure_labels
        assert spec.pattern.startswith(HARD_PATTERN_PREFIX)
    lengths = {spec.gate_count_range for spec in HARD_FAMILY_SPECS}
    assert min(low for low, _ in lengths) == 2, "two-gate transitions covered"
    assert max(high for _, high in lengths) >= 13, "long chains covered"
