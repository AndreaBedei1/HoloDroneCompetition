"""Failure families, survivor-bias-free survival curves and the seed guard."""

from __future__ import annotations

import json

import pytest

from marine_race_arena.learning.rl_failure_taxonomy import (
    ACCUMULATION_MIN_POSITION,
    DEFAULT_LARGE_TURN_DEG,
    FAILURE_FAMILIES,
    FAILURE_FAMILY_NAMES,
    FAMILY_BY_NAME,
    FailureRecord,
    FamilyThresholds,
    SeedMemorisationError,
    assert_no_seed_memorisation,
    classify_failure,
    classify_many,
    failure_position_distribution,
    load_evaluation_records,
    mine_failure_families,
    survival_curve,
)
from marine_race_arena.learning.rl_holdout_policy import seed_group

DEFAULTS = FamilyThresholds.defaults()


def _record(**overrides) -> FailureRecord:
    """A mid-course failure whose geometry is deliberately unremarkable."""

    base = dict(
        gate_count=8,
        gates_completed=4,
        episode_type="full_sequence",
        pattern="combined",
        difficulty="G6",
        spacings_m=(6.5,) * 7,
        turn_deltas_deg=(2.0,) * 7,
        vertical_deltas_m=(0.05,) * 7,
        initial_lateral_offset_m=0.05,
        crossing_offset_m=0.05,
        first_gate_crossed=True,
        target_switched=True,
        new_target_aligned=True,
        new_target_range_decreasing=True,
        succeeded=False,
    )
    base.update(overrides)
    return FailureRecord(**base)


#: The control: a failure that exemplifies nothing.  Every family predicate is
#: checked against it, so a predicate that fires on anything is caught.
INERT = _record()

#: One hand-built record per family, each engineered so the named family fires.
EXEMPLARS = {
    "first_gate_acquisition": _record(
        gates_completed=0, first_gate_crossed=False, acquisition_timeout=True,
        target_switched=False, new_target_aligned=False,
    ),
    "off_centre_crossing": _record(gates_completed=1, crossing_offset_m=1.10),
    "poor_target_switch": _record(gates_completed=1, target_switched=False),
    "reacquisition_after_crossing": _record(
        gates_completed=1, target_switched=True, new_target_aligned=False),
    "large_left_turn": _record(
        turn_deltas_deg=(2.0, 2.0, 2.0, 45.0, 2.0, 2.0, 2.0)),
    "large_right_turn": _record(
        turn_deltas_deg=(2.0, 2.0, 2.0, -45.0, 2.0, 2.0, 2.0)),
    "vertical_climb": _record(
        vertical_deltas_m=(0.05, 0.05, 0.05, 1.80, 0.05, 0.05, 0.05)),
    "vertical_descent": _record(
        vertical_deltas_m=(0.05, 0.05, 0.05, -1.80, 0.05, 0.05, 0.05)),
    # Neither axis is large on its own; the coupling is the failure.
    "combined_yaw_vertical": _record(
        turn_deltas_deg=(2.0, 2.0, 2.0, 25.0, 2.0, 2.0, 2.0),
        vertical_deltas_m=(0.05, 0.05, 0.05, 0.90, 0.05, 0.05, 0.05)),
    "short_spacing": _record(
        spacings_m=(6.5, 6.5, 6.5, 3.20, 6.5, 6.5, 6.5)),
    "long_spacing": _record(
        spacings_m=(6.5, 6.5, 6.5, 9.50, 6.5, 6.5, 6.5)),
    "consecutive_same_direction_turns": _record(
        turn_deltas_deg=(40.0, 40.0, 40.0, 40.0, 2.0, 2.0, 2.0)),
    "alternating_s_transitions": _record(
        turn_deltas_deg=(40.0, -40.0, 40.0, -40.0, 2.0, 2.0, 2.0)),
    "collision_after_crossing": _record(gates_completed=2, collision_events=1),
    "missed_gate": _record(missed_gate_events=1),
    "wrong_direction": _record(wrong_direction_events=1),
    "out_of_bounds_excursion": _record(out_of_bounds_events=1),
    "accumulated_lateral_error": _record(
        gates_completed=5, turn_deltas_deg=(25.0,) * 7),
    "accumulated_vertical_error": _record(
        gates_completed=5, vertical_deltas_m=(0.90,) * 7),
}


# ------------------------------------------------------------- the taxonomy

def test_taxonomy_declares_every_required_family_in_a_stable_order():
    required = (
        "first_gate_acquisition", "off_centre_crossing", "poor_target_switch",
        "reacquisition_after_crossing", "large_left_turn", "large_right_turn",
        "vertical_climb", "vertical_descent", "combined_yaw_vertical",
        "short_spacing", "long_spacing", "consecutive_same_direction_turns",
        "alternating_s_transitions", "collision_after_crossing", "missed_gate",
        "wrong_direction", "accumulated_lateral_error",
        "accumulated_vertical_error",
    )
    assert set(required) <= set(FAILURE_FAMILY_NAMES)
    assert len(FAILURE_FAMILY_NAMES) == len(FAILURE_FAMILIES)
    assert all(family.description.strip() for family in FAILURE_FAMILIES)
    assert all(1 <= family.specificity <= 3 for family in FAILURE_FAMILIES)
    assert all(0.0 < family.severity <= 1.0 for family in FAILURE_FAMILIES)
    # The exemplar table below must keep pace with the taxonomy.
    assert set(EXEMPLARS) == set(FAILURE_FAMILY_NAMES)


@pytest.mark.parametrize("family", sorted(EXEMPLARS))
def test_each_family_fires_on_its_exemplar_and_not_on_an_unrelated_record(family):
    predicate = FAMILY_BY_NAME[family]
    assert predicate.matches(EXEMPLARS[family], DEFAULTS), family
    assert family in classify_failure(EXEMPLARS[family], DEFAULTS)
    assert not predicate.matches(INERT, DEFAULTS), family


def test_an_unremarkable_failure_matches_nothing():
    """Not every failure is a family; forcing a label would invent signal."""

    assert classify_failure(INERT, DEFAULTS) == []


def test_a_successful_episode_never_reports_geometry_families():
    """Geometry families are anchored at the leg the episode died on."""

    survivor = _record(
        gates_completed=8, succeeded=True,
        turn_deltas_deg=(50.0,) * 7, spacings_m=(3.0,) * 7,
    )
    assert survivor.failing_leg_index is None
    assert classify_failure(survivor, DEFAULTS) == []


def test_classification_returns_every_matching_family_most_specific_first():
    record = _record(
        gates_completed=4,
        collision_events=1,
        turn_deltas_deg=(40.0, 40.0, 40.0, 40.0, 2.0, 2.0, 2.0),
        spacings_m=(6.5, 6.5, 6.5, 3.0, 6.5, 6.5, 6.5),
    )
    families = classify_failure(record, DEFAULTS)
    assert families == [
        "collision_after_crossing",       # specificity 3: the terminal event
        "consecutive_same_direction_turns",  # 2: the multi-leg pattern
        "large_left_turn",                # 1: single-axis geometry
        "short_spacing",
    ]
    specificities = [FAMILY_BY_NAME[name].specificity for name in families]
    assert specificities == sorted(specificities, reverse=True)


def test_unrecorded_signals_never_read_as_negative_ones():
    """Train rows carry no target-switch flags; absence is not failure."""

    unknown = _record(
        gates_completed=1, target_switched=None,
        new_target_aligned=None, new_target_range_decreasing=None,
    )
    assert "poor_target_switch" not in classify_failure(unknown, DEFAULTS)
    assert "reacquisition_after_crossing" not in classify_failure(unknown, DEFAULTS)


def test_off_centre_falls_back_to_the_spawn_offset_only_on_the_first_gate():
    """Nothing but the spawn has acted on the lateral axis before gate one."""

    first = _record(
        gates_completed=1, crossing_offset_m=None, initial_lateral_offset_m=1.15)
    assert first.crossing_offset_is_estimated
    assert "off_centre_crossing" in classify_failure(first, DEFAULTS)
    deep = _record(
        gates_completed=4, crossing_offset_m=None, initial_lateral_offset_m=1.15)
    assert deep.crossing_offset_estimate_m is None
    assert "off_centre_crossing" not in classify_failure(deep, DEFAULTS)


def test_a_completed_run_is_never_labelled_by_a_logged_event():
    """A sequence can finish while the referee still logged a bump."""

    events = dict(
        collision_events=1, missed_gate_events=2,
        wrong_direction_events=1, out_of_bounds_events=1,
    )
    finished = _record(gates_completed=8, succeeded=True, **events)
    assert classify_failure(finished, DEFAULTS) == []
    assert classify_many([finished], DEFAULTS).unclassified == 1
    # The same events on a genuine failure are exactly what the families exist
    # for, so the gate is on the outcome and not on the events being ignored.
    died = _record(gates_completed=2, **events)
    assert set(classify_failure(died, DEFAULTS)) == {
        "collision_after_crossing", "missed_gate", "wrong_direction",
        "out_of_bounds_excursion",
    }


def test_accumulation_families_need_a_chained_failure_not_a_bad_corner():
    shallow = _record(gates_completed=1, turn_deltas_deg=(25.0,) * 7)
    assert shallow.failure_position < ACCUMULATION_MIN_POSITION
    assert "accumulated_lateral_error" not in classify_failure(shallow, DEFAULTS)
    # A discrete event explains the failure, so drift must not also claim it.
    with_event = _record(
        gates_completed=5, turn_deltas_deg=(25.0,) * 7, collision_events=1)
    assert "accumulated_lateral_error" not in classify_failure(with_event, DEFAULTS)


# ------------------------------------------------------------- thresholds

def test_thresholds_are_relative_to_the_observed_distribution():
    """A 12 degree turn is large in a gentle cohort and ordinary by default."""

    cohort = [
        _record(
            turn_deltas_deg=(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0),
            spacings_m=(4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0),
            vertical_deltas_m=(0.05,) * 7,
        )
        for _ in range(10)
    ]
    observed = FamilyThresholds.from_records(cohort)
    assert observed.large_turn_deg == pytest.approx(12.0)
    assert observed.large_turn_deg < DEFAULT_LARGE_TURN_DEG
    assert observed.short_spacing_m == pytest.approx(5.0)
    assert observed.long_spacing_m == pytest.approx(9.0)
    assert observed.source in {"observed_distribution", "mixed"}

    borderline = _record(
        gates_completed=6,
        turn_deltas_deg=(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0),
        spacings_m=(6.5,) * 7,
    )
    assert "large_left_turn" in classify_failure(borderline, observed)
    assert "large_left_turn" not in classify_failure(borderline, DEFAULTS)


def test_too_few_samples_keep_the_curriculum_derived_defaults():
    thresholds = FamilyThresholds.from_records([_record()])
    assert thresholds.source == "curriculum_defaults"
    assert thresholds.large_turn_deg == pytest.approx(DEFAULT_LARGE_TURN_DEG)


def test_distribution_counts_every_matching_family_and_reports_shares():
    records = [EXEMPLARS["collision_after_crossing"]] * 3 + [INERT]
    distribution = classify_many(records, DEFAULTS)
    assert distribution["collision_after_crossing"] == 3
    assert distribution["missed_gate"] == 0
    assert distribution.share("collision_after_crossing") == pytest.approx(0.75)
    assert distribution.unclassified == 1
    assert distribution.most_common(1)[0][0] == "collision_after_crossing"
    assert distribution.as_dict()["unclassified_share"] == pytest.approx(0.25)


# ------------------------------------------- survivor-bias-free survival

def _cohort():
    """100 starters, 50 reach gate 2, 25 reach gate 3, none finish."""

    rows = []
    for completed, count in ((1, 50), (2, 25), (3, 25)):
        rows.extend([{
            "gate_count": 5,
            "gates_completed": completed,
            "episode_type": "full_sequence",
            "full_sequence_completion": False,
        }] * count)
    return rows


def test_survival_from_start_is_not_the_conditional_rate():
    curve = survival_curve(_cohort())
    assert curve["n_episodes"] == 100
    from_start = curve["reach_from_start"]
    conditional = curve["conditional_on_previous"]
    assert from_start["1"] == pytest.approx(1.00)
    assert from_start["2"] == pytest.approx(0.50)
    assert from_start["3"] == pytest.approx(0.25)
    assert conditional["1"] == pytest.approx(1.00)
    assert conditional["2"] == pytest.approx(0.50)
    assert conditional["3"] == pytest.approx(0.50)
    # The whole point: at position 3 the two numbers differ by 2x.
    assert from_start["3"] != conditional["3"]
    assert curve["max_conditional_minus_from_start"] == pytest.approx(0.25)


def test_conditional_rates_alone_make_a_dead_policy_look_perfect():
    """One survivor out of a hundred reads 1.00 when conditioned."""

    rows = [{"gate_count": 12, "gates_completed": 0} for _ in range(99)]
    rows.append({"gate_count": 12, "gates_completed": 12})
    curve = survival_curve(rows)
    assert curve["conditional_on_previous"]["12"] == pytest.approx(1.00)
    assert curve["reach_from_start"]["12"] == pytest.approx(0.01)


def test_half_survival_position_is_the_first_strict_drop_below_half():
    curve = survival_curve(_cohort())
    # Position 2 sits exactly at 0.5 and therefore has not dropped below it.
    assert curve["reach_from_start"]["2"] == pytest.approx(0.5)
    assert curve["first_position_below_half"] == 3


def test_expected_gates_completed_matches_the_hand_computed_mean():
    curve = survival_curve(_cohort())
    # (50*1 + 25*2 + 25*3) / 100
    assert curve["expected_gates_completed"] == pytest.approx(1.75)
    # E[gates] is the sum of the unconditional survival probabilities.
    assert sum(
        curve["reach_from_start"][position] for position in curve["positions"]
    ) == pytest.approx(1.75)


def test_transition_focus_episodes_can_be_excluded_from_the_curve():
    rows = _cohort() + [{
        "gate_count": 2, "gates_completed": 1,
        "episode_type": "transition_focus",
        "universal_transition_success": True,
    }] * 200
    full_only = survival_curve(rows, episode_types=("full_sequence",))
    assert full_only["n_episodes"] == 100
    assert full_only["reach_from_start"]["2"] == pytest.approx(0.5)


def test_failure_positions_are_normalised_by_exposure():
    distribution = failure_position_distribution(_cohort())
    assert distribution["n_failures"] == 100
    assert distribution["failures_at_position"] == {
        "1": 0, "2": 50, "3": 25, "4": 25, "5": 0}
    assert distribution["at_risk_at_position"] == {
        "1": 100, "2": 100, "3": 50, "4": 25, "5": 0}
    # Raw share peaks early only because everyone gets there.
    assert distribution["failure_share"]["2"] == pytest.approx(0.50)
    assert distribution["peak_share_position"] == 2
    # Normalised by exposure, position 4 is where the policy always dies.
    assert distribution["hazard_rate"]["4"] == pytest.approx(1.0)
    assert distribution["peak_hazard_position"] == 4
    assert distribution["hazard_rate"]["5"] is None


def _mixed_length_cohort():
    """10 short courses finished, 10 long courses stopped six gates in."""

    short = [{
        "gate_count": 3, "gates_completed": 3, "episode_type": "full_sequence",
        "full_sequence_completion": True,
    }] * 10
    long = [{
        "gate_count": 10, "gates_completed": 6, "episode_type": "full_sequence",
        "full_sequence_completion": False,
    }] * 10
    return short + long


def test_deep_positions_are_divided_by_the_courses_that_contained_them():
    """A 3-gate course cannot be evidence about gate 4 either way."""

    curve = survival_curve(_mixed_length_cohort())
    assert curve["exposure"] == {
        "1": 20, "2": 20, "3": 20, "4": 10, "5": 10,
        "6": 10, "7": 10, "8": 10, "9": 10, "10": 10,
    }
    # Every course that *had* a gate 4 reached it; dividing by the 20 starters
    # instead would report 0.5 and blame the policy for the shorter cohort.
    assert curve["reach_from_start"]["4"] == pytest.approx(1.0)
    assert curve["reach_from_start"]["7"] == pytest.approx(0.0)
    assert curve["first_position_below_half"] == 7
    assert curve["expected_gates_completed"] == pytest.approx(4.5)

    distribution = failure_position_distribution(_mixed_length_cohort())
    assert distribution["n_failures"] == 10
    assert distribution["at_risk_at_position"]["4"] == 10
    assert distribution["at_risk_at_position"]["8"] == 0
    assert distribution["hazard_rate"]["7"] == pytest.approx(1.0)
    assert distribution["peak_hazard_position"] == 7


def test_a_truncated_transition_success_is_not_a_failure_at_its_second_gate():
    """Focus cases end a window after crossing; that is censoring, not death."""

    rows = [{
        "gate_count": 2, "gates_completed": 1,
        "episode_type": "transition_focus",
        "universal_transition_success": success,
    } for success in [True] * 100 + [False] * 20]
    distribution = failure_position_distribution(rows)
    assert distribution["n_failures"] == 20
    assert distribution["failures_at_position"]["2"] == 20
    assert distribution["hazard_rate"]["2"] == pytest.approx(20 / 120)
    curve = survival_curve(rows)
    assert curve["reach_from_start"]["2"] == pytest.approx(100 / 120)
    # A policy that passes every transition must not read as failing them all.
    perfect = failure_position_distribution(rows[:100])
    assert perfect["n_failures"] == 0
    assert survival_curve(rows[:100])["first_position_below_half"] is None


def test_hazard_is_the_complement_of_the_conditional_survival_rate():
    curve = survival_curve(_cohort())
    distribution = failure_position_distribution(_cohort())
    for position in curve["positions"]:
        conditional = curve["conditional_on_previous"][position]
        hazard = distribution["hazard_rate"][position]
        if conditional is None or hazard is None:
            continue
        assert hazard == pytest.approx(1.0 - conditional), position


# ----------------------------------------------------------------- mining

def test_mining_ranks_a_dominant_family_above_a_rare_one():
    dominant = [
        _record(
            gate_count=6, gates_completed=2,
            spacings_m=(6.0,) * 5,
            turn_deltas_deg=(2.0, 50.0, 3.0, 1.0, 1.0),
            vertical_deltas_m=(0.05,) * 5,
        )
        for _ in range(30)
    ]
    rare = [
        _record(
            gate_count=6, gates_completed=2,
            spacings_m=(6.0, 9.0, 6.0, 6.0, 6.0),
            turn_deltas_deg=(2.0, 3.0, 3.0, 1.0, 1.0),
            vertical_deltas_m=(0.05,) * 5,
        )
        for _ in range(3)
    ]
    report = mine_failure_families(dominant + rare, thresholds=DEFAULTS)
    assert report["n_failures"] == 33
    assert report["unclassified_failures"] == 0
    names = [row["family"] for row in report["families"]]
    assert names[0] == "large_left_turn"
    assert names.index("large_left_turn") < names.index("long_spacing")
    ranked = {row["family"]: row for row in report["families"]}
    assert ranked["large_left_turn"]["rank"] == 1
    assert ranked["large_left_turn"]["count"] == 30
    assert ranked["long_spacing"]["count"] == 3
    assert ranked["large_left_turn"]["priority"] > ranked["long_spacing"]["priority"]
    assert report["top_families"][0] == "large_left_turn"


def test_ranking_is_frequency_times_severity_not_frequency_alone():
    """Ten race-ending collisions outrank twenty legs that were merely long."""

    mild = [
        _record(
            gate_count=6, gates_completed=2,
            spacings_m=(6.0, 9.0, 6.0, 6.0, 6.0),
            turn_deltas_deg=(2.0,) * 5, vertical_deltas_m=(0.05,) * 5,
        )
        for _ in range(20)
    ]
    severe = [
        _record(
            gate_count=6, gates_completed=2, collision_events=1,
            spacings_m=(6.0,) * 5,
            turn_deltas_deg=(2.0,) * 5, vertical_deltas_m=(0.05,) * 5,
        )
        for _ in range(10)
    ]
    report = mine_failure_families(mild + severe, thresholds=DEFAULTS)
    ranked = {row["family"]: row for row in report["families"]}
    assert ranked["long_spacing"]["count"] == 20
    assert ranked["collision_after_crossing"]["count"] == 10
    assert report["families"][0]["family"] == "collision_after_crossing"
    assert ranked["collision_after_crossing"]["priority"] == pytest.approx(
        10 / 30 * FAMILY_BY_NAME["collision_after_crossing"].severity, abs=1e-6)
    assert ranked["long_spacing"]["priority"] == pytest.approx(
        20 / 30 * FAMILY_BY_NAME["long_spacing"].severity, abs=1e-6)


def test_the_report_separates_the_chaining_curve_from_the_blended_one():
    """A 2-gate focus case and a 22-gate sequence are not the same question."""

    focus = [{
        "gate_count": 2, "gates_completed": 1,
        "episode_type": "transition_focus",
        "universal_transition_success": True,
    }] * 40
    sequences = [{
        "gate_count": 8, "gates_completed": 3,
        "episode_type": "full_sequence",
        "full_sequence_completion": False,
    }] * 10
    report = mine_failure_families(focus + sequences, thresholds=DEFAULTS)
    assert report["n_failures"] == 10
    assert report["survival"]["n_episodes"] == 50
    chaining = report["survival_chaining"]
    assert chaining["n_episodes"] == 10
    # Blended, position 2 is dominated by the focus cohort; alone, the full
    # sequences say every run died on the fourth gate.
    assert report["survival"]["reach_from_start"]["2"] == pytest.approx(1.0)
    assert chaining["reach_from_start"]["4"] == pytest.approx(0.0)
    assert chaining["first_position_below_half"] == 4
    assert mine_failure_families(focus, thresholds=DEFAULTS)["survival_chaining"] is None


def test_mined_families_carry_the_context_a_generator_needs():
    records = [
        _record(
            gate_count=6, gates_completed=3,
            spacings_m=(6.0, 4.0, 5.0, 6.0, 6.0),
            turn_deltas_deg=(2.0, 10.0, 50.0, 1.0, 1.0),
            vertical_deltas_m=(0.05, 0.10, 0.05, 0.05, 0.05),
        )
        for _ in range(12)
    ]
    report = mine_failure_families(records, thresholds=DEFAULTS)
    context = {row["family"]: row["context"] for row in report["families"]}
    left = context["large_left_turn"]
    assert left["turn_abs_deg"]["p50"] == pytest.approx(50.0)
    assert left["spacing_m"]["p50"] == pytest.approx(5.0)
    assert left["failure_position"]["p50"] == pytest.approx(4.0)
    assert left["gate_count"]["p50"] == pytest.approx(6.0)
    # Preceding-transition geometry is what makes the family reproducible.
    assert left["preceding_turn_abs_deg"]["p50"] == pytest.approx(10.0)
    assert left["preceding_spacing_m"]["p50"] == pytest.approx(4.0)
    assert left["preceding_vertical_abs_m"]["p50"] == pytest.approx(0.10)
    assert left["difficulty_share"] == {"G6": 1.0}


def test_evaluation_cases_supply_geometry_but_never_episode_identity(tmp_path):
    """Per-case ``track.json`` carries the legs; its seed must not survive."""

    seed = seed_group("validation").start + 11
    case = tmp_path / "transitions" / "case_0000"
    case.mkdir(parents=True)
    (case / "episode.json").write_text(json.dumps({
        "seed": seed,
        "difficulty": "G6",
        "episode_type": "transition_focus",
        "gate_count": 2,
        "gates_completed": 1,
        "first_gate_crossed": True,
        "correct_target_switch": False,
        "universal_transition_success": False,
        "collision_events": 0,
        "missed_gate_dnf": 0,
        "wrong_direction_events": 0,
        "out_of_bounds_events": 0,
    }), encoding="utf-8")
    (case / "track.json").write_text(json.dumps({
        "universal_transition": {
            "seed": seed,
            "difficulty": "G6",
            "episode_type": "transition_focus",
            "pattern": "yaw",
            "gate_count": 2,
            "spacings_m": [3.1],
            "turn_deltas_deg": [-48.0],
            "vertical_deltas_m": [0.1],
            "initial_yaw_error_deg": 12.0,
            "initial_lateral_offset_m": 0.2,
        },
    }), encoding="utf-8")

    records = load_evaluation_records(tmp_path)
    assert len(records) == 1
    assert not hasattr(records[0], "seed")
    families = classify_failure(records[0], DEFAULTS)
    assert "poor_target_switch" in families
    assert "large_right_turn" in families
    assert "short_spacing" in families

    # mine_failure_families guards itself, so a leaked seed would raise here.
    report = mine_failure_families(tmp_path, thresholds=DEFAULTS)
    assert report["n_failures"] == 1
    assert_no_seed_memorisation(report)


def test_train_rows_contribute_aggregate_only_families(tmp_path):
    """Composition rows have no signed legs, so signed families stay silent."""

    rows = [{
        "gate_count": 8,
        "gates_completed": 4,
        "episode_type": "full_sequence",
        "difficulty": "G6",
        "pattern": "yaw",
        "spacing_mean_m": 4.0,
        "turn_abs_mean_deg": 30.0,
        "turn_abs_max_deg": 45.0,
        "turn_sign_changes": 0,
        "vertical_abs_mean_m": 0.2,
        "vertical_abs_max_m": 0.3,
        "outcome": "DNF",
    } for _ in range(4)]
    report = mine_failure_families(None, train_episode_rows=rows, thresholds=DEFAULTS)
    names = {row["family"] for row in report["families"]}
    assert "consecutive_same_direction_turns" in names
    assert "short_spacing" in names
    assert not {"large_left_turn", "large_right_turn"} & names
    assert report["record_sources"] == {"train_log": 1.0}


# ------------------------------------------------------------------ guard

def test_guard_rejects_a_concrete_validation_seed_anywhere_in_the_report():
    seed = seed_group("validation").start + 17
    with pytest.raises(SeedMemorisationError, match="validation"):
        assert_no_seed_memorisation(
            {"families": [{"family": "large_left_turn", "worst_cases": [seed]}]}
        )


def test_guard_rejects_a_seed_named_in_prose():
    seed = seed_group("validation").start + 3
    with pytest.raises(SeedMemorisationError, match="validation"):
        assert_no_seed_memorisation({"note": f"reproduce with seed {seed}"})


def test_guard_reads_a_whole_list_of_seeds_not_just_the_first():
    """"seeds: a, b" is the natural way to leak two cases in one sentence."""

    train = seed_group("train").start + 4
    validation = seed_group("validation").start + 4
    with pytest.raises(SeedMemorisationError, match="validation"):
        assert_no_seed_memorisation({"note": f"seeds: {train}, {validation}"})
    with pytest.raises(SeedMemorisationError, match="validation"):
        assert_no_seed_memorisation({"note": f"worst seeds {validation}; {train}"})


def test_guard_rejects_a_seed_valued_key_whatever_the_band():
    with pytest.raises(SeedMemorisationError, match="seed-valued"):
        assert_no_seed_memorisation({"context": {"episode_seed": 1234}})


def test_guard_passes_a_family_report_and_the_mined_one():
    clean = {
        "families": [{
            "family": "short_spacing",
            "count": 42,
            "context": {"spacing_m": {"p50": 3.4}, "gate_count": {"p50": 8}},
        }],
        # Step-count file names are not seeds and must not trip the guard.
        "checkpoint": f"ppo_{seed_group('validation').start + 128}_steps.zip",
    }
    assert assert_no_seed_memorisation(clean) is None
    report = mine_failure_families([EXEMPLARS["missed_gate"]], thresholds=DEFAULTS)
    assert assert_no_seed_memorisation(report) is None


def test_guard_scope_is_selectable_but_holdout_bands_are_the_default():
    train_seed = seed_group("train").start + 123
    assert assert_no_seed_memorisation({"value": train_seed}) is None
    with pytest.raises(SeedMemorisationError, match="train"):
        assert_no_seed_memorisation(
            {"value": train_seed}, roles=("train", "validation", "test"))
    test_seed = seed_group("test").start + 5
    with pytest.raises(SeedMemorisationError, match="test"):
        assert_no_seed_memorisation({"value": test_seed})
