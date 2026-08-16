"""Procedural hard TRAIN episodes for the observed failure families.

The normal S0 distribution is deliberately gentle, so the geometries that
actually break the policy -- near-maximal turns, hairpin reversals, coupled
turn-plus-depth steps, offset entries -- appear a few times per thousand
episodes.  A failure taxonomy that only *reports* those families changes
nothing; the training distribution has to contain them.

Two rules shape everything here.  First, a hard family is a *distribution*, not
a track: spacing, magnitude, sign, entry state and the neighbouring transitions
are all redrawn per episode, because a fixed hard course is memorised in a few
thousand episodes and teaches nothing about the next unseen one.  A previous
generator in this project collapsed to a handful of repeated layouts and the
resulting "improvement" did not transfer, which is why ``assert_diverse`` exists
and why the geometry signature deliberately ignores the episode seed.  Second,
hard cases are an *addition* to the curriculum, never a replacement: the mixer
refuses any request that pushes hard sampling past ``MAX_HARD_FRACTION``, so the
distribution the policy is already competent on always keeps the majority.

Hard episodes stretch the combination of geometric factors, never the physical
envelope the track generator and reward contract were validated against (G6),
and they are TRAIN-only by construction: the seed comes from the train band and
any other split is refused rather than silently accepted.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from marine_race_arena.learning.generic_sequence_curriculum import (
    GenericSequenceCurriculum,
)
from marine_race_arena.learning.rl_holdout_policy import (
    assert_not_final_circuit,
    assert_seed_role,
    seed_group,
)
from marine_race_arena.learning.transition_curriculum import (
    DIFFICULTY_BY_KEY,
    TransitionGeometry,
)

HARD_FAMILY_VERSION = "rl_hard_families_v1"

#: Written into ``TransitionGeometry.pattern`` so the family survives into the
#: episode composition log without touching that schema.
HARD_PATTERN_PREFIX = "hard_"

#: Hard families stay inside the geometry envelope of the hardest curriculum
#: level; what makes them hard is the *combination*, not out-of-contract values.
HARD_ENVELOPE = DIFFICULTY_BY_KEY["G6"]
#: Label carried by hard episodes.  A hard case is drawn at full envelope
#: strength regardless of the run's current geometric difficulty, so labelling
#: it with that difficulty would misreport what the policy actually saw.
HARD_DIFFICULTY_LABEL = "G6"

MAX_HARD_FRACTION = 0.50
MIN_BASE_FRACTION = 1.0 - MAX_HARD_FRACTION
DEFAULT_BASE_FRACTION = 0.65
#: The realised-fraction guard only engages once a mixture is meaningful; below
#: this many episodes a single hard draw is already 50% and the guard would
#: distort the very mixture it protects.
HARD_CAP_ENFORCEMENT_EPISODES = 20

#: Neighbouring transitions that are not themselves hard.
FILLER_TURN_RANGE_DEG = (2.0, 18.0)
FILLER_VERTICAL_RANGE_M = (0.05, 0.70)
#: "straight" means no run-up: the policy gets no advance cue that a large
#: heading change is about to be required.
STRAIGHT_TURN_RANGE_DEG = (0.4, 3.5)
#: A constrained neighbour keeps most of the hard magnitude, otherwise
#: "opposite sign" degenerates into a mild wobble.
NEIGHBOUR_MAGNITUDE_FLOOR = 0.6

TURN_SIGN_POLICIES = ("random", "alternating", "monotone")
VERTICAL_SIGN_POLICIES = ("random", "alternating", "monotone")
#: Constraint applied to the transition immediately before/after a hard one.
NEIGHBOUR_POLICIES = ("any", "straight", "opposite_sign", "same_sign", "hard")
#: Policies that are meaningless without a neighbouring transition to constrain.
NEIGHBOUR_DEPENDENT_POLICIES = ("opposite_sign", "same_sign")
#: Where inside the episode the hard transitions land.  ``late`` weights the
#: choice toward the end without ever excluding the early positions, because a
#: fixed late block is a layout the policy can memorise.
HARD_POSITION_BIASES = ("uniform", "late")

#: What the failure analysis can report: the benchmark metrics that observe a
#: failure, plus the geometric families the taxonomy attributes it to.  Every
#: label must be attacked by at least one hard family -- a diagnosed failure
#: with no matching training distribution is a dead end.  This is a vocabulary
#: snapshot, not an import: the taxonomy must stay free to evolve, and
#: ``families_for_failure`` matches loosely so a renamed label still resolves.
TAXONOMY_FAILURE_LABELS = (
    # Observed outcomes.
    "collision_episodes",
    "missed_gate_dnf",
    "wrong_direction_events",
    "out_of_bounds_episodes",
    "acquisition_timeouts",
    "previous_gate_returns",
    "first_gate_crossing_failure",
    "target_switch_failure",
    "new_target_alignment_failure",
    "new_target_range_increase",
    "late_position_failure",
    "long_sequence_dropout",
    # Attributed geometric families.
    "first_gate_acquisition",
    "off_centre_crossing",
    "poor_target_switch",
    "reacquisition_after_crossing",
    "large_left_turn",
    "large_right_turn",
    "vertical_climb",
    "vertical_descent",
    "combined_yaw_vertical",
    "short_spacing",
    "long_spacing",
    "consecutive_same_direction_turns",
    "alternating_s_transitions",
    "collision_after_crossing",
    "missed_gate",
    "wrong_direction",
    "out_of_bounds_excursion",
    "accumulated_lateral_error",
    "accumulated_vertical_error",
)


class DiversityCollapseError(RuntimeError):
    """Raised when generation degenerated into a small repeated pool."""


@dataclass(frozen=True)
class HardFamilySpec:
    """One broad family of geometries that share a failure mechanism.

    Ranges are magnitudes; signs are drawn separately so no family is one-sided.
    """

    name: str
    description: str
    #: Taxonomy labels this family is meant to move.
    failure_labels: Tuple[str, ...]
    gate_count_range: Tuple[int, int]
    #: Fraction of the episode's transitions drawn at hard strength.
    hard_share_range: Tuple[float, float]
    turn_magnitude_range_deg: Tuple[float, float]
    turn_sign_policy: str
    vertical_magnitude_range_m: Tuple[float, float]
    vertical_sign_policy: str
    spacing_range_m: Tuple[float, float]
    entry_yaw_error_range_deg: Tuple[float, float]
    entry_lateral_offset_range_m: Tuple[float, float]
    #: Multiplier on the envelope's maximum initial speed.
    entry_speed_scale_range: Tuple[float, float]
    start_distance_range_m: Tuple[float, float]
    preceding_policy: str = "any"
    following_policy: str = "any"
    #: Where the hard transitions sit inside the episode; see
    #: ``HARD_POSITION_BIASES``.
    hard_position_bias: str = "uniform"

    @property
    def pattern(self) -> str:
        return f"{HARD_PATTERN_PREFIX}{self.name}"

    def as_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["pattern"] = self.pattern
        return value


HARD_FAMILY_SPECS: Tuple[HardFamilySpec, ...] = (
    HardFamilySpec(
        name="sharp_turn",
        description=(
            "Near-maximal heading change with no run-up: the policy must brake "
            "and re-aim inside one gate interval."
        ),
        # Sign is redrawn per episode, so one family trains the taxonomy's left
        # and right variants of the same mechanism.
        failure_labels=(
            "missed_gate_dnf", "wrong_direction_events", "target_switch_failure",
            "new_target_alignment_failure", "large_left_turn", "large_right_turn",
            "off_centre_crossing", "poor_target_switch", "missed_gate",
        ),
        gate_count_range=(2, 4),
        hard_share_range=(0.34, 0.60),
        turn_magnitude_range_deg=(34.0, 55.0),
        turn_sign_policy="random",
        vertical_magnitude_range_m=(0.15, 0.90),
        vertical_sign_policy="random",
        spacing_range_m=(3.2, 5.5),
        entry_yaw_error_range_deg=(6.0, 30.0),
        entry_lateral_offset_range_m=(0.20, 1.20),
        entry_speed_scale_range=(0.15, 0.70),
        start_distance_range_m=(0.9, 2.2),
        preceding_policy="straight",
        following_policy="any",
    ),
    HardFamilySpec(
        name="hairpin_reversal",
        description=(
            "A large turn immediately after an equally large opposite one, the "
            "geometry that produces returns to the previous gate."
        ),
        failure_labels=(
            "wrong_direction_events", "previous_gate_returns",
            "new_target_range_increase", "wrong_direction",
            "reacquisition_after_crossing", "large_left_turn", "large_right_turn",
        ),
        gate_count_range=(3, 6),
        hard_share_range=(0.30, 0.50),
        turn_magnitude_range_deg=(38.0, 55.0),
        turn_sign_policy="random",
        vertical_magnitude_range_m=(0.10, 0.80),
        vertical_sign_policy="random",
        spacing_range_m=(3.0, 4.6),
        entry_yaw_error_range_deg=(4.0, 26.0),
        entry_lateral_offset_range_m=(0.10, 1.00),
        entry_speed_scale_range=(0.10, 0.55),
        start_distance_range_m=(1.0, 3.0),
        preceding_policy="opposite_sign",
        following_policy="any",
    ),
    HardFamilySpec(
        name="serpentine_alternation",
        description=(
            "Every transition reverses at high magnitude, so lateral error "
            "accumulates instead of cancelling."
        ),
        failure_labels=(
            "new_target_alignment_failure", "previous_gate_returns",
            "late_position_failure", "long_sequence_dropout",
            "alternating_s_transitions", "accumulated_lateral_error",
        ),
        gate_count_range=(5, 12),
        hard_share_range=(1.0, 1.0),
        turn_magnitude_range_deg=(28.0, 50.0),
        turn_sign_policy="alternating",
        vertical_magnitude_range_m=(0.10, 0.90),
        vertical_sign_policy="alternating",
        spacing_range_m=(3.0, 5.0),
        entry_yaw_error_range_deg=(3.0, 22.0),
        entry_lateral_offset_range_m=(0.10, 0.90),
        entry_speed_scale_range=(0.10, 0.50),
        start_distance_range_m=(3.5, 5.5),
        preceding_policy="any",
        following_policy="any",
    ),
    HardFamilySpec(
        name="steep_vertical",
        description=(
            "Near-maximal depth step with the heading almost unchanged: pure "
            "vertical tracking, where a yaw-dominated policy misses high or low."
        ),
        failure_labels=(
            "missed_gate_dnf", "long_sequence_dropout", "vertical_climb",
            "vertical_descent", "accumulated_vertical_error", "missed_gate",
        ),
        gate_count_range=(2, 5),
        hard_share_range=(0.40, 0.80),
        turn_magnitude_range_deg=(3.0, 16.0),
        turn_sign_policy="random",
        vertical_magnitude_range_m=(1.20, 2.00),
        vertical_sign_policy="random",
        spacing_range_m=(3.0, 5.0),
        entry_yaw_error_range_deg=(3.0, 20.0),
        entry_lateral_offset_range_m=(0.10, 0.90),
        entry_speed_scale_range=(0.10, 0.50),
        start_distance_range_m=(1.0, 3.0),
        preceding_policy="straight",
        following_policy="any",
    ),
    HardFamilySpec(
        name="turn_with_vertical",
        description=(
            "Large heading change and large depth change at the same gate, then "
            "the turn continues: the coupled-axis case."
        ),
        failure_labels=(
            "collision_episodes", "missed_gate_dnf",
            "new_target_alignment_failure", "combined_yaw_vertical",
            "vertical_climb", "vertical_descent",
        ),
        gate_count_range=(3, 6),
        hard_share_range=(0.40, 0.80),
        turn_magnitude_range_deg=(30.0, 52.0),
        turn_sign_policy="random",
        vertical_magnitude_range_m=(1.10, 2.00),
        vertical_sign_policy="random",
        spacing_range_m=(3.2, 5.5),
        entry_yaw_error_range_deg=(5.0, 26.0),
        entry_lateral_offset_range_m=(0.20, 1.10),
        entry_speed_scale_range=(0.15, 0.60),
        start_distance_range_m=(1.2, 3.5),
        preceding_policy="any",
        following_policy="same_sign",
    ),
    HardFamilySpec(
        name="tight_spacing",
        description=(
            "Back-to-back hard transitions at minimum spacing: almost no time "
            "to re-acquire the next gate after crossing."
        ),
        failure_labels=(
            "collision_episodes", "target_switch_failure",
            "late_position_failure", "short_spacing", "poor_target_switch",
            "reacquisition_after_crossing", "collision_after_crossing",
        ),
        gate_count_range=(4, 9),
        hard_share_range=(0.60, 1.00),
        turn_magnitude_range_deg=(22.0, 45.0),
        turn_sign_policy="random",
        vertical_magnitude_range_m=(0.10, 0.80),
        vertical_sign_policy="random",
        spacing_range_m=(3.0, 3.9),
        entry_yaw_error_range_deg=(4.0, 24.0),
        entry_lateral_offset_range_m=(0.10, 1.00),
        entry_speed_scale_range=(0.10, 0.60),
        start_distance_range_m=(2.0, 4.5),
        preceding_policy="hard",
        following_policy="hard",
    ),
    HardFamilySpec(
        name="wide_spacing_drift",
        description=(
            "Long, nearly straight runs where a small heading bias integrates "
            "into a large lateral error before the gate is reachable."
        ),
        failure_labels=(
            "out_of_bounds_episodes", "acquisition_timeouts",
            "new_target_range_increase", "long_spacing",
            "out_of_bounds_excursion",
        ),
        gate_count_range=(2, 5),
        hard_share_range=(0.40, 1.00),
        turn_magnitude_range_deg=(4.0, 18.0),
        turn_sign_policy="random",
        vertical_magnitude_range_m=(0.10, 0.70),
        vertical_sign_policy="monotone",
        spacing_range_m=(8.0, 10.0),
        entry_yaw_error_range_deg=(8.0, 30.0),
        entry_lateral_offset_range_m=(0.40, 1.20),
        entry_speed_scale_range=(0.00, 0.35),
        start_distance_range_m=(2.5, 5.0),
        preceding_policy="any",
        following_policy="any",
    ),
    HardFamilySpec(
        name="offset_entry",
        description=(
            "Spawn far off the approach axis and badly aimed, so the first gate "
            "has to be acquired rather than merely flown through."
        ),
        failure_labels=(
            "first_gate_crossing_failure", "acquisition_timeouts",
            "missed_gate_dnf", "first_gate_acquisition", "off_centre_crossing",
        ),
        gate_count_range=(2, 4),
        hard_share_range=(0.50, 1.00),
        turn_magnitude_range_deg=(12.0, 40.0),
        turn_sign_policy="random",
        vertical_magnitude_range_m=(0.10, 1.00),
        vertical_sign_policy="random",
        spacing_range_m=(3.2, 6.0),
        entry_yaw_error_range_deg=(18.0, 30.0),
        entry_lateral_offset_range_m=(0.75, 1.20),
        entry_speed_scale_range=(0.10, 0.60),
        start_distance_range_m=(0.8, 1.6),
        preceding_policy="any",
        following_policy="any",
    ),
    HardFamilySpec(
        name="fast_entry",
        description=(
            "Arrive at the first gate near the maximum initial speed with a "
            "substantial turn waiting behind it: overshoot territory."
        ),
        failure_labels=(
            "collision_episodes", "out_of_bounds_episodes",
            "first_gate_crossing_failure", "missed_gate_dnf",
            "collision_after_crossing", "out_of_bounds_excursion",
            "first_gate_acquisition", "off_centre_crossing",
        ),
        gate_count_range=(2, 5),
        hard_share_range=(0.40, 0.90),
        turn_magnitude_range_deg=(20.0, 48.0),
        turn_sign_policy="random",
        vertical_magnitude_range_m=(0.10, 1.00),
        vertical_sign_policy="random",
        spacing_range_m=(3.0, 5.0),
        entry_yaw_error_range_deg=(4.0, 24.0),
        entry_lateral_offset_range_m=(0.10, 1.00),
        entry_speed_scale_range=(0.72, 1.00),
        start_distance_range_m=(0.8, 1.8),
        preceding_policy="straight",
        following_policy="any",
    ),
    HardFamilySpec(
        name="long_chain_accumulation",
        description=(
            "Long sequences with hard transitions scattered late, which is where "
            "per-position success actually falls off."
        ),
        failure_labels=(
            "late_position_failure", "long_sequence_dropout",
            "acquisition_timeouts", "accumulated_lateral_error",
            "accumulated_vertical_error",
        ),
        gate_count_range=(13, 22),
        hard_share_range=(0.25, 0.50),
        turn_magnitude_range_deg=(24.0, 48.0),
        turn_sign_policy="random",
        vertical_magnitude_range_m=(0.20, 1.20),
        vertical_sign_policy="random",
        spacing_range_m=(3.2, 7.5),
        entry_yaw_error_range_deg=(3.0, 18.0),
        entry_lateral_offset_range_m=(0.10, 0.80),
        entry_speed_scale_range=(0.10, 0.50),
        start_distance_range_m=(3.5, 5.5),
        preceding_policy="any",
        following_policy="any",
        # The mechanism is depth, not length: the hard transitions have to land
        # where per-position success actually falls off.
        hard_position_bias="late",
    ),
    HardFamilySpec(
        name="sustained_same_direction",
        description=(
            "Every transition turns the same way: a spiral where heading bias "
            "never gets cancelled by the next gate and lateral error integrates."
        ),
        failure_labels=(
            "consecutive_same_direction_turns", "accumulated_lateral_error",
            "new_target_alignment_failure", "out_of_bounds_episodes",
            "out_of_bounds_excursion", "late_position_failure",
        ),
        gate_count_range=(4, 10),
        hard_share_range=(1.0, 1.0),
        turn_magnitude_range_deg=(20.0, 42.0),
        turn_sign_policy="monotone",
        vertical_magnitude_range_m=(0.10, 0.90),
        vertical_sign_policy="random",
        spacing_range_m=(3.2, 6.0),
        entry_yaw_error_range_deg=(3.0, 22.0),
        entry_lateral_offset_range_m=(0.10, 1.00),
        entry_speed_scale_range=(0.10, 0.55),
        start_distance_range_m=(3.0, 5.0),
        preceding_policy="any",
        following_policy="any",
    ),
)

HARD_FAMILY_BY_NAME: Dict[str, HardFamilySpec] = {
    spec.name: spec for spec in HARD_FAMILY_SPECS
}
HARD_FAMILY_NAMES: Tuple[str, ...] = tuple(HARD_FAMILY_BY_NAME)
DEFAULT_FAMILY_WEIGHTS: Dict[str, float] = {
    name: 1.0 / len(HARD_FAMILY_NAMES) for name in HARD_FAMILY_NAMES
}


# --------------------------------------------------------------- validation

def _check_range(
    spec_name: str, field: str, value: Sequence[float],
    *, minimum: float, maximum: float, allow_point: bool = False,
) -> None:
    low, high = float(value[0]), float(value[1])
    if high < low or (high == low and not allow_point):
        raise ValueError(f"{spec_name}.{field} is not an increasing range: {value}")
    if low < minimum or high > maximum:
        raise ValueError(
            f"{spec_name}.{field}={value} leaves the validated G6 envelope "
            f"[{minimum}, {maximum}]"
        )


def validate_specs(specs: Iterable[HardFamilySpec] = HARD_FAMILY_SPECS) -> None:
    """Fail at import rather than after a week of training on bad geometry."""

    seen: set[str] = set()
    for spec in specs:
        if spec.name in seen:
            raise ValueError(f"duplicate hard family {spec.name!r}")
        seen.add(spec.name)
        # A family named after a sealed circuit would leak that identity into
        # every generated track name.
        assert_not_final_circuit(f"{spec.name}.json", purpose="train")
        assert_not_final_circuit(f"{spec.pattern}.json", purpose="train")
        if spec.turn_sign_policy not in TURN_SIGN_POLICIES:
            raise ValueError(f"{spec.name}: unknown turn sign policy")
        if spec.vertical_sign_policy not in VERTICAL_SIGN_POLICIES:
            raise ValueError(f"{spec.name}: unknown vertical sign policy")
        if spec.hard_position_bias not in HARD_POSITION_BIASES:
            raise ValueError(
                f"{spec.name}: unknown hard_position_bias "
                f"{spec.hard_position_bias!r}"
            )
        for field in ("preceding_policy", "following_policy"):
            policy = getattr(spec, field)
            if policy not in NEIGHBOUR_POLICIES:
                raise ValueError(f"{spec.name}: unknown {field} {policy!r}")
            if policy in NEIGHBOUR_DEPENDENT_POLICIES and spec.gate_count_range[0] < 3:
                raise ValueError(
                    f"{spec.name}: {field}={policy!r} needs at least two "
                    "transitions, so the family cannot start at two gates"
                )
        low, high = spec.gate_count_range
        if not 2 <= low <= high <= 24:
            raise ValueError(f"{spec.name}: invalid gate count range {spec.gate_count_range}")
        # A family may legitimately make every transition hard (serpentine).
        _check_range(spec.name, "hard_share_range", spec.hard_share_range,
                     minimum=0.0, maximum=1.0, allow_point=True)
        _check_range(spec.name, "turn_magnitude_range_deg",
                     spec.turn_magnitude_range_deg,
                     minimum=0.0, maximum=HARD_ENVELOPE.max_turn_deg)
        _check_range(spec.name, "vertical_magnitude_range_m",
                     spec.vertical_magnitude_range_m,
                     minimum=0.0, maximum=HARD_ENVELOPE.max_vertical_step_m)
        _check_range(spec.name, "spacing_range_m", spec.spacing_range_m,
                     minimum=HARD_ENVELOPE.min_spacing_m,
                     maximum=HARD_ENVELOPE.max_spacing_m)
        _check_range(spec.name, "entry_yaw_error_range_deg",
                     spec.entry_yaw_error_range_deg,
                     minimum=0.0, maximum=HARD_ENVELOPE.max_initial_yaw_error_deg)
        _check_range(spec.name, "entry_lateral_offset_range_m",
                     spec.entry_lateral_offset_range_m,
                     minimum=0.0, maximum=HARD_ENVELOPE.max_lateral_offset_m)
        _check_range(spec.name, "entry_speed_scale_range",
                     spec.entry_speed_scale_range, minimum=0.0, maximum=1.0)
        _check_range(spec.name, "start_distance_range_m",
                     spec.start_distance_range_m, minimum=0.5, maximum=6.0)
        if spec.turn_magnitude_range_deg[0] <= 0.0:
            raise ValueError(
                f"{spec.name}: a zero turn magnitude makes the sign meaningless"
            )
        if spec.vertical_magnitude_range_m[0] <= 0.0:
            raise ValueError(
                f"{spec.name}: a zero vertical magnitude makes the sign meaningless"
            )


validate_specs()


def uncovered_failure_labels(
    labels: Sequence[str] = TAXONOMY_FAILURE_LABELS,
) -> Tuple[str, ...]:
    """Taxonomy labels no family currently attacks."""

    return tuple(label for label in labels if not families_for_failure(label))


def _normalize_label(label: str) -> str:
    return str(label).strip().lower().replace("-", "_").replace(" ", "_")


def families_for_failure(label: str) -> Tuple[str, ...]:
    """Families that target a taxonomy failure label, most specific first."""

    wanted = _normalize_label(label)
    if not wanted:
        return ()
    exact, loose = [], []
    for spec in HARD_FAMILY_SPECS:
        declared = [_normalize_label(value) for value in spec.failure_labels]
        if wanted in declared:
            exact.append(spec.name)
        elif len(wanted) >= 3 and any(
            wanted in value or value in wanted for value in declared
        ):
            # Substring matching keeps a renamed taxonomy label resolving, but a
            # one- or two-character fragment matches nearly everything and would
            # report coverage that does not exist.
            loose.append(spec.name)
    return tuple(exact + loose)


# ------------------------------------------------------------- generation

def _as_generator(rng: Any) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(int(rng))


def _sign_array(
    rng: np.random.Generator, count: int, policy: str
) -> np.ndarray:
    """Per-transition signs; every policy reaches both signs across episodes."""

    if count <= 0:
        return np.zeros(0)
    if policy == "alternating":
        start = float(rng.choice((-1.0, 1.0)))
        return np.asarray([start * (1.0 if i % 2 == 0 else -1.0) for i in range(count)])
    if policy == "monotone":
        return np.full(count, float(rng.choice((-1.0, 1.0))))
    return np.asarray(rng.choice((-1.0, 1.0), size=count), dtype=float)


def _unconstrained_sign(
    rng: np.random.Generator,
    signs: np.ndarray,
    index: int,
    policy: str,
) -> float:
    """Turn sign for a transition that no neighbour rule constrains.

    The draw happens either way so the rng stream does not depend on the sign
    policy, but a non-random policy overrides it: an ``alternating`` or
    ``monotone`` family whose filler transitions turned freely would silently
    stop being that family as soon as its hard share dropped below one.
    """

    drawn = float(rng.choice((-1.0, 1.0)))
    return drawn if policy == "random" else float(signs[index])


def _hard_positions(
    rng: np.random.Generator, count: int, hard_count: int, bias: str
) -> Tuple[int, ...]:
    """Which transitions are drawn at hard strength.

    ``late`` skews the choice toward the end of the chain -- where per-position
    success actually falls off -- with a linear weight rather than a fixed late
    block, which would be a memorisable layout.
    """

    if hard_count <= 0 or count <= 0:
        return ()
    if bias == "late" and count > 1:
        weights = np.arange(1, count + 1, dtype=float)
        chosen = rng.choice(
            count, size=hard_count, replace=False, p=weights / weights.sum()
        )
    else:
        chosen = rng.choice(count, size=hard_count, replace=False)
    return tuple(sorted(int(value) for value in chosen))


def _neighbour_roles(
    hard_positions: Sequence[int],
    count: int,
    *,
    preceding_policy: str,
    following_policy: str,
) -> Dict[int, Tuple[str, int]]:
    """Which non-hard transitions are constrained by an adjacent hard one.

    A gap between two hard transitions is both a follower and a predecessor.
    The preceding rule wins whenever it constrains anything, because the
    approach *into* the hard transition is the mechanism a family targets;
    resolving this by iteration order instead silently dropped the constraint.
    """

    hard = set(int(v) for v in hard_positions)
    roles: Dict[int, Tuple[str, int]] = {}
    for index in range(count):
        if index in hard:
            continue
        precedes = index + 1 in hard
        follows = index - 1 >= 0 and index - 1 in hard
        if precedes and follows:
            take_preceding = (
                preceding_policy != "any" or following_policy == "any"
            )
        elif precedes:
            take_preceding = True
        elif follows:
            take_preceding = False
        else:
            continue
        roles[index] = (
            ("preceding", index + 1) if take_preceding else ("following", index - 1)
        )
    return roles


@dataclass(frozen=True)
class HardEpisode:
    """A sampled hard episode plus the plan that produced it."""

    family: str
    hard_positions: Tuple[int, ...]
    geometry: TransitionGeometry

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": HARD_FAMILY_VERSION,
            "family": self.family,
            "hard_positions": list(self.hard_positions),
            "geometry": asdict(self.geometry),
        }


def sample_hard_episode(
    spec: HardFamilySpec,
    rng: Any,
    *,
    dataset_split: str = "train",
    difficulty: str = HARD_DIFFICULTY_LABEL,
    curriculum_stage: Optional[str] = None,
) -> HardEpisode:
    """Draw one episode from a hard family, with its hard-transition plan."""

    if str(dataset_split) != "train":
        # Hard cases are a training intervention.  Refusing here is what makes
        # "a validation or test seed can never be produced" a property of the
        # code rather than of the caller's discipline.
        raise ValueError(
            f"hard families are TRAIN-only; refused dataset_split="
            f"{dataset_split!r}"
        )
    if difficulty not in DIFFICULTY_BY_KEY:
        raise ValueError(f"unknown difficulty label {difficulty!r}")
    generator = _as_generator(rng)

    low, high = spec.gate_count_range
    gate_count = int(generator.integers(low, high + 1))
    count = max(0, gate_count - 1)

    turn_signs = _sign_array(generator, count, spec.turn_sign_policy)
    vertical_signs = _sign_array(generator, count, spec.vertical_sign_policy)
    share = float(generator.uniform(*spec.hard_share_range))
    hard_count = int(max(1, min(count, int(np.rint(share * count))))) if count else 0
    hard_positions = _hard_positions(
        generator, count, hard_count, spec.hard_position_bias
    )
    hard = set(hard_positions)
    roles = _neighbour_roles(
        hard_positions, count,
        preceding_policy=spec.preceding_policy,
        following_policy=spec.following_policy,
    )

    turns: List[float] = []
    verticals: List[float] = []
    spacings: List[float] = []
    for index in range(count):
        if index in hard:
            magnitude = float(generator.uniform(*spec.turn_magnitude_range_deg))
            sign = float(turn_signs[index])
            vertical = float(generator.uniform(*spec.vertical_magnitude_range_m))
        else:
            kind, anchor = roles.get(index, ("filler", -1))
            policy = (
                spec.preceding_policy if kind == "preceding"
                else spec.following_policy if kind == "following"
                else "any"
            )
            if policy == "straight":
                magnitude = float(generator.uniform(*STRAIGHT_TURN_RANGE_DEG))
                sign = _unconstrained_sign(
                    generator, turn_signs, index, spec.turn_sign_policy
                )
                vertical = float(generator.uniform(*FILLER_VERTICAL_RANGE_M))
            elif policy in NEIGHBOUR_DEPENDENT_POLICIES:
                turn_low, turn_high = spec.turn_magnitude_range_deg
                magnitude = float(generator.uniform(
                    NEIGHBOUR_MAGNITUDE_FLOOR * turn_low, turn_high
                ))
                anchor_sign = float(turn_signs[anchor])
                sign = -anchor_sign if policy == "opposite_sign" else anchor_sign
                vertical = float(generator.uniform(*FILLER_VERTICAL_RANGE_M))
            elif policy == "hard":
                magnitude = float(generator.uniform(*spec.turn_magnitude_range_deg))
                sign = _unconstrained_sign(
                    generator, turn_signs, index, spec.turn_sign_policy
                )
                vertical = float(generator.uniform(*spec.vertical_magnitude_range_m))
            else:
                magnitude = float(generator.uniform(*FILLER_TURN_RANGE_DEG))
                sign = _unconstrained_sign(
                    generator, turn_signs, index, spec.turn_sign_policy
                )
                vertical = float(generator.uniform(*FILLER_VERTICAL_RANGE_M))
        turns.append(sign * magnitude)
        verticals.append(float(vertical_signs[index]) * vertical)
        spacings.append(float(generator.uniform(*spec.spacing_range_m)))

    speed_scale = float(generator.uniform(*spec.entry_speed_scale_range))
    speed_cap = HARD_ENVELOPE.max_initial_speed_m_s * speed_scale
    yaw_error = float(generator.choice((-1.0, 1.0))) * float(
        generator.uniform(*spec.entry_yaw_error_range_deg)
    )
    lateral_offset = float(generator.choice((-1.0, 1.0))) * float(
        generator.uniform(*spec.entry_lateral_offset_range_m)
    )

    group = seed_group("train")
    seed = int(group.start) + int(generator.integers(0, group.size))
    assert_seed_role(seed, expected="train")

    geometry = TransitionGeometry(
        difficulty=difficulty,
        episode_type=(
            "transition_focus" if gate_count <= 2 else "full_sequence"
        ),
        pattern=spec.pattern,
        seed=seed,
        gate_count=gate_count,
        spacings_m=tuple(spacings),
        turn_deltas_deg=tuple(turns),
        vertical_deltas_m=tuple(verticals),
        initial_yaw_error_deg=yaw_error,
        initial_lateral_offset_m=lateral_offset,
        start_distance_m=float(generator.uniform(*spec.start_distance_range_m)),
        initial_body_velocity_m_s=(
            speed_cap,
            float(generator.uniform(-0.5, 0.5)) * speed_cap,
            float(generator.uniform(-0.35, 0.35)) * speed_cap,
        ),
        dataset_split="train",
        sequence_bucket=GenericSequenceCurriculum.bucket_of_gate_count(gate_count),
        curriculum_stage=curriculum_stage,
    )
    assert_no_final_circuit_reference(geometry)
    return HardEpisode(
        family=spec.name, hard_positions=hard_positions, geometry=geometry
    )


def sample_hard_geometry(
    spec: HardFamilySpec,
    rng: Any,
    *,
    dataset_split: str = "train",
    difficulty: str = HARD_DIFFICULTY_LABEL,
    curriculum_stage: Optional[str] = None,
) -> TransitionGeometry:
    """Geometry for one hard episode, ready for the transition track generator."""

    return sample_hard_episode(
        spec, rng, dataset_split=dataset_split, difficulty=difficulty,
        curriculum_stage=curriculum_stage,
    ).geometry


def family_of(geometry: Any) -> Optional[str]:
    """Family behind a geometry, or ``None`` for a normal curriculum episode."""

    pattern = str(getattr(geometry, "pattern", "") or "")
    if not pattern.startswith(HARD_PATTERN_PREFIX):
        return None
    return pattern[len(HARD_PATTERN_PREFIX):]


def track_label(geometry: Any) -> str:
    """The race name ``generate_transition_track`` derives from a geometry."""

    return f"Universal transition {geometry.geometry_group}"


def assert_no_final_circuit_reference(geometry: Any) -> None:
    """No hard episode may carry a sealed circuit's identity into a track."""

    for text in (geometry.pattern, geometry.geometry_group, track_label(geometry)):
        assert_not_final_circuit(f"{text}.json", purpose="train")


# --------------------------------------------------------------- diversity

def geometry_signature(geometry: Any) -> str:
    """Stable digest of the *shape* of an episode.

    The episode seed is deliberately excluded: a collapsed generator that emits
    the same twenty layouts with fresh seeds every time would otherwise report
    perfect diversity, which is exactly the bug this guards against.
    """

    payload = (
        int(getattr(geometry, "gate_count", 0)),
        tuple(round(float(v), 3) for v in getattr(geometry, "spacings_m", ()) or ()),
        tuple(round(float(v), 3) for v in getattr(geometry, "turn_deltas_deg", ()) or ()),
        tuple(round(float(v), 3) for v in getattr(geometry, "vertical_deltas_m", ()) or ()),
        round(float(getattr(geometry, "initial_yaw_error_deg", 0.0)), 3),
        round(float(getattr(geometry, "initial_lateral_offset_m", 0.0)), 3),
        round(float(getattr(geometry, "start_distance_m", 0.0)), 3),
        tuple(round(float(v), 3)
              for v in getattr(geometry, "initial_body_velocity_m_s", ()) or ()),
    )
    return hashlib.blake2s(repr(payload).encode("utf-8"), digest_size=12).hexdigest()


def _extent(values: Sequence[float]) -> Dict[str, Optional[float]]:
    clean = [float(v) for v in values]
    if not clean:
        return {"min": None, "max": None, "span": None}
    return {
        "min": round(min(clean), 4),
        "max": round(max(clean), 4),
        "span": round(max(clean) - min(clean), 4),
    }


def _sign_share(values: Sequence[float]) -> Dict[str, float]:
    total = max(1, len(values))
    positive = sum(1 for v in values if float(v) > 0.0)
    negative = sum(1 for v in values if float(v) < 0.0)
    return {
        "positive": round(positive / total, 4),
        "negative": round(negative / total, 4),
    }


def diversity_report(samples: Sequence[Any]) -> Dict[str, Any]:
    """What a batch of generated episodes actually covers, JSON-serialisable."""

    geometries = [getattr(s, "geometry", s) for s in samples]
    total = len(geometries)
    if not total:
        return {"schema_version": HARD_FAMILY_VERSION, "samples": 0}
    signatures = {geometry_signature(g) for g in geometries}
    families: Dict[str, int] = {}
    for geometry in geometries:
        key = family_of(geometry) or "base"
        families[key] = families.get(key, 0) + 1
    spacings = [v for g in geometries for v in (g.spacings_m or ())]
    turns = [v for g in geometries for v in (g.turn_deltas_deg or ())]
    verticals = [v for g in geometries for v in (g.vertical_deltas_m or ())]
    splits: Dict[str, int] = {}
    for geometry in geometries:
        key = str(getattr(geometry, "dataset_split", None))
        splits[key] = splits.get(key, 0) + 1
    return {
        "schema_version": HARD_FAMILY_VERSION,
        "samples": total,
        "distinct_signatures": len(signatures),
        "distinct_fraction": round(len(signatures) / total, 6),
        "distinct_seeds": len({int(g.seed) for g in geometries}),
        "family_share": {
            name: round(count / total, 6) for name, count in sorted(families.items())
        },
        "family_counts": dict(sorted(families.items())),
        "gate_counts": sorted({int(g.gate_count) for g in geometries}),
        "spacing_m": _extent(spacings),
        "turn_deg": _extent(turns),
        "abs_turn_deg": _extent([abs(float(v)) for v in turns]),
        "vertical_m": _extent(verticals),
        "abs_vertical_m": _extent([abs(float(v)) for v in verticals]),
        "turn_sign_share": _sign_share(turns),
        "vertical_sign_share": _sign_share(verticals),
        "dataset_split_share": {
            key: round(count / total, 6) for key, count in sorted(splits.items())
        },
    }


def assert_diverse(
    samples: Sequence[Any],
    minimum_distinct_fraction: float = 0.95,
    *,
    minimum_for_sign_check: int = 20,
) -> Dict[str, Any]:
    """Refuse a batch that collapsed to a small repeated pool.

    Returns the diversity report so callers can log what they verified.
    """

    report = diversity_report(samples)
    if not report.get("samples"):
        raise DiversityCollapseError("no samples to check for diversity")
    fraction = float(report["distinct_fraction"])
    if fraction < float(minimum_distinct_fraction):
        raise DiversityCollapseError(
            f"only {report['distinct_signatures']} distinct layouts in "
            f"{report['samples']} samples ({fraction:.3f} < "
            f"{float(minimum_distinct_fraction):.3f}): generation collapsed"
        )
    if report["samples"] >= minimum_for_sign_check:
        for field in ("spacing_m", "abs_turn_deg", "abs_vertical_m"):
            span = report[field]["span"]
            if not span:
                raise DiversityCollapseError(
                    f"{field} never varies: the family is a fixed track"
                )
        for field, values in (
            ("turn", report["turn_sign_share"]),
            ("vertical", report["vertical_sign_share"]),
        ):
            if values["positive"] <= 0.0 or values["negative"] <= 0.0:
                raise DiversityCollapseError(
                    f"{field} deltas are one-sided ({values}): the family only "
                    "teaches one direction"
                )
    return report


# ------------------------------------------------------------------ mixing

class HardCaseMixer:
    """Per-episode choice between the normal S0 distribution and a hard family.

    Hard cases never become the curriculum: the requested hard share is refused
    above ``MAX_HARD_FRACTION`` and the realised share is capped as well, so a
    pathological rng cannot drift the run onto hard geometry only.
    """

    def __init__(
        self,
        base_fraction: float = DEFAULT_BASE_FRACTION,
        family_weights: Optional[Mapping[str, float]] = None,
        rng: Any = 0,
    ) -> None:
        base = float(base_fraction)
        if not 0.0 < base <= 1.0:
            raise ValueError(f"base_fraction must be in (0, 1]; got {base_fraction!r}")
        hard = 1.0 - base
        if hard > MAX_HARD_FRACTION + 1e-9:
            raise ValueError(
                f"hard fraction {hard:.4f} exceeds MAX_HARD_FRACTION="
                f"{MAX_HARD_FRACTION}: base_fraction must stay at or above "
                f"{MIN_BASE_FRACTION} so the normal S0 distribution keeps the "
                "majority of episodes"
            )
        weights = {
            str(key): float(value)
            for key, value in dict(family_weights or DEFAULT_FAMILY_WEIGHTS).items()
        }
        unknown = sorted(set(weights) - set(HARD_FAMILY_BY_NAME))
        if unknown:
            raise ValueError(f"unknown hard families {unknown}")
        if any(value < 0.0 for value in weights.values()):
            raise ValueError("family weights cannot be negative")
        total = sum(weights.values())
        if total <= 0.0:
            raise ValueError("family weights must have positive mass")
        self.base_fraction = base
        self.hard_fraction = hard
        self.family_weights = {
            name: value / total for name, value in weights.items()
        }
        self.rng = _as_generator(rng)
        self._base_episodes = 0
        self._hard_episodes = 0
        self._family_counts: Dict[str, int] = {name: 0 for name in weights}
        self._capped_draws = 0
        self._refused_draws = 0

    # ------------------------------------------------------------- choice

    def _draw_family(self) -> str:
        names = list(self.family_weights)
        probabilities = [self.family_weights[name] for name in names]
        return str(self.rng.choice(names, p=probabilities))

    def next_source(self) -> Tuple[str, Optional[str]]:
        """Decide the next episode: ``("base", None)`` or ``("hard", family)``."""

        total = self._base_episodes + self._hard_episodes
        take_hard = float(self.rng.random()) >= self.base_fraction
        if (
            take_hard
            and total + 1 >= HARD_CAP_ENFORCEMENT_EPISODES
            and (self._hard_episodes + 1) / (total + 1) > MAX_HARD_FRACTION
        ):
            take_hard = False
            self._capped_draws += 1
        if not take_hard:
            self._base_episodes += 1
            return "base", None
        family = self._draw_family()
        self._hard_episodes += 1
        self._family_counts[family] = self._family_counts.get(family, 0) + 1
        return "hard", family

    def sample(
        self,
        base_sampler: Any,
        *,
        curriculum_stage: Optional[str] = None,
    ) -> TransitionGeometry:
        """One episode geometry from either the base sampler or a hard family."""

        source, family = self.next_source()
        # ``next_source`` books the episode before it exists, so a draw that
        # cannot be served has to be un-booked: an episode that was never flown
        # must not appear in realised_composition(), which is the only record of
        # what the policy actually saw.
        try:
            if source == "base":
                return base_sampler.sample()
            stage = curriculum_stage
            if stage is None:
                curriculum = getattr(base_sampler, "sequence_curriculum", None)
                stage = curriculum.stage["name"] if curriculum is not None else None
            return sample_hard_geometry(
                HARD_FAMILY_BY_NAME[family],
                self.rng,
                dataset_split=str(getattr(base_sampler, "dataset_split", "train")),
                curriculum_stage=stage,
            )
        except BaseException:
            if source == "base":
                self._base_episodes -= 1
            else:
                self._hard_episodes -= 1
                self._family_counts[family] -= 1
            self._refused_draws += 1
            raise

    # ------------------------------------------------------------ reporting

    def realised_composition(self) -> Dict[str, Any]:
        """The mixture that actually happened, not the one that was requested."""

        total = self._base_episodes + self._hard_episodes
        hard_share = (self._hard_episodes / total) if total else 0.0
        return {
            "schema_version": HARD_FAMILY_VERSION,
            "episodes": total,
            "base_episodes": self._base_episodes,
            "hard_episodes": self._hard_episodes,
            "requested_base_fraction": round(self.base_fraction, 6),
            "requested_hard_fraction": round(self.hard_fraction, 6),
            "realised_base_fraction": round(
                (self._base_episodes / total) if total else 0.0, 6
            ),
            "realised_hard_fraction": round(hard_share, 6),
            "max_hard_fraction": MAX_HARD_FRACTION,
            "hard_fraction_within_cap": hard_share <= MAX_HARD_FRACTION + 1e-9,
            "capped_draws": self._capped_draws,
            "refused_draws": self._refused_draws,
            "family_counts": dict(sorted(self._family_counts.items())),
            "family_share_of_hard": {
                name: round(count / self._hard_episodes, 6)
                for name, count in sorted(self._family_counts.items())
            } if self._hard_episodes else {},
            "requested_family_weights": {
                name: round(value, 6)
                for name, value in sorted(self.family_weights.items())
            },
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": HARD_FAMILY_VERSION,
            "base_fraction": self.base_fraction,
            "family_weights": dict(self.family_weights),
            "rng_state": self.rng.bit_generator.state,
            "base_episodes": self._base_episodes,
            "hard_episodes": self._hard_episodes,
            "family_counts": dict(self._family_counts),
            "capped_draws": self._capped_draws,
            "refused_draws": self._refused_draws,
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if value.get("schema_version") != HARD_FAMILY_VERSION:
            raise ValueError("unsupported hard case mixer state")
        if not np.isclose(float(value["base_fraction"]), self.base_fraction):
            raise ValueError("hard case mixture changed across resume")
        stored = {str(k): float(v) for k, v in dict(value["family_weights"]).items()}
        if sorted(stored) != sorted(self.family_weights):
            raise ValueError("hard family set changed across resume")
        # Same families at different weights is the same silent-mixture-change
        # bug as a different base fraction, only harder to see in the logs.
        if not all(
            np.isclose(weight, self.family_weights[name])
            for name, weight in stored.items()
        ):
            raise ValueError("hard family weights changed across resume")
        self.rng.bit_generator.state = value["rng_state"]
        self._base_episodes = int(value["base_episodes"])
        self._hard_episodes = int(value["hard_episodes"])
        self._family_counts = {
            str(k): int(v) for k, v in dict(value["family_counts"]).items()
        }
        self._capped_draws = int(value.get("capped_draws", 0))
        self._refused_draws = int(value.get("refused_draws", 0))


def hard_family_manifest() -> Dict[str, Any]:
    """Everything a run needs to record about how hard cases were generated."""

    return {
        "schema_version": HARD_FAMILY_VERSION,
        "max_hard_fraction": MAX_HARD_FRACTION,
        "min_base_fraction": MIN_BASE_FRACTION,
        "default_base_fraction": DEFAULT_BASE_FRACTION,
        "difficulty_envelope": HARD_DIFFICULTY_LABEL,
        "pattern_prefix": HARD_PATTERN_PREFIX,
        "dataset_split": "train",
        "taxonomy_failure_labels": list(TAXONOMY_FAILURE_LABELS),
        "uncovered_failure_labels": list(uncovered_failure_labels()),
        "families": [spec.as_dict() for spec in HARD_FAMILY_SPECS],
        "default_family_weights": dict(DEFAULT_FAMILY_WEIGHTS),
    }
