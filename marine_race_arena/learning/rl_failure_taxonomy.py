"""Where and why the transition policy fails, as generic geometry families.

An aggregate success rate says a policy is bad; it never says *where* or *why*.
This module turns evaluation outputs and TRAIN episode logs into a small set of
GENERIC geometry/behaviour families -- "large left turn", "short spacing",
"reacquisition after crossing" -- so the next curriculum can procedurally
generate more of what actually breaks instead of replaying the individual
courses that broke.  Nothing here ever carries an episode seed: a report that
pointed at concrete cases would invite memorising them, which is precisely the
failure mode the TRAIN/VALIDATION/TEST split exists to prevent.  The record
type deliberately has no ``seed`` field at all, and
:func:`assert_no_seed_memorisation` re-checks the finished report.

The headline function is :func:`survival_curve`.  Per-position success was
previously reported *conditioned on having reached the previous gate*, which
made a policy that almost never got past gate 3 look like it had 1.00 success
at gate 12 -- the only episodes measured there were the survivors.  Every curve
here is also reported from EPISODE START, and the two live under names that
cannot be confused: ``reach_from_start`` versus ``conditional_on_previous``.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, replace
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple,
)

from marine_race_arena.learning.rl_holdout_policy import SEED_GROUPS
from marine_race_arena.learning.transition_curriculum import (
    DIFFICULTIES,
    DIFFICULTY_BY_KEY,
)

FAILURE_TAXONOMY_VERSION = "rl_failure_taxonomy_v1"


# --------------------------------------------------------------- thresholds
#
# "Large" and "short" are defined against the distribution the policy actually
# flew, because a 30 degree turn is extreme under G2 and unremarkable under G6.
# The fallback constants are derived from the curriculum's own hardest stage
# rather than typed in by hand, so they move if the curriculum moves.

_HARDEST = DIFFICULTY_BY_KEY["G6"]
_SPACING_LOW = min(level.min_spacing_m for level in DIFFICULTIES)
_SPACING_HIGH = max(level.max_spacing_m for level in DIFFICULTIES)

#: Fraction of the hardest stage's per-axis cap that counts as a "large" step.
LARGE_MAGNITUDE_FRACTION = 0.60
#: Fraction of the hardest stage's spawn offset cap that counts as off centre.
OFF_CENTRE_FRACTION = 0.75

LARGE_TURN_QUANTILE = 0.75
LARGE_VERTICAL_QUANTILE = 0.75
SHORT_SPACING_QUANTILE = 0.25
LONG_SPACING_QUANTILE = 0.75
OFF_CENTRE_QUANTILE = 0.75

DEFAULT_LARGE_TURN_DEG = LARGE_MAGNITUDE_FRACTION * _HARDEST.max_turn_deg
DEFAULT_LARGE_VERTICAL_M = LARGE_MAGNITUDE_FRACTION * _HARDEST.max_vertical_step_m
DEFAULT_SHORT_SPACING_M = (
    _SPACING_LOW + SHORT_SPACING_QUANTILE * (_SPACING_HIGH - _SPACING_LOW)
)
DEFAULT_LONG_SPACING_M = (
    _SPACING_LOW + LONG_SPACING_QUANTILE * (_SPACING_HIGH - _SPACING_LOW)
)
DEFAULT_OFF_CENTRE_OFFSET_M = OFF_CENTRE_FRACTION * _HARDEST.max_lateral_offset_m

#: Below this many observed legs an empirical quantile is noise, so the
#: curriculum-derived default is kept instead.
MIN_SAMPLES_FOR_EMPIRICAL_THRESHOLD = 32

#: Both axes must clear this fraction of their "large" bar simultaneously for a
#: failure to count as combined: the point of the family is the *coupling*, so
#: neither axis needs to be extreme on its own.
COMBINED_AXIS_FRACTION = 0.70
#: A turn below this fraction of the "large" bar is course noise, not a turn.
RUN_TURN_FRACTION = 0.50
#: Three real turns the same way is a sustained spiral, not a single corner.
CONSECUTIVE_TURN_RUN = 3
#: Two sign changes means left-right-left: the shortest genuine S.
ALTERNATING_SIGN_CHANGES = 2
#: Error accumulation is only a meaningful diagnosis once several transitions
#: have been chained; before that a failure is simply a bad single transition.
ACCUMULATION_MIN_POSITION = 4
ACCUMULATION_TURN_MULTIPLE = 3.0
ACCUMULATION_VERTICAL_MULTIPLE = 3.0

#: Survival is reported against this bar; the position where cumulative
#: survival first drops below it is the honest "how deep does it get" number.
SURVIVAL_HALF_LIFE = 0.5

#: Sign conventions, read off ``sequence_curriculum.generate_sequence_track``:
#: gate heading accumulates ``turn_deltas_deg`` in the x/y plane of a z-up
#: frame, so a positive delta rotates counter-clockwise, i.e. to port (left);
#: gate depth accumulates ``vertical_deltas_m`` toward the surface, so a
#: positive delta is a climb.
POSITIVE_TURN_IS_LEFT = True
POSITIVE_VERTICAL_IS_CLIMB = True


class SeedMemorisationError(RuntimeError):
    """Raised when a failure report names concrete holdout episode seeds."""


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    clean = sorted(float(v) for v in values if v is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = float(q) * (len(clean) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(clean) - 1)
    weight = position - low
    return clean[low] * (1.0 - weight) + clean[high] * weight


def _summary(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """Compact spread of a geometry axis, enough to seed a generator."""

    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"n": 0, "min": None, "p25": None, "p50": None,
                "p75": None, "max": None, "mean": None}
    return {
        "n": len(clean),
        "min": round(min(clean), 4),
        "p25": round(float(_quantile(clean, 0.25)), 4),
        "p50": round(float(_quantile(clean, 0.50)), 4),
        "p75": round(float(_quantile(clean, 0.75)), 4),
        "max": round(max(clean), 4),
        "mean": round(sum(clean) / len(clean), 4),
    }


def _shares(values: Sequence[Any]) -> Dict[str, float]:
    clean = [str(value) for value in values if value is not None]
    total = len(clean)
    if not total:
        return {}
    counts: Dict[str, int] = {}
    for value in clean:
        counts[value] = counts.get(value, 0) + 1
    return {
        key: round(count / total, 4)
        for key, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    }


@dataclass(frozen=True)
class FamilyThresholds:
    """The numbers that decide "large", "short" and "off centre".

    Held in one object so every predicate reads the same bar and a report can
    state which bar it used; nothing in this module compares against a literal.
    """

    large_turn_deg: float = DEFAULT_LARGE_TURN_DEG
    large_vertical_m: float = DEFAULT_LARGE_VERTICAL_M
    short_spacing_m: float = DEFAULT_SHORT_SPACING_M
    long_spacing_m: float = DEFAULT_LONG_SPACING_M
    off_centre_offset_m: float = DEFAULT_OFF_CENTRE_OFFSET_M
    source: str = "curriculum_defaults"
    n_leg_samples: int = 0

    @classmethod
    def defaults(cls) -> "FamilyThresholds":
        return cls()

    @classmethod
    def from_records(
        cls, records: Iterable["FailureRecord"]
    ) -> "FamilyThresholds":
        """Derive the bars from the geometry the cohort actually contains.

        Successes are included on purpose: the question "is this turn large?"
        is about the exposure distribution, not about the failures.
        """

        records = list(records)
        turns: List[float] = []
        vertical: List[float] = []
        spacing: List[float] = []
        offsets: List[float] = []
        for record in records:
            turns.extend(abs(value) for value in record.turn_deltas_deg)
            vertical.extend(abs(value) for value in record.vertical_deltas_m)
            spacing.extend(record.spacings_m)
            if not record.turn_deltas_deg and record.turn_abs_mean_deg is not None:
                turns.append(abs(record.turn_abs_mean_deg))
            if not record.vertical_deltas_m and record.vertical_abs_mean_m is not None:
                vertical.append(abs(record.vertical_abs_mean_m))
            if not record.spacings_m and record.spacing_mean_m is not None:
                spacing.append(record.spacing_mean_m)
            offset = record.crossing_offset_estimate_m
            if offset is not None:
                offsets.append(offset)
        empirical = 0

        def bar(values: List[float], q: float, fallback: float) -> float:
            nonlocal empirical
            if len(values) < MIN_SAMPLES_FOR_EMPIRICAL_THRESHOLD:
                return float(fallback)
            empirical += 1
            return float(_quantile(values, q))

        resolved = cls(
            large_turn_deg=bar(turns, LARGE_TURN_QUANTILE, DEFAULT_LARGE_TURN_DEG),
            large_vertical_m=bar(
                vertical, LARGE_VERTICAL_QUANTILE, DEFAULT_LARGE_VERTICAL_M),
            short_spacing_m=bar(
                spacing, SHORT_SPACING_QUANTILE, DEFAULT_SHORT_SPACING_M),
            long_spacing_m=bar(
                spacing, LONG_SPACING_QUANTILE, DEFAULT_LONG_SPACING_M),
            off_centre_offset_m=bar(
                offsets, OFF_CENTRE_QUANTILE, DEFAULT_OFF_CENTRE_OFFSET_M),
            n_leg_samples=len(turns),
        )
        source = (
            "curriculum_defaults" if empirical == 0
            else "observed_distribution" if empirical == 5
            else "mixed"
        )
        return replace(resolved, source=source)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "large_turn_deg": round(float(self.large_turn_deg), 4),
            "large_vertical_m": round(float(self.large_vertical_m), 4),
            "short_spacing_m": round(float(self.short_spacing_m), 4),
            "long_spacing_m": round(float(self.long_spacing_m), 4),
            "off_centre_offset_m": round(float(self.off_centre_offset_m), 4),
            "source": str(self.source),
            "n_leg_samples": int(self.n_leg_samples),
        }


# ------------------------------------------------------------ failure record

def _as_tuple(value: Any) -> Tuple[float, ...]:
    if value is None:
        return ()
    return tuple(float(item) for item in value)


def _optional_bool(value: Any) -> Optional[bool]:
    """Tri-state: an unrecorded signal must never look like a negative one."""

    return None if value is None else bool(value)


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


@dataclass(frozen=True)
class FailureRecord:
    """Geometry plus outcome for one episode, with no episode identity.

    Per-leg arrays are present when the case's ``track.json`` is available;
    train composition rows only ever carried aggregates, so the aggregate
    fields stand in and the sign-dependent families simply do not fire.
    """

    gate_count: int
    gates_completed: int = 0
    episode_type: Optional[str] = None
    pattern: Optional[str] = None
    difficulty: Optional[str] = None
    #: Per-leg geometry; ``spacings_m[i]`` is the leg from gate i+1 to gate i+2.
    spacings_m: Tuple[float, ...] = ()
    turn_deltas_deg: Tuple[float, ...] = ()
    vertical_deltas_m: Tuple[float, ...] = ()
    #: Whole-episode aggregates, for logs that never stored the arrays.
    spacing_mean_m: Optional[float] = None
    turn_abs_mean_deg: Optional[float] = None
    turn_abs_max_deg: Optional[float] = None
    turn_sign_changes: Optional[int] = None
    vertical_abs_mean_m: Optional[float] = None
    vertical_abs_max_m: Optional[float] = None
    initial_yaw_error_deg: float = 0.0
    initial_lateral_offset_m: float = 0.0
    crossing_offset_m: Optional[float] = None
    first_gate_crossed: bool = False
    target_switched: Optional[bool] = None
    new_target_aligned: Optional[bool] = None
    new_target_range_decreasing: Optional[bool] = None
    collision_events: int = 0
    missed_gate_events: int = 0
    wrong_direction_events: int = 0
    out_of_bounds_events: int = 0
    previous_gate_return: bool = False
    acquisition_timeout: bool = False
    timeout: bool = False
    succeeded: bool = False
    source: str = "unknown"

    @property
    def is_failure(self) -> bool:
        return not self.succeeded

    @property
    def failure_position(self) -> Optional[int]:
        """1-based gate the episode failed to reach; ``None`` on success."""

        if self.succeeded:
            return None
        return max(1, min(int(self.gates_completed) + 1, int(self.gate_count)))

    @property
    def failing_leg_index(self) -> Optional[int]:
        """Index into the per-leg arrays of the leg the episode died on.

        ``None`` when the episode succeeded or never reached the first gate --
        there is no leg to blame in either case.  Anchoring every geometry
        family here is what stops a successful run's geometry from being
        counted as a failure mode.
        """

        position = self.failure_position
        if position is None or position < 2:
            return None
        return position - 2

    @property
    def preceding_leg_index(self) -> Optional[int]:
        index = self.failing_leg_index
        if index is None or index < 1:
            return None
        return index - 1

    def _leg(self, values: Tuple[float, ...], index: Optional[int]) -> Optional[float]:
        if index is None or index < 0 or index >= len(values):
            return None
        return float(values[index])

    @property
    def failing_turn_deg(self) -> Optional[float]:
        return self._leg(self.turn_deltas_deg, self.failing_leg_index)

    @property
    def failing_vertical_m(self) -> Optional[float]:
        return self._leg(self.vertical_deltas_m, self.failing_leg_index)

    @property
    def failing_spacing_m(self) -> Optional[float]:
        return self._leg(self.spacings_m, self.failing_leg_index)

    @property
    def preceding_turn_deg(self) -> Optional[float]:
        return self._leg(self.turn_deltas_deg, self.preceding_leg_index)

    @property
    def preceding_vertical_m(self) -> Optional[float]:
        return self._leg(self.vertical_deltas_m, self.preceding_leg_index)

    @property
    def preceding_spacing_m(self) -> Optional[float]:
        return self._leg(self.spacings_m, self.preceding_leg_index)

    @property
    def spacing_at_failure_m(self) -> Optional[float]:
        """Per-leg spacing, or the episode mean when only aggregates exist."""

        value = self.failing_spacing_m
        if value is not None:
            return value
        return self.spacing_mean_m if self.failing_leg_index is not None else None

    @property
    def turn_magnitude_at_failure_deg(self) -> Optional[float]:
        value = self.failing_turn_deg
        if value is not None:
            return abs(value)
        return self.turn_abs_max_deg if self.failing_leg_index is not None else None

    @property
    def vertical_magnitude_at_failure_m(self) -> Optional[float]:
        value = self.failing_vertical_m
        if value is not None:
            return abs(value)
        return self.vertical_abs_max_m if self.failing_leg_index is not None else None

    @property
    def flown_legs(self) -> int:
        position = self.failure_position
        return 0 if position is None else max(0, position - 1)

    @property
    def cumulative_abs_turn_deg(self) -> Optional[float]:
        """Total heading change the policy had to absorb before dying.

        A proxy for how much lateral correction the course demanded, which is
        what "accumulated lateral error" actually means on a chained course.
        """

        index = self.failing_leg_index
        if index is None:
            return None
        if self.turn_deltas_deg:
            return float(sum(abs(v) for v in self.turn_deltas_deg[: index + 1]))
        if self.turn_abs_mean_deg is None:
            return None
        return float(self.turn_abs_mean_deg) * self.flown_legs

    @property
    def cumulative_abs_vertical_m(self) -> Optional[float]:
        index = self.failing_leg_index
        if index is None:
            return None
        if self.vertical_deltas_m:
            return float(sum(abs(v) for v in self.vertical_deltas_m[: index + 1]))
        if self.vertical_abs_mean_m is None:
            return None
        return float(self.vertical_abs_mean_m) * self.flown_legs

    @property
    def crossing_offset_is_estimated(self) -> bool:
        return self.crossing_offset_m is None

    @property
    def crossing_offset_estimate_m(self) -> Optional[float]:
        """Lateral error at the crossing that preceded the failure, metres.

        Recorders that measure it win.  When they do not and the episode died
        on the *first* transition, the spawn lateral offset is the best proxy
        available: nothing else has acted on the lateral axis yet.
        """

        if self.crossing_offset_m is not None:
            return abs(float(self.crossing_offset_m))
        if self.first_gate_crossed and int(self.gates_completed) <= 1:
            return abs(float(self.initial_lateral_offset_m))
        return None


_SUCCESS_OUTCOMES = {"FINISHED", "SUCCESS", "COMPLETED", "COMPLETE"}


def _episode_succeeded(
    row: Mapping[str, Any], *, episode_type: Optional[str],
    gate_count: int, gates_completed: int,
) -> bool:
    """Success means the thing the episode was *for*, not gates crossed.

    A transition-focus episode is truncated a short window after the first
    crossing, so it can succeed with one gate completed; a full sequence only
    succeeds by finishing.
    """

    if episode_type == "transition_focus" and "universal_transition_success" in row:
        return bool(row["universal_transition_success"])
    if "full_sequence_completion" in row:
        return bool(row["full_sequence_completion"])
    if "universal_transition_success" in row:
        return bool(row["universal_transition_success"])
    outcome = row.get("outcome")
    if outcome is not None:
        return str(outcome).upper() in _SUCCESS_OUTCOMES
    return gate_count > 0 and gates_completed >= gate_count


def failure_record_from_mapping(row: Mapping[str, Any]) -> FailureRecord:
    """Build a record from an evaluation episode row or a train log row.

    Both schemas are read by one builder because their keys are disjoint where
    they differ and identical where they agree; the episode seed present in
    either is deliberately dropped on the floor.
    """

    gate_count = int(row.get("gate_count", 0) or 0)
    gates_completed = int(row.get("gates_completed", 0) or 0)
    gates_completed = max(0, min(gates_completed, gate_count) if gate_count else gates_completed)
    episode_type = row.get("episode_type")
    first_crossed = row.get("first_gate_crossed")
    return FailureRecord(
        gate_count=gate_count,
        gates_completed=gates_completed,
        episode_type=None if episode_type is None else str(episode_type),
        pattern=None if row.get("pattern") is None else str(row.get("pattern")),
        difficulty=(
            None if row.get("difficulty") is None else str(row.get("difficulty"))
        ),
        spacings_m=_as_tuple(row.get("spacings_m")),
        turn_deltas_deg=_as_tuple(row.get("turn_deltas_deg")),
        vertical_deltas_m=_as_tuple(row.get("vertical_deltas_m")),
        spacing_mean_m=_optional_float(row.get("spacing_mean_m")),
        turn_abs_mean_deg=_optional_float(row.get("turn_abs_mean_deg")),
        turn_abs_max_deg=_optional_float(row.get("turn_abs_max_deg")),
        turn_sign_changes=(
            None if row.get("turn_sign_changes") is None
            else int(row["turn_sign_changes"])
        ),
        vertical_abs_mean_m=_optional_float(row.get("vertical_abs_mean_m")),
        vertical_abs_max_m=_optional_float(row.get("vertical_abs_max_m")),
        initial_yaw_error_deg=float(row.get("initial_yaw_error_deg", 0.0) or 0.0),
        initial_lateral_offset_m=float(
            row.get("initial_lateral_offset_m", 0.0) or 0.0),
        crossing_offset_m=_optional_float(row.get("crossing_offset_m")),
        first_gate_crossed=(
            gates_completed >= 1 if first_crossed is None else bool(first_crossed)
        ),
        target_switched=_optional_bool(row.get("correct_target_switch")),
        new_target_aligned=_optional_bool(
            row.get("new_target_acquired_and_aligned")),
        new_target_range_decreasing=_optional_bool(
            row.get("new_target_range_decreasing")),
        collision_events=int(
            row.get("collision_events", row.get("collision", 0)) or 0),
        missed_gate_events=int(
            row.get("missed_gate_dnf", row.get("missed_gate", 0)) or 0),
        wrong_direction_events=int(
            row.get("wrong_direction_events", row.get("wrong_direction", 0)) or 0),
        out_of_bounds_events=int(
            row.get("out_of_bounds_events", row.get("out_of_bounds", 0)) or 0),
        previous_gate_return=bool(row.get("previous_gate_return", False)),
        acquisition_timeout=bool(row.get("acquisition_timeout", False)),
        timeout=bool(row.get("timeout", False)),
        succeeded=_episode_succeeded(
            row, episode_type=episode_type, gate_count=gate_count,
            gates_completed=gates_completed,
        ),
        source=str(row.get("source", "episode_row")),
    )


def as_failure_record(value: Any) -> FailureRecord:
    if isinstance(value, FailureRecord):
        return value
    if isinstance(value, MappingABC):
        return failure_record_from_mapping(value)
    raise TypeError(f"cannot read a failure record from {type(value).__name__}")


# ----------------------------------------------------------------- families

Predicate = Callable[[FailureRecord, FamilyThresholds], bool]


@dataclass(frozen=True)
class FailureFamily:
    """One generic failure mode: what it is, how specific, how much it costs."""

    name: str
    description: str
    #: 3 = names the terminal event, 2 = multi-leg pattern, 1 = single-axis
    #: geometry.  Matches are reported most specific first.
    specificity: int
    #: Cost of one occurrence in [0, 1]; a collision ends the race, an
    #: off-centre crossing only costs time.
    severity: float
    predicate: Predicate

    def matches(self, record: FailureRecord, thresholds: FamilyThresholds) -> bool:
        return bool(self.predicate(record, thresholds))


def _first_gate_acquisition(r: FailureRecord, t: FamilyThresholds) -> bool:
    return r.is_failure and not r.first_gate_crossed


def _off_centre_crossing(r: FailureRecord, t: FamilyThresholds) -> bool:
    offset = r.crossing_offset_estimate_m
    return (
        r.is_failure and r.first_gate_crossed
        and offset is not None and offset >= t.off_centre_offset_m
    )


def _poor_target_switch(r: FailureRecord, t: FamilyThresholds) -> bool:
    return r.is_failure and r.first_gate_crossed and r.target_switched is False


def _reacquisition_after_crossing(r: FailureRecord, t: FamilyThresholds) -> bool:
    return (
        r.is_failure and r.first_gate_crossed and r.target_switched is True
        and (r.new_target_aligned is False
             or r.new_target_range_decreasing is False)
    )


def _large_left_turn(r: FailureRecord, t: FamilyThresholds) -> bool:
    turn = r.failing_turn_deg
    if turn is None:
        return False
    signed = turn if POSITIVE_TURN_IS_LEFT else -turn
    return signed >= t.large_turn_deg


def _large_right_turn(r: FailureRecord, t: FamilyThresholds) -> bool:
    turn = r.failing_turn_deg
    if turn is None:
        return False
    signed = turn if POSITIVE_TURN_IS_LEFT else -turn
    return signed <= -t.large_turn_deg


def _vertical_climb(r: FailureRecord, t: FamilyThresholds) -> bool:
    step = r.failing_vertical_m
    if step is None:
        return False
    signed = step if POSITIVE_VERTICAL_IS_CLIMB else -step
    return signed >= t.large_vertical_m


def _vertical_descent(r: FailureRecord, t: FamilyThresholds) -> bool:
    step = r.failing_vertical_m
    if step is None:
        return False
    signed = step if POSITIVE_VERTICAL_IS_CLIMB else -step
    return signed <= -t.large_vertical_m


def _combined_yaw_vertical(r: FailureRecord, t: FamilyThresholds) -> bool:
    turn = r.turn_magnitude_at_failure_deg
    step = r.vertical_magnitude_at_failure_m
    if turn is None or step is None:
        return False
    return (
        turn >= COMBINED_AXIS_FRACTION * t.large_turn_deg
        and step >= COMBINED_AXIS_FRACTION * t.large_vertical_m
    )


def _short_spacing(r: FailureRecord, t: FamilyThresholds) -> bool:
    spacing = r.spacing_at_failure_m
    return spacing is not None and spacing <= t.short_spacing_m


def _long_spacing(r: FailureRecord, t: FamilyThresholds) -> bool:
    spacing = r.spacing_at_failure_m
    return spacing is not None and spacing >= t.long_spacing_m


def _significant_turns(
    r: FailureRecord, t: FamilyThresholds
) -> Optional[List[float]]:
    """Signed turns on the legs actually flown, course noise removed."""

    index = r.failing_leg_index
    if index is None or not r.turn_deltas_deg:
        return None
    floor = RUN_TURN_FRACTION * t.large_turn_deg
    return [
        float(value) for value in r.turn_deltas_deg[: index + 1]
        if abs(value) >= floor
    ]


def _turns_are_significant_on_average(r: FailureRecord, t: FamilyThresholds) -> bool:
    mean = r.turn_abs_mean_deg
    return mean is not None and mean >= RUN_TURN_FRACTION * t.large_turn_deg


def _consecutive_same_direction_turns(
    r: FailureRecord, t: FamilyThresholds
) -> bool:
    turns = _significant_turns(r, t)
    if turns is None:
        # Aggregate-only rows: no sign change anywhere plus real turn magnitude
        # is the same statement, coarser.
        return (
            r.turn_sign_changes == 0
            and r.flown_legs >= CONSECUTIVE_TURN_RUN
            and _turns_are_significant_on_average(r, t)
        )
    if len(turns) < CONSECUTIVE_TURN_RUN:
        return False
    run = 1
    for index in range(len(turns) - 1, 0, -1):
        if turns[index] * turns[index - 1] > 0:
            run += 1
        else:
            break
    return run >= CONSECUTIVE_TURN_RUN


def _alternating_s_transitions(r: FailureRecord, t: FamilyThresholds) -> bool:
    turns = _significant_turns(r, t)
    if turns is None:
        return (
            r.turn_sign_changes is not None
            and r.turn_sign_changes >= ALTERNATING_SIGN_CHANGES
            and _turns_are_significant_on_average(r, t)
        )
    changes = sum(1 for a, b in zip(turns, turns[1:]) if a * b < 0)
    return changes >= ALTERNATING_SIGN_CHANGES


# Every terminal-event predicate is gated on ``is_failure`` for the same reason
# the geometry ones are anchored at ``failing_leg_index``: a full sequence can
# be logged as FINISHED while the referee still counted a bump or a missed
# attempt, and a taxonomy of failures must not hand a completed run a failure
# family just because something was logged along the way.

def _collision_after_crossing(r: FailureRecord, t: FamilyThresholds) -> bool:
    return r.is_failure and r.collision_events > 0 and int(r.gates_completed) >= 1


def _missed_gate(r: FailureRecord, t: FamilyThresholds) -> bool:
    return r.is_failure and r.missed_gate_events > 0


def _wrong_direction(r: FailureRecord, t: FamilyThresholds) -> bool:
    return r.is_failure and (r.wrong_direction_events > 0 or r.previous_gate_return)


def _out_of_bounds_excursion(r: FailureRecord, t: FamilyThresholds) -> bool:
    return r.is_failure and r.out_of_bounds_events > 0


def _accumulated(r: FailureRecord, t: FamilyThresholds) -> bool:
    """Deep failure with no discrete event to blame -- i.e. drift."""

    position = r.failure_position
    return (
        position is not None
        and position >= ACCUMULATION_MIN_POSITION
        and r.collision_events == 0
        and r.wrong_direction_events == 0
        and not r.previous_gate_return
    )


def _accumulated_lateral_error(r: FailureRecord, t: FamilyThresholds) -> bool:
    total = r.cumulative_abs_turn_deg
    return (
        _accumulated(r, t) and total is not None
        and total >= ACCUMULATION_TURN_MULTIPLE * t.large_turn_deg
    )


def _accumulated_vertical_error(r: FailureRecord, t: FamilyThresholds) -> bool:
    total = r.cumulative_abs_vertical_m
    return (
        _accumulated(r, t) and total is not None
        and total >= ACCUMULATION_VERTICAL_MULTIPLE * t.large_vertical_m
    )


#: The taxonomy, in narrative order.  Matches are returned most specific first,
#: which is declaration order within a specificity tier.
FAILURE_FAMILIES: Tuple[FailureFamily, ...] = (
    FailureFamily(
        "first_gate_acquisition",
        "Never reached the first gate: acquisition or approach, not chaining.",
        3, 0.80, _first_gate_acquisition,
    ),
    FailureFamily(
        "off_centre_crossing",
        "Crossed, but far enough off centre that the next transition started "
        "compromised.",
        3, 0.45, _off_centre_crossing,
    ),
    FailureFamily(
        "poor_target_switch",
        "Crossed a gate and never adopted the next gate as its target.",
        3, 0.70, _poor_target_switch,
    ),
    FailureFamily(
        "reacquisition_after_crossing",
        "Switched target but never aligned on it or never closed range.",
        3, 0.65, _reacquisition_after_crossing,
    ),
    FailureFamily(
        "large_left_turn",
        "Died on a leg demanding a large heading change to port.",
        1, 0.50, _large_left_turn,
    ),
    FailureFamily(
        "large_right_turn",
        "Died on a leg demanding a large heading change to starboard.",
        1, 0.50, _large_right_turn,
    ),
    FailureFamily(
        "vertical_climb",
        "Died on a leg demanding a large climb toward the surface.",
        1, 0.45, _vertical_climb,
    ),
    FailureFamily(
        "vertical_descent",
        "Died on a leg demanding a large descent.",
        1, 0.45, _vertical_descent,
    ),
    FailureFamily(
        "combined_yaw_vertical",
        "Died where yaw and depth had to change together; neither axis alone "
        "was extreme.",
        2, 0.60, _combined_yaw_vertical,
    ),
    FailureFamily(
        "short_spacing",
        "Died on a tightly spaced leg with little room to settle after the "
        "crossing.",
        1, 0.55, _short_spacing,
    ),
    FailureFamily(
        "long_spacing",
        "Died on a long leg where small heading error integrates into large "
        "lateral error.",
        1, 0.40, _long_spacing,
    ),
    FailureFamily(
        "consecutive_same_direction_turns",
        "Died inside a sustained same-direction turn sequence (a spiral).",
        2, 0.55, _consecutive_same_direction_turns,
    ),
    FailureFamily(
        "alternating_s_transitions",
        "Died inside alternating left/right transitions (an S), where each "
        "correction must be unwound immediately.",
        2, 0.60, _alternating_s_transitions,
    ),
    FailureFamily(
        "collision_after_crossing",
        "Hit something after having crossed at least one gate.",
        3, 1.00, _collision_after_crossing,
    ),
    FailureFamily(
        "missed_gate",
        "Passed the gate plane outside the gate: a DNF, not a slow lap.",
        3, 0.90, _missed_gate,
    ),
    FailureFamily(
        "wrong_direction",
        "Crossed a gate the wrong way or returned to an already-passed gate.",
        3, 1.00, _wrong_direction,
    ),
    FailureFamily(
        "out_of_bounds_excursion",
        "Left the course volume entirely.",
        3, 0.85, _out_of_bounds_excursion,
    ),
    FailureFamily(
        "accumulated_lateral_error",
        "Failed deep in a sequence with no discrete event, after absorbing a "
        "large total heading change: lateral drift, not a single bad corner.",
        2, 0.75, _accumulated_lateral_error,
    ),
    FailureFamily(
        "accumulated_vertical_error",
        "Failed deep in a sequence with no discrete event, after absorbing a "
        "large total depth change: vertical drift.",
        2, 0.70, _accumulated_vertical_error,
    ),
)

FAMILY_BY_NAME: Dict[str, FailureFamily] = {
    family.name: family for family in FAILURE_FAMILIES
}
FAILURE_FAMILY_NAMES: Tuple[str, ...] = tuple(FAMILY_BY_NAME)
_FAMILY_ORDER: Dict[str, int] = {
    family.name: index for index, family in enumerate(FAILURE_FAMILIES)
}


def classify_failure(
    record: Any, thresholds: Optional[FamilyThresholds] = None
) -> List[str]:
    """All families a record belongs to, most specific first.

    A record legitimately matches several: "collided after crossing" and "on a
    large left turn at short spacing" are the same failure described at
    different altitudes, and procedural generation needs both.
    """

    resolved = as_failure_record(record)
    limits = thresholds or FamilyThresholds.defaults()
    matched = [
        family for family in FAILURE_FAMILIES if family.matches(resolved, limits)
    ]
    matched.sort(key=lambda family: (-family.specificity, _FAMILY_ORDER[family.name]))
    return [family.name for family in matched]


class FamilyDistribution(MappingABC):
    """Counter-like family tally that also knows its denominator.

    Shares deliberately sum to more than one: a record belongs to every family
    it matches, and forcing a single label would hide exactly the geometry
    combinations that need generating.
    """

    def __init__(
        self,
        counts: Mapping[str, int],
        *,
        n_records: int,
        unclassified: int,
        primary_counts: Mapping[str, int],
        thresholds: FamilyThresholds,
    ) -> None:
        self._counts = {
            name: int(counts.get(name, 0)) for name in FAILURE_FAMILY_NAMES
            if int(counts.get(name, 0)) > 0
        }
        self.n_records = int(n_records)
        self.unclassified = int(unclassified)
        self.primary_counts = dict(primary_counts)
        self.thresholds = thresholds

    def __getitem__(self, family: str) -> int:
        return self._counts.get(str(family), 0)

    def __iter__(self):
        return iter(self._counts)

    def __len__(self) -> int:
        return len(self._counts)

    def most_common(self, n: Optional[int] = None) -> List[Tuple[str, int]]:
        ordered = sorted(
            self._counts.items(),
            key=lambda item: (-item[1], _FAMILY_ORDER[item[0]]),
        )
        return ordered if n is None else ordered[: int(n)]

    def share(self, family: str) -> float:
        return self[family] / self.n_records if self.n_records else 0.0

    @property
    def shares(self) -> Dict[str, float]:
        return {
            name: round(self.share(name), 4) for name, _ in self.most_common()
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": FAILURE_TAXONOMY_VERSION,
            "n_records": self.n_records,
            "counts": dict(self.most_common()),
            "shares": self.shares,
            "primary_counts": dict(self.primary_counts),
            "unclassified": self.unclassified,
            "unclassified_share": (
                round(self.unclassified / self.n_records, 4)
                if self.n_records else 0.0
            ),
            "thresholds": self.thresholds.as_dict(),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FamilyDistribution({self.most_common(5)}, n={self.n_records})"


def classify_many(
    records: Iterable[Any], thresholds: Optional[FamilyThresholds] = None
) -> FamilyDistribution:
    """Tally families over a cohort, deriving the bars from it when unset."""

    resolved = [as_failure_record(value) for value in records]
    limits = thresholds or FamilyThresholds.from_records(resolved)
    counts: Dict[str, int] = {}
    primary: Dict[str, int] = {}
    unclassified = 0
    for record in resolved:
        families = classify_failure(record, limits)
        if not families:
            unclassified += 1
            continue
        primary[families[0]] = primary.get(families[0], 0) + 1
        for name in families:
            counts[name] = counts.get(name, 0) + 1
    return FamilyDistribution(
        counts, n_records=len(resolved), unclassified=unclassified,
        primary_counts=primary, thresholds=limits,
    )


# ------------------------------------------------------- survivor-bias-free

def _episode_gates(value: Any) -> Tuple[int, int]:
    """``(gate_count, gates_completed)`` for a record or a raw row.

    A succeeded episode is credited with its whole course even when it was
    truncated by design.  A transition-focus case ends a short window after its
    crossing having already met its own success criterion, so reading it as
    "failed to reach gate 2" would report a *perfect* policy as failing every
    focus episode; that is right censoring, not failure.  Blending the two
    cohorts still blends two questions, which is what ``episode_types`` is for.
    """

    record = as_failure_record(value)
    gate_count = max(0, int(record.gate_count))
    completed = max(0, min(int(record.gates_completed), gate_count))
    if record.succeeded:
        completed = gate_count
    return gate_count, completed


def _episode_type_of(value: Any) -> Optional[str]:
    return as_failure_record(value).episode_type


def survival_curve(
    episodes: Iterable[Any], *, episode_types: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """P(reach gate k) measured two ways, named so they cannot be swapped.

    ``reach_from_start[k]`` divides by every episode whose course *had* a gate
    k, so it answers "how deep does a run get".  ``conditional_on_previous[k]``
    divides by the episodes that reached k-1, so it answers "given that we got
    here, do we get one more".  Reporting only the second is what made a policy
    that rarely passed gate 3 show 1.00 success at deep positions: the only
    episodes it measured there were the survivors.

    Transition-focus episodes truncate a short window after their first
    crossing, so pass ``episode_types=("full_sequence",)`` when mixing cohorts.
    """

    rows = list(episodes)
    if episode_types is not None:
        allowed = {str(value) for value in episode_types}
        rows = [row for row in rows if _episode_type_of(row) in allowed]
    pairs = [_episode_gates(row) for row in rows]
    n_episodes = len(pairs)
    max_position = max((gate_count for gate_count, _ in pairs), default=0)
    positions = list(range(1, max_position + 1))

    exposure: Dict[int, int] = {}
    reached: Dict[int, int] = {}
    reached_previous: Dict[int, int] = {}
    for position in positions:
        exposed = [pair for pair in pairs if pair[0] >= position]
        exposure[position] = len(exposed)
        reached[position] = sum(1 for _, done in exposed if done >= position)
        reached_previous[position] = sum(
            1 for _, done in exposed if done >= position - 1
        )

    reach_from_start = {
        str(position): (
            reached[position] / exposure[position] if exposure[position] else None
        )
        for position in positions
    }
    conditional = {
        str(position): (
            reached[position] / reached_previous[position]
            if reached_previous[position] else None
        )
        for position in positions
    }
    expected = (
        sum(done for _, done in pairs) / n_episodes if n_episodes else 0.0
    )
    below_half = next(
        (
            position for position in positions
            if reach_from_start[str(position)] is not None
            and reach_from_start[str(position)] < SURVIVAL_HALF_LIFE
        ),
        None,
    )
    gaps = [
        conditional[str(position)] - reach_from_start[str(position)]
        for position in positions
        if conditional[str(position)] is not None
        and reach_from_start[str(position)] is not None
    ]
    return {
        "schema_version": FAILURE_TAXONOMY_VERSION,
        "n_episodes": n_episodes,
        "positions": [str(position) for position in positions],
        "exposure": {str(p): exposure[p] for p in positions},
        "reached": {str(p): reached[p] for p in positions},
        "reach_from_start": reach_from_start,
        "conditional_on_previous": conditional,
        "expected_gates_completed": expected,
        "half_survival_threshold": SURVIVAL_HALF_LIFE,
        "first_position_below_half": below_half,
        "max_conditional_minus_from_start": max(gaps) if gaps else 0.0,
        "definitions": {
            "reach_from_start": (
                "episodes reaching gate k / episodes whose course contained "
                "gate k -- measured from episode start, never conditioned"
            ),
            "conditional_on_previous": (
                "episodes reaching gate k / episodes that reached gate k-1 -- "
                "survivor-conditioned, reads high at deep positions"
            ),
        },
    }


def failure_position_distribution(
    episodes: Iterable[Any], *, episode_types: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """Where failures concentrate, raw and normalised by exposure.

    ``failure_share`` alone is misleading in the same way the conditional
    survival curve is: early positions dominate simply because every episode
    reaches them.  ``hazard_rate`` divides by the at-risk set, which is the
    number that identifies the position the policy genuinely cannot handle.

    ``episode_types`` filters the cohort exactly as in :func:`survival_curve`,
    so a chaining hazard can be read off the full sequences alone.
    """

    rows = list(episodes)
    if episode_types is not None:
        allowed = {str(value) for value in episode_types}
        rows = [row for row in rows if _episode_type_of(row) in allowed]
    pairs = [_episode_gates(row) for row in rows]
    n_episodes = len(pairs)
    max_position = max((gate_count for gate_count, _ in pairs), default=0)
    positions = list(range(1, max_position + 1))
    failures: Dict[int, int] = {position: 0 for position in positions}
    at_risk: Dict[int, int] = {position: 0 for position in positions}
    n_failures = 0
    for gate_count, done in pairs:
        for position in positions:
            if gate_count >= position and done >= position - 1:
                at_risk[position] += 1
        if done >= gate_count:
            continue
        n_failures += 1
        failures[min(done + 1, gate_count)] += 1
    hazard = {
        str(position): (
            failures[position] / at_risk[position] if at_risk[position] else None
        )
        for position in positions
    }
    share = {
        str(position): (failures[position] / n_failures if n_failures else 0.0)
        for position in positions
    }
    peak_hazard = max(
        (p for p in positions if hazard[str(p)] is not None),
        key=lambda p: (hazard[str(p)], -p), default=None,
    )
    peak_share = max(
        positions, key=lambda p: (share[str(p)], -p), default=None,
    )
    return {
        "schema_version": FAILURE_TAXONOMY_VERSION,
        "n_episodes": n_episodes,
        "n_failures": n_failures,
        "positions": [str(position) for position in positions],
        "failures_at_position": {str(p): failures[p] for p in positions},
        "at_risk_at_position": {str(p): at_risk[p] for p in positions},
        "hazard_rate": hazard,
        "failure_share": share,
        "peak_hazard_position": peak_hazard,
        "peak_share_position": peak_share,
    }


# ------------------------------------------------------------------ mining

def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _records_from_report(payload: Any) -> List[FailureRecord]:
    rows = payload.get("episodes") if isinstance(payload, MappingABC) else payload
    if not isinstance(rows, list):
        return []
    return [
        failure_record_from_mapping({**row, "source": "evaluation_report"})
        for row in rows if isinstance(row, MappingABC)
    ]


def load_evaluation_records(path: str | Path) -> List[FailureRecord]:
    """Read an evaluation output into records, geometry included when present.

    Per-case directories are preferred over ``evaluation.json`` because the
    sibling ``track.json`` carries the per-leg geometry that the aggregated
    episode rows never stored, and every geometry family needs it.
    """

    target = Path(path)
    if target.is_file():
        return _records_from_report(_read_json(target))
    if not target.is_dir():
        raise FileNotFoundError(str(target))
    records: List[FailureRecord] = []
    for episode_path in sorted(target.rglob("episode.json")):
        row = _read_json(episode_path)
        if not isinstance(row, MappingABC):
            continue
        track = _read_json(episode_path.parent / "track.json")
        geometry = (
            track.get("universal_transition")
            if isinstance(track, MappingABC) else None
        )
        merged = {**dict(row), **dict(geometry or {}), "source": "evaluation_case"}
        records.append(failure_record_from_mapping(merged))
    if records:
        return records
    report = target / "evaluation.json"
    return _records_from_report(_read_json(report)) if report.exists() else []


def _family_context(records: Sequence[FailureRecord]) -> Dict[str, Any]:
    """Everything a generator needs to make more of this failure."""

    return {
        "gate_count": _summary([r.gate_count for r in records]),
        "failure_position": _summary(
            [r.failure_position for r in records if r.failure_position]),
        "spacing_m": _summary(
            [r.spacing_at_failure_m for r in records]),
        "turn_abs_deg": _summary(
            [r.turn_magnitude_at_failure_deg for r in records]),
        "vertical_abs_m": _summary(
            [r.vertical_magnitude_at_failure_m for r in records]),
        "preceding_spacing_m": _summary([r.preceding_spacing_m for r in records]),
        "preceding_turn_abs_deg": _summary([
            None if r.preceding_turn_deg is None else abs(r.preceding_turn_deg)
            for r in records
        ]),
        "preceding_vertical_abs_m": _summary([
            None if r.preceding_vertical_m is None else abs(r.preceding_vertical_m)
            for r in records
        ]),
        "cumulative_abs_turn_deg": _summary(
            [r.cumulative_abs_turn_deg for r in records]),
        "cumulative_abs_vertical_m": _summary(
            [r.cumulative_abs_vertical_m for r in records]),
        "difficulty_share": _shares([r.difficulty for r in records]),
        "pattern_share": _shares([r.pattern for r in records]),
        "episode_type_share": _shares([r.episode_type for r in records]),
    }


def mine_failure_families(
    evaluation_dir_or_records: Any = None,
    *,
    train_episode_rows: Optional[Iterable[Mapping[str, Any]]] = None,
    thresholds: Optional[FamilyThresholds] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Rank generic failure families by frequency x severity.

    Thresholds are derived from the whole cohort (successes included) because
    "is this turn large?" is a question about exposure; only the failures are
    then classified.  The report carries per-family geometry context so a
    procedural generator can be pointed at the family, and no episode identity
    so it cannot be pointed at an individual course.
    """

    records: List[FailureRecord] = []
    if isinstance(evaluation_dir_or_records, (str, Path)):
        records.extend(load_evaluation_records(evaluation_dir_or_records))
    elif evaluation_dir_or_records is not None:
        records.extend(
            as_failure_record(value) for value in evaluation_dir_or_records
        )
    if train_episode_rows is not None:
        records.extend(
            failure_record_from_mapping({**dict(row), "source": "train_log"})
            for row in train_episode_rows
        )

    limits = thresholds or FamilyThresholds.from_records(records)
    failures = [record for record in records if record.is_failure]
    distribution = classify_many(failures, limits)
    n_failures = len(failures)

    ranked: List[Dict[str, Any]] = []
    for family in FAILURE_FAMILIES:
        members = [
            record for record in failures
            if family.matches(record, limits)
        ]
        if not members:
            continue
        share = len(members) / n_failures if n_failures else 0.0
        ranked.append({
            "family": family.name,
            "description": family.description,
            "specificity": family.specificity,
            "severity": family.severity,
            "count": len(members),
            "share_of_failures": round(share, 4),
            "priority": round(share * family.severity, 6),
            "context": _family_context(members),
        })
    ranked.sort(
        key=lambda row: (-row["priority"], -row["count"], _FAMILY_ORDER[row["family"]])
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank

    report = {
        "schema_version": FAILURE_TAXONOMY_VERSION,
        "n_episodes": len(records),
        "n_failures": n_failures,
        "failure_rate": round(n_failures / len(records), 4) if records else None,
        "thresholds": limits.as_dict(),
        "families": ranked,
        "top_families": [row["family"] for row in ranked[: max(0, int(top_k))]],
        "unclassified_failures": distribution.unclassified,
        "family_distribution": distribution.as_dict(),
        "survival": survival_curve(records),
        "failure_positions": failure_position_distribution(records),
        # The cohort curve blends a 2-gate transition case with a 22-gate
        # sequence at position 2.  The chaining question is only answerable on
        # the full sequences, so it is reported separately rather than left for
        # a reader to misquote the blended one.
        "survival_chaining": (
            survival_curve(records, episode_types=("full_sequence",))
            if any(record.episode_type == "full_sequence" for record in records)
            else None
        ),
        "record_sources": _shares([record.source for record in records]),
    }
    # Self-enforcing: a report that leaked a holdout case would defeat the
    # reason this module exists, so it never leaves the function unchecked.
    assert_no_seed_memorisation(report)
    return report


# ------------------------------------------------------------------- guard

_SEED_KEY = re.compile(r"seed", re.IGNORECASE)
#: "seed 5000123", "seed=5000123", "episode_seed_5000123", and the plural list
#: form "seeds: 5000123, 5000124" -- a seed named in prose.  The trailing list
#: is captured whole because naming the second seed is memorisation exactly as
#: much as naming the first.  Bare integers inside strings are deliberately not
#: scanned: ``ppo_5000128_steps.zip`` is a step count, not a seed.
_SEED_IN_TEXT = re.compile(
    r"seeds?[^0-9a-zA-Z]{0,3}(\d+(?:\s*[,;]\s*\d+)*)", re.IGNORECASE
)
_INTEGER_IN_LIST = re.compile(r"\d+")
_BARE_INTEGER = re.compile(r"^-?\d+$")


def _guarded_bands(roles: Sequence[str]) -> List[Tuple[str, int, int]]:
    bands = []
    for role in roles:
        group = SEED_GROUPS.get(str(role))
        if group is None:
            raise ValueError(f"unknown seed role {role!r}")
        bands.append((group.name, group.start, group.end))
    return bands


def assert_no_seed_memorisation(
    report: Any, *, roles: Sequence[str] = ("validation", "test"),
) -> None:
    """Refuse a report that names concrete holdout episodes.

    The taxonomy exists so training can target geometry *families*.  A report
    that carried validation seeds would invite regenerating exactly those
    courses, which is memorisation wearing a diagnosis as a hat -- and would
    quietly turn the validation band into training data.

    Two rules: any integer anywhere inside a guarded seed band, and any
    ``seed``-like key holding concrete integers regardless of band.
    """

    bands = _guarded_bands(roles)

    def offending_band(value: Any) -> Optional[str]:
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None
        if isinstance(value, float) and not float(value).is_integer():
            return None
        for name, start, end in bands:
            if start <= number < end:
                return name
        return None

    def fail(reason: str, path: str) -> None:
        raise SeedMemorisationError(
            f"{reason} at {path or '<root>'}; failure reports must describe "
            f"geometry families, never individual holdout episodes"
        )

    def walk(node: Any, path: str) -> None:
        if isinstance(node, MappingABC):
            for key, value in node.items():
                child = f"{path}.{key}" if path else str(key)
                if _SEED_KEY.search(str(key)) and _contains_integer(value):
                    fail(f"seed-valued key {key!r}", path)
                walk(key, child)
                walk(value, child)
            return
        if isinstance(node, (list, tuple, set)):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")
            return
        if isinstance(node, bool) or node is None:
            return
        if isinstance(node, (int, float)):
            band = offending_band(node)
            if band is not None:
                fail(f"{band} seed {int(node)}", path)
            return
        if isinstance(node, str):
            candidates = [
                number
                for run in _SEED_IN_TEXT.findall(node)
                for number in _INTEGER_IN_LIST.findall(run)
            ]
            if _BARE_INTEGER.match(node.strip()):
                candidates.append(node.strip())
            for candidate in candidates:
                band = offending_band(candidate)
                if band is not None:
                    fail(f"{band} seed {candidate}", path)

    walk(report, "")


def _contains_integer(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return float(value).is_integer()
    if isinstance(value, str):
        return bool(_BARE_INTEGER.match(value.strip()))
    if isinstance(value, (list, tuple, set)):
        return any(_contains_integer(item) for item in value)
    return False


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir")
    parser.add_argument("--train-run-dir", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    train_rows = None
    if args.train_run_dir:
        from marine_race_arena.learning.episode_composition_log import read_episodes

        train_rows = read_episodes(args.train_run_dir)
    print(json.dumps(mine_failure_families(
        args.evaluation_dir, train_episode_rows=train_rows, top_k=args.top_k,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
