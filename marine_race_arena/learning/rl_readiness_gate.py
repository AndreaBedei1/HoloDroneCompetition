"""Pre-registered readiness gate, policy freeze record, and holdout access guard.

This module is the scientific-integrity boundary of the project.  The three
official circuits are a single-use holdout: the moment their results are seen,
every subsequent tuning decision is contaminated by them, and the generalization
claim collapses.  Three mechanisms enforce that, in order:

``readiness gate``
    A frozen, pre-registered set of criteria measured on the *validation* band.
    The bars are written down here, before the numbers exist, so a policy cannot
    be declared "ready enough" by moving the bar afterwards.  Scientific judgment
    is still allowed, but only through :class:`ReadinessOverride`: it must name
    the criteria, it must record a reason, it can forgive only misses inside
    measurement granularity, and it can never forgive a safety or evidence
    criterion.

``freeze record``
    Freezing is once-only and content-hashed.  The record pins the exact
    checkpoint bytes, the contracts, the curriculum, the validation evidence, the
    readiness verdict *and the final-circuit protocol*, so the protocol is fixed
    before any final-circuit result is seen.

``access guard``
    :func:`assert_final_circuits_unlocked` is the only sanctioned door to the
    sealed circuits, and it stays shut without a valid, ready, untampered freeze
    record.  After the door opens, :func:`record_final_circuit_evaluation`
    appends to a hash-chained ledger and
    :func:`assert_no_tuning_after_holdout` refuses any further training or
    configuration change.

The content hashes here are tamper-*evident*, not tamper-proof: anyone may
recompute them.  They exist to make an accidental or casual edit of a frozen
record impossible to miss, not to resist a determined adversary.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from marine_race_arena.learning.generic_sequence_curriculum import CURRICULUM_VERSION
from marine_race_arena.learning.longrun_checkpoint import canonical_hash, sha256_file
from marine_race_arena.learning.provenance import now_utc
from marine_race_arena.learning.rl_holdout_policy import (
    FINAL_CIRCUIT_NAMES,
    is_final_circuit,
    role_of_seed,
)
from marine_race_arena.learning.seed_registry import RESERVED_FINAL_MULTIGATE_SEEDS
from marine_race_arena.learning.transition_curriculum import DIFFICULTY_LEVELS

READINESS_GATE_VERSION = "rl_readiness_gate_v1"
FREEZE_RECORD_VERSION = "rl_policy_freeze_record_v1"
FINAL_CIRCUIT_LEDGER_VERSION = "final_circuit_evaluation_ledger_v1"
FINAL_CIRCUIT_PROTOCOL_VERSION = "final_circuit_protocol_v1"

FREEZE_RECORD_FILENAME = "final_policy_freeze.json"
FINAL_CIRCUIT_LEDGER_FILENAME = "final_circuit_evaluations.jsonl"


class ReadinessGateError(RuntimeError):
    """Base class for every refusal raised by the integrity boundary."""


class ReadinessNotMet(ReadinessGateError):
    """Raised when a policy that has not passed the gate is being frozen."""


class PolicyAlreadyFrozen(ReadinessGateError):
    """Raised on any attempt to overwrite an existing freeze record."""


class FinalCircuitsLocked(ReadinessGateError):
    """Raised when the sealed circuits are requested without a valid freeze."""


class LedgerTampered(ReadinessGateError):
    """Raised when the append-only holdout ledger fails its chain check."""


class HoldoutContaminated(ReadinessGateError):
    """Raised when tuning is attempted after the holdout has been opened."""


class ProtocolViolation(ReadinessGateError):
    """Raised when a trial would vary something the protocol pins down."""


# --------------------------------------------------------------- thresholds

GROUP_COMPETENCE = "competence"
GROUP_COMPLETION = "sequence_completion"
GROUP_SAFETY = "safety"
GROUP_EVIDENCE = "evidence"

#: Safety criteria protect the vehicle and the referee, and evidence criteria
#: protect the measurement itself.  Neither may ever be forgiven by judgment:
#: "we did not measure it" and "it crashed" are not close calls.
NON_OVERRIDABLE_GROUPS = frozenset({GROUP_SAFETY, GROUP_EVIDENCE})

#: 17- and 22-gate courses are not required to be reliable, only to be *possible*
#: in a reproducible way: at least one completion in four attempts, which at the
#: mandated minimum of 20 cases per length means at least five completions.  A
#: single lucky run cannot clear it.
ROBUST_COMPLETION_MINIMUM = 0.25

#: Where a difficulty-ladder run is stored inside the evidence, once normalised.
LADDER_KEY = "difficulty_ladder"

#: Rate differences are computed in binary floating point, where a one-case gap
#: lands a few ulps either side of ``1/n``.  Only the narrow-miss *margin* is
#: padded by this; the thresholds themselves are compared exactly.
_MARGIN_EPS = 1e-9


@dataclass(frozen=True)
class ReadinessThresholds:
    """The pre-registered bars, fixed before the numbers exist.

    ``min_full_cases_per_length`` is deliberately larger than the benchmark
    default of five: a 0.95 completion bar cannot even be *expressed* with five
    cases, whose resolution is 0.20.  Twenty cases resolve every bound below to
    0.05, so each criterion is measurable rather than nominal.

    The difficulty bar is a *ladder*, not a single level, because a single level
    is unusable in both directions.  Every candidate trained exclusively at G1 --
    promotion needs >=0.99 success with zero safety events and no evaluation ever
    cleared it -- so demanding matched evidence at G6 can only ever return "not
    ready", while accepting G1 alone would certify a controller on near-straight,
    near-level, near-aligned geometry and call it circuit-ready.  So the matched
    benchmark is pinned to the regime the candidates share
    (``primary_difficulty``) and a separate ladder run must show the easy regime
    was not the only regime tested.
    """

    # -- universal transition competence ---------------------------------
    min_universal_transition_success: float = 0.90
    min_first_gate_crossing: float = 0.97
    min_target_switch: float = 0.85
    min_new_target_alignment: float = 0.85
    # -- whole-sequence completion, by gate count ------------------------
    completion_by_length: Tuple[Tuple[int, float], ...] = (
        (3, 0.95),
        (5, 0.90),
        (8, 0.75),
        (12, 0.50),
        (17, ROBUST_COMPLETION_MINIMUM),
        (22, ROBUST_COMPLETION_MINIMUM),
    )
    # -- difficulty-ladder competence (overridable like any other bar) -----
    #: G3 is the first rung with vertical structure a real circuit would call
    #: structure at all (1.20 m steps against G1's 0.20 m and G2's 0.35 m), which
    #: is why it, and not G2, is the mid-ladder rung the evidence must reach.
    ladder_min_rung: str = "G3"
    #: The ladder is a coarse stress probe (60 transition cases per rung by
    #: default), so the bar is set where a coarse sample can still tell "weak"
    #: from "collapsed": half the transitions completed and four first gates in
    #: five.  Below that, the G1 numbers describe a controller a circuit would
    #: never meet.
    min_ladder_transition_success: float = 0.50
    min_ladder_first_gate_crossing: float = 0.80
    # -- safety (never overridable) ---------------------------------------
    max_out_of_bounds_rate: float = 0.01
    max_collision_episode_rate: float = 0.20
    # -- matched evidence (never overridable) -----------------------------
    required_dataset_split: str = "validation"
    #: The difficulty the matched benchmark is measured at: the one regime every
    #: candidate actually trained in, so the comparison between them is matched.
    #: It is a statement about provenance, never about readiness.
    primary_difficulty: str = "G1"
    min_transition_cases: int = 500
    min_full_cases_per_length: int = 20
    #: A rung counts as *run* only with enough cases to mean something.  Half the
    #: ladder tool's default keeps shortened probes usable while excluding the
    #: token sample -- and n=0 -- that a fabricated block would claim.
    min_ladder_rung_cases: int = 30
    # -- judgment budget ---------------------------------------------------
    #: A miss is "narrow" when it is within two percentage points *or* within one
    #: measured case, whichever is looser; a bar measured on 20 cases cannot be
    #: missed by less than 0.05, so a fixed absolute margin alone would make
    #: completion criteria un-forgivable in principle rather than in policy.
    narrow_miss_margin: float = 0.02
    max_overridden_criteria: int = 2

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": READINESS_GATE_VERSION,
            "min_universal_transition_success": self.min_universal_transition_success,
            "min_first_gate_crossing": self.min_first_gate_crossing,
            "min_target_switch": self.min_target_switch,
            "min_new_target_alignment": self.min_new_target_alignment,
            "completion_by_length": {
                str(length): float(bound) for length, bound in self.completion_by_length
            },
            "ladder_min_rung": self.ladder_min_rung,
            "min_ladder_transition_success": self.min_ladder_transition_success,
            "min_ladder_first_gate_crossing": self.min_ladder_first_gate_crossing,
            "max_out_of_bounds_rate": self.max_out_of_bounds_rate,
            "max_collision_episode_rate": self.max_collision_episode_rate,
            "required_dataset_split": self.required_dataset_split,
            "primary_difficulty": self.primary_difficulty,
            "min_transition_cases": self.min_transition_cases,
            "min_full_cases_per_length": self.min_full_cases_per_length,
            "min_ladder_rung_cases": self.min_ladder_rung_cases,
            "narrow_miss_margin": self.narrow_miss_margin,
            "max_overridden_criteria": self.max_overridden_criteria,
        }

    def sha256(self) -> str:
        """Content hash, so a freeze record proves which bars were in force."""

        return canonical_hash(self.as_dict())


READINESS_THRESHOLDS = ReadinessThresholds()


# ---------------------------------------------------------------- criteria

@dataclass(frozen=True)
class ReadinessCriterion:
    """One pre-registered bar and the value measured against it."""

    name: str
    metric: str
    group: str
    direction: str  # "min" (observed >= bound) or "max" (observed <= bound)
    bound: float
    observed: Optional[float]
    narrow_miss_margin: float

    @property
    def evaluated(self) -> bool:
        return self.observed is not None

    @property
    def satisfied(self) -> bool:
        """An unmeasured criterion never counts as satisfied.

        Unlike checkpoint ranking, where missing metrics keep frozen historical
        reports usable, readiness is a claim about evidence: absence of evidence
        must fail, or the gate could be passed by reporting less.
        """

        if self.observed is None:
            return False
        if self.direction == "min":
            return float(self.observed) >= float(self.bound)
        return float(self.observed) <= float(self.bound)

    @property
    def shortfall(self) -> Optional[float]:
        """How far the measurement is on the wrong side of the bar."""

        if self.observed is None:
            return None
        gap = (
            float(self.bound) - float(self.observed)
            if self.direction == "min"
            else float(self.observed) - float(self.bound)
        )
        return max(0.0, gap)

    @property
    def is_safety(self) -> bool:
        return self.group == GROUP_SAFETY

    @property
    def overridable(self) -> bool:
        return self.group not in NON_OVERRIDABLE_GROUPS

    @property
    def narrow_miss(self) -> bool:
        """Whether the miss is inside the declared margin.

        The comparison is epsilon-padded because the margin is usually one
        measured case (``1/n``) and rate differences are computed in binary
        floating point: ``0.75 - 0.70`` evaluates to 0.05000000000000004, so a
        genuine one-of-twenty miss on the 8-gate bucket would otherwise be
        classified as too large to forgive while the same miss on the 3-gate
        bucket is not.  The margin is a judgment boundary, not a bar, so the
        padding cannot admit anything a human would call a different case.
        """

        shortfall = self.shortfall
        return (
            not self.satisfied
            and shortfall is not None
            and shortfall - float(self.narrow_miss_margin) <= _MARGIN_EPS
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric,
            "group": self.group,
            "direction": self.direction,
            "bound": float(self.bound),
            "observed": None if self.observed is None else float(self.observed),
            "evaluated": self.evaluated,
            "satisfied": self.satisfied,
            "shortfall": self.shortfall,
            "narrow_miss": self.narrow_miss,
            "overridable": self.overridable,
        }


# ---------------------------------------------------------------- override

MIN_OVERRIDE_REASON_CHARS = 20

GRANT_NARROW_MISS = "narrow_miss_forgiven"
REFUSAL_NOT_FAILING = "criterion_already_satisfied"
REFUSAL_SAFETY = "safety_criterion_is_never_overridable"
REFUSAL_EVIDENCE = "evidence_criterion_is_never_overridable"
REFUSAL_NOT_MEASURED = "criterion_was_not_measured"
REFUSAL_TOO_LARGE = "miss_is_larger_than_the_narrow_margin"
REFUSAL_BUDGET = "override_budget_exhausted"


@dataclass(frozen=True)
class ReadinessOverride:
    """A recorded scientific-judgment exception to the pre-registered gate.

    The reason is mandatory and substantive: an override that nobody has to
    justify in writing is just a lower threshold with extra steps.
    """

    reason: str
    approved_criteria: Tuple[str, ...]
    approver: Optional[str] = None
    utc: Optional[str] = None

    def __post_init__(self) -> None:
        reason = str(self.reason or "").strip()
        if len(reason) < MIN_OVERRIDE_REASON_CHARS:
            raise ValueError(
                "a readiness override must record a written reason of at least "
                f"{MIN_OVERRIDE_REASON_CHARS} characters"
            )
        names = tuple(dict.fromkeys(str(name) for name in self.approved_criteria))
        if not names:
            raise ValueError("a readiness override must name the criteria it forgives")
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "approved_criteria", names)
        object.__setattr__(self, "utc", self.utc or now_utc())

    def as_dict(self) -> Dict[str, Any]:
        return {
            "reason": self.reason,
            "approved_criteria": list(self.approved_criteria),
            "approver": self.approver,
            "utc": self.utc,
        }


@dataclass(frozen=True)
class OverrideDecision:
    """What actually happened to one requested override, and why."""

    criterion: str
    granted: bool
    rationale: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "criterion": self.criterion,
            "granted": self.granted,
            "rationale": self.rationale,
        }


# ----------------------------------------------------------------- verdict

@dataclass(frozen=True)
class ReadinessVerdict:
    """Per-criterion outcome plus the overall decision and its exceptions."""

    ready: bool
    criteria: Tuple[ReadinessCriterion, ...]
    thresholds: ReadinessThresholds
    override: Optional[ReadinessOverride] = None
    decisions: Tuple[OverrideDecision, ...] = ()

    @property
    def overall(self) -> bool:
        """Alias for :attr:`ready`; the gate has exactly one overall answer."""

        return self.ready

    @property
    def granted_overrides(self) -> Tuple[str, ...]:
        return tuple(item.criterion for item in self.decisions if item.granted)

    @property
    def failures(self) -> Tuple[str, ...]:
        """Criteria still failing after any granted overrides."""

        granted = set(self.granted_overrides)
        return tuple(
            item.name
            for item in self.criteria
            if not item.satisfied and item.name not in granted
        )

    @property
    def narrow_misses(self) -> Tuple[str, ...]:
        return tuple(item.name for item in self.criteria if item.narrow_miss)

    @property
    def unmeasured(self) -> Tuple[str, ...]:
        return tuple(item.name for item in self.criteria if not item.evaluated)

    @property
    def ready_without_override(self) -> bool:
        return all(item.satisfied for item in self.criteria)

    def criterion(self, name: str) -> ReadinessCriterion:
        for item in self.criteria:
            if item.name == name:
                return item
        raise KeyError(name)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": READINESS_GATE_VERSION,
            "ready": self.ready,
            "ready_without_override": self.ready_without_override,
            "failures": list(self.failures),
            "narrow_misses": list(self.narrow_misses),
            "unmeasured": list(self.unmeasured),
            "thresholds": self.thresholds.as_dict(),
            "thresholds_sha256": self.thresholds.sha256(),
            "criteria": [item.as_dict() for item in self.criteria],
            "override": None if self.override is None else self.override.as_dict(),
            "override_decisions": [item.as_dict() for item in self.decisions],
        }


# --------------------------------------------------------------- extraction

def benchmark_inputs(report_or_metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Accept either a full benchmark report or just its ``metrics`` block.

    The run configuration that makes the set *matched* (difficulty, case seeds)
    lives at the report's top level while the measurements live under
    ``metrics``, and the gate needs both.  The per-episode rows are replaced by
    the one thing the gate needs from them -- the seed *band* each case was
    drawn from -- so the evidence survives without the episode log.
    """

    inputs = dict(report_or_metrics)
    nested = inputs.get("metrics")
    merged = {
        key: value
        for key, value in inputs.items()
        if key not in {"metrics", "episodes"}
    }
    if isinstance(nested, Mapping):
        merged.update(nested)
    rungs = ladder_rungs(inputs)
    if rungs is None:
        # An unparseable block is no evidence, and leaving it in place would let
        # it be mistaken for some.
        merged.pop(LADDER_KEY, None)
    else:
        merged[LADDER_KEY] = {"rungs": rungs}
    roles = _roles_from_seeds(inputs.get("episodes"))
    if roles is None:
        roles = _roles_from_seeds(
            merged.get("episode_seeds") or merged.get("case_seeds")
        )
    if roles is not None:
        # Roles derived from real case seeds replace any summary the caller
        # supplied: a pre-computed ``episode_seed_roles`` must never outrank the
        # seeds it claims to summarise.
        merged["episode_seed_roles"] = sorted(set(roles))
        merged["episode_seed_n"] = len(roles)
    return merged


def _number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _positive_int(value: Any) -> Optional[int]:
    number = _number(value)
    if number is None or number <= 0:
        return None
    return int(number)


def _margin(base: float, sample_size: Optional[int]) -> float:
    """Narrow-miss margin widened to the granularity of the measurement."""

    if not sample_size or int(sample_size) <= 0:
        return float(base)
    return max(float(base), 1.0 / float(sample_size))


def _length_entry(
    by_length: Optional[Mapping[Any, Any]], length: int
) -> Optional[Mapping[str, Any]]:
    if by_length is None:
        return None
    entry = by_length.get(str(length), by_length.get(length))
    return entry if isinstance(entry, Mapping) else None


# ----------------------------------------------------------- difficulty ladder

#: How far down to look for the ladder block.  ``run_difficulty_ladder`` writes
#: ``{"rungs": [...]}`` at the top of its own report, but callers hang that
#: report off an evidence bundle, a benchmark report or a metrics block, and
#: refusing a correctly-run ladder because of where it was pasted would punish
#: the wrong thing.
_LADDER_SEARCH_DEPTH = 4


def _ladder_index(value: Any) -> Optional[int]:
    """Position on the geometry ladder, or ``None`` for an unknown rung."""

    try:
        return DIFFICULTY_LEVELS.index(str(value).strip().upper())
    except (ValueError, AttributeError):
        return None


def _ladder_block(root: Any) -> Optional[Mapping[str, Any]]:
    """The nearest mapping carrying a ``rungs`` list, wherever it was nested."""

    queue: List[Tuple[Any, int]] = [(root, 0)]
    while queue:
        node, depth = queue.pop(0)
        if not isinstance(node, Mapping):
            continue
        if isinstance(node.get("rungs"), (list, tuple)):
            return node
        if depth < _LADDER_SEARCH_DEPTH:
            queue.extend((child, depth + 1) for child in node.values())
    return None


def _confidence_interval(value: Any) -> Optional[List[float]]:
    bounds = (
        [_number(item) for item in value]
        if isinstance(value, (list, tuple)) else []
    )
    if len(bounds) != 2 or any(bound is None for bound in bounds):
        return None
    return [float(bound) for bound in bounds]


def ladder_rungs(report_or_metrics: Any) -> Optional[List[Dict[str, Any]]]:
    """Normalise a ``run_difficulty_ladder`` report into comparable rows.

    ``None`` means no ladder block was found at all; an empty list means one was
    found but claimed nothing recognisable.  Rows keep the Wilson interval the
    ladder tool reports, because the probe is deliberately coarse and the freeze
    record should say so rather than pin a bare rate.
    """

    block = _ladder_block(report_or_metrics)
    if block is None:
        return None
    rows: List[Dict[str, Any]] = []
    for item in block.get("rungs") or ():
        if not isinstance(item, Mapping):
            continue
        index = _ladder_index(item.get("difficulty"))
        if index is None:
            continue
        rows.append({
            "difficulty": DIFFICULTY_LEVELS[index],
            "n": int(_positive_int(item.get("n")) or 0),
            "success": _number(item.get("success")),
            "success_ci": _confidence_interval(item.get("success_ci")),
            "first_gate": _number(item.get("first_gate")),
        })
    return rows


def difficulty_ladder_summary(
    report_or_metrics: Mapping[str, Any],
    thresholds: ReadinessThresholds = READINESS_THRESHOLDS,
) -> Optional[Dict[str, Any]]:
    """Summarise a ladder run against the pre-registered mid rung.

    ``None`` means no ladder was ever run, which is a different thing from a
    ladder that ran and stopped below the mid rung: the first is missing
    evidence, the second is evidence that the easy regime was the only regime
    tested.  Both are fatal, but only the second is informative.

    The bar is read at the *easiest* qualifying rung, so probing further up the
    ladder can never make the gate harder to pass -- otherwise the honest thing
    (measure the whole curve) would be the costly thing.  A rung repeated within
    one report is taken at its worst row, so a duplicate cannot improve on
    itself.
    """

    rows = ladder_rungs(report_or_metrics)
    if rows is None:
        return None
    min_cases = int(thresholds.min_ladder_rung_cases)
    min_index = _ladder_index(thresholds.ladder_min_rung)
    mid = (
        []
        if min_index is None
        else [
            row for row in rows
            if row["n"] >= min_cases and _ladder_index(row["difficulty"]) >= min_index
        ]
    )
    governing: Optional[Dict[str, Any]] = None
    if mid:
        lowest = min(_ladder_index(row["difficulty"]) for row in mid)
        governing = min(
            (row for row in mid if _ladder_index(row["difficulty"]) == lowest),
            # An unmeasured rate sorts below every measured one: it must not be
            # able to hide a worse sibling.
            key=lambda row: -1.0 if row["success"] is None else float(row["success"]),
        )
    return {
        "rungs": rows,
        "rungs_claimed": [row["difficulty"] for row in rows],
        "required_min_rung": str(thresholds.ladder_min_rung),
        "min_rung_cases": min_cases,
        "mid_ladder_rungs": sorted({row["difficulty"] for row in mid}),
        "governing_rung": None if governing is None else governing["difficulty"],
        "governing_rung_n": None if governing is None else governing["n"],
        "success": None if governing is None else governing["success"],
        "first_gate": None if governing is None else governing["first_gate"],
    }


#: Resolved split values that are *not* a band, each of which fails the matched
#: evidence criterion for a different reason.
SPLIT_UNREGISTERED = "unregistered_seed_band"
SPLIT_MIXED = "mixed_seed_bands"
SPLIT_CONTRADICTORY = "contradictory"
SPLIT_UNVERIFIED = "unverified_label"


def _roles_from_seeds(seeds: Any) -> Optional[List[str]]:
    """Seed bands of a case list, accepting episode rows or bare seeds.

    ``None`` means "no seed evidence at all", which is different from "seeds
    outside every band": the first is unmeasured, the second is disproof.
    """

    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, (list, tuple)):
        return None
    roles: List[str] = []
    for item in seeds:
        seed = _number(item.get("seed")) if isinstance(item, Mapping) else _number(item)
        if seed is None:
            continue
        roles.append(role_of_seed(int(seed)) or SPLIT_UNREGISTERED)
    return roles or None


def _rate(
    inputs: Mapping[str, Any],
    rate_key: str,
    count_key: str,
    n_eval: Optional[int],
) -> Optional[float]:
    """Prefer a reported rate, else derive one from episode counts."""

    reported = _number(inputs.get(rate_key))
    if reported is not None:
        return reported
    count = _number(inputs.get(count_key))
    if count is None or n_eval is None:
        return None
    return count / float(n_eval)


def resolved_dataset_split(inputs: Mapping[str, Any]) -> Optional[str]:
    """The role of the evaluation set, with the *case* seed bands as authority.

    The band-bearing seeds are the per-episode seeds
    (``TransitionGeometrySampler._episode_seed`` draws them from the band that
    ``dataset_split`` selects).  The report's top-level ``seed`` is only the
    sampler's master seed and carries no band membership at all -- the full
    sequence sampler even runs at ``seed + 1_000_003``, which lands inside the
    train band for small master seeds -- so reading it as a band would both
    reject genuine validation runs and accept train-band evidence whose master
    seed happens to fall in the validation range.

    Every case seed must therefore resolve to the same registered band; a
    declared label that contradicts it loses, and a label with no seed evidence
    behind it is reported as unverified rather than believed.
    """

    declared = None
    for key in ("dataset_split", "split", "evaluation_role", "data_role"):
        value = inputs.get(key)
        if value:
            declared = str(value).strip().lower()
            break
    roles = inputs.get("episode_seed_roles")
    if roles is None:
        roles = _roles_from_seeds(
            inputs.get("episode_seeds") or inputs.get("case_seeds")
        )
    if not roles:
        return None if declared is None else SPLIT_UNVERIFIED
    bands = sorted({str(role) for role in roles})
    if len(bands) > 1:
        return SPLIT_MIXED
    band = bands[0]
    if band == SPLIT_UNREGISTERED:
        return SPLIT_UNREGISTERED
    if declared is not None and declared != band:
        return SPLIT_CONTRADICTORY
    return band


def _match_indicator(value: Optional[str], required: str) -> Optional[float]:
    if value is None:
        return None
    return 1.0 if str(value).strip().lower() == str(required).strip().lower() else 0.0


def _build_criteria(
    inputs: Mapping[str, Any], thresholds: ReadinessThresholds
) -> Tuple[ReadinessCriterion, ...]:
    n_eval = _positive_int(inputs.get("n_eval"))
    transition_n = _positive_int(inputs.get("transition_n"))
    by_length = inputs.get("full_sequence_success_by_length")
    by_length = by_length if isinstance(by_length, Mapping) else None
    base = float(thresholds.narrow_miss_margin)
    transition_margin = _margin(base, transition_n)

    criteria: List[ReadinessCriterion] = [
        ReadinessCriterion(
            "universal_transition_success", "universal_transition_success_rate",
            GROUP_COMPETENCE, "min", thresholds.min_universal_transition_success,
            _number(inputs.get("universal_transition_success_rate")),
            transition_margin,
        ),
        ReadinessCriterion(
            "first_gate_crossing", "first_gate_crossing_rate",
            GROUP_COMPETENCE, "min", thresholds.min_first_gate_crossing,
            _number(inputs.get("first_gate_crossing_rate")), transition_margin,
        ),
        ReadinessCriterion(
            "target_switch", "target_switch_rate",
            GROUP_COMPETENCE, "min", thresholds.min_target_switch,
            _number(inputs.get("target_switch_rate")), transition_margin,
        ),
        ReadinessCriterion(
            "new_target_alignment", "new_target_alignment_rate",
            GROUP_COMPETENCE, "min", thresholds.min_new_target_alignment,
            _number(inputs.get("new_target_alignment_rate")), transition_margin,
        ),
    ]
    # The ladder bars are competence, not evidence: a policy that is merely
    # *worse* at G3 than at G1 is expected, and a narrow miss there is a
    # judgment call like any other.  Only the question of whether the mid rung
    # was run at all belongs to the evidence group below.
    ladder = difficulty_ladder_summary(inputs, thresholds)
    rung = (ladder or {}).get("governing_rung") or f">={thresholds.ladder_min_rung}"
    ladder_margin = _margin(base, (ladder or {}).get("governing_rung_n"))
    criteria.extend((
        ReadinessCriterion(
            "ladder_transition_success",
            f"{LADDER_KEY}[{rung}].success",
            GROUP_COMPETENCE, "min", thresholds.min_ladder_transition_success,
            None if ladder is None else ladder["success"], ladder_margin,
        ),
        ReadinessCriterion(
            "ladder_first_gate_crossing",
            f"{LADDER_KEY}[{rung}].first_gate",
            GROUP_COMPETENCE, "min", thresholds.min_ladder_first_gate_crossing,
            None if ladder is None else ladder["first_gate"], ladder_margin,
        ),
    ))
    for length, bound in thresholds.completion_by_length:
        entry = _length_entry(by_length, length)
        cases = None if entry is None else _positive_int(entry.get("n"))
        criteria.append(
            ReadinessCriterion(
                f"completion_{length}_gate",
                f"full_sequence_success_by_length[{length}].completion_rate",
                GROUP_COMPLETION, "min", float(bound),
                None if entry is None else _number(entry.get("completion_rate")),
                _margin(base, cases),
            )
        )
    criteria.extend((
        ReadinessCriterion(
            "out_of_bounds_rate", "out_of_bounds_episodes/n_eval",
            GROUP_SAFETY, "max", thresholds.max_out_of_bounds_rate,
            _rate(inputs, "out_of_bounds_rate", "out_of_bounds_episodes", n_eval),
            _margin(base, n_eval),
        ),
        ReadinessCriterion(
            "collision_episode_rate", "collision_episodes/n_eval",
            GROUP_SAFETY, "max", thresholds.max_collision_episode_rate,
            _rate(inputs, "collision_episode_rate", "collision_episodes", n_eval),
            _margin(base, n_eval),
        ),
        ReadinessCriterion(
            "matched_validation_split", "dataset_split",
            GROUP_EVIDENCE, "min", 1.0,
            _match_indicator(
                resolved_dataset_split(inputs), thresholds.required_dataset_split
            ),
            0.0,
        ),
        ReadinessCriterion(
            "matched_difficulty", "difficulty",
            GROUP_EVIDENCE, "min", 1.0,
            _match_indicator(inputs.get("difficulty"), thresholds.primary_difficulty),
            0.0,
        ),
        # The bar that G1-only evidence cannot clear by being good at G1.  It
        # counts rungs that were actually *run*, so a policy cannot pass by
        # having been tested on the easy rung alone -- nor by claiming a mid rung
        # it never sampled.
        ReadinessCriterion(
            "ladder_mid_rung_coverage",
            f"{LADDER_KEY}.rungs[>={thresholds.ladder_min_rung}, "
            f"n>={thresholds.min_ladder_rung_cases}]",
            GROUP_EVIDENCE, "min", 1.0,
            None if ladder is None else float(len(ladder["mid_ladder_rungs"])),
            0.0,
        ),
        ReadinessCriterion(
            "transition_cases", "transition_n",
            GROUP_EVIDENCE, "min", float(thresholds.min_transition_cases),
            _number(inputs.get("transition_n")), 0.0,
        ),
        ReadinessCriterion(
            "full_sequence_cases_per_length", "full_sequence_success_by_length[*].n",
            GROUP_EVIDENCE, "min", float(thresholds.min_full_cases_per_length),
            None if by_length is None else min(
                float(_number((_length_entry(by_length, length) or {}).get("n")) or 0.0)
                for length, _ in thresholds.completion_by_length
            ),
            0.0,
        ),
    ))
    return tuple(criteria)


def _refusal(criterion: ReadinessCriterion) -> Optional[str]:
    """Why an override cannot be granted, or ``None`` when it can.

    Safety and evidence are checked before "was it measured", so an unmeasured
    safety criterion is refused as a safety criterion rather than looking like a
    reporting gap.
    """

    if criterion.satisfied:
        return REFUSAL_NOT_FAILING
    if criterion.is_safety:
        return REFUSAL_SAFETY
    if not criterion.overridable:
        return REFUSAL_EVIDENCE
    if not criterion.evaluated:
        return REFUSAL_NOT_MEASURED
    if not criterion.narrow_miss:
        return REFUSAL_TOO_LARGE
    return None


def evaluate_readiness(
    metrics: Mapping[str, Any],
    thresholds: ReadinessThresholds = READINESS_THRESHOLDS,
    *,
    override: Optional[ReadinessOverride] = None,
) -> ReadinessVerdict:
    """Judge one validation benchmark against the pre-registered gate.

    ``metrics`` may be a full benchmark report or its ``metrics`` block.  An
    override is applied criterion by criterion in the order it lists them, and
    every request -- granted or refused -- is recorded on the verdict, so a
    refused override is visible rather than silent.  Naming an unknown criterion
    raises, because a typo must never be mistaken for a forgiven criterion.
    """

    inputs = benchmark_inputs(metrics)
    criteria = _build_criteria(inputs, thresholds)
    index = {item.name: item for item in criteria}
    decisions: List[OverrideDecision] = []
    if override is not None:
        unknown = [name for name in override.approved_criteria if name not in index]
        if unknown:
            raise ValueError(
                f"unknown readiness criteria in override: {sorted(unknown)}; "
                f"known criteria are {sorted(index)}"
            )
        budget = int(thresholds.max_overridden_criteria)
        granted = 0
        for name in override.approved_criteria:
            rationale = _refusal(index[name])
            if rationale is None and granted >= budget:
                rationale = REFUSAL_BUDGET
            if rationale is None:
                granted += 1
                decisions.append(OverrideDecision(name, True, GRANT_NARROW_MISS))
            else:
                decisions.append(OverrideDecision(name, False, rationale))
    forgiven = {item.criterion for item in decisions if item.granted}
    ready = all(
        item.satisfied or item.name in forgiven for item in criteria
    )
    return ReadinessVerdict(
        ready=ready,
        criteria=criteria,
        thresholds=thresholds,
        override=override,
        decisions=tuple(decisions),
    )


# ------------------------------------------------------- final-circuit protocol

TRIALS_PER_CIRCUIT = 10

#: Trials need seeds that belong to no procedural role -- reusing the TEST band
#: would spend a holdout that exists for a different question.  The project
#: already publishes exactly such a reservation: seed_registry's
#: RESERVED_FINAL_MULTIGATE_SEEDS (1800-1899), marked "DO NOT USE until the final
#: multi-gate eval", and final_benchmark's official groups already draw from it.
#: Minting a fresh band here instead would have left the ledger unable to record
#: the very episodes the benchmark runs, so the protocol conforms to the existing
#: reservation rather than competing with it; 1800-1804 also match the published
#: current-free circuit runs, keeping the new results directly comparable.
FINAL_CIRCUIT_TRIAL_SEEDS: Tuple[int, ...] = tuple(
    RESERVED_FINAL_MULTIGATE_SEEDS[:TRIALS_PER_CIRCUIT]
)

#: Seeds and start-state perturbation only.  Nothing here may describe the
#: course itself.
PROTOCOL_MAY_VARY: Tuple[str, ...] = (
    "evaluation_seed",
    "initial_position_offset_m",
    "initial_yaw_error_deg",
    "initial_body_velocity_m_s",
)
PROTOCOL_MUST_NOT_VARY: Tuple[str, ...] = (
    "circuit_geometry",
    "gate_positions",
    "gate_orientations",
    "gate_order",
    "gate_count",
    "gate_dimensions",
    "course_scale",
    "start_gate",
    "policy_weights",
    "policy_determinism",
    "observation_contract",
    "action_contract",
    "reward_contract",
    "difficulty",
    "max_steps",
    "adapter",
    "trials_per_circuit",
)
PROTOCOL_METRICS: Tuple[str, ...] = (
    "circuit_completed",
    "gates_crossed",
    "gates_crossed_fraction",
    "completion_time_s",
    "collision_events",
    "collision_episodes",
    "out_of_bounds_events",
    "wrong_direction_events",
    "missed_gate_dnf",
    "previous_gate_returns",
    "mean_action_jerk",
    "dnf_reason",
)

#: Any of these substrings in a "may vary" entry would mean the course itself is
#: being resampled, which is the one thing the protocol exists to forbid.
_GEOMETRY_TOKENS = (
    "geometry", "gate", "course", "track", "circuit", "layout", "waypoint",
)


def _assert_variation_whitelist_is_geometry_free() -> None:
    offenders = sorted(
        name for name in PROTOCOL_MAY_VARY
        if any(token in name.lower() for token in _GEOMETRY_TOKENS)
    )
    if offenders:
        raise ProtocolViolation(
            f"the final-circuit protocol may never permit varying {offenders}: "
            "circuit geometry is fixed by definition"
        )


# Fail at import: a future edit that lets geometry vary must break loudly and
# immediately, not at the moment the holdout is opened.
_assert_variation_whitelist_is_geometry_free()


def _normalize_variation(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def protocol_permits_variation(name: str) -> bool:
    """Default-deny: only the explicitly enumerated fields may vary."""

    return _normalize_variation(name) in {
        _normalize_variation(item) for item in PROTOCOL_MAY_VARY
    }


def assert_protocol_variations_allowed(names: Iterable[str]) -> None:
    disallowed = sorted(
        str(name) for name in names if not protocol_permits_variation(name)
    )
    if disallowed:
        raise ProtocolViolation(
            f"the final-circuit protocol pins {disallowed}; only "
            f"{list(PROTOCOL_MAY_VARY)} may vary between trials"
        )


def final_circuit_protocol() -> Dict[str, Any]:
    """The evaluation protocol, fixed before any final-circuit result is seen."""

    _assert_variation_whitelist_is_geometry_free()
    return {
        "schema_version": FINAL_CIRCUIT_PROTOCOL_VERSION,
        "circuits": list(FINAL_CIRCUIT_NAMES),
        "trials_per_circuit": TRIALS_PER_CIRCUIT,
        "total_trials": TRIALS_PER_CIRCUIT * len(FINAL_CIRCUIT_NAMES),
        "trial_seeds": list(FINAL_CIRCUIT_TRIAL_SEEDS),
        "seed_band": "reserved final-circuit band, disjoint from train/validation/test",
        "deterministic_policy": True,
        "may_vary_between_trials": list(PROTOCOL_MAY_VARY),
        "must_not_vary": list(PROTOCOL_MUST_NOT_VARY),
        "metrics": list(PROTOCOL_METRICS),
        "primary_metric": "circuit_completion_rate",
        "reporting": {
            "report_every_trial": True,
            "post_hoc_trial_exclusion_permitted": False,
            "post_hoc_threshold_changes_permitted": False,
            "executions_permitted": 1,
        },
    }


# ------------------------------------------------------------- freeze record

_CONTRACT_ALIASES: Dict[str, Tuple[str, ...]] = {
    "reward": ("reward", "reward_contract_version", "reward_version"),
    "observation": (
        "observation", "observation_contract_version", "observation_version",
        "obs_version",
    ),
    "action": ("action", "action_contract_version", "action_version"),
}


def _json_safe(value: Any) -> Any:
    """Recursively coerce numpy scalars and paths into JSON-hashable values."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    number = _number(value)
    return number if number is not None else str(value)


def _resolved_contracts(contracts: Mapping[str, Any]) -> Dict[str, Any]:
    resolved = {str(key): _json_safe(value) for key, value in contracts.items()}
    for name, aliases in _CONTRACT_ALIASES.items():
        value = next(
            (contracts[alias] for alias in aliases if contracts.get(alias)), None
        )
        if not value:
            raise ValueError(
                f"the freeze record requires the {name} contract version "
                f"(one of {list(aliases)})"
            )
        resolved[name] = str(value)
    return resolved


def validation_benchmark_summary(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """The validation evidence the freeze record pins, without the episode log."""

    inputs = benchmark_inputs(metrics)
    keys = (
        "seed", "difficulty", "n_eval", "transition_n", "algorithm",
        "universal_transition_success_rate", "first_gate_crossing_rate",
        "target_switch_rate", "new_target_alignment_rate",
        "new_target_range_decrease_rate", "long_sequence_completion_score",
        "collision_episodes", "collision_events", "out_of_bounds_episodes",
        "missed_gate_dnf", "wrong_direction_events", "previous_gate_returns",
        "acquisition_timeouts", "mean_absolute_action",
        "nontrivial_action_fraction", "mean_action_jerk",
        "full_sequence_success_by_length", "transition_success_by_position",
        "episode_seed_roles", "episode_seed_n", LADDER_KEY,
    )
    summary = {key: _json_safe(inputs[key]) for key in keys if key in inputs}
    # ``seed`` above is the sampler master seed; the band evidence is the case
    # seeds, so the resolved split is recorded next to it and never inferred
    # from it.
    summary["dataset_split"] = resolved_dataset_split(inputs)
    return summary


def freeze_record_id(record: Mapping[str, Any]) -> str:
    """Content hash of a freeze record, excluding the identifier itself."""

    body = {key: value for key, value in record.items() if key != "freeze_id"}
    return canonical_hash(_json_safe(body))


def freeze_record_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / FREEZE_RECORD_FILENAME


def freeze_policy(
    checkpoint: str | Path,
    *,
    run_dir: str | Path,
    metrics: Mapping[str, Any],
    verdict: ReadinessVerdict,
    git_sha: Optional[str],
    contracts: Mapping[str, Any],
    training_transitions: Optional[int] = None,
    curriculum_version: str = CURRICULUM_VERSION,
    algorithm: str = "ppo",
    path: Optional[str | Path] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """Write the once-only, content-hashed freeze record for a ready policy.

    The record is created with an exclusive open, so two concurrent processes
    cannot both believe they froze the policy, and a re-freeze after a "small
    extra training run" is impossible by construction.  ``training_transitions``
    may be supplied directly, through ``contracts`` or through the benchmark
    report; it is never guessed.
    """

    if not verdict.ready:
        raise ReadinessNotMet(
            "refusing to freeze a policy that failed the readiness gate: "
            f"{list(verdict.failures)}"
        )
    # A verdict carries whatever thresholds it was judged against, which is what
    # makes the gate testable -- and would otherwise make it trivially
    # bypassable, since home-made bars need no reason, no budget and no safety
    # protection.  Freezing is the moment of commitment, so only the
    # pre-registered bars may be committed to.
    if verdict.thresholds.sha256() != READINESS_THRESHOLDS.sha256():
        raise ReadinessNotMet(
            "refusing to freeze against thresholds that are not the "
            f"pre-registered ones ({verdict.thresholds.sha256()[:12]} != "
            f"{READINESS_THRESHOLDS.sha256()[:12]}): moving the bar is not a "
            "way of passing the gate"
        )
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint to freeze does not exist: {checkpoint}")
    inputs = benchmark_inputs(metrics)
    # The record pins *these* metrics as the evidence, so the verdict must be
    # reproducible from them; a ready verdict computed on some other benchmark
    # would otherwise unlock the circuits on evidence nobody ever checked.
    recomputed = evaluate_readiness(inputs, verdict.thresholds, override=verdict.override)
    if not recomputed.ready:
        raise ReadinessNotMet(
            "the supplied verdict is ready but the metrics being recorded do "
            f"not reproduce it: {list(recomputed.failures)}"
        )
    transitions = training_transitions
    if transitions is None:
        transitions = contracts.get("training_transitions")
    if transitions is None:
        transitions = inputs.get("total_environment_transitions")
    if transitions is None:
        raise ValueError(
            "the freeze record requires the training transition count; pass "
            "training_transitions explicitly"
        )
    target = Path(path) if path is not None else freeze_record_path(run_dir)
    record: Dict[str, Any] = {
        "schema_version": FREEZE_RECORD_VERSION,
        "readiness_gate_version": READINESS_GATE_VERSION,
        "utc": now_utc(),
        "git_sha": None if git_sha is None else str(git_sha),
        "policy": {
            "checkpoint": checkpoint_path.name,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_bytes": int(checkpoint_path.stat().st_size),
            "run_dir": str(run_dir),
            "algorithm": str(algorithm),
            "training_transitions": int(transitions),
        },
        "contracts": _resolved_contracts(contracts),
        "curriculum_version": str(curriculum_version),
        "validation_benchmark": validation_benchmark_summary(inputs),
        "readiness_verdict": verdict.as_dict(),
        "readiness_thresholds_sha256": verdict.thresholds.sha256(),
        "final_circuit_protocol": final_circuit_protocol(),
        "note": note,
    }
    record = _json_safe(record)
    record["freeze_id"] = freeze_record_id(record)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(record, indent=2, sort_keys=True))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise PolicyAlreadyFrozen(
            f"a freeze record already exists at {target}; freezing is once-only "
            "and a second freeze would silently replace the evaluated policy"
        ) from exc
    return record


# --------------------------------------------------------------- access guard

def _validated_freeze_record(
    freeze_record_path: str | Path, *, verify_checkpoint: bool
) -> Dict[str, Any]:
    path = Path(freeze_record_path)
    if not path.is_file():
        raise FinalCircuitsLocked(
            f"no freeze record at {path}: the three final circuits stay sealed "
            "until a policy has passed the pre-registered readiness gate and "
            "been frozen"
        )
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FinalCircuitsLocked(f"freeze record at {path} is unreadable: {exc}") from exc
    if not isinstance(record, Mapping):
        raise FinalCircuitsLocked(f"freeze record at {path} is not an object")
    if record.get("schema_version") != FREEZE_RECORD_VERSION:
        raise FinalCircuitsLocked(
            f"freeze record at {path} has unsupported schema "
            f"{record.get('schema_version')!r}"
        )
    recorded_id = record.get("freeze_id")
    if not recorded_id or recorded_id != freeze_record_id(record):
        raise FinalCircuitsLocked(
            f"freeze record at {path} does not match its own content hash: it "
            "was edited after freezing and cannot be trusted"
        )
    verdict = record.get("readiness_verdict")
    if not isinstance(verdict, Mapping) or verdict.get("ready") is not True:
        failures = (
            list(verdict.get("failures", []))
            if isinstance(verdict, Mapping) else ["no readiness verdict"]
        )
        raise FinalCircuitsLocked(
            f"the frozen policy did not pass the readiness gate ({failures}); "
            "the final circuits stay sealed"
        )
    # The record is self-consistent by its own hash, so the remaining question
    # is whether the bars it was judged against are the pre-registered ones.  A
    # record frozen by a build with relaxed thresholds -- or thresholds relaxed
    # after freezing -- must lock the door, not open it.
    preregistered = READINESS_THRESHOLDS.sha256()
    recorded_bars = record.get("readiness_thresholds_sha256")
    verdict_bars = verdict.get("thresholds_sha256")
    if recorded_bars != preregistered or (
        verdict_bars is not None and verdict_bars != recorded_bars
    ):
        raise FinalCircuitsLocked(
            f"the freeze record at {path} was judged against thresholds "
            f"{recorded_bars!r}, not the pre-registered "
            f"{preregistered!r}; the final circuits stay sealed"
        )
    if verify_checkpoint:
        policy = record.get("policy") or {}
        candidate = Path(str(policy.get("checkpoint_path", "")))
        if not candidate.is_file():
            candidate = path.parent / str(policy.get("checkpoint", ""))
        if not candidate.is_file():
            raise FinalCircuitsLocked(
                f"the frozen checkpoint {policy.get('checkpoint_path')!r} is "
                "missing; the evaluated policy cannot be the frozen one"
            )
        if sha256_file(candidate) != policy.get("checkpoint_sha256"):
            raise FinalCircuitsLocked(
                f"the checkpoint at {candidate} no longer matches the frozen "
                "sha256: the policy changed after it was frozen"
            )
    return dict(record)


def assert_final_circuits_unlocked(
    freeze_record_path: str | Path, *, verify_checkpoint: bool = True
) -> Dict[str, Any]:
    """Open the door to the sealed circuits, or refuse and explain why.

    This is the pre-freeze peek guard: without a valid, ready, untampered freeze
    record whose checkpoint bytes still hash to the frozen value, there is no
    sanctioned way to reach the final circuits.
    """

    return _validated_freeze_record(
        freeze_record_path, verify_checkpoint=verify_checkpoint
    )


def final_circuits_unlocked(
    freeze_record_path: str | Path, *, verify_checkpoint: bool = True
) -> bool:
    """Non-raising form of :func:`assert_final_circuits_unlocked`."""

    try:
        assert_final_circuits_unlocked(
            freeze_record_path, verify_checkpoint=verify_checkpoint
        )
    except FinalCircuitsLocked:
        return False
    return True


# ------------------------------------------------------------- holdout ledger

def _entry_id(entry: Mapping[str, Any]) -> str:
    body = {key: value for key, value in entry.items() if key != "entry_sha256"}
    return canonical_hash(_json_safe(body))


def read_final_circuit_ledger(ledger_path: str | Path) -> List[Dict[str, Any]]:
    """Ledger entries in append order; an absent ledger is simply empty."""

    path = Path(ledger_path)
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def verify_ledger_chain(ledger_path: str | Path) -> List[Dict[str, Any]]:
    """Verify the hash chain, so a deleted or edited entry cannot hide.

    Each entry commits to its predecessor, which makes the ledger append-only in
    evidence as well as in intent: dropping an inconvenient holdout result
    breaks every entry after it.
    """

    try:
        entries = read_final_circuit_ledger(ledger_path)
    except (OSError, ValueError) as exc:
        raise LedgerTampered(f"holdout ledger {ledger_path} is unreadable: {exc}") from exc
    previous = None
    for index, entry in enumerate(entries):
        if entry.get("schema_version") != FINAL_CIRCUIT_LEDGER_VERSION:
            raise LedgerTampered(f"ledger entry {index} has an unsupported schema")
        if int(entry.get("entry_index", -1)) != index:
            raise LedgerTampered(
                f"ledger entry {index} is out of order or an entry was removed"
            )
        if entry.get("previous_sha256") != previous:
            raise LedgerTampered(f"ledger entry {index} breaks the hash chain")
        if entry.get("entry_sha256") != _entry_id(entry):
            raise LedgerTampered(f"ledger entry {index} was edited after it was written")
        previous = entry.get("entry_sha256")
    return entries


def record_final_circuit_evaluation(
    ledger_path: str | Path,
    *,
    freeze_record_path: str | Path,
    circuit: str,
    trial_seeds: Sequence[int],
    metrics: Mapping[str, Any],
    note: Optional[str] = None,
    verify_checkpoint: bool = True,
) -> Dict[str, Any]:
    """Append one final-circuit result to the hash-chained, append-only ledger.

    The guard runs here rather than in the caller so a result cannot be recorded
    unless the policy really was frozen and ready first, and the seeds are
    checked against the pre-registered protocol so trials cannot be re-rolled
    until a circuit is completed.
    """

    record = assert_final_circuits_unlocked(
        freeze_record_path, verify_checkpoint=verify_checkpoint
    )
    if not is_final_circuit(circuit):
        raise ValueError(
            f"{circuit!r} is not one of the sealed final circuits "
            f"{list(FINAL_CIRCUIT_NAMES)}"
        )
    seeds = [int(seed) for seed in trial_seeds]
    if not seeds:
        raise ValueError("a final-circuit evaluation must record its trial seeds")
    unregistered = sorted(set(seeds) - set(FINAL_CIRCUIT_TRIAL_SEEDS))
    if unregistered:
        raise ProtocolViolation(
            f"trial seeds {unregistered} are not the pre-registered protocol "
            f"seeds {list(FINAL_CIRCUIT_TRIAL_SEEDS)}"
        )
    entries = verify_ledger_chain(ledger_path)
    policy = record.get("policy") or {}
    entry: Dict[str, Any] = {
        "schema_version": FINAL_CIRCUIT_LEDGER_VERSION,
        "entry_index": len(entries),
        "previous_sha256": entries[-1]["entry_sha256"] if entries else None,
        "utc": now_utc(),
        "freeze_id": record.get("freeze_id"),
        "git_sha": record.get("git_sha"),
        "checkpoint_sha256": policy.get("checkpoint_sha256"),
        "checkpoint": policy.get("checkpoint"),
        "protocol_version": FINAL_CIRCUIT_PROTOCOL_VERSION,
        "circuit": str(circuit),
        "trial_seeds": seeds,
        "metrics": _json_safe(metrics),
        "note": note,
    }
    entry = _json_safe(entry)
    entry["entry_sha256"] = _entry_id(entry)
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, separators=(",", ":"), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return entry


def assert_no_tuning_after_holdout(
    ledger_path: str | Path,
    current_git_sha: Optional[str],
    *,
    policy_sha256: Optional[str] = None,
    change: str = "training_or_configuration_change",
    allow_reproduction: bool = False,
) -> None:
    """Refuse any training or configuration change once the holdout is open.

    The holdout is single-use.  Once a final-circuit result exists, any further
    training, hyperparameter change, reward change or checkpoint reselection is
    informed by that result, whether or not the author intends it to be, so the
    rule is unconditional rather than a judgment call.  ``allow_reproduction``
    permits exactly one narrow exception: re-running the *identical* policy from
    the *identical* commit, which produces no new decisions.
    """

    entries = verify_ledger_chain(ledger_path)
    if not entries:
        return
    frozen_shas = {entry.get("git_sha") for entry in entries}
    policy_shas = {entry.get("checkpoint_sha256") for entry in entries}
    current = None if current_git_sha is None else str(current_git_sha)
    if (
        allow_reproduction
        and current is not None
        and current in frozen_shas
        and policy_sha256 is not None
        and str(policy_sha256) in policy_shas
    ):
        return
    circuits = sorted({str(entry.get("circuit")) for entry in entries})
    raise HoldoutContaminated(
        f"the final circuits {circuits} were already evaluated with git sha "
        f"{sorted(str(sha) for sha in frozen_shas)}; refusing {change} at "
        f"{current!r}: any change after the holdout is opened is post-hoc "
        "tuning on the holdout"
    )


def gate_manifest() -> Dict[str, Any]:
    """Everything a report needs to state the gate that was pre-registered."""

    return {
        "schema_version": READINESS_GATE_VERSION,
        "thresholds": READINESS_THRESHOLDS.as_dict(),
        "thresholds_sha256": READINESS_THRESHOLDS.sha256(),
        "non_overridable_groups": sorted(NON_OVERRIDABLE_GROUPS),
        "max_overridden_criteria": READINESS_THRESHOLDS.max_overridden_criteria,
        "min_override_reason_chars": MIN_OVERRIDE_REASON_CHARS,
        "final_circuit_protocol": final_circuit_protocol(),
    }
