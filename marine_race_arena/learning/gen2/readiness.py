"""The pre-registered Generation-2 readiness gate.

Committed **before** the sealed holdout is opened and **before** the numbers it
judges exist.  After that commit the thresholds are fixed: the freeze record
pins :data:`GEN2_READINESS_THRESHOLDS`'s own SHA-256, and every later read
re-compares it, so moving a bar retroactively invalidates the freeze rather
than quietly passing.

The value objects (``ReadinessCriterion``, ``ReadinessOverride``,
``OverrideDecision``, ``ReadinessVerdict``, the refusal ladder) are reused
verbatim from :mod:`marine_race_arena.learning.rl_readiness_gate` -- they carry
no Gen-1 numbers.  Only the thresholds object and the criteria builder are
Gen-2's own, because Gen-1's criteria are hashed into a completed experiment
and must not be touched.

Threshold rationale, in the brief's own hierarchy (completion > safety >
efficiency):

* **Transition quality.** Gen-1's defining failure was 1.00 at gate 1 against
  0.7333 for the complete gate1 -> gate2 transition.  Gen-2 therefore gates on
  the *unconditional* transition, not on first-gate crossing, which is why
  ``gate1_to_gate2_transition`` sits at the same 0.95 bar as the headline
  universal-transition rate.
* **Composition.** The completion ladder decays with length (0.95 / 0.90 /
  0.85 / 0.70 / 0.50 / 0.45) because the claim is that compounding error is
  *reduced*, not eliminated.  A flat bar would be either trivially passed at
  3 gates or unreachable at 22.
* **Safety is not tradeable.** Out-of-bounds is effectively zero-tolerance and
  collisions are capped; both live in the non-overridable safety group, so no
  amount of completion can buy them off.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from marine_race_arena.learning.gen2 import (
    GEN2_ACTION_CONTRACT,
    GEN2_EXPERIMENT_ID,
    GEN2_OBS_CONTRACT,
)
from marine_race_arena.learning.rl_readiness_gate import (
    GROUP_EVIDENCE,
    GROUP_SAFETY,
    NON_OVERRIDABLE_GROUPS,
    OverrideDecision,
    ReadinessCriterion,
    ReadinessOverride,
    _refusal,
)

GEN2_GATE_ID = "gen2_readiness_gate_v1"

GROUP_TRANSITION = "transition"
GROUP_COMPLETION = "completion"

#: A miss smaller than this may be forgiven by a written override, and only for
#: transition/completion criteria.  Safety and evidence are never overridable.
GEN2_NARROW_MISS_MARGIN = 0.03
GEN2_MAX_OVERRIDDEN_CRITERIA = 2


def canonical_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class Gen2ReadinessThresholds:
    """The Gen-2 bars.  Frozen at commit; the hash is the pre-registration."""

    gate_id: str = GEN2_GATE_ID

    # --- transition quality (the Gen-1 failure mode) ----------------------
    universal_transition_success: float = 0.95
    gate1_to_gate2_transition: float = 0.95

    # --- composition across sequence length -------------------------------
    completion_3_gate: float = 0.95
    completion_5_gate: float = 0.90
    completion_8_gate: float = 0.85
    completion_12_gate: float = 0.70
    completion_17_gate: float = 0.50
    completion_22_gate: float = 0.45

    # --- safety (non-overridable) -----------------------------------------
    max_out_of_bounds_episode_rate: float = 0.005
    max_collision_episode_rate: float = 0.10
    max_wrong_direction_episode_rate: float = 0.01

    # --- evidence (non-overridable) ---------------------------------------
    min_transition_cases: int = 500
    min_cases_per_length: int = 40
    min_gate1_to_gate2_cases: int = 200

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def sha256(self) -> str:
        return canonical_hash(self.as_dict())

    def completion_by_length(self) -> Dict[int, float]:
        return {
            3: self.completion_3_gate,
            5: self.completion_5_gate,
            8: self.completion_8_gate,
            12: self.completion_12_gate,
            17: self.completion_17_gate,
            22: self.completion_22_gate,
        }


#: THE pre-registration.  Do not edit after the committing change lands.
GEN2_READINESS_THRESHOLDS = Gen2ReadinessThresholds()


@dataclass(frozen=True)
class Gen2ReadinessVerdict:
    ready: bool
    criteria: Tuple[ReadinessCriterion, ...]
    thresholds: Gen2ReadinessThresholds
    override: Optional[ReadinessOverride] = None
    decisions: Tuple[OverrideDecision, ...] = ()

    @property
    def failures(self) -> Tuple[ReadinessCriterion, ...]:
        granted = {item.criterion for item in self.decisions if item.granted}
        return tuple(c for c in self.criteria if not c.satisfied and c.name not in granted)

    @property
    def unmeasured(self) -> Tuple[ReadinessCriterion, ...]:
        return tuple(c for c in self.criteria if not c.evaluated)

    @property
    def ready_without_override(self) -> bool:
        return all(c.satisfied for c in self.criteria)

    def criterion(self, name: str) -> ReadinessCriterion:
        for item in self.criteria:
            if item.name == name:
                return item
        raise KeyError(name)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "gate_id": GEN2_GATE_ID,
            "ready": self.ready,
            "ready_without_override": self.ready_without_override,
            "thresholds": self.thresholds.as_dict(),
            "thresholds_sha256": self.thresholds.sha256(),
            "criteria": [c.as_dict() for c in self.criteria],
            "failures": [c.name for c in self.failures],
            "unmeasured": [c.name for c in self.unmeasured],
            "override": self.override.as_dict() if self.override else None,
            "decisions": [d.as_dict() for d in self.decisions],
        }

    def sha256(self) -> str:
        return canonical_hash(self.as_dict())

    def summary(self) -> str:
        lines = [
            f"Gen-2 readiness: {'PASS' if self.ready else 'FAIL'} "
            f"({sum(c.satisfied for c in self.criteria)}/{len(self.criteria)} criteria met)",
        ]
        for item in self.criteria:
            mark = "ok " if item.satisfied else "MISS"
            observed = "unmeasured" if item.observed is None else f"{item.observed:.4f}"
            arrow = ">=" if item.direction == "min" else "<="
            lines.append(
                f"  [{mark}] {item.name:<34} {observed:>11} {arrow} {item.bound:<8} ({item.group})"
            )
        return "\n".join(lines)


def _value(metrics: Mapping[str, Any], *names: str) -> Optional[float]:
    for name in names:
        if name in metrics and metrics[name] is not None:
            try:
                return float(metrics[name])
            except (TypeError, ValueError):
                return None
    return None


def _nested(metrics: Mapping[str, Any], block: str, key: Any) -> Optional[float]:
    section = metrics.get(block)
    if not isinstance(section, Mapping):
        return None
    for candidate in (key, str(key), int(key) if str(key).isdigit() else key):
        if candidate in section and section[candidate] is not None:
            try:
                return float(section[candidate])
            except (TypeError, ValueError):
                return None
    return None


def build_gen2_criteria(
    metrics: Mapping[str, Any],
    thresholds: Gen2ReadinessThresholds = GEN2_READINESS_THRESHOLDS,
) -> Tuple[ReadinessCriterion, ...]:
    """Turn a Gen-2 benchmark report into the 15 pre-registered criteria."""
    margin = GEN2_NARROW_MISS_MARGIN
    criteria = [
        ReadinessCriterion(
            name="universal_transition_success",
            metric="universal_transition_success_rate",
            group=GROUP_TRANSITION, direction="min",
            bound=thresholds.universal_transition_success,
            observed=_value(metrics, "universal_transition_success_rate"),
            narrow_miss_margin=margin,
        ),
        ReadinessCriterion(
            name="gate1_to_gate2_transition",
            metric="unconditional_survival.2",
            group=GROUP_TRANSITION, direction="min",
            bound=thresholds.gate1_to_gate2_transition,
            observed=(
                _nested(metrics, "unconditional_survival", 2)
                if _nested(metrics, "unconditional_survival", 2) is not None
                else _value(metrics, "gate1_to_gate2_transition_rate")
            ),
            narrow_miss_margin=margin,
        ),
    ]
    for length, bound in thresholds.completion_by_length().items():
        criteria.append(ReadinessCriterion(
            name=f"completion_{length}_gate",
            metric=f"completion_by_length.{length}",
            group=GROUP_COMPLETION, direction="min", bound=bound,
            observed=_nested(metrics, "completion_by_length", length),
            narrow_miss_margin=margin,
        ))
    criteria += [
        ReadinessCriterion(
            name="out_of_bounds_rate", metric="out_of_bounds_episode_rate",
            group=GROUP_SAFETY, direction="max",
            bound=thresholds.max_out_of_bounds_episode_rate,
            observed=_value(metrics, "out_of_bounds_episode_rate"),
            narrow_miss_margin=0.0,
        ),
        ReadinessCriterion(
            name="collision_rate", metric="collision_episode_rate",
            group=GROUP_SAFETY, direction="max",
            bound=thresholds.max_collision_episode_rate,
            observed=_value(metrics, "collision_episode_rate"),
            narrow_miss_margin=0.0,
        ),
        ReadinessCriterion(
            name="wrong_direction_rate", metric="wrong_direction_episode_rate",
            group=GROUP_SAFETY, direction="max",
            bound=thresholds.max_wrong_direction_episode_rate,
            observed=_value(metrics, "wrong_direction_episode_rate"),
            narrow_miss_margin=0.0,
        ),
        ReadinessCriterion(
            name="transition_case_count", metric="transition_cases",
            group=GROUP_EVIDENCE, direction="min",
            bound=float(thresholds.min_transition_cases),
            observed=_value(metrics, "transition_cases"),
            narrow_miss_margin=0.0,
        ),
        ReadinessCriterion(
            name="min_cases_per_length", metric="min_cases_per_length",
            group=GROUP_EVIDENCE, direction="min",
            bound=float(thresholds.min_cases_per_length),
            observed=_value(metrics, "min_cases_per_length"),
            narrow_miss_margin=0.0,
        ),
        ReadinessCriterion(
            name="gate1_to_gate2_case_count", metric="gate1_to_gate2_cases",
            group=GROUP_EVIDENCE, direction="min",
            bound=float(thresholds.min_gate1_to_gate2_cases),
            observed=_value(metrics, "gate1_to_gate2_cases"),
            narrow_miss_margin=0.0,
        ),
        # The learned controller must be measured with no expert in the loop and
        # on the right contracts.  These are booleans coerced to 0/1 so a missing
        # value fails rather than passes.
        ReadinessCriterion(
            name="no_expert_dependency", metric="inference_expert_free",
            group=GROUP_EVIDENCE, direction="min", bound=1.0,
            observed=_value(metrics, "inference_expert_free"),
            narrow_miss_margin=0.0,
        ),
        ReadinessCriterion(
            name="contract_compliance", metric="contracts_match",
            group=GROUP_EVIDENCE, direction="min", bound=1.0,
            observed=_value(metrics, "contracts_match"),
            narrow_miss_margin=0.0,
        ),
    ]
    return tuple(criteria)


def evaluate_gen2_readiness(
    metrics: Mapping[str, Any],
    thresholds: Gen2ReadinessThresholds = GEN2_READINESS_THRESHOLDS,
    *,
    override: Optional[ReadinessOverride] = None,
) -> Gen2ReadinessVerdict:
    """Judge a Gen-2 benchmark report against the pre-registered gate."""
    criteria = build_gen2_criteria(metrics, thresholds)
    by_name = {item.name: item for item in criteria}
    decisions: list = []
    granted: set = set()
    if override is not None:
        for name in override.approved_criteria:
            if name not in by_name:
                raise ValueError(
                    f"override names unknown Gen-2 criterion {name!r}; "
                    f"known criteria are {sorted(by_name)}"
                )
            criterion = by_name[name]
            refusal = _refusal(criterion)
            if refusal is None and len(granted) >= GEN2_MAX_OVERRIDDEN_CRITERIA:
                refusal = "override budget exhausted"
            if refusal is None:
                granted.add(name)
                decisions.append(OverrideDecision(name, True, "granted: narrow miss"))
            else:
                decisions.append(OverrideDecision(name, False, refusal))
    ready = all(item.satisfied or item.name in granted for item in criteria)
    return Gen2ReadinessVerdict(
        ready=ready,
        criteria=criteria,
        thresholds=thresholds,
        override=override,
        decisions=tuple(decisions),
    )


# ------------------------------------------------------------------ freeze

FREEZE_RECORD_PATH = Path("results/rl_public/gen2_frozen_policy/freeze_record.json")
PREREGISTRATION_PATH = Path("results/rl_public/gen2_readiness/preregistration.json")


def write_preregistration(root: Optional[str | Path] = None) -> Path:
    """Publish the thresholds and their hash before any Gen-2 result exists."""
    base = Path(root) if root else Path.cwd()
    target = base / PREREGISTRATION_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "gen2_readiness_preregistration_v1",
        "experiment": GEN2_EXPERIMENT_ID,
        "gate_id": GEN2_GATE_ID,
        "observation_contract": GEN2_OBS_CONTRACT,
        "action_contract": GEN2_ACTION_CONTRACT,
        "thresholds": GEN2_READINESS_THRESHOLDS.as_dict(),
        "thresholds_sha256": GEN2_READINESS_THRESHOLDS.sha256(),
        "criteria": [
            {"name": c.name, "metric": c.metric, "group": c.group,
             "direction": c.direction, "bound": c.bound}
            for c in build_gen2_criteria({})
        ],
        "non_overridable_groups": sorted(NON_OVERRIDABLE_GROUPS),
        "narrow_miss_margin": GEN2_NARROW_MISS_MARGIN,
        "max_overridden_criteria": GEN2_MAX_OVERRIDDEN_CRITERIA,
        "note": (
            "Committed before the Gen-2 holdout was opened and before the "
            "measurements existed. Editing these bars after this file is "
            "committed invalidates the Gen-2 freeze record."
        ),
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def freeze_gen2_policy(
    *,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    verdict: Gen2ReadinessVerdict,
    training_method: Mapping[str, Any],
    root: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Freeze exactly one Gen-2 policy.  Once only, by exclusive create."""
    base = Path(root) if root else Path.cwd()
    target = base / FREEZE_RECORD_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    if verdict.thresholds.sha256() != GEN2_READINESS_THRESHOLDS.sha256():
        raise ValueError(
            "the verdict was produced against different thresholds than the "
            "live pre-registration; the Gen-2 gate has been edited"
        )
    from marine_race_arena.learning.gen2.holdout_seal import _git_sha

    payload: Dict[str, Any] = {
        "schema_version": "gen2_freeze_record_v1",
        "experiment": GEN2_EXPERIMENT_ID,
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _git_sha(base),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": str(checkpoint_sha256),
        "observation_contract": GEN2_OBS_CONTRACT,
        "action_contract": GEN2_ACTION_CONTRACT,
        "readiness_thresholds_sha256": GEN2_READINESS_THRESHOLDS.sha256(),
        "readiness_verdict_sha256": verdict.sha256(),
        "readiness_ready": bool(verdict.ready),
        "training_method": dict(training_method),
    }
    payload["record_sha256"] = canonical_hash(
        {k: v for k, v in payload.items() if k != "record_sha256"}
    )
    try:
        handle = open(target, "x", encoding="utf-8")
    except FileExistsError as exc:
        raise PermissionError(
            f"a Gen-2 policy has already been frozen; see {target}"
        ) from exc
    with handle:
        handle.write(json.dumps(payload, indent=2))
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def read_freeze_record(root: Optional[str | Path] = None) -> Dict[str, Any]:
    """Read the freeze record, re-validating its hash and the live thresholds."""
    base = Path(root) if root else Path.cwd()
    record = json.loads((base / FREEZE_RECORD_PATH).read_text(encoding="utf-8"))
    recomputed = canonical_hash(
        {k: v for k, v in record.items() if k != "record_sha256"}
    )
    if record.get("record_sha256") != recomputed:
        raise PermissionError("the Gen-2 freeze record has been modified")
    if record.get("readiness_thresholds_sha256") != GEN2_READINESS_THRESHOLDS.sha256():
        raise PermissionError(
            "the Gen-2 readiness thresholds have changed since the policy was "
            "frozen; the pre-registration no longer matches the freeze record"
        )
    return record
