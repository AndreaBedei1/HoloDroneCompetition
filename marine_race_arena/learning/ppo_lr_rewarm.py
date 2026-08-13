"""KL-aware bounded learning-rate re-warm for a stalled PPO optimizer.

Measured on the live v3 run: relative policy-parameter movement of ~1.3e-5 per
2048-transition rollout, 3.98e-3 total across 645,000 transitions, ``log_std``
frozen at -2.526, and a linear schedule still decaying toward 1e-6.  The policy
is competent and non-degenerate -- this is not collapse -- but at that step size
the remaining budget cannot produce useful learning.

The controller only ever nudges the learning rate, inside hard bounds, and only
on *sustained* evidence.  It raises the rate when the optimizer is provably
doing nothing (KL far below its target, clipping inactive, parameters barely
moving) and lowers it the moment the update becomes too large.  It never
touches the reward, the architecture, the curriculum or the clip range.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Any, Dict, List, Mapping, Optional

REWARM_VERSION = "ppo_lr_rewarm_v1"

STATE_STALLED = "PPO_OPTIMIZATION_STALLED"
STATE_HEALTHY = "PPO_OPTIMIZATION_HEALTHY"
STATE_TOO_HOT = "PPO_OPTIMIZATION_TOO_HOT"


@dataclass(frozen=True)
class RewarmPolicy:
    """Bounded, KL-aware adaptation of the PPO learning rate."""

    minimum_learning_rate: float = 3.0e-6
    maximum_learning_rate: float = 1.5e-5
    #: Where re-warm starts when the optimizer is found stalled.
    initial_learning_rate: float = 7.0e-6
    #: PPO's own target_kl; "useful" updates sit within a band around it.
    target_kl: float = 0.004
    #: Below this fraction of target_kl the update is doing essentially nothing.
    stalled_kl_fraction: float = 0.10
    #: Above this fraction the update is too large and the rate must come down.
    hot_kl_fraction: float = 1.00
    #: Clipping this inactive means the ratio never leaves the trust region.
    stalled_clip_fraction: float = 0.02
    max_clip_fraction: float = 0.30
    #: Relative ||dtheta||/||theta|| per rollout below which movement is nil.
    stalled_relative_update: float = 5.0e-5
    #: Consecutive rollouts of evidence before the rate moves at all.
    sustained_rollouts: int = 5
    increase_factor: float = 1.25
    decrease_factor: float = 0.7

    def as_dict(self) -> Dict[str, Any]:
        return {"schema_version": REWARM_VERSION, **asdict(self)}


DEFAULT_REWARM = RewarmPolicy()


def rewarm_policy_from_mapping(
    value: Optional[Mapping[str, Any]], base: RewarmPolicy = DEFAULT_REWARM
) -> RewarmPolicy:
    if not value:
        return base
    known = set(asdict(base))
    unknown = sorted(set(value) - known)
    if unknown:
        raise ValueError(f"unknown rewarm keys {unknown}")
    typed = {key: type(getattr(base, key))(value[key]) for key in value}
    return replace(base, **typed)


def classify_update(
    sample: Mapping[str, Any], policy: RewarmPolicy = DEFAULT_REWARM
) -> str:
    """Label one rollout's optimizer activity.

    TOO_HOT wins over STALLED: an update can show a large KL while clipping is
    still low, and shrinking the step is always the safe response.
    """

    kl = float(sample.get("approx_kl", 0.0) or 0.0)
    clip = float(sample.get("clip_fraction", 0.0) or 0.0)
    move = float(sample.get("relative_policy_update", 0.0) or 0.0)
    if kl > policy.hot_kl_fraction * policy.target_kl or clip > policy.max_clip_fraction:
        return STATE_TOO_HOT
    stalled = (
        kl < policy.stalled_kl_fraction * policy.target_kl
        and clip < policy.stalled_clip_fraction
        and move < policy.stalled_relative_update
    )
    return STATE_STALLED if stalled else STATE_HEALTHY


class LearningRateRewarm:
    """Track sustained optimizer activity and adapt the rate within bounds."""

    def __init__(
        self,
        policy: RewarmPolicy = DEFAULT_REWARM,
        *,
        learning_rate: Optional[float] = None,
    ) -> None:
        self.policy = policy
        self.learning_rate = float(
            policy.initial_learning_rate if learning_rate is None else learning_rate
        )
        self.consecutive_stalled = 0
        self.consecutive_hot = 0
        self.state = STATE_HEALTHY
        self.adjustments: List[Dict[str, Any]] = []

    def _clamp(self, value: float) -> float:
        return max(
            float(self.policy.minimum_learning_rate),
            min(float(self.policy.maximum_learning_rate), float(value)),
        )

    def observe(self, sample: Mapping[str, Any], *, timesteps: int = 0) -> Dict[str, Any]:
        policy = self.policy
        self.state = classify_update(sample, policy)
        changed = False
        before = self.learning_rate

        if self.state == STATE_TOO_HOT:
            self.consecutive_hot += 1
            self.consecutive_stalled = 0
            # React immediately to an oversized update: protecting the competent
            # policy matters more than a smooth schedule.
            self.learning_rate = self._clamp(self.learning_rate * policy.decrease_factor)
            changed = self.learning_rate != before
        elif self.state == STATE_STALLED:
            self.consecutive_stalled += 1
            self.consecutive_hot = 0
            if self.consecutive_stalled >= int(policy.sustained_rollouts):
                self.learning_rate = self._clamp(
                    self.learning_rate * policy.increase_factor
                )
                changed = self.learning_rate != before
                self.consecutive_stalled = 0
        else:
            self.consecutive_stalled = 0
            self.consecutive_hot = 0

        record = {
            "schema_version": REWARM_VERSION,
            "timesteps": int(timesteps),
            "state": self.state,
            "learning_rate_before": before,
            "learning_rate": self.learning_rate,
            "changed": changed,
            "consecutive_stalled": self.consecutive_stalled,
            "consecutive_hot": self.consecutive_hot,
            "approx_kl": float(sample.get("approx_kl", 0.0) or 0.0),
            "clip_fraction": float(sample.get("clip_fraction", 0.0) or 0.0),
            "relative_policy_update": float(
                sample.get("relative_policy_update", 0.0) or 0.0
            ),
        }
        if changed:
            self.adjustments.append(record)
        return record

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": REWARM_VERSION,
            "learning_rate": float(self.learning_rate),
            "state": self.state,
            "consecutive_stalled": int(self.consecutive_stalled),
            "consecutive_hot": int(self.consecutive_hot),
            "policy": self.policy.as_dict(),
            "adjustments": list(self.adjustments[-100:]),
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if value.get("schema_version") != REWARM_VERSION:
            raise ValueError("unsupported PPO rewarm state")
        self.learning_rate = float(value.get("learning_rate", self.learning_rate))
        self.state = str(value.get("state", STATE_HEALTHY))
        self.consecutive_stalled = int(value.get("consecutive_stalled", 0))
        self.consecutive_hot = int(value.get("consecutive_hot", 0))
        self.adjustments = list(value.get("adjustments") or [])
        policy = dict(value.get("policy") or {})
        policy.pop("schema_version", None)
        if policy:
            self.policy = rewarm_policy_from_mapping(policy)
