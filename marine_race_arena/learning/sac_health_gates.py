"""Cheap training-only health gates that detect SAC collapse early.

The v1 run needed a full 100-case unseen evaluation to discover that the actor
had saturated: by then it had already burned tens of thousands of transitions.
These gates run on learner-side statistics every few hundred updates and stop
the run at the first *sustained* symptom, long before an official evaluation.

Nothing here is ever visible to the policy: the gates read learner diagnostics
and completed-episode counters only.  Compact probes never replace the official
unseen evaluation; they only decide whether it is still worth reaching one.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

HEALTH_GATE_VERSION = "sac_health_gates_v1"


@dataclass(frozen=True)
class HealthThresholds:
    """Sustained limits; a single noisy update never stops a run."""

    max_mean_absolute_action: float = 0.30
    max_absolute_yaw_action: float = 0.40
    max_action_saturation_fraction: float = 0.20
    max_anchor_mean_drift: float = 0.10
    max_anchor_p95_drift: float = 0.25
    max_absolute_q: float = 5_000.0
    max_out_of_bounds_rate: float = 0.20
    min_probe_first_gate_rate: float = 0.60
    min_probe_target_switch_rate: float = 0.50
    consecutive_violations: int = 3
    minimum_recent_episodes: int = 20

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": HEALTH_GATE_VERSION,
            "max_mean_absolute_action": self.max_mean_absolute_action,
            "max_absolute_yaw_action": self.max_absolute_yaw_action,
            "max_action_saturation_fraction": self.max_action_saturation_fraction,
            "max_anchor_mean_drift": self.max_anchor_mean_drift,
            "max_anchor_p95_drift": self.max_anchor_p95_drift,
            "max_absolute_q": self.max_absolute_q,
            "max_out_of_bounds_rate": self.max_out_of_bounds_rate,
            "min_probe_first_gate_rate": self.min_probe_first_gate_rate,
            "min_probe_target_switch_rate": self.min_probe_target_switch_rate,
            "consecutive_violations": self.consecutive_violations,
            "minimum_recent_episodes": self.minimum_recent_episodes,
        }


DEFAULT_HEALTH_THRESHOLDS = HealthThresholds()


def thresholds_from_mapping(
    value: Optional[Mapping[str, Any]],
    base: HealthThresholds = DEFAULT_HEALTH_THRESHOLDS,
) -> HealthThresholds:
    if not value:
        return base
    known = set(base.as_dict()) - {"schema_version"}
    unknown = sorted(set(value) - known)
    if unknown:
        raise ValueError(f"unknown health threshold keys {unknown}")
    typed = {key: type(getattr(base, key))(value[key]) for key in value}
    return replace(base, **typed)


def _finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and abs(number) != float("inf")


def evaluate_health(
    metrics: Mapping[str, Any],
    thresholds: HealthThresholds = DEFAULT_HEALTH_THRESHOLDS,
    *,
    recent_episodes: int = 0,
    out_of_bounds_episodes: int = 0,
    probe: Optional[Mapping[str, Any]] = None,
) -> Tuple[bool, Sequence[str]]:
    """Return ``(violated, reasons)`` for one health sample.

    A non-finite learner value is fatal on its own; everything else must persist
    for ``consecutive_violations`` samples before the caller acts.
    """

    reasons = []
    for key in (
        "critic_loss", "actor_loss", "mean_target_q", "mean_q",
        "actor_gradient_norm", "critic_gradient_norm", "entropy_coefficient",
    ):
        if key in metrics and not _finite(metrics[key]):
            reasons.append(f"non_finite:{key}")
    if float(metrics.get("mean_absolute_action", 0.0) or 0.0) > thresholds.max_mean_absolute_action:
        reasons.append("mean_absolute_action")
    yaw = metrics.get("mean_absolute_action_yaw")
    if yaw is not None and abs(float(yaw)) > thresholds.max_absolute_yaw_action:
        reasons.append("absolute_yaw_action")
    if float(
        metrics.get("action_saturation_fraction", 0.0) or 0.0
    ) > thresholds.max_action_saturation_fraction:
        reasons.append("action_saturation_fraction")
    if float(metrics.get("anchor_mean_drift", 0.0) or 0.0) > thresholds.max_anchor_mean_drift:
        reasons.append("anchor_mean_drift")
    if float(metrics.get("anchor_p95_drift", 0.0) or 0.0) > thresholds.max_anchor_p95_drift:
        reasons.append("anchor_p95_drift")
    for key in ("mean_q", "mean_target_q", "q_p95", "target_q_p95"):
        value = metrics.get(key)
        if value is not None and _finite(value) and abs(float(value)) > thresholds.max_absolute_q:
            reasons.append(f"exploding_q:{key}")
            break
    if int(recent_episodes) >= int(thresholds.minimum_recent_episodes):
        rate = float(out_of_bounds_episodes) / max(1, int(recent_episodes))
        if rate > thresholds.max_out_of_bounds_rate:
            reasons.append("out_of_bounds_rate")
    if probe:
        if float(
            probe.get("first_gate_crossing_rate", 1.0) or 0.0
        ) < thresholds.min_probe_first_gate_rate:
            reasons.append("probe_first_gate_rate")
        if float(
            probe.get("target_switch_rate", 1.0) or 0.0
        ) < thresholds.min_probe_target_switch_rate:
            reasons.append("probe_target_switch_rate")
    return bool(reasons), tuple(reasons)


class HealthMonitor:
    """Track sustained violations across successive health samples."""

    def __init__(self, thresholds: HealthThresholds = DEFAULT_HEALTH_THRESHOLDS) -> None:
        self.thresholds = thresholds
        self.consecutive = 0
        self.last_reasons: Sequence[str] = ()
        self.history: list[Dict[str, Any]] = []

    def observe(
        self,
        metrics: Mapping[str, Any],
        *,
        recent_episodes: int = 0,
        out_of_bounds_episodes: int = 0,
        probe: Optional[Mapping[str, Any]] = None,
        updates: int = 0,
        transitions: int = 0,
    ) -> Dict[str, Any]:
        violated, reasons = evaluate_health(
            metrics, self.thresholds,
            recent_episodes=recent_episodes,
            out_of_bounds_episodes=out_of_bounds_episodes,
            probe=probe,
        )
        fatal = any(reason.startswith("non_finite:") for reason in reasons)
        self.consecutive = self.consecutive + 1 if violated else 0
        self.last_reasons = reasons
        should_pause = fatal or (
            violated and self.consecutive >= self.thresholds.consecutive_violations
        )
        record = {
            "schema_version": HEALTH_GATE_VERSION,
            "updates": int(updates),
            "total_environment_transitions": int(transitions),
            "violated": violated,
            "fatal": fatal,
            "consecutive_violations": self.consecutive,
            "reasons": list(reasons),
            "should_pause": should_pause,
            "metrics": {
                key: metrics[key] for key in (
                    "mean_absolute_action", "mean_absolute_action_yaw",
                    "action_saturation_fraction", "anchor_mean_drift",
                    "anchor_p95_drift", "anchor_coefficient", "mean_q",
                    "mean_target_q", "td_error_p95", "actor_gradient_norm",
                    "critic_gradient_norm", "entropy_coefficient",
                    "pre_tanh_jacobian_max",
                ) if key in metrics
            },
        }
        self.history.append(record)
        return record

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": HEALTH_GATE_VERSION,
            "consecutive": int(self.consecutive),
            "last_reasons": list(self.last_reasons),
            "history": list(self.history[-200:]),
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if value.get("schema_version") != HEALTH_GATE_VERSION:
            raise ValueError("unsupported SAC health monitor state")
        self.consecutive = int(value.get("consecutive", 0))
        self.last_reasons = tuple(value.get("last_reasons") or ())
        self.history = list(value.get("history") or [])
