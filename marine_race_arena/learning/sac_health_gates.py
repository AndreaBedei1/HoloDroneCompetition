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


@dataclass(frozen=True)
class CriticHealthThresholds:
    """Divergence limits measured against a rolling recent baseline.

    SAC v2 was only stopped after two catastrophic *evaluations*; by then the
    critics had been drifting for a long time (mean target Q -63, p05 -216,
    TD-error tail 192, critic loss 181). These limits catch that drift from the
    learner statistics alone, long before an evaluation is due.
    """

    max_q_drift_rate_per_1k_updates: float = 5.0
    max_q_spread_growth_ratio: float = 4.0
    max_td_p95_growth_ratio: float = 4.0
    max_critic_loss_growth_ratio: float = 8.0
    max_clipped_gradient_fraction: float = 0.80
    minimum_baseline_samples: int = 20
    consecutive_violations: int = 3

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": HEALTH_GATE_VERSION,
            "max_q_drift_rate_per_1k_updates": self.max_q_drift_rate_per_1k_updates,
            "max_q_spread_growth_ratio": self.max_q_spread_growth_ratio,
            "max_td_p95_growth_ratio": self.max_td_p95_growth_ratio,
            "max_critic_loss_growth_ratio": self.max_critic_loss_growth_ratio,
            "max_clipped_gradient_fraction": self.max_clipped_gradient_fraction,
            "minimum_baseline_samples": self.minimum_baseline_samples,
            "consecutive_violations": self.consecutive_violations,
        }


DEFAULT_CRITIC_HEALTH = CriticHealthThresholds()


class CriticHealthMonitor:
    """Rolling critic-divergence detector with a protective pause action."""

    def __init__(
        self,
        thresholds: CriticHealthThresholds = DEFAULT_CRITIC_HEALTH,
        *,
        window: int = 200,
        gradient_clip: float = 5.0,
    ) -> None:
        self.thresholds = thresholds
        self.window = int(window)
        self.gradient_clip = float(gradient_clip)
        self.samples: list[Dict[str, Any]] = []
        self.consecutive = 0
        self.baseline: Optional[Dict[str, float]] = None

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        values = [float(v) for v in values if _finite(v)]
        return sum(values) / len(values) if values else 0.0

    def observe(self, metrics: Mapping[str, Any], *, updates: int = 0) -> Dict[str, Any]:
        sample = {
            "updates": int(updates),
            "mean_q": float(metrics.get("mean_q", 0.0) or 0.0),
            "q_p05": float(metrics.get("q_p05", 0.0) or 0.0),
            "q_p95": float(metrics.get("q_p95", 0.0) or 0.0),
            "td_error_p95": float(metrics.get("td_error_p95", 0.0) or 0.0),
            "critic_loss": float(metrics.get("critic_loss", 0.0) or 0.0),
            "critic_gradient_norm": float(
                metrics.get("critic_gradient_norm", 0.0) or 0.0
            ),
        }
        self.samples.append(sample)
        if len(self.samples) > self.window:
            self.samples.pop(0)
        reasons: list[str] = []
        if len(self.samples) >= int(self.thresholds.minimum_baseline_samples):
            half = len(self.samples) // 2
            early, late = self.samples[:half], self.samples[half:]
            if self.baseline is None:
                self.baseline = {
                    "q_spread": abs(self._mean([s["q_p95"] for s in early])
                                    - self._mean([s["q_p05"] for s in early])),
                    "td_p95": abs(self._mean([s["td_error_p95"] for s in early])),
                    "critic_loss": abs(self._mean([s["critic_loss"] for s in early])),
                }
            span = max(1.0, late[-1]["updates"] - early[0]["updates"])
            drift = abs(self._mean([s["mean_q"] for s in late])
                        - self._mean([s["mean_q"] for s in early]))
            if drift / span * 1000.0 > self.thresholds.max_q_drift_rate_per_1k_updates:
                reasons.append("q_median_drift")
            spread = abs(self._mean([s["q_p95"] for s in late])
                         - self._mean([s["q_p05"] for s in late]))
            base_spread = max(1e-6, self.baseline["q_spread"])
            if spread / base_spread > self.thresholds.max_q_spread_growth_ratio:
                reasons.append("q_spread_expansion")
            td = abs(self._mean([s["td_error_p95"] for s in late]))
            if td / max(1e-6, self.baseline["td_p95"]) > self.thresholds.max_td_p95_growth_ratio:
                reasons.append("td_error_growth")
            loss = abs(self._mean([s["critic_loss"] for s in late]))
            if loss / max(1e-6, self.baseline["critic_loss"]) > self.thresholds.max_critic_loss_growth_ratio:
                reasons.append("critic_loss_explosion")
            clipped = sum(
                1 for s in late
                if s["critic_gradient_norm"] >= self.gradient_clip - 1e-6
            ) / max(1, len(late))
            if clipped > self.thresholds.max_clipped_gradient_fraction:
                reasons.append("critic_gradients_pinned_at_clip")
        self.consecutive = self.consecutive + 1 if reasons else 0
        should_pause = self.consecutive >= int(self.thresholds.consecutive_violations)
        return {
            "schema_version": HEALTH_GATE_VERSION,
            "updates": int(updates),
            "diverging": bool(reasons),
            "reasons": reasons,
            "consecutive_violations": self.consecutive,
            "should_pause": should_pause,
            "action": (
                # The actor is competent; only the critics are rebuilt.
                "checkpoint_freeze_actor_and_rebuild_critics"
                if should_pause else None
            ),
            "baseline": dict(self.baseline or {}),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": HEALTH_GATE_VERSION,
            "consecutive": int(self.consecutive),
            "baseline": dict(self.baseline or {}),
            "samples": list(self.samples[-self.window:]),
        }


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
