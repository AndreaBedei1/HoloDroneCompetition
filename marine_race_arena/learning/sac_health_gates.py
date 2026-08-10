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


#: Explicit lifecycle of a critic generation.  A freshly rebuilt critic starts
#: in BASELINE_BUILDING and is *never* judged until it has settled.
PHASE_BASELINE = "BASELINE_BUILDING"
PHASE_HEALTHY = "HEALTHY"
PHASE_WARNING = "WARNING"
PHASE_DIVERGING = "DIVERGING"
PHASE_RECOVERING = "RECOVERING"
CRITIC_PHASES = (
    PHASE_BASELINE, PHASE_HEALTHY, PHASE_WARNING, PHASE_DIVERGING, PHASE_RECOVERING,
)


@dataclass(frozen=True)
class CriticHealthThresholds:
    """Divergence limits measured against an *established* baseline.

    SAC v2 was only stopped after two catastrophic evaluations; by then the
    critics had drifted a long way (mean target Q -63, p05 -216, TD-error tail
    192, critic loss 181).  These limits catch that from learner statistics
    alone.  All of them are ratios against a baseline this critic generation
    actually reached, never the literal v2 numbers, so an equivalent future
    divergence at a different scale is caught just the same.

    The baseline requirements exist because SAC v3 fired at 22 updates: a fresh
    critic necessarily moves away from its random initial Q distribution while
    it learns, and reading that as divergence caused an endless rebuild loop.
    """

    max_q_drift_rate_per_1k_updates: float = 5.0
    max_q_spread_growth_ratio: float = 4.0
    max_td_p95_growth_ratio: float = 4.0
    max_critic_loss_growth_ratio: float = 8.0
    max_clipped_gradient_fraction: float = 0.80
    # A critic generation must produce this much evidence before it may be judged.
    minimum_baseline_samples: int = 200
    minimum_baseline_updates: int = 2_000
    minimum_baseline_windows: int = 2
    window_samples: int = 100
    consecutive_violations: int = 3
    recovery_windows: int = 2

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": HEALTH_GATE_VERSION,
            "max_q_drift_rate_per_1k_updates": self.max_q_drift_rate_per_1k_updates,
            "max_q_spread_growth_ratio": self.max_q_spread_growth_ratio,
            "max_td_p95_growth_ratio": self.max_td_p95_growth_ratio,
            "max_critic_loss_growth_ratio": self.max_critic_loss_growth_ratio,
            "max_clipped_gradient_fraction": self.max_clipped_gradient_fraction,
            "minimum_baseline_samples": self.minimum_baseline_samples,
            "minimum_baseline_updates": self.minimum_baseline_updates,
            "minimum_baseline_windows": self.minimum_baseline_windows,
            "window_samples": self.window_samples,
            "consecutive_violations": self.consecutive_violations,
            "recovery_windows": self.recovery_windows,
        }


DEFAULT_CRITIC_HEALTH = CriticHealthThresholds()


def critic_health_thresholds_from_mapping(
    value: Optional[Mapping[str, Any]],
    base: CriticHealthThresholds = DEFAULT_CRITIC_HEALTH,
) -> CriticHealthThresholds:
    if not value:
        return base
    known = set(base.as_dict()) - {"schema_version"}
    unknown = sorted(set(value) - known)
    if unknown:
        raise ValueError(f"unknown critic health threshold keys {unknown}")
    typed = {key: type(getattr(base, key))(value[key]) for key in value}
    return replace(base, **typed)


class CriticHealthMonitor:
    """Stateful critic-divergence detector scoped to one critic generation.

    ``generation`` increments on every rebuild.  Each generation re-enters
    BASELINE_BUILDING, so statistics are never compared across a rebuild -- the
    comparison that made SAC v3 rebuild itself forever.
    """

    def __init__(
        self,
        thresholds: CriticHealthThresholds = DEFAULT_CRITIC_HEALTH,
        *,
        window: int = 2_000,
        gradient_clip: float = 5.0,
        generation: int = 0,
    ) -> None:
        self.thresholds = thresholds
        self.window = int(window)
        self.gradient_clip = float(gradient_clip)
        self.generation = int(generation)
        self.samples: list[Dict[str, Any]] = []
        self.consecutive = 0
        self.healthy_windows = 0
        self.baseline: Optional[Dict[str, float]] = None
        self.phase = PHASE_BASELINE
        self.updates_at_generation_start: Optional[int] = None

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        values = [float(v) for v in values if _finite(v)]
        return sum(values) / len(values) if values else 0.0

    @property
    def samples_in_generation(self) -> int:
        return len(self.samples)

    @property
    def updates_in_generation(self) -> int:
        if not self.samples or self.updates_at_generation_start is None:
            return 0
        return int(self.samples[-1]["updates"]) - int(self.updates_at_generation_start)

    def is_healthy(self) -> bool:
        """Only a settled, non-drifting critic generation counts as healthy."""

        return self.phase == PHASE_HEALTHY

    def baseline_established(self) -> bool:
        return self.baseline is not None

    def begin_generation(self, generation: Optional[int] = None) -> None:
        """Restart the lifecycle after a critic rebuild or restore."""

        self.generation = (
            self.generation + 1 if generation is None else int(generation)
        )
        self.samples.clear()
        self.consecutive = 0
        self.healthy_windows = 0
        self.baseline = None
        self.phase = PHASE_BASELINE
        self.updates_at_generation_start = None

    def _baseline_ready(self) -> bool:
        t = self.thresholds
        return (
            len(self.samples) >= int(t.minimum_baseline_samples)
            and self.updates_in_generation >= int(t.minimum_baseline_updates)
            and len(self.samples) >= int(t.minimum_baseline_windows) * int(t.window_samples)
        )

    def _establish_baseline(self) -> None:
        """Freeze the baseline from the *settled tail* of the baseline phase.

        Using the earliest samples would anchor on a randomly initialised
        critic, which is exactly what made normal early learning look like
        divergence.
        """

        t = self.thresholds
        tail = self.samples[-int(t.minimum_baseline_windows) * int(t.window_samples):]
        self.baseline = {
            "mean_q": self._mean([s["mean_q"] for s in tail]),
            "q_spread": abs(self._mean([s["q_p95"] for s in tail])
                            - self._mean([s["q_p05"] for s in tail])),
            "td_p95": abs(self._mean([s["td_error_p95"] for s in tail])),
            "critic_loss": abs(self._mean([s["critic_loss"] for s in tail])),
            "updates": float(tail[-1]["updates"]),
        }

    def _window_reasons(self) -> list:
        t = self.thresholds
        window = self.samples[-int(t.window_samples):]
        base = self.baseline or {}
        reasons: list = []

        span = max(1.0, float(window[-1]["updates"]) - float(base.get("updates", 0.0)))
        drift = abs(self._mean([s["mean_q"] for s in window]) - float(base.get("mean_q", 0.0)))
        if drift / span * 1000.0 > t.max_q_drift_rate_per_1k_updates:
            reasons.append("q_median_drift")

        spread = abs(self._mean([s["q_p95"] for s in window])
                     - self._mean([s["q_p05"] for s in window]))
        if spread / max(1e-6, float(base.get("q_spread", 0.0))) > t.max_q_spread_growth_ratio:
            reasons.append("q_spread_expansion")

        td = abs(self._mean([s["td_error_p95"] for s in window]))
        if td / max(1e-6, float(base.get("td_p95", 0.0))) > t.max_td_p95_growth_ratio:
            reasons.append("td_error_growth")

        loss = abs(self._mean([s["critic_loss"] for s in window]))
        if loss / max(1e-6, float(base.get("critic_loss", 0.0))) > t.max_critic_loss_growth_ratio:
            reasons.append("critic_loss_explosion")

        clipped = sum(
            1 for s in window
            if s["critic_gradient_norm"] >= self.gradient_clip - 1e-6
        ) / max(1, len(window))
        if clipped > t.max_clipped_gradient_fraction:
            reasons.append("critic_gradients_pinned_at_clip")

        for s in window[-1:]:
            if not all(_finite(s[k]) for k in ("mean_q", "critic_loss", "td_error_p95")):
                reasons.append("non_finite_critic_statistics")
        return reasons

    # ------------------------------------------------------------- public

    def observe(self, metrics: Mapping[str, Any], *, updates: int = 0) -> Dict[str, Any]:
        sample = {
            "updates": int(updates),
            "mean_q": float(metrics.get("mean_q", 0.0) or 0.0),
            "q_p05": float(metrics.get("q_p05", 0.0) or 0.0),
            "q_p95": float(metrics.get("q_p95", 0.0) or 0.0),
            "mean_target_q": float(metrics.get("mean_target_q", 0.0) or 0.0),
            "td_error_p95": float(metrics.get("td_error_p95", 0.0) or 0.0),
            "td_error_max": float(metrics.get("td_error_max", 0.0) or 0.0),
            "critic_loss": float(metrics.get("critic_loss", 0.0) or 0.0),
            "critic_gradient_norm": float(
                metrics.get("critic_gradient_norm", 0.0) or 0.0
            ),
        }
        if self.updates_at_generation_start is None:
            self.updates_at_generation_start = int(updates)
        self.samples.append(sample)
        if len(self.samples) > self.window:
            self.samples.pop(0)

        reasons: list = []
        if self.baseline is None:
            if self._baseline_ready():
                self._establish_baseline()
                self.phase = PHASE_HEALTHY
            # While BASELINE_BUILDING nothing can trigger, by construction.
        else:
            reasons = self._window_reasons()
            if reasons:
                self.consecutive += 1
                self.healthy_windows = 0
                self.phase = (
                    PHASE_DIVERGING
                    if self.consecutive >= int(self.thresholds.consecutive_violations)
                    else PHASE_WARNING
                )
            else:
                self.consecutive = 0
                self.healthy_windows += 1
                if self.phase in (PHASE_WARNING, PHASE_DIVERGING, PHASE_RECOVERING):
                    self.phase = (
                        PHASE_HEALTHY
                        if self.healthy_windows >= int(self.thresholds.recovery_windows)
                        else PHASE_RECOVERING
                    )
                else:
                    self.phase = PHASE_HEALTHY

        should_pause = self.phase == PHASE_DIVERGING
        return {
            "schema_version": HEALTH_GATE_VERSION,
            "updates": int(updates),
            "generation": int(self.generation),
            "phase": self.phase,
            "baseline_established": self.baseline is not None,
            "samples_in_generation": len(self.samples),
            "updates_in_generation": self.updates_in_generation,
            "diverging": bool(reasons),
            "reasons": reasons,
            "consecutive_violations": self.consecutive,
            "healthy_windows": self.healthy_windows,
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
            "generation": int(self.generation),
            "phase": self.phase,
            "consecutive": int(self.consecutive),
            "healthy_windows": int(self.healthy_windows),
            "baseline": dict(self.baseline or {}),
            "updates_at_generation_start": self.updates_at_generation_start,
            "samples": list(self.samples[-self.window:]),
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if value.get("schema_version") != HEALTH_GATE_VERSION:
            raise ValueError("unsupported SAC critic health state")
        self.generation = int(value.get("generation", 0))
        self.phase = str(value.get("phase", PHASE_BASELINE))
        self.consecutive = int(value.get("consecutive", 0))
        self.healthy_windows = int(value.get("healthy_windows", 0))
        baseline = value.get("baseline") or {}
        self.baseline = {k: float(v) for k, v in baseline.items()} or None
        start = value.get("updates_at_generation_start")
        self.updates_at_generation_start = None if start is None else int(start)
        self.samples = [dict(s) for s in (value.get("samples") or [])]


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
