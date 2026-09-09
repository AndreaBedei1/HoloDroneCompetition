"""Detect a degenerate inactive PPO policy from rollout statistics.

The lost continuation spent ~51k transitions on a policy whose deterministic
output was ~0 on every axis.  Nothing noticed until an unseen evaluation
returned 0% success, because training reward alone never said "the drone is not
moving".  This detector reads rollout-side behaviour instead, and is deliberately
independent of the reward: the reward contract is frozen and is *not* adjusted to
bribe the policy into moving.

It is a safety net, not the cure.  The actual cause of that run was a
continuation that silently started from random weights; see
``assert_policy_matches_parent``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Any, Dict, Mapping, Optional, Sequence

ACTIVITY_GATE_VERSION = "ppo_activity_gate_v1"

STATUS_HEALTHY = "HEALTHY"
STATUS_WARNING = "WARNING"
STATUS_INACTIVE = "INACTIVE"


@dataclass(frozen=True)
class ActivityThresholds:
    """Limits mirroring the competence gate that already rejects such policies."""

    min_mean_absolute_action: float = 0.02
    min_nontrivial_action_fraction: float = 0.25
    min_mean_distance_travelled_m: float = 1.0
    min_first_gate_reach_fraction: float = 0.20
    min_completed_gate_rate: float = 0.05
    #: Rollout windows a symptom must persist for; one bad rollout is noise.
    warning_windows: int = 2
    inactive_windows: int = 4
    minimum_episodes: int = 10

    def as_dict(self) -> Dict[str, Any]:
        return {"schema_version": ACTIVITY_GATE_VERSION, **asdict(self)}


DEFAULT_ACTIVITY = ActivityThresholds()


def activity_thresholds_from_mapping(
    value: Optional[Mapping[str, Any]],
    base: ActivityThresholds = DEFAULT_ACTIVITY,
) -> ActivityThresholds:
    if not value:
        return base
    known = set(asdict(base))
    unknown = sorted(set(value) - known)
    if unknown:
        raise ValueError(f"unknown activity threshold keys {unknown}")
    typed = {key: type(getattr(base, key))(value[key]) for key in value}
    return replace(base, **typed)


def evaluate_activity(
    window: Mapping[str, Any],
    thresholds: ActivityThresholds = DEFAULT_ACTIVITY,
) -> Sequence[str]:
    """Reasons this rollout window looks like an inactive policy."""

    reasons = []
    if float(window.get("mean_absolute_action", 1.0)) < thresholds.min_mean_absolute_action:
        reasons.append("mean_absolute_action")
    if float(
        window.get("nontrivial_action_fraction", 1.0)
    ) < thresholds.min_nontrivial_action_fraction:
        reasons.append("nontrivial_action_fraction")
    return tuple(reasons)


def evaluate_task_failure(
    window: Mapping[str, Any],
    thresholds: ActivityThresholds = DEFAULT_ACTIVITY,
) -> Sequence[str]:
    """Corroborating evidence that the inactivity is actually hurting the task."""

    reasons = []
    if float(
        window.get("mean_distance_travelled_m", 1e9)
    ) < thresholds.min_mean_distance_travelled_m:
        reasons.append("mean_distance_travelled")
    if float(
        window.get("first_gate_reach_fraction", 1.0)
    ) < thresholds.min_first_gate_reach_fraction:
        reasons.append("first_gate_reach_fraction")
    if float(window.get("completed_gate_rate", 1.0)) < thresholds.min_completed_gate_rate:
        reasons.append("completed_gate_rate")
    return tuple(reasons)


class ActivityMonitor:
    """Escalate WARNING -> INACTIVE only on sustained, corroborated evidence."""

    def __init__(self, thresholds: ActivityThresholds = DEFAULT_ACTIVITY) -> None:
        self.thresholds = thresholds
        self.consecutive_quiet = 0
        self.consecutive_failing = 0
        self.status = STATUS_HEALTHY
        self.history: list = []

    def observe(self, window: Mapping[str, Any], *, timesteps: int = 0) -> Dict[str, Any]:
        episodes = int(window.get("episodes", 0) or 0)
        if episodes < int(self.thresholds.minimum_episodes):
            # Too little evidence to judge; never escalate on a partial window.
            record = {
                "schema_version": ACTIVITY_GATE_VERSION,
                "timesteps": int(timesteps),
                "status": self.status,
                "reasons": [],
                "task_reasons": [],
                "consecutive_quiet": self.consecutive_quiet,
                "should_rollback": False,
                "note": "insufficient_episodes",
            }
            self.history.append(record)
            return record

        quiet = evaluate_activity(window, self.thresholds)
        failing = evaluate_task_failure(window, self.thresholds)
        self.consecutive_quiet = self.consecutive_quiet + 1 if quiet else 0
        # Inactivity only counts as collapse when the task is demonstrably
        # suffering too: a policy can legitimately be gentle for a while.
        corroborated = bool(quiet) and bool(failing)
        self.consecutive_failing = self.consecutive_failing + 1 if corroborated else 0

        if self.consecutive_failing >= int(self.thresholds.inactive_windows):
            self.status = STATUS_INACTIVE
        elif self.consecutive_quiet >= int(self.thresholds.warning_windows):
            self.status = STATUS_WARNING
        else:
            self.status = STATUS_HEALTHY

        record = {
            "schema_version": ACTIVITY_GATE_VERSION,
            "timesteps": int(timesteps),
            "status": self.status,
            "reasons": list(quiet),
            "task_reasons": list(failing),
            "consecutive_quiet": self.consecutive_quiet,
            "consecutive_failing": self.consecutive_failing,
            "should_rollback": self.status == STATUS_INACTIVE,
            "action": (
                "checkpoint_and_restore_latest_competent"
                if self.status == STATUS_INACTIVE else None
            ),
            "window": {
                key: window.get(key) for key in (
                    "mean_absolute_action", "nontrivial_action_fraction",
                    "mean_distance_travelled_m", "first_gate_reach_fraction",
                    "completed_gate_rate", "mean_absolute_action_surge",
                    "mean_absolute_action_yaw", "episodes",
                ) if key in window
            },
        }
        self.history.append(record)
        return record

    def reset_after_rollback(self) -> None:
        self.consecutive_quiet = 0
        self.consecutive_failing = 0
        self.status = STATUS_HEALTHY

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": ACTIVITY_GATE_VERSION,
            "status": self.status,
            "consecutive_quiet": int(self.consecutive_quiet),
            "consecutive_failing": int(self.consecutive_failing),
            "history": list(self.history[-200:]),
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if value.get("schema_version") != ACTIVITY_GATE_VERSION:
            raise ValueError("unsupported PPO activity gate state")
        self.status = str(value.get("status", STATUS_HEALTHY))
        self.consecutive_quiet = int(value.get("consecutive_quiet", 0))
        self.consecutive_failing = int(value.get("consecutive_failing", 0))
        self.history = list(value.get("history") or [])
