"""Explicit actor freeze/unfreeze state, scoped to a critic generation.

SAC v3 froze the actor on its first critic rebuild and never unfroze it, so
362,496 environment transitions trained critics only and every reported score
was the preserved v2 actor re-measured.  The defect was arithmetic: the unfreeze
threshold was computed as ``agent.gradient_updates + warmup`` *before* the
rebuild, but ``rebuild_critics_from_actor`` returns a fresh agent whose
``gradient_updates`` restarts at zero.  13,548 + 10,000 = 23,548 could never be
reached by a counter that had just gone back to 0.

The fix is to stop deriving anything from the absolute counter.  This object
tracks ``critic_updates_since_rebuild``, which resets to zero exactly when a new
critic generation begins, and the actor may only wake when that generation has
both trained enough *and* been declared healthy by the critic monitor.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Any, Dict, Mapping, Optional

ACTOR_GATE_VERSION = "sac_actor_gate_v1"

STATE_FROZEN = "FROZEN"
STATE_ACTIVE = "ACTIVE"
STATE_LIVELOCK = "CRITIC_WARMUP_LIVELOCK"


class CriticWarmupLivelock(RuntimeError):
    """Critic warm-up keeps restarting, so the actor can never activate.

    SAC v5 rebuilt its critics seven times, every time at exactly
    gradient_updates=4004, and spent 471,108 transitions with actor_updates=0.
    The individual rebuilds each looked locally reasonable, so nothing surfaced
    a fault.  This makes never-activating warm-up an explicit, loud condition
    rather than an invisible loop.
    """ 


@dataclass(frozen=True)
class ActorGatePolicy:
    """How much healthy critic training the actor must wait for."""

    #: Critic updates that must accumulate *within the current generation*.
    actor_unfreeze_min_updates: int = 10_000
    #: The critic monitor must also report a settled, non-drifting generation.
    require_critic_healthy: bool = True
    #: Start frozen on a brand-new run so the actor never chases random critics.
    start_frozen: bool = True
    #: Critic rebuilds tolerated *before the actor has ever activated* before
    #: declaring a livelock.  This is a safety trip, never a shortcut: it pauses
    #: the run for diagnosis rather than activating onto unhealthy critics.
    max_critic_rebuilds_before_actor_activation: int = 3

    def as_dict(self) -> Dict[str, Any]:
        return {"schema_version": ACTOR_GATE_VERSION, **asdict(self)}


DEFAULT_ACTOR_GATE = ActorGatePolicy()


def actor_gate_policy_from_mapping(
    value: Optional[Mapping[str, Any]],
    base: ActorGatePolicy = DEFAULT_ACTOR_GATE,
) -> ActorGatePolicy:
    if not value:
        return base
    known = set(asdict(base))
    unknown = sorted(set(value) - known)
    if unknown:
        raise ValueError(f"unknown actor gate keys {unknown}")
    typed = {key: type(getattr(base, key))(value[key]) for key in value}
    return replace(base, **typed)


class ActorFreezeGate:
    """Decide, per update, whether the actor is allowed to learn.

    Nothing here consults an absolute optimizer step count.  The only counter
    that matters is ``critic_updates_since_rebuild``, which is reset by
    :meth:`on_critic_rebuild` and by nothing else.
    """

    def __init__(
        self,
        policy: ActorGatePolicy = DEFAULT_ACTOR_GATE,
        *,
        critic_rebuild_generation: int = 0,
    ) -> None:
        self.policy = policy
        self.critic_rebuild_generation = int(critic_rebuild_generation)
        self.critic_updates_since_rebuild = 0
        self.actor_frozen = bool(policy.start_frozen)
        self.total_actor_updates = 0
        self.rebuilds_before_activation = 0
        self.ever_activated = not bool(policy.start_frozen)
        self.unfroze_at_generation: Optional[int] = None
        self.unfroze_at_updates_since_rebuild: Optional[int] = None
        self.last_reason = "initialised_frozen" if policy.start_frozen else "initialised_active"

    # ---------------------------------------------------------- lifecycle

    def on_critic_rebuild(self) -> Dict[str, Any]:
        """A new critic generation begins: freeze and restart the counter."""

        self.critic_rebuild_generation += 1
        self.critic_updates_since_rebuild = 0
        self.actor_frozen = True
        if not self.ever_activated:
            # Only rebuilds that happen before the actor has *ever* trained
            # count toward the livelock trip; ordinary recovery afterwards is
            # healthy behaviour.
            self.rebuilds_before_activation += 1
        self.unfroze_at_generation = None
        self.unfroze_at_updates_since_rebuild = None
        self.last_reason = "frozen_after_critic_rebuild"
        return self.describe()

    def livelock_detected(self) -> bool:
        """Has critic warm-up restarted too many times without ever activating?"""

        limit = int(self.policy.max_critic_rebuilds_before_actor_activation)
        return (
            not self.ever_activated
            and limit > 0
            and self.rebuilds_before_activation >= limit
        )

    def assert_not_livelocked(self) -> None:
        if self.livelock_detected():
            raise CriticWarmupLivelock(
                f"{STATE_LIVELOCK}: critic warm-up restarted "
                f"{self.rebuilds_before_activation} times without the actor "
                "ever activating; pausing for diagnosis rather than training "
                "critics indefinitely"
            )

    def record_critic_update(self, count: int = 1) -> None:
        self.critic_updates_since_rebuild += int(count)

    # ------------------------------------------------------------ decision

    def enough_critic_training(self) -> bool:
        return (
            self.critic_updates_since_rebuild
            >= int(self.policy.actor_unfreeze_min_updates)
        )

    def consider_unfreeze(self, *, critic_healthy: bool) -> bool:
        """Unfreeze only on sufficient *and* healthy critic training."""

        if not self.actor_frozen:
            return False
        if not self.enough_critic_training():
            self.last_reason = (
                f"waiting_for_critic_updates "
                f"{self.critic_updates_since_rebuild}/"
                f"{self.policy.actor_unfreeze_min_updates}"
            )
            return False
        if self.policy.require_critic_healthy and not critic_healthy:
            self.last_reason = "waiting_for_critic_health"
            return False
        self.actor_frozen = False
        self.ever_activated = True
        self.unfroze_at_generation = self.critic_rebuild_generation
        self.unfroze_at_updates_since_rebuild = self.critic_updates_since_rebuild
        self.last_reason = "unfrozen_after_healthy_critic_warmup"
        return True

    def should_update_actor(self) -> bool:
        return not self.actor_frozen

    def record_actor_update(self, count: int = 1) -> None:
        self.total_actor_updates += int(count)

    @property
    def state(self) -> str:
        if self.livelock_detected():
            return STATE_LIVELOCK
        return STATE_FROZEN if self.actor_frozen else STATE_ACTIVE

    def describe(self) -> Dict[str, Any]:
        return {
            "schema_version": ACTOR_GATE_VERSION,
            "state": self.state,
            "actor_frozen": bool(self.actor_frozen),
            "critic_rebuild_generation": int(self.critic_rebuild_generation),
            "critic_updates_since_rebuild": int(self.critic_updates_since_rebuild),
            "actor_unfreeze_min_updates": int(self.policy.actor_unfreeze_min_updates),
            "total_actor_updates": int(self.total_actor_updates),
            "rebuilds_before_activation": int(self.rebuilds_before_activation),
            "ever_activated": bool(self.ever_activated),
            "livelock_detected": self.livelock_detected(),
            "unfroze_at_generation": self.unfroze_at_generation,
            "unfroze_at_updates_since_rebuild": self.unfroze_at_updates_since_rebuild,
            "reason": self.last_reason,
        }

    # ---------------------------------------------------------- persistence

    def state_dict(self) -> Dict[str, Any]:
        return {
            **self.describe(),
            "policy": self.policy.as_dict(),
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if value.get("schema_version") != ACTOR_GATE_VERSION:
            raise ValueError("unsupported SAC actor gate state")
        self.critic_rebuild_generation = int(value.get("critic_rebuild_generation", 0))
        self.critic_updates_since_rebuild = int(
            value.get("critic_updates_since_rebuild", 0)
        )
        self.actor_frozen = bool(value.get("actor_frozen", True))
        self.total_actor_updates = int(value.get("total_actor_updates", 0))
        self.rebuilds_before_activation = int(value.get("rebuilds_before_activation", 0))
        self.ever_activated = bool(value.get("ever_activated", False))
        self.unfroze_at_generation = value.get("unfroze_at_generation")
        self.unfroze_at_updates_since_rebuild = value.get(
            "unfroze_at_updates_since_rebuild"
        )
        self.last_reason = str(value.get("reason", ""))
        policy = dict(value.get("policy") or {})
        policy.pop("schema_version", None)
        if policy:
            self.policy = actor_gate_policy_from_mapping(policy)
