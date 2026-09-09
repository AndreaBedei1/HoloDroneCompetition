"""Global HoloOcean engine allocation across PPO, SAC and evaluators.

The governor owns one number that the whole system must respect: the maximum
number of simultaneously active Unreal engines that the machine sustains with
headroom.  Everything else -- rollout workers per algorithm, evaluator workers,
lending slots between algorithms -- is derived from it.

Two rules keep this from degenerating into thrash:

* a layout change is only ever applied at an explicit atomic rollout boundary,
  and never within ``minimum_dwell_seconds`` of the previous change;
* an algorithm that is training keeps at least ``minimum_workers_per_algorithm``
  engines, so lending can never starve a live learner.

The goal is maximum *aggregate valid environment transitions per second*, not
maximum engine count: a layout that raises engine count while lowering measured
throughput is rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

GOVERNOR_VERSION = "rl_resource_governor_v1"

# States in which an algorithm is not collecting rollouts and its engine slots
# may be lent to the other algorithm at the next safe boundary.
IDLE_STATES = frozenset({
    "stopped", "paused", "evaluating", "checkpointing", "collapsed",
    "completed", "failed", "not_started",
})


@dataclass(frozen=True)
class HeadroomLimits:
    """Sustained resource ceilings; exceeding any one rejects a layout."""

    max_cpu_percent: float = 80.0
    max_memory_percent: float = 80.0
    max_gpu_percent: float = 90.0
    max_vram_percent: float = 85.0
    max_gpu_temperature_c: float = 80.0
    target_headroom_fraction: float = 0.175

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": GOVERNOR_VERSION,
            "max_cpu_percent": self.max_cpu_percent,
            "max_memory_percent": self.max_memory_percent,
            "max_gpu_percent": self.max_gpu_percent,
            "max_vram_percent": self.max_vram_percent,
            "max_gpu_temperature_c": self.max_gpu_temperature_c,
            "target_headroom_fraction": self.target_headroom_fraction,
        }


DEFAULT_HEADROOM = HeadroomLimits()


def headroom_violations(
    resources: Mapping[str, Any], limits: HeadroomLimits = DEFAULT_HEADROOM
) -> Tuple[str, ...]:
    """Sustained resource ceilings a sample breaches (empty when acceptable)."""

    checks = (
        ("cpu_percent", limits.max_cpu_percent, "cpu"),
        ("memory_percent", limits.max_memory_percent, "system_ram"),
        ("gpu_percent", limits.max_gpu_percent, "gpu"),
        ("vram_percent", limits.max_vram_percent, "vram"),
        ("gpu_temperature_c", limits.max_gpu_temperature_c, "gpu_temperature"),
    )
    out = []
    for key, ceiling, label in checks:
        value = resources.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number > float(ceiling):
            out.append(label)
    return tuple(out)


def reliability_violations(sample: Mapping[str, Any]) -> Tuple[str, ...]:
    """Non-resource reasons a layout must be rejected."""

    out = []
    if int(sample.get("engine_startup_failures", 0) or 0) > 0:
        out.append("engine_startup_failures")
    if int(sample.get("worker_crashes", 0) or 0) > 0:
        out.append("worker_crashes")
    if int(sample.get("orphan_processes", 0) or 0) > 0:
        out.append("orphan_processes")
    if int(sample.get("stale_observations", 0) or 0) > 0:
        out.append("stale_observations")
    if sample.get("sensor_contract_valid") is False:
        out.append("sensor_contract_invalid")
    if sample.get("checkpoint_write_failures"):
        out.append("checkpoint_write_failures")
    if float(sample.get("sac_update_backlog_growth", 0.0) or 0.0) > 0.0:
        out.append("sac_update_backlog_growing")
    return tuple(out)


def layout_rejected(
    sample: Mapping[str, Any], limits: HeadroomLimits = DEFAULT_HEADROOM
) -> Tuple[str, ...]:
    return headroom_violations(
        dict(sample.get("resources") or {}), limits
    ) + reliability_violations(sample)


@dataclass(frozen=True)
class GovernorPolicy:
    maximum_total_engines: int = 10
    minimum_workers_per_algorithm: int = 2
    minimum_dwell_seconds: float = 1800.0
    evaluator_workers: int = 6

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": GOVERNOR_VERSION,
            "maximum_total_engines": self.maximum_total_engines,
            "minimum_workers_per_algorithm": self.minimum_workers_per_algorithm,
            "minimum_dwell_seconds": self.minimum_dwell_seconds,
            "evaluator_workers": self.evaluator_workers,
        }


def policy_from_mapping(
    value: Optional[Mapping[str, Any]], base: GovernorPolicy = GovernorPolicy()
) -> GovernorPolicy:
    if not value:
        return base
    known = set(base.as_dict()) - {"schema_version"}
    unknown = sorted(set(value) - known)
    if unknown:
        raise ValueError(f"unknown governor policy keys {unknown}")
    return replace(base, **{key: type(getattr(base, key))(value[key]) for key in value})


def allocate_workers(
    algorithm_states: Mapping[str, str],
    policy: GovernorPolicy,
    *,
    supported: Mapping[str, Sequence[int]],
    marginal_throughput: Optional[Mapping[str, float]] = None,
) -> Dict[str, int]:
    """Split the engine budget between algorithms.

    Active algorithms are first guaranteed their floor; the remainder is offered
    to the algorithm with the higher measured marginal throughput.  Idle
    algorithms lend their whole share and receive zero engines.
    """

    names = sorted(algorithm_states)
    active = [name for name in names if algorithm_states[name] not in IDLE_STATES]
    allocation = {name: 0 for name in names}
    if not active:
        return allocation

    budget = int(policy.maximum_total_engines)
    floor = int(policy.minimum_workers_per_algorithm)
    if len(active) * floor > budget:
        # Not enough engines to honour every floor: give whole floors in a
        # deterministic order rather than starving everyone equally.
        for name in active:
            grant = min(floor, budget)
            allocation[name] = grant
            budget -= grant
        return {name: _snap(allocation[name], supported.get(name, ())) for name in names}

    for name in active:
        allocation[name] = floor
        budget -= floor

    weights = dict(marginal_throughput or {})
    order = sorted(active, key=lambda name: (-float(weights.get(name, 0.0)), name))
    # Offer the surplus to the strongest algorithm first, snapping to a
    # sharding it can actually run, then pass the remainder on.
    for name in order:
        if budget <= 0:
            break
        options = sorted(supported.get(name, ()))
        if not options:
            allocation[name] += budget
            budget = 0
            break
        best = allocation[name]
        for option in options:
            if allocation[name] < option <= allocation[name] + budget:
                best = option
        budget -= best - allocation[name]
        allocation[name] = best
    return {name: _snap(allocation[name], supported.get(name, ())) for name in names}


def _snap(count: int, options: Sequence[int]) -> int:
    """Round a raw slot count down to a sharding the algorithm supports."""

    if not options or count <= 0:
        return int(count)
    feasible = [value for value in sorted(options) if value <= count]
    return int(feasible[-1]) if feasible else int(min(options))


class ResourceGovernor:
    """Stateful allocator enforcing the dwell time and the engine ceiling."""

    def __init__(
        self,
        policy: GovernorPolicy,
        *,
        supported: Mapping[str, Sequence[int]],
        limits: HeadroomLimits = DEFAULT_HEADROOM,
    ) -> None:
        self.policy = policy
        self.supported = {key: tuple(value) for key, value in supported.items()}
        self.limits = limits
        self.current: Dict[str, int] = {}
        self.last_change_time: Optional[float] = None
        self.history: list[Dict[str, Any]] = []

    def plan(
        self,
        algorithm_states: Mapping[str, str],
        *,
        now: float,
        at_atomic_boundary: bool,
        marginal_throughput: Optional[Mapping[str, float]] = None,
        resources: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        desired = allocate_workers(
            algorithm_states, self.policy,
            supported=self.supported, marginal_throughput=marginal_throughput,
        )
        total = sum(desired.values())
        if total > self.policy.maximum_total_engines:
            raise ValueError(
                f"allocation {desired} exceeds the engine ceiling "
                f"{self.policy.maximum_total_engines}"
            )
        breaches = headroom_violations(dict(resources or {}), self.limits)
        dwell_remaining = 0.0
        if self.last_change_time is not None:
            dwell_remaining = max(
                0.0,
                float(self.policy.minimum_dwell_seconds) - (now - self.last_change_time),
            )
        blocked = []
        if not at_atomic_boundary:
            blocked.append("not_at_atomic_boundary")
        if dwell_remaining > 0.0:
            blocked.append("minimum_dwell_not_elapsed")
        if breaches:
            blocked.append("headroom_breach")
        changed = desired != self.current
        apply = changed and not blocked
        result = {
            "schema_version": GOVERNOR_VERSION,
            "current": dict(self.current),
            "desired": desired,
            "total_engines": total,
            "maximum_total_engines": self.policy.maximum_total_engines,
            "changed": changed,
            "apply": apply,
            "blocked_by": blocked,
            "dwell_remaining_seconds": dwell_remaining,
            "headroom_breaches": list(breaches),
        }
        self.history.append(result)
        return result

    def commit(self, allocation: Mapping[str, int], *, now: float) -> None:
        self.current = dict(allocation)
        self.last_change_time = float(now)

    def evaluation_slots(self, algorithm: str, allocation: Mapping[str, int]) -> int:
        """Engines available to evaluators while ``algorithm`` is evaluating."""

        others = sum(
            count for name, count in allocation.items() if name != algorithm
        )
        return max(0, int(self.policy.maximum_total_engines) - int(others))


def evaluation_order(pending: Sequence[str], durations: Mapping[str, float]) -> list:
    """Sequential evaluation order: shortest first minimises total wall clock.

    Running PPO and SAC evaluations concurrently halves the engines available to
    each; when the machine is engine-bound, sequential scheduling finishes both
    sooner, so the governor never grants two evaluation leases at once.
    """

    return sorted(pending, key=lambda name: (float(durations.get(name, 0.0)), name))
