"""Engine allocation, headroom rejection, dwell time and capacity stopping."""

from __future__ import annotations

import pytest

from marine_race_arena.learning.holoocean_capacity_benchmark import (
    MINIMUM_IMPROVEMENT,
    allocation_candidates,
    aggregate_resources,
    headroom_report,
    next_engine_counts,
    select_layout,
    should_stop_increasing,
)
from marine_race_arena.learning.rl_resource_governor import (
    DEFAULT_HEADROOM,
    GovernorPolicy,
    HeadroomLimits,
    ResourceGovernor,
    allocate_workers,
    evaluation_order,
    headroom_violations,
    layout_rejected,
    policy_from_mapping,
    reliability_violations,
)

PPO_OPTIONS = (2, 4, 6, 8, 10, 12, 16)
SAC_OPTIONS = (2, 4, 6, 8, 10, 12)
SUPPORTED = {"ppo": PPO_OPTIONS, "sac": SAC_OPTIONS}


# ------------------------------------------------------------------ allocation


def test_both_active_algorithms_keep_their_minimum():
    policy = GovernorPolicy(maximum_total_engines=12, minimum_workers_per_algorithm=2)
    allocation = allocate_workers(
        {"ppo": "training", "sac": "training"}, policy, supported=SUPPORTED,
    )
    assert allocation["ppo"] >= 2 and allocation["sac"] >= 2
    assert sum(allocation.values()) <= 12


def test_surplus_goes_to_the_higher_marginal_throughput():
    policy = GovernorPolicy(maximum_total_engines=12, minimum_workers_per_algorithm=2)
    ppo_first = allocate_workers(
        {"ppo": "training", "sac": "training"}, policy, supported=SUPPORTED,
        marginal_throughput={"ppo": 4.0, "sac": 1.0},
    )
    sac_first = allocate_workers(
        {"ppo": "training", "sac": "training"}, policy, supported=SUPPORTED,
        marginal_throughput={"ppo": 1.0, "sac": 4.0},
    )
    assert ppo_first["ppo"] > ppo_first["sac"]
    assert sac_first["sac"] > sac_first["ppo"]
    # The weaker competent algorithm still keeps its floor.
    assert ppo_first["sac"] >= 2 and sac_first["ppo"] >= 2


@pytest.mark.parametrize("idle", ["stopped", "paused", "evaluating", "collapsed"])
def test_idle_algorithm_lends_all_of_its_slots(idle):
    policy = GovernorPolicy(maximum_total_engines=10, minimum_workers_per_algorithm=2)
    allocation = allocate_workers(
        {"ppo": "training", "sac": idle}, policy, supported=SUPPORTED,
    )
    assert allocation["sac"] == 0
    assert allocation["ppo"] == 10


def test_allocation_never_exceeds_the_engine_ceiling():
    for ceiling in (8, 10, 12, 14, 16, 18, 20):
        policy = GovernorPolicy(maximum_total_engines=ceiling)
        allocation = allocate_workers(
            {"ppo": "training", "sac": "training"}, policy, supported=SUPPORTED,
        )
        assert sum(allocation.values()) <= ceiling


def test_allocation_snaps_to_supported_shardings():
    policy = GovernorPolicy(maximum_total_engines=11, minimum_workers_per_algorithm=2)
    allocation = allocate_workers(
        {"ppo": "training", "sac": "training"}, policy, supported=SUPPORTED,
        marginal_throughput={"ppo": 9.0},
    )
    assert allocation["ppo"] in PPO_OPTIONS
    assert allocation["sac"] in SAC_OPTIONS


def test_allocation_is_deterministic():
    policy = GovernorPolicy(maximum_total_engines=14)
    first = allocate_workers({"ppo": "training", "sac": "training"}, policy,
                             supported=SUPPORTED, marginal_throughput={"ppo": 2.0})
    second = allocate_workers({"sac": "training", "ppo": "training"}, policy,
                              supported=SUPPORTED, marginal_throughput={"ppo": 2.0})
    assert first == second


# ------------------------------------------------------------------ dwell time


def _governor(ceiling=12, dwell=1800.0):
    return ResourceGovernor(
        GovernorPolicy(maximum_total_engines=ceiling, minimum_dwell_seconds=dwell),
        supported=SUPPORTED,
    )


def test_layout_changes_only_at_an_atomic_boundary():
    governor = _governor()
    plan = governor.plan({"ppo": "training", "sac": "training"},
                         now=0.0, at_atomic_boundary=False)
    assert plan["changed"] and not plan["apply"]
    assert "not_at_atomic_boundary" in plan["blocked_by"]


def test_minimum_dwell_time_blocks_rapid_resharding():
    governor = _governor(dwell=1800.0)
    first = governor.plan({"ppo": "training", "sac": "training"},
                          now=0.0, at_atomic_boundary=True)
    assert first["apply"]
    governor.commit(first["desired"], now=0.0)
    early = governor.plan({"ppo": "training", "sac": "stopped"},
                          now=600.0, at_atomic_boundary=True)
    assert not early["apply"]
    assert "minimum_dwell_not_elapsed" in early["blocked_by"]
    assert early["dwell_remaining_seconds"] == pytest.approx(1200.0)
    late = governor.plan({"ppo": "training", "sac": "stopped"},
                         now=1801.0, at_atomic_boundary=True)
    assert late["apply"]


def test_unchanged_allocation_is_never_reapplied():
    governor = _governor()
    plan = governor.plan({"ppo": "training", "sac": "training"},
                         now=0.0, at_atomic_boundary=True)
    governor.commit(plan["desired"], now=0.0)
    again = governor.plan({"ppo": "training", "sac": "training"},
                          now=99_999.0, at_atomic_boundary=True)
    assert not again["changed"] and not again["apply"]


# ------------------------------------------------------------------- headroom


def test_each_resource_ceiling_rejects_a_layout():
    for key, value in (
        ("cpu_percent", 92.0), ("memory_percent", 88.0),
        ("gpu_percent", 97.0), ("vram_percent", 91.0),
        ("gpu_temperature_c", 87.0),
    ):
        assert headroom_violations({key: value}), key
    healthy = {"cpu_percent": 68.0, "memory_percent": 61.0, "gpu_percent": 74.0,
               "vram_percent": 55.0, "gpu_temperature_c": 66.0}
    assert headroom_violations(healthy) == ()


def test_reliability_problems_reject_a_layout():
    for key in ("engine_startup_failures", "worker_crashes", "orphan_processes",
                "stale_observations"):
        assert reliability_violations({key: 2}) == (key,)
    assert reliability_violations({"sensor_contract_valid": False}) == (
        "sensor_contract_invalid",
    )
    assert reliability_violations({"sac_update_backlog_growth": 3.0}) == (
        "sac_update_backlog_growing",
    )
    assert reliability_violations({"worker_crashes": 0}) == ()


def test_headroom_breach_blocks_applying_a_new_layout():
    governor = _governor()
    plan = governor.plan(
        {"ppo": "training", "sac": "training"}, now=0.0, at_atomic_boundary=True,
        resources={"cpu_percent": 95.0},
    )
    assert not plan["apply"] and "headroom_breach" in plan["blocked_by"]


def test_headroom_report_measures_remaining_margin():
    report = headroom_report({"cpu_percent": 66.0, "memory_percent": 60.0,
                              "gpu_percent": 72.0, "vram_percent": 51.0})
    assert report["cpu_percent"]["headroom_fraction"] == pytest.approx(0.175, abs=1e-3)
    assert report["meets_target_headroom"] is True
    tight = headroom_report({"cpu_percent": 79.0})
    assert tight["meets_target_headroom"] is False


def test_custom_limits_are_honoured():
    strict = HeadroomLimits(max_cpu_percent=50.0)
    assert headroom_violations({"cpu_percent": 60.0}, strict) == ("cpu",)
    assert headroom_violations({"cpu_percent": 60.0}, DEFAULT_HEADROOM) == ()
    with pytest.raises(ValueError):
        policy_from_mapping({"unknown_key": 1})


# ------------------------------------------------------------------- capacity


def test_capacity_stops_after_two_non_improvements():
    improving = {8: 10.0, 10: 12.0, 12: 14.0}
    assert not should_stop_increasing(improving)
    plateau = {8: 10.0, 10: 12.0, 12: 12.2, 14: 12.3}
    assert should_stop_increasing(plateau)
    assert next_engine_counts(plateau) == []
    assert next_engine_counts(improving) == [14, 16, 18, 20]


def test_capacity_requires_the_full_improvement_margin():
    just_under = {8: 10.0, 10: 10.0 * (1 + MINIMUM_IMPROVEMENT * 0.5),
                  12: 10.0 * (1 + MINIMUM_IMPROVEMENT * 0.6)}
    assert should_stop_increasing(just_under)


def test_a_slower_larger_layout_is_never_selected():
    layouts = [
        {"total_engines": 8, "ppo_workers": 4, "sac_workers": 4,
         "aggregate_valid_transitions_per_second": 18.0, "resources": {"cpu_percent": 60.0}},
        {"total_engines": 16, "ppo_workers": 8, "sac_workers": 8,
         "aggregate_valid_transitions_per_second": 15.0, "resources": {"cpu_percent": 70.0}},
    ]
    selection = select_layout(layouts)
    assert selection["selected"]["total_engines"] == 8


def test_layouts_breaching_headroom_are_rejected_outright():
    layouts = [
        {"total_engines": 20, "aggregate_valid_transitions_per_second": 40.0,
         "resources": {"cpu_percent": 95.0}},
        {"total_engines": 10, "aggregate_valid_transitions_per_second": 20.0,
         "resources": {"cpu_percent": 60.0}},
    ]
    selection = select_layout(layouts)
    assert selection["selected"]["total_engines"] == 10
    assert selection["rejected"][0]["rejected_reasons"] == ["cpu"]
    assert layout_rejected(layouts[0]) == ("cpu",)


def test_allocation_candidates_respect_the_algorithm_floor():
    pairs = allocation_candidates(12, ppo_options=PPO_OPTIONS, sac_options=SAC_OPTIONS)
    assert (4, 8) in pairs and (8, 4) in pairs and (6, 6) in pairs
    assert all(ppo >= 2 and sac >= 2 for ppo, sac in pairs)
    assert all(ppo + sac == 12 for ppo, sac in pairs)


def test_resource_aggregation_reports_mean_and_peak():
    out = aggregate_resources([
        {"cpu_percent": 50.0}, {"cpu_percent": 70.0},
    ])
    assert out["cpu_percent"] == pytest.approx(60.0)
    assert out["peak_cpu_percent"] == pytest.approx(70.0)
    assert out["samples"] == 2


# ---------------------------------------------------------------- evaluation


def test_evaluations_are_scheduled_sequentially_shortest_first():
    assert evaluation_order(["ppo", "sac"], {"ppo": 900.0, "sac": 300.0}) == ["sac", "ppo"]


def test_evaluation_lease_can_use_the_idle_algorithms_slots():
    governor = _governor(ceiling=12)
    slots = governor.evaluation_slots("ppo", {"ppo": 6, "sac": 0})
    assert slots == 12
    shared = governor.evaluation_slots("ppo", {"ppo": 6, "sac": 4})
    assert shared == 8
