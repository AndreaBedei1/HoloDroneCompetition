import pytest

# The RL stack (gymnasium/torch/SB3) lives in requirements-rl.txt and is not
# installed in the benchmark environment; skip rather than fail collection.
pytest.importorskip("gymnasium")

from marine_race_arena.learning import benchmark_evaluation_parallelism
from marine_race_arena.learning import benchmark_sac_parallelism
from marine_race_arena.learning import rl_evaluation_scheduler as scheduler


def test_dynamic_evaluator_allocation_respects_measured_choice_and_free_slots():
    result = scheduler.select_evaluation_workers(
        {
            "dynamic_parallelism": {
                "enabled": True,
                "candidates": [4, 6, 8],
                "selected_workers": 6,
            }
        },
        fixed_workers=1,
        snapshot={"available": 5, "limit": 10},
    )
    assert result["requested_workers"] == 6
    assert result["selected_workers"] == 5
    assert result["degraded_for_current_capacity"] is True


def test_fixed_evaluator_allocation_is_backward_compatible():
    result = scheduler.select_evaluation_workers(
        {}, fixed_workers=2, snapshot={"available": 8, "limit": 10}
    )
    assert result["selected_workers"] == 2
    assert result["selection_source"] == "fixed_config"


def test_evaluator_selection_requires_ten_percent_gain():
    rows = [
        {"workers": 4, "stable": True, "cases_per_second": 1.0},
        {"workers": 6, "stable": True, "cases_per_second": 1.09},
        {"workers": 8, "stable": True, "cases_per_second": 1.11},
    ]
    assert benchmark_evaluation_parallelism.select_evaluator_count(rows) == 8


def test_training_layout_uses_aggregate_not_ppo_slowdown():
    rows = [
        {"workers": 1, "stable": True, "aggregate_transitions_per_second": 10.0},
        {
            "workers": 2,
            "stable": True,
            "aggregate_transitions_per_second": 11.2,
            "ppo_throughput_degradation_fraction": 0.50,
        },
    ]
    assert benchmark_sac_parallelism._select(rows) == 2


def test_capacity_wait_times_out_if_ppo_is_not_in_rollout(monkeypatch):
    monkeypatch.setattr(
        benchmark_sac_parallelism, "ppo_is_in_normal_rollout", lambda run: False
    )
    monkeypatch.setattr(
        benchmark_sac_parallelism.time, "monotonic", lambda: 10.0
    )
    with pytest.raises(TimeoutError):
        benchmark_sac_parallelism.wait_for_ppo_normal_rollout(
            "run", timeout_seconds=0.0
        )
