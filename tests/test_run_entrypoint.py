from __future__ import annotations

import json
from pathlib import Path

import pytest

import run
from marine_race_arena.scripts import run_benchmark
from marine_race_arena.scripts import run_marine_race


ROOT = Path(__file__).resolve().parents[1]


def _load(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_benchmark_config_maps_only_supported_runner_flags() -> None:
    scenario, argv = run.build_argv(_load("configs/benchmark.json"))

    assert scenario == "benchmark"
    assert "--log-dir" not in argv
    parsed = run_benchmark._build_arg_parser().parse_args(argv)
    assert parsed.seeds == [0, 1, 2]
    assert parsed.output_dir == "results/benchmarks/config_benchmark"


def test_default_config_maps_only_supported_runner_flags() -> None:
    scenario, argv = run.build_argv(_load("config.json"))

    assert scenario == "single"
    parsed = run_marine_race._build_arg_parser().parse_args(argv)
    assert parsed.controller == "rule_gate_baseline"
    assert parsed.official is True
    assert parsed.current_profile == "none"


def test_fleet_config_maps_only_supported_runner_flags() -> None:
    scenario, argv = run.build_argv(_load("configs/fleet.json"))

    assert scenario == "fleet"
    parsed = run_marine_race._build_arg_parser().parse_args(argv)
    assert parsed.num_rovers == 2


def test_unknown_scenario_is_rejected() -> None:
    config = _load("config.json")
    config["run"]["scenario"] = "smoke"
    with pytest.raises(SystemExit):
        run.build_argv(config)
