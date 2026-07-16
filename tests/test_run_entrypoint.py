from __future__ import annotations

import json
from pathlib import Path

import run
from marine_race_arena.scripts import run_benchmark
from marine_race_arena.scripts import run_staggered_multi_rover_smoke


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


def test_smoke_scenario_maps_only_supported_runner_flags() -> None:
    config = _load("config.json")
    config["run"]["scenario"] = "smoke"
    scenario, argv = run.build_argv(config)

    assert scenario == "smoke"
    assert "--headless" not in argv
    assert "--controller" not in argv
    assert "--log-dir" not in argv
    parsed = run_staggered_multi_rover_smoke._build_arg_parser().parse_args(argv)
    assert parsed.num_rovers == 2
    assert parsed.start_gap_s == 90.0
    assert parsed.wall_timeout_s == 900.0
