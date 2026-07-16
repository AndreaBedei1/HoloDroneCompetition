#!/usr/bin/env python3
"""Single entry point for Marine Race Arena.

Run an entire simulation from one configuration file::

    python run.py                 # uses ./config.json
    python run.py my_config.json  # uses an explicit config file
    python run.py --config configs/fleet.json

The configuration file is plain JSON and fully describes the run. The
``run.scenario`` field selects what to launch:

    single     one rover on a track (the default official benchmark run)
    fleet      several staggered rovers scored as one team
    benchmark  repeated single-rover trials over several seeds
    smoke      the staggered multi-rover release smoke test

Each scenario maps only the supported subset of the configuration onto its
underlying runner, so no command-line flags are needed. See ``config.json`` for
a documented example and ``configs/`` for ready-made fleet and benchmark
configurations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG = "config.json"
SCENARIOS = ("single", "fleet", "benchmark", "smoke")


def _load_config(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(
            f"Configuration file not found: {path}\n"
            f"Create one (see {DEFAULT_CONFIG}) or pass a path: python run.py <config.json>"
        )
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise SystemExit(f"Top-level config in {path} must be a JSON object.")
    return config


def _flag(argv: list[str], name: str, value, *, store_true: bool = False) -> None:
    """Append a CLI flag to ``argv`` when the config supplied a value."""
    if store_true:
        if value:
            argv.append(name)
    elif value is not None:
        argv.extend([name, str(value)])


def _controller_flags(argv: list[str], controller: dict) -> None:
    module_or_file = controller.get("module_or_file")
    if module_or_file:
        argv.extend(["--participant-controller", str(module_or_file)])
        _flag(argv, "--controller-class", controller.get("class"))
    else:
        _flag(argv, "--controller", controller.get("name"))


def _common_race_flags(argv: list[str], config: dict) -> None:
    run = config.get("run", {})
    obstacles = config.get("obstacles", {})
    currents = config.get("currents", {})
    output = config.get("output", {})

    track = config.get("track")
    if not track:
        raise SystemExit("Config is missing the required 'track' field (path to a track JSON).")
    argv.extend(["--track", str(track)])

    _controller_flags(argv, config.get("controller", {}))
    _flag(argv, "--adapter", run.get("adapter"))
    _flag(argv, "--allow-fallback", run.get("allow_fallback"), store_true=True)
    _flag(argv, "--official", run.get("official"), store_true=True)
    _flag(argv, "--seed", run.get("seed"))
    _flag(argv, "--dt", run.get("dt"))
    _flag(argv, "--duration", run.get("duration_s"))
    _flag(argv, "--motion-compensation", run.get("motion_compensation"))
    _flag(argv, "--gate-timeout-s", run.get("gate_timeout_s"))
    _flag(argv, "--benchmark-task", config.get("benchmark_task"))
    _flag(argv, "--obstacles", obstacles.get("mode"))
    _flag(argv, "--obstacle-density", obstacles.get("density"))
    _flag(argv, "--obstacle-physics", obstacles.get("physics"))
    _flag(argv, "--current-profile", currents.get("profile"))
    _flag(argv, "--log-dir", output.get("log_dir"))


def _fleet_flags(argv: list[str], config: dict) -> None:
    fleet = config.get("fleet", {})
    ivc = fleet.get("inter_vehicle_collision", {})
    argv.append("--staggered-start")
    _flag(argv, "--num-rovers", fleet.get("num_rovers"))
    _flag(argv, "--start-gap-s", fleet.get("start_gap_s"))
    _flag(argv, "--staggered-lateral-offset-m", fleet.get("lateral_offset_m"))
    _flag(argv, "--team-id", fleet.get("team_id"))
    _flag(argv, "--inter-vehicle-collision-mode", ivc.get("mode"))
    _flag(argv, "--inter-vehicle-collision-xy-threshold-m", ivc.get("xy_threshold_m"))
    _flag(argv, "--inter-vehicle-collision-z-threshold-m", ivc.get("z_threshold_m"))
    _flag(argv, "--inter-vehicle-collision-release-threshold-m", ivc.get("release_threshold_m"))
    _flag(argv, "--inter-vehicle-collision-cooldown-s", ivc.get("cooldown_s"))
    comms = fleet.get("comms", {})
    _flag(argv, "--comms-enabled", comms.get("enabled"), store_true=True)
    _flag(argv, "--comms-sound-speed-m-s", comms.get("sound_speed_m_s"))
    _flag(argv, "--comms-max-range-m", comms.get("max_range_m"))
    _flag(argv, "--comms-processing-delay-s", comms.get("processing_delay_s"))
    _flag(argv, "--comms-packet-loss-prob", comms.get("packet_loss_prob"))
    _flag(argv, "--comms-max-payload-bytes", comms.get("max_payload_bytes"))
    _flag(argv, "--comms-min-send-interval-s", comms.get("min_send_interval_s"))


def _benchmark_flags(argv: list[str], config: dict) -> None:
    """Map only options accepted by ``run_benchmark``."""
    run = config.get("run", {})
    obstacles = config.get("obstacles", {})
    currents = config.get("currents", {})
    benchmark = config.get("benchmark", {})
    controller = config.get("controller", {})

    track = config.get("track")
    if not track:
        raise SystemExit("Config is missing the required 'track' field (path to a track JSON).")
    argv.extend(["--track", str(track)])

    controller_value = controller.get("module_or_file") or controller.get("name")
    if not controller_value:
        raise SystemExit("Benchmark config requires controller.name or controller.module_or_file.")
    argv.extend(["--controller", str(controller_value)])
    _flag(argv, "--controller-class", controller.get("class"))
    _flag(argv, "--benchmark-task", config.get("benchmark_task"))
    _flag(argv, "--adapter", run.get("adapter"))
    _flag(argv, "--allow-fallback", run.get("allow_fallback"), store_true=True)
    _flag(argv, "--official", run.get("official"), store_true=True)
    _flag(argv, "--dt", run.get("dt"))
    _flag(argv, "--duration", run.get("duration_s"))
    _flag(argv, "--motion-compensation", run.get("motion_compensation"))
    _flag(argv, "--gate-timeout-s", run.get("gate_timeout_s"))
    _flag(argv, "--obstacles", obstacles.get("mode"))
    _flag(argv, "--obstacle-density", obstacles.get("density"))
    _flag(argv, "--obstacle-physics", obstacles.get("physics"))
    _flag(argv, "--current-profile", currents.get("profile"))

    seeds = benchmark.get("seeds")
    if seeds:
        argv.append("--seeds")
        argv.extend(str(seed) for seed in seeds)
    _flag(argv, "--output-dir", benchmark.get("output_dir"))


def _smoke_flags(argv: list[str], config: dict) -> None:
    """Map only options accepted by ``run_staggered_multi_rover_smoke``."""
    run = config.get("run", {})
    fleet = config.get("fleet", {})
    ivc = fleet.get("inter_vehicle_collision", {})
    smoke = config.get("smoke", {})

    track = config.get("track")
    if not track:
        raise SystemExit("Config is missing the required 'track' field (path to a track JSON).")
    argv.extend(["--track", str(track)])
    _flag(argv, "--num-rovers", fleet.get("num_rovers"))
    _flag(argv, "--start-gap-s", fleet.get("start_gap_s"))
    _flag(argv, "--staggered-lateral-offset-m", fleet.get("lateral_offset_m"))
    _flag(argv, "--dt", run.get("dt"))
    _flag(argv, "--seed", run.get("seed"))
    _flag(argv, "--duration", run.get("duration_s"))
    _flag(argv, "--inter-vehicle-collision-mode", ivc.get("mode"))
    _flag(argv, "--inter-vehicle-collision-xy-threshold-m", ivc.get("xy_threshold_m"))
    _flag(argv, "--inter-vehicle-collision-z-threshold-m", ivc.get("z_threshold_m"))
    _flag(argv, "--inter-vehicle-collision-cooldown-s", ivc.get("cooldown_s"))
    _flag(argv, "--wall-timeout-s", smoke.get("wall_timeout_s"))
    _flag(argv, "--output-dir", smoke.get("output_dir"))


def build_argv(config: dict) -> tuple[str, list[str]]:
    """Return ``(scenario, argv)`` for the runner selected by the config."""
    run = config.get("run", {})
    scenario = run.get("scenario", "single")
    if scenario not in SCENARIOS:
        raise SystemExit(f"Unknown run.scenario '{scenario}'. Choose one of {SCENARIOS}.")

    argv: list[str] = []
    debug = config.get("debug", {})
    sensors = config.get("sensors", {})
    output = config.get("output", {})

    if scenario in ("single", "fleet"):
        _flag(argv, "--headless", run.get("headless"), store_true=True)
        _flag(argv, "--record", run.get("record"), store_true=True)
        _common_race_flags(argv, config)
        _flag(argv, "--disable-front-camera", sensors.get("disable_front_camera"), store_true=True)
        _flag(argv, "--show-front-camera", debug.get("show_front_camera"), store_true=True)
        _flag(argv, "--print-beacons", debug.get("print_beacons"), store_true=True)
        _flag(argv, "--log-participant-states", output.get("log_participant_states"), store_true=True)
        if scenario == "fleet":
            _fleet_flags(argv, config)
        return scenario, argv

    if scenario == "benchmark":
        _benchmark_flags(argv, config)
        _flag(argv, "--print-beacons", debug.get("print_beacons"), store_true=True)
        _flag(argv, "--log-participant-states", output.get("log_participant_states"), store_true=True)
        return scenario, argv

    # smoke
    _smoke_flags(argv, config)
    return scenario, argv


def dispatch(scenario: str, argv: list[str]) -> int:
    if scenario in ("single", "fleet"):
        from marine_race_arena.scripts import run_marine_race as runner
    elif scenario == "benchmark":
        from marine_race_arena.scripts import run_benchmark as runner
    else:
        from marine_race_arena.scripts import run_staggered_multi_rover_smoke as runner
    return runner.main(argv)


def main(cli_argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Marine Race Arena from a single config file.")
    parser.add_argument(
        "config",
        nargs="?",
        default=DEFAULT_CONFIG,
        help=f"Path to the JSON configuration file (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument("--config", dest="config_opt", default=None, help="Alternative way to pass the config path.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved scenario and arguments, then exit.")
    args = parser.parse_args(cli_argv)

    config_path = Path(args.config_opt or args.config)
    config = _load_config(config_path)
    scenario, argv = build_argv(config)

    print(f"[run] config: {config_path}")
    print(f"[run] scenario: {scenario}")
    print(f"[run] -> {scenario_command(scenario)} {' '.join(argv)}")
    if args.dry_run:
        return 0
    return dispatch(scenario, argv)


def scenario_command(scenario: str) -> str:
    return {
        "single": "run_marine_race",
        "fleet": "run_marine_race",
        "benchmark": "run_benchmark",
        "smoke": "run_staggered_multi_rover_smoke",
    }[scenario]


if __name__ == "__main__":
    raise SystemExit(main())
