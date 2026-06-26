# Marine Race Arena

Marine Race Arena is a HoloOcean-based underwater drone racing benchmark for BlueROV2-style vehicles. It is inspired by underwater gate and corridor racing, and it is designed to evaluate controllers against configurable tracks, gates, sensors, currents, obstacles, participants, referee rules, and scoring.

The v0.1 release scope is deliberately narrow and reproducible:

- Official single-rover clean-gate benchmark.
- HoloOcean BlueROV2 integration plus a simulator-independent fallback adapter for tests and plumbing.
- Three official track JSON files with official 1.5 m x 1.5 m gate openings.
- `rule_gate_baseline`, a deterministic beacon plus `FrontCamera` baseline controller.
- Custom controller loading for users and students.
- Staggered fleet/team evaluation for multiple rovers belonging to one team.
- `team_summary` aggregation for fleet runs.
- Inter-vehicle collision diagnostics with an optional penalty mode.

The following are available but not claimed as solved v0.1 results: current compensation, DVL PI compensation, close-proximity multi-rover racing, fully calibrated rover-rover collision penalties, and full empirical inter-vehicle collision calibration.

## 1. Project Overview

Marine Race Arena separates the race definition, simulator adapter, controller observation, and referee. Controllers decide commands from observations; the referee validates progress and scoring from simulator state.

Modes used in this repository:

| Mode | Meaning | Intended use |
| --- | --- | --- |
| Official benchmark mode | Uses official tracks, official gate size, official sensor filtering, and reproducible scoring. Run with `--official` and an explicit `--benchmark-task`. | Main v0.1 single-rover clean-gate benchmark. |
| Diagnostic mode | Adds instrumentation or controlled smoke tests without changing official tracks or rules. | Adapter checks, staggered fleet smoke, collision analysis, calibration attempts. |
| Experimental mode | Components useful for research but not release claims. | DVL PI compensation, current stress tests, obstacle/current variants, close-proximity fleet experiments. |

## 2. Repository Structure

```text
marine_race_arena/
  adapters/        HoloOcean and fallback simulator adapters
  arena/           gates, bounds, beacon manager, currents, obstacles
  config/          dataclasses, JSON loader, validation, benchmark task modes
  controllers/     built-in controllers, official baselines, manual controllers, student template
  participants/    participant model, sensor filtering, controller loader/interface
  referee/         gate validation, race state, scoring, event logger, team summary
  scripts/         runnable CLIs for race execution, validation, smoke tests, diagnostics
  tracks/          official tracks and smaller validation/tuning tracks
tests/             simulator-independent pytest coverage
results/           generated outputs; normally ignored and not committed
diagnostics/       generated diagnostic images/logs; normally ignored and not committed
docs/              release notes and longer project notes
```

`create_best_tracks.py` is a track-authoring helper kept for reproducibility of the current track family. It is not part of the normal benchmark execution path.

## 3. Installation / Environment

The expected local environment is:

- A conda environment named `ocean`.
- HoloOcean installed in that environment.
- Python able to import this repository from the checkout root.
- `pytest` available in the environment.

This repository does not currently provide a packaged environment file. Use the existing local `ocean` environment assumptions and run commands from the repository root.

## 4. Quick Validation

Run these from the repository root:

```bash
python -m compileall -q marine_race_arena tests
conda run -n ocean python -m pytest -q
conda run -n ocean python -m marine_race_arena.scripts.run_staggered_multi_rover_smoke
```

The smoke test runs the stable two-rover fleet/team demonstration on Horseshoe Bay with no currents, no obstacles, no motion compensation, and diagnostic inter-vehicle collision detection.

To print the same release-check commands without running them:

```bash
python -m marine_race_arena.scripts.run_release_v0_1_checks
```

To run them sequentially:

```bash
python -m marine_race_arena.scripts.run_release_v0_1_checks --run
```

## 5. Single-Rover Official Clean Benchmark

POSIX-style line continuation:

```bash
conda run -n ocean python -m marine_race_arena.scripts.run_marine_race \
  --track marine_race_arena/tracks/marine_race_horseshoe_bay.json \
  --benchmark-task clean_gate \
  --controller rule_gate_baseline \
  --adapter holoocean \
  --official \
  --headless \
  --seed 0 \
  --dt 0.033 \
  --duration 560 \
  --obstacles none \
  --current-profile none \
  --motion-compensation none \
  --log-dir results/benchmarks/single_rover_clean_manual_run
```

Windows caret-continuation form:

```bat
conda run -n ocean python -m marine_race_arena.scripts.run_marine_race ^
  --track marine_race_arena/tracks/marine_race_horseshoe_bay.json ^
  --benchmark-task clean_gate ^
  --controller rule_gate_baseline ^
  --adapter holoocean ^
  --official ^
  --headless ^
  --seed 0 ^
  --dt 0.033 ^
  --duration 560 ^
  --obstacles none ^
  --current-profile none ^
  --motion-compensation none ^
  --log-dir results/benchmarks/single_rover_clean_manual_run
```

PowerShell users can also paste the command as one line; native PowerShell line continuation uses backticks rather than `^`.

## 6. Stable Fleet/Team Demo

This run evaluates two BlueROV2 agents as one fleet/team. Rovers are released 90 seconds apart and laterally offset by 3 m to avoid close physical interaction. Per-rover rows are diagnostics; `team_summary` is the fleet-level result.

```bash
conda run -n ocean python -m marine_race_arena.scripts.run_marine_race \
  --track marine_race_arena/tracks/marine_race_horseshoe_bay.json \
  --benchmark-task clean_gate \
  --controller rule_gate_baseline \
  --adapter holoocean \
  --official \
  --headless \
  --seed 0 \
  --dt 0.033 \
  --duration 560 \
  --obstacles none \
  --current-profile none \
  --motion-compensation none \
  --staggered-start \
  --num-rovers 2 \
  --start-gap-s 90.0 \
  --staggered-lateral-offset-m 3.0 \
  --inter-vehicle-collision-mode diagnostic \
  --inter-vehicle-collision-xy-threshold-m 0.8 \
  --inter-vehicle-collision-z-threshold-m 0.75 \
  --inter-vehicle-collision-cooldown-s 1.0 \
  --team-id fleet_01 \
  --log-participant-states \
  --log-dir results/benchmarks/staggered_multi_rover_manual_run
```

Shortcut:

```bash
conda run -n ocean python -m marine_race_arena.scripts.run_staggered_multi_rover_smoke
```

## 7. Command-Line Parameters

| Parameter | What it does | Typical values | Scope |
| --- | --- | --- | --- |
| `--track` | Path to a track JSON. | `marine_race_arena/tracks/marine_race_horseshoe_bay.json` | Official |
| `--benchmark-task` | Validates/runs the track under a task mode. | `clean_gate`, `obstacle_gate`, `current_gate`, `multi_rov` | Official/experimental by task |
| `--controller` | Built-in controller alias. Overrides the participant config. | `rule_gate_baseline`, `student_template`, `keyboard`, `pygame`, `oracle` | Official if non-debug |
| `--participant-controller` | External controller module, `module:Class`, fully qualified class, or `.py` file. | `path/to/my_controller.py`, `my_pkg.ctrl:MyController` | Official if controller follows rules |
| `--controller-class` | Class name for a module or file-path controller. | `MyController` | Official if controller follows rules |
| `--adapter` | Simulator adapter selection. | `holoocean`, `fallback`, `auto` | `holoocean` official; `fallback` diagnostic |
| `--official` | Forces official sensor/timing mode and blocks debug-only controllers. | flag | Official |
| `--headless` | Requests HoloOcean without viewport. | flag | Official/diagnostic |
| `--seed` | Seed for deterministic beacons/adapters/generation. | `0` | Official |
| `--dt` | Race-loop step size in seconds. | `0.033` for HoloOcean | Official |
| `--duration` | Maximum run duration in seconds. | Horseshoe `560`, Vertical `900`, Mixed `1300` | Official |
| `--obstacles` | Runtime obstacle mode. | `none`, `fixed`, `random` | `none` official clean; others experimental |
| `--current-profile` | Runtime current profile override. | `none`, `medium`, `strong` | `none` official clean; currents experimental in v0.1 |
| `--motion-compensation` | Optional command compensation layer. | `none`, `dvl_pi` | `none` official; `dvl_pi` experimental |
| `--staggered-start` | Clones the base participant into multiple delayed rovers. | flag | Diagnostic fleet/team |
| `--num-rovers` | Number of generated staggered participants. | `2` for stable demo | Diagnostic fleet/team |
| `--start-gap-s` | Release delay between generated rovers. | `90.0` stable, `20.0` diagnostic stress | Diagnostic |
| `--staggered-lateral-offset-m` | Lateral spawn spacing around the base start. | `3.0` stable | Diagnostic |
| `--inter-vehicle-collision-mode` | Referee-side rover-rover proximity detector. | `off`, `diagnostic`, `penalize` | Diagnostic; `penalize` experimental |
| `--inter-vehicle-collision-xy-threshold-m` | Horizontal threshold for inter-vehicle proximity events. | `0.8` | Diagnostic/experimental |
| `--inter-vehicle-collision-z-threshold-m` | Vertical threshold for inter-vehicle proximity events. | `0.75` | Diagnostic/experimental |
| `--inter-vehicle-collision-cooldown-s` | Per-pair cooldown between counted events. | `1.0` | Diagnostic/experimental |
| `--team-id` | Team identifier in fleet summaries and events. | `fleet_01` | Diagnostic fleet/team |
| `--log-participant-states` | Logs per-tick participant positions/status for analysis. | flag | Diagnostic |
| `--log-dir` | Output directory for summary JSON and event JSONL. | `results/benchmarks/...` | Official/diagnostic |

Additional useful diagnostic options include `--print-beacon-targets`, `--show-front-camera`, `--allow-fallback`, and `--gate-timeout-s`.

## 8. Tracks / Circuits

All three official tracks use `gate_inner_size_m = [1.5, 1.5]` and `timing_mode = first_gate_to_last_gate`.

| Track | File | Gates | Laps | Total gates | Declared length | Purpose | Difficulty |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Marine Race Horseshoe Bay | `marine_race_arena/tracks/marine_race_horseshoe_bay.json` | 12 | 1 | 12 | 93.8 m | Primary clean-gate route and stable fleet demo. | Medium, mostly clean-gate/horseshoe. |
| Marine Race Vertical Serpent | `marine_race_arena/tracks/marine_race_vertical_serpent.json` | 17 | 1 | 17 | 118.3 m | Vertical and slalom-style gate sequencing. | Higher, serpent/slalom. |
| Marine Race Mixed Endurance | `marine_race_arena/tracks/marine_race_mixed_endurance.json` | 22 | 1 | 22 | 206.3 m | Longer endurance route with configured current profiles. | Highest, endurance/current-oriented. |

Recommended clean commands:

```bash
conda run -n ocean python -m marine_race_arena.scripts.run_marine_race --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --benchmark-task clean_gate --controller rule_gate_baseline --adapter holoocean --official --headless --seed 0 --dt 0.033 --duration 560 --obstacles none --current-profile none --motion-compensation none --log-dir results/benchmarks/horseshoe_clean_manual
conda run -n ocean python -m marine_race_arena.scripts.run_marine_race --track marine_race_arena/tracks/marine_race_vertical_serpent.json --benchmark-task clean_gate --controller rule_gate_baseline --adapter holoocean --official --headless --seed 0 --dt 0.033 --duration 900 --obstacles none --current-profile none --motion-compensation none --log-dir results/benchmarks/vertical_clean_manual
conda run -n ocean python -m marine_race_arena.scripts.run_marine_race --track marine_race_arena/tracks/marine_race_mixed_endurance.json --benchmark-task clean_gate --controller rule_gate_baseline --adapter holoocean --official --headless --seed 0 --dt 0.033 --duration 1300 --obstacles none --current-profile none --motion-compensation none --log-dir results/benchmarks/mixed_clean_manual
```

Mixed Endurance also contains `medium` and `strong` current profiles, but current robustness is experimental in v0.1.

## 9. Configuration System

Track JSON files are loaded into dataclasses from `marine_race_arena/config/schema.py` and semantically validated before execution.

Main sections:

- `race`: race name, format, laps, expected gates per lap, timing mode, max duration, official-mode default.
- `world`: HoloOcean package/map names, arena origin, preferred/fallback environments, world bounds.
- `start`: default spawn position and rotation.
- `finish`: final gate id.
- `track`: declared length, gate-size defaults, gate-bar dimensions, depth defaults, gate sequence.
- `gates`: each gate id, type, position, rotation, inner size, bar thickness, color, passage direction, optional linked gate, optional beacon override.
- `beacon`: default acoustic beacon settings for gates.
- `currents`: active current fields.
- `current_profiles`: named current presets such as `none`, `medium`, and `strong`.
- `obstacle_generation`: random obstacle mode, density, clearance, seed, and physics settings.
- `obstacles`: fixed static obstacle definitions.
- `participants`: vehicle type, controller, controller class, spawn, sensor profile, control mode, `start_delay_s`.
- `referee`: gate validation settings, penalties, and ranking/scoring options.
- `benchmark_task`: intended task mode for validation and benchmark reporting.

Practical edits:

- Change gate positions in `gates[*].position`.
- Change the required order in `track.gate_sequence`.
- Change lap count in `race.laps`; expected total gates are `laps * len(gate_sequence)`.
- Select current behavior by defining `currents` or named `current_profiles`, then run with `--current-profile`.
- Select obstacles with fixed `obstacles` or generated obstacles via `obstacle_generation` and `--obstacles`.
- Set participant spawn in `participants[*].spawn.position` and `rotation_rpy_deg`.
- Add staggered release with `participants[*].spawn.start_delay_s` or the `--staggered-start` runner options.
- Configure sensors through `participants[*].sensors.allowed_sensors` and `holoocean_sensors`.
- Configure controller loading with `participants[*].controller` and `controller_class`, or override from the CLI.
- Adjust penalties under `referee.penalties`; adjust stuck/collision cooldowns under `referee.gate_validation`.

Do not enlarge official gate sizes or change official track geometry when reporting v0.1 official benchmark results.

## 10. Controller Architecture

A controller object only needs:

- `reset(self, race_info)`
- `step(self, observation) -> command dict`
- `close(self)`

Minimal example:

```python
class MyController:
    def reset(self, race_info):
        self.target_gate = race_info.get("initial_target_gate_id")

    def step(self, observation):
        sensors = observation.get("sensors", {})
        race = observation.get("race", {})
        beacon = observation.get("beacon", {})

        return {
            "surge": 0.3,
            "sway": 0.0,
            "heave": 0.0,
            "yaw": 0.0,
        }

    def close(self):
        pass
```

Command fields:

- `surge`: forward/back command in the vehicle body frame.
- `sway`: lateral body-frame command.
- `heave`: vertical command. Negative goes deeper, positive moves toward the surface.
- `yaw`: yaw-rate command.

High-level commands are clamped to `[-1.0, 1.0]` by the adapter. The BlueROV2 HoloOcean adapter maps high-level commands to 8 thruster values. Controllers should not directly access HoloOcean, referee internals, other rover positions, or ground truth outside the provided observation.

## 11. Controller Observations

`build_observation()` returns:

```python
{
    "participant_id": "...",
    "time_s": 0.0,
    "sensors": {...},
    "beacon": {...},
    "race": {...},
}
```

`debug_ground_truth` is only included for debug controllers in non-official mode.

`observation["race"]` contains:

- `status`
- `lap`
- `laps`
- `completed_gates`
- `target_gate_id`
- `target_sequence_index`
- `official_time_started`

`observation["beacon"]` contains fields such as:

- `valid`
- `reason`
- `active_beacon_id`
- `target_gate_id`
- `sequence_index`
- `bearing_deg`
- `elevation_deg`
- `range_m`
- `signal_strength`
- `noise_level`
- `mode`
- `message`

In non-official oracle mode, beacon observations can include exact gate/beacon fields; those are not available in official mode.

Official sensor profiles can include:

- `FrontCamera`: RGB/BGRA image buffer from the front camera.
- `DepthSensor` and derived `depth_m`.
- `IMUSensor`.
- `DVLSensor` and `VelocitySensor` when configured.
- `CollisionSensor`.
- `heading_yaw_deg`.
- `current_physical_coupling_active`, `current_coupling_method`, and `control_mode` metadata.
- `environment_current_m_s` may appear as current metadata in local configs, but it should not be used for no-cheat official controller claims unless the benchmark explicitly allows direct current-vector input.

Ground-truth sensors such as `PoseSensor`, `LocationSensor`, `RotationSensor`, and `DynamicsSensor` are filtered out in official mode.

Practical examples:

```python
image = observation["sensors"].get("FrontCamera")
beacon = observation["beacon"]
race = observation["race"]

if beacon.get("valid") and beacon.get("bearing_deg") is not None:
    yaw = max(-0.2, min(0.2, beacon["bearing_deg"] / 90.0))
else:
    yaw = 0.0

command = {
    "surge": 0.25 if race.get("status") == "RUNNING" else 0.0,
    "sway": 0.0,
    "heave": 0.0,
    "yaw": yaw,
}
```

## 12. Loading a Custom Controller

Built-in aliases include:

- `rule_gate_baseline`
- `acoustic_baseline`
- `student_template`
- `keyboard`, `manual`, `manual_keyboard`
- `pygame`, `pygame_keyboard`
- `oracle` for debug only, blocked in official mode

External controller syntaxes supported by `ControllerLoader`:

- Python file path ending in `.py`; requires `--controller-class`.
- `module:Class`.
- Fully qualified `module.Class`.
- Module plus `--controller-class`.

Example file-path command:

```bash
conda run -n ocean python -m marine_race_arena.scripts.run_marine_race \
  --track marine_race_arena/tracks/marine_race_horseshoe_bay.json \
  --benchmark-task clean_gate \
  --participant-controller path/to/my_controller.py \
  --controller-class MyController \
  --adapter holoocean \
  --official \
  --headless \
  --obstacles none \
  --current-profile none \
  --motion-compensation none
```

Example module/class command:

```bash
conda run -n ocean python -m marine_race_arena.scripts.run_marine_race --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --benchmark-task clean_gate --participant-controller my_package.my_controller:MyController --adapter holoocean --official --headless --obstacles none --current-profile none --motion-compensation none
```

## 13. Architecture Diagram

```text
Track JSON
  |
  v
Config Loader
  |
  v
ArenaBuilder ----> Gates / Bounds / Obstacles / Currents / Beacons
  |
  v
Race Runner
  |
  +--> Adapter: HoloOcean or fallback
  |
  +--> Controller observation
  |       |
  |       v
  |   User Controller
  |       |
  |       v
  |   Command: surge / sway / heave / yaw
  |
  +--> Referee
          |
          +--> gate validation
          +--> collisions
          +--> stuck / out-of-bounds
          +--> penalties
          +--> per-rover summary
          +--> team_summary for fleet mode
```

## 14. Referee And Arbitration

The referee is not part of the controller observation. It uses simulator state to validate and score the race.

Referee logic:

- Gate validation checks that the vehicle center crosses the expected gate plane, in the expected direction, through the aperture.
- `wrong_direction` is logged when the expected gate is crossed backwards.
- `missed_gate` is logged when a different gate is crossed before the expected gate; by default this causes DNF.
- Collision events add the configured collision penalty subject to cooldown.
- Obstacle collisions add obstacle-specific penalties when obstacles are active.
- Out-of-bounds events add penalties subject to cooldown.
- Stuck detection accumulates time below a movement threshold and adds stuck penalties.
- Optional gate-timeout stuck handling can mark a participant stuck in experiment runners.
- Ranking sorts finished participants by penalized time. Unfinished participants are ranked by completed gates, collisions, penalties, distance to next gate, and participant id.

Fleet/team mode:

- Each rover still has its own referee state, progress, events, and diagnostic row.
- All rovers belong to one team.
- `team_summary` aggregates gates, collisions, penalties, and team elapsed time.
- `team_summary` is the official fleet-level result.
- `participants` rows are diagnostics.

Inter-vehicle collision modes:

- `off`: disabled; preserves old behavior.
- `diagnostic`: detects and logs rover-rover proximity events without penalty. Recommended default for v0.1.
- `penalize`: adds one team-level penalty per inter-vehicle collision event.

One inter-vehicle event counts once at team level, not once per rover. Per-rover `involved_inter_vehicle_collisions` is diagnostic only. Detection uses referee-side geometric proximity from participant positions; controllers do not receive other rover positions.

Recommended v0.1 default: `--inter-vehicle-collision-mode diagnostic` until full calibration succeeds.

## 15. Calibration Script

Quick calibration:

```bash
conda run -n ocean python -m marine_race_arena.scripts.calibrate_inter_vehicle_collision_threshold --quick
```

Full calibration:

```bash
conda run -n ocean python -m marine_race_arena.scripts.calibrate_inter_vehicle_collision_threshold
```

The calibration script tries to estimate a geometric BlueROV2-vs-BlueROV2 contact threshold in HoloOcean by spawning two BlueROV2 agents with zero commands and sweeping relative placements/yaw angles. It writes:

- `results/benchmarks/inter_vehicle_collision_calibration/calibration_samples.csv`
- `results/benchmarks/inter_vehicle_collision_calibration/calibration_summary.json`
- `results/benchmarks/inter_vehicle_collision_calibration/calibration_report.md`

Current limitation: quick calibration may fail or timeout while repeatedly loading HoloOcean scenarios on some machines. The current conservative defaults are:

- `xy threshold = 0.8 m`
- `z threshold = 0.75 m`
- `release threshold = 1.05 m`
- `cooldown = 1.0 s`

Do not claim these thresholds are fully empirically validated until the full calibration completes with enough contact and non-contact samples.

## 16. Results And Outputs

Race outputs are written under `--log-dir`:

- Summary JSON, usually `*_summary.json`.
- Event JSONL, one event per line.
- Optional stdout/stderr files from wrapper scripts.
- CSV tables and markdown reports from benchmark/smoke scripts.

Useful fields/events:

- `participants`: per-rover diagnostic summaries.
- `ranking`: per-rover diagnostic order.
- `team_summary`: fleet-level score for multi-rover runs.
- `inter_vehicle_collision`: diagnostic or penalized rover-rover proximity event.
- `participant_released`: staggered release event.
- `participant_state`: per-tick diagnostic state when `--log-participant-states` is enabled.

Generated `results/` and `diagnostics/` outputs are ignored by Git by default. Do not commit large logs, videos, JSONL files, HoloOcean recordings, or generated cache directories.

## 17. Known Limitations

- Inter-vehicle collision calibration is incomplete.
- `inter_vehicle_collision_mode=diagnostic` is recommended for now.
- Close-proximity fleet racing is not fully validated.
- `inter_vehicle_collision_mode=penalize` is available but should be treated as experimental until calibration is complete.
- Current compensation is experimental and is not part of v0.1 official results.
- `dvl_pi` modifies surge/sway using DVL or velocity feedback, but it is not an official v0.1 robustness claim.
- The fallback adapter is useful for unit tests and control-flow plumbing, but it is not a physical simulator.
- HoloOcean loading can be slow or can timeout for calibration sweeps.
- Manual controllers are demos, not benchmark baselines.
- The debug `oracle` controller uses ground truth and is blocked in official mode.

## 18. Release Checklist

- `python -m compileall -q marine_race_arena tests` passes.
- `conda run -n ocean python -m pytest -q` passes.
- `conda run -n ocean python -m marine_race_arena.scripts.run_staggered_multi_rover_smoke` passes.
- At least one single-rover official clean command runs from a clean checkout.
- README is updated.
- `docs/release_v0_1.md` is updated.
- No large generated results, logs, videos, caches, or recordings are committed.
- Official examples run from the repository root.
- Release tag is created only after the above checks pass.
