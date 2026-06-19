# Marine Race Arena

This repository contains a modular marine racing arena library for BlueROV2-style underwater racing in HoloOcean-inspired scenarios. The racing format is inspired by the Abu Dhabi A2RL x DCL Autonomous Drone Championship, adapted from aerial drones to underwater marine vehicles.

The package focuses on race infrastructure, not autonomy. It provides configurable tracks, gates, acoustic beacons, currents, participant/controller interfaces, referee logic, scoring, JSONL logs, example tracks, and test coverage for the simulator-independent logic.

## Race Format

The standard marine race uses the same high-level structure as the Abu Dhabi reference format:

- 11 gates per lap.
- 2 laps for a valid standard finish.
- 22 valid gate crossings.
- 1.5 m x 1.5 m internal gate opening.
- Official timing can start on the first valid G01 crossing on lap 1 and stop on the valid G11 crossing on lap 2.

The marine version supports single gates, double gates, vertical double gates, and a marine split-S where an upper gate is followed by a lower gate with a different passage direction.

## Package Layout

```text
marine_race_arena/
  config/          JSON schema dataclasses, loader, validation
  arena/           bounds, gates, gate factory, beacons, currents, obstacles
  adapters/        fallback and HoloOcean simulator adapters
  participants/    participant state, controller interface, controller loader
  controllers/     Pygame/manual keyboard controllers, student template, acoustic baseline, debug oracle
  referee/         gate validation, race state, scoring, logger, referee
  tracks/          benchmark JSON tracks
  scripts/         validation and race runner entry points
tests/             simulator-independent pytest tests
```

## Arena, Beacon, Controller, Referee

The arena owns the static race definition: bounds, gate geometry, debug visual gate bars, acoustic beacons, currents, and optional static obstacles.

The beacon system guides the rover toward the next expected gate. It does not validate gate passage. In official mode it returns bearing, elevation, range, signal strength, and metadata, but not exact gate positions.

Controllers receive observations and return commands. The default sample-track controller is the manual `pygame` controller. Student controllers should use only allowed sensors and beacon observations. The built-in acoustic controller is a simple baseline.

The referee validates gates using simulation ground truth. This is allowed because the referee is not a participant controller. The first implementation validates the vehicle center point and keeps the interface ready for future full-body validation.

## JSON Track Configuration

Track files live in `marine_race_arena/tracks/`. A track JSON contains:

- `race`: name, format, laps, timing mode, duration, official mode.
- `benchmark_task`: optional benchmark mode, either a string or `{ "mode": "..." }`.
- `world`: HoloOcean package/map preference, arena origin, and explicit bounds.
- `track`: declared path length, gate size defaults, gate sequence.
- `start`: spawn pose.
- `finish`: final gate id.
- `gates`: gate id, type, position, rotation, size, color, passage direction, optional linked gate and beacon override.
- `beacon`: global acoustic beacon defaults.
- `currents`: constant, localized jet, sinusoidal, and vortex fields.
- `obstacle_generation`: optional obstacle mode, density, clearance, and seed settings.
- `obstacles`: optional fixed static box obstacle definitions.
- `participants`: vehicle, controller, sensors, spawn, and control mode.
- `referee`: validation, penalties, and scoring settings.

All configured starts and gates must remain inside `world.bounds`. Underwater depth safety is enforced with `z_min` and `z_max`; values below `z_min` are unsafe and are logged as out-of-bounds events with penalties. The benchmark tracks use safe depths around `z = -4.0` to `z = -5.9` and avoid the seabed.

## Benchmark Task Modes

Marine Race Arena supports explicit benchmark task modes inspired by F1TENTH-style task definitions. The task mode is optional for backward compatibility; older/custom track JSON files without `benchmark_task` keep the existing validation behavior.

Supported modes:

- `clean_gate`: one ROV, gates only, no obstacles, no currents.
- `obstacle_gate`: one ROV, gates plus active static box obstacles between adjacent gates, no currents.
- `current_gate`: one ROV, gates plus at least one configured current with speed >= 0.50 m/s. Obstacles can be ignored with `--obstacles none` unless intentionally enabled.
- `multi_rov`: future-ready multi-participant task; requires at least two participants but is not the default execution mode.

Add a mode to JSON:

```json
"benchmark_task": {
  "mode": "current_gate"
}
```

Or validate/run an existing track under an explicit task mode:

```bash
conda run -n ocean python marine_race_arena/scripts/validate_track_config.py --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --benchmark-task clean_gate
conda run -n ocean python marine_race_arena/scripts/run_marine_race.py --track marine_race_arena/tracks/marine_race_mixed_endurance.json --benchmark-task current_gate --controller pygame --adapter fallback
```

For `obstacle_gate`, active obstacles can come from fixed JSON definitions or deterministic random generation. `--obstacles none` removes active obstacles at runtime and validation time, `--obstacles fixed` uses the JSON `obstacles` array, and `--obstacles random` generates boxes between adjacent gates using `--seed`. Generated obstacles are placed at the midpoint between each selected gate pair, aligned to that local corridor direction, and given small symmetric lateral offsets around the centerline. In HoloOcean, benchmark obstacles are static suspended props by default, so they remain fixed at their configured underwater positions.

A fixed box obstacle entry uses:

```json
{
  "id": "OBS01",
  "type": "box",
  "position": [-28.2, -6.25, -4.05],
  "size": [0.8, 0.8, 0.8],
  "rotation_rpy_deg": [0.0, 0.0, 33.7],
  "collision": true,
  "penalty_s": 5.0,
  "between_gates": ["G01", "G02"]
}
```

Random obstacle default box sizes are `0.8 m`, `1.0 m`, and `1.2 m` cubes for `low`, `medium`, and `high` density respectively. Obstacle validation requires box obstacles to be inside bounds, between adjacent gates, away from gate apertures and start/finish spawn, and small enough or offset enough to leave passable clearance on at least one side of the corridor.

## Benchmark Evaluation Protocol

Use `marine_race_arena/scripts/run_benchmark.py` for repeated benchmark trials with consistent runner settings and aggregate paper-style metrics. The benchmark runner calls the normal `run_marine_race.py` pipeline once per seed, writes each run's normal JSONL event log and summary JSON under `OUTPUT_DIR/runs/`, writes a `benchmark_metadata.json` next to each run, and produces:

- `OUTPUT_DIR/benchmark_summary.csv`
- `OUTPUT_DIR/benchmark_summary.json`

The aggregate files report completion rate, official and penalized time mean/std, completed gates, collision/event averages, DNF totals and reasons, manual-stop count, and controller-error count. Pygame and keyboard controllers are accepted for manual demos, but their metadata is marked `manual_demo` and they should not be treated as main scientific baselines. The oracle controller is marked `debug_only`.

Clean-gate example:

```bash
conda run -n ocean python marine_race_arena/scripts/run_benchmark.py --benchmark-task clean_gate --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --controller acoustic --adapter fallback --seeds 0 1 2 3 4 --duration 500 --dt 0.1 --output-dir results/benchmarks/clean_gate_acoustic
```

Obstacle-gate example:

```bash
conda run -n ocean python marine_race_arena/scripts/run_benchmark.py --benchmark-task obstacle_gate --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --controller acoustic --adapter fallback --seeds 0 1 2 3 4 --duration 500 --dt 0.1 --obstacles random --obstacle-density low --obstacle-physics static --output-dir results/benchmarks/obstacle_gate_acoustic
```

Current-gate example:

```bash
conda run -n ocean python marine_race_arena/scripts/run_benchmark.py --benchmark-task current_gate --track marine_race_arena/tracks/marine_race_mixed_endurance.json --controller acoustic --adapter fallback --seeds 0 1 2 3 4 --duration 1300 --dt 0.1 --obstacles none --output-dir results/benchmarks/current_gate_acoustic
```

## Gate Validation Rule

A gate crossing is valid only when:

- The vehicle center crosses the expected gate plane between the previous and current pose.
- The crossing direction matches the gate `passage_direction`.
- The segment intersection point is inside the internal opening.
- The gate is the expected id in `track.gate_sequence`.
- The participant has not skipped the expected sequence by crossing a different gate.

The referee uses the abstract gate geometry, not visual collision geometry.
Set `referee.gate_validation.vehicle_clearance_margin_m` to shrink the valid center-point aperture by a safety margin. The default is `0.0` for backward compatibility; the sample tracks use small positive margins to reduce valid-but-too-close center crossings near physical bars.

## Timing

Two timing modes are supported:

- `green_to_finish`: official time starts at the GREEN/simulation start event and ends at the final gate.
- `first_gate_to_last_gate`: official time starts at the first valid crossing of G01 on lap 1 and ends at the final valid crossing of the finish gate.

The logger also saves green-to-finish time for every finished participant.

## Scoring And Ranking

Final benchmark rules:

- Minor collision: +5 s.
- Gate collision: +10 s.
- Obstacle collision: obstacle-specific `penalty_s`, default examples use +5 s.
- Out of bounds: +10 s.
- Stuck episode: +15 s.
- Wrong direction: logged only, no default penalty or DSQ.
- Collision, obstacle collision, out-of-bounds, and stuck are penalty/events, not DNF.
- Missing or skipping a gate by crossing a different gate than expected is DNF.
- Timeout is disabled by default. The runner may still stop at the configured duration guard, but the referee does not mark TIMEOUT unless `referee.gate_validation.timeout_enabled` is true.
- Controller failure remains terminal as `CONTROLLER_ERROR`.

Finished participants rank ahead of unfinished participants. Finished racers rank by lower penalized official time. Non-finished racers rank by more completed gates, fewer collisions, lower accumulated penalty, then shorter distance to the next expected gate.

## Official And Debug Modes

Official mode does not expose ground-truth pose, exact gate positions, the full track geometry, or referee internals to participant controllers.

The debug oracle controller is explicitly a cheating feasibility tool:

```text
marine_race_arena.controllers.oracle_gate_follower.OracleGateFollowerController
```

It uses own ground-truth pose and exact target gate geometry. It is blocked by the runner when `--official` is set and is not competition-valid. The current oracle is a simplified no-yaw translator: it commands surge, sway, and heave toward each gate centerline and always returns `yaw = 0.0`.

## Manual Pygame Control

Use the built-in `pygame` controller for manual local testing:

```bash
conda run -n ocean python marine_race_arena/scripts/run_marine_race.py --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --controller pygame --adapter holoocean --duration 500 --dt 0.033
```

Controls:

- `W` / `S`: forward and backward on the horizontal plane.
- `A` / `D`: left and right sway on the horizontal plane.
- `Q` / `E`: yaw left and right.
- Up arrow / Down arrow: raise and lower the rover.
- Space: stop motion.
- Esc: cleanly stop the race.

The controller opens a small Pygame control window. Keep focus on that Pygame window while driving. Pressing Esc or closing the Pygame control window prints `Manual stop requested.`, marks the participant as `MANUAL_STOP`, and closes the controller, camera viewer, adapter, and logger through the normal runner cleanup path. It does not use ground truth.

## Acoustic Beacons

Supported beacon activation modes:

- `active_when_target`: only the expected gate beacon is active.
- `always_on`: all beacon ids are visible, while the target observation remains focused on the expected gate.
- `sequential_channel`: simple channel index based on sequence position.

Supported observation modes:

- `oracle`: exact gate pose, debug only.
- `acoustic_ideal`: bearing/range/elevation without noise.
- `acoustic_noisy`: bearing/range/elevation with configured noise and dropout.

Official observations never include exact gate center or full track ground truth.

## Currents

The current manager supports:

- `constant`: uniform velocity vector.
- `localized_jet`: radius-limited current with Gaussian or linear falloff.
- `sinusoidal`: oscillating velocity component.
- `vortex`: analytic horizontal swirl around a configured center, with optional vertical component and linear or Gaussian falloff.

In the fallback adapter, currents are applied to the simple point-vehicle kinematics and exposed in logs/observations. In the tested HoloOcean 2.3.0 installation, the HoloOcean adapter applies currents through `env.set_ocean_currents(agent_name, velocity)`. If that method is missing in another installation, the adapter reports the limitation and exposes configured currents in observations/logs only.

The first two benchmark tracks intentionally have no currents. `marine_race_mixed_endurance.json` is the current benchmark and combines constant flow, localized jets, a sinusoidal vertical component, and a vortex. Use the stopped-rover diagnostic to verify physical drift:

```bash
conda run -n ocean python marine_race_arena/scripts/diagnose_currents.py --track marine_race_arena/tracks/marine_race_mixed_endurance.json --adapter holoocean --duration 10 --zero-command
```

To test physical drift inside a strong current zone instead of the track start, pass an explicit drift spawn position. For the mixed endurance benchmark, this uses the second localized jet:

```bash
conda run -n ocean python marine_race_arena/scripts/diagnose_currents.py --track marine_race_arena/tracks/marine_race_mixed_endurance.json --adapter holoocean --duration 10 --zero-command --drift-position 58.0 12.0 -5.3
```

## Validate Tracks

Run validation with either module or script style:

```bash
conda run -n ocean python -m marine_race_arena.scripts.validate_track_config --track marine_race_arena/tracks/marine_race_horseshoe_bay.json
conda run -n ocean python marine_race_arena/scripts/validate_track_config.py --track marine_race_arena/tracks/marine_race_vertical_serpent.json
conda run -n ocean python marine_race_arena/scripts/validate_track_config.py --track marine_race_arena/tracks/marine_race_mixed_endurance.json
```

Validation checks required fields, unique gate ids, sequence references, finish gate, bounds, depth safety, positive sizes, nonzero passage directions, declared length, linked gates, split-S consistency, beacon ids, participant controller references, supported current types, active obstacle generation/layout, and any active benchmark task requirements.

## Run Races

The runner builds the arena, loads controllers, starts the referee/logger, and runs the selected simulator adapter.

```bash
conda run -n ocean python marine_race_arena/scripts/run_marine_race.py --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --controller pygame --adapter fallback --duration 500
conda run -n ocean python marine_race_arena/scripts/run_marine_race.py --track marine_race_arena/tracks/marine_race_vertical_serpent.json --controller pygame --adapter fallback --duration 850
conda run -n ocean python marine_race_arena/scripts/run_marine_race.py --track marine_race_arena/tracks/marine_race_mixed_endurance.json --controller pygame --adapter fallback --duration 1300
```

Obstacle benchmark examples:

```bash
conda run -n ocean python marine_race_arena/scripts/validate_track_config.py --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --benchmark-task obstacle_gate --obstacles random --obstacle-density low --seed 7
conda run -n ocean python marine_race_arena/scripts/run_marine_race.py --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --benchmark-task obstacle_gate --obstacles random --obstacle-density low --seed 7 --controller pygame --adapter fallback --duration 500
conda run -n ocean python marine_race_arena/scripts/run_marine_race.py --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --benchmark-task obstacle_gate --obstacles random --obstacle-density low --obstacle-physics static --seed 7 --controller pygame --adapter holoocean --duration 300 --dt 0.033
```

Current diagnostics:

```bash
conda run -n ocean python marine_race_arena/scripts/diagnose_currents.py --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --adapter fallback
conda run -n ocean python marine_race_arena/scripts/diagnose_currents.py --track marine_race_arena/tracks/marine_race_vertical_serpent.json --adapter fallback
conda run -n ocean python marine_race_arena/scripts/diagnose_currents.py --track marine_race_arena/tracks/marine_race_mixed_endurance.json --adapter fallback
```

Useful flags:

- `--adapter auto`: try HoloOcean first. If it fails, fallback is used only when `--allow-fallback` is also set.
- `--adapter fallback`: run the deterministic point-vehicle adapter.
- `--adapter holoocean`: require the HoloOcean adapter.
- `--benchmark-task`: validate this run as `clean_gate`, `obstacle_gate`, `current_gate`, or `multi_rov`.
- `--obstacles`: `none` removes active obstacles, `fixed` uses JSON obstacles, `random` generates deterministic boxes between gates.
- `--obstacle-density`: `low`, `medium`, or `high` density for random obstacle generation.
- `--obstacle-physics`: `static` keeps HoloOcean obstacles suspended and fixed; `dynamic` enables simulator physics for experiments.
- `--allow-fallback`: explicitly allow fallback kinematics after HoloOcean initialization failure.
- `--official`: force official mode and block oracle ground truth.
- `--headless`: request headless HoloOcean mode when supported.
- `--record`: request HoloOcean recording when supported.
- `--participant-controller`: external `module:Class`, fully qualified `module.Class`, or file path.
- `--controller-class`: class name to instantiate when using a Python file or module path.
- `--disable-front-camera`: disable `FrontCamera` capture for non-official live/debug runs when the viewport or control loop is too slow. This is rejected in `--official` mode.
- `--show-front-camera`: open a live viewer for `observation["sensors"]["FrontCamera"]`. Press `V` or `Esc` in the camera viewer to close only that viewer while the race continues.
- `--log-dir`: output directory for JSONL events and summary JSON.
- `--seed`: deterministic beacon noise/dropout seed and random-obstacle seed.

## HoloOcean Adapter

Two simulator adapters are available:

- `fallback`: simple point-vehicle kinematics, no Unreal/HoloOcean process, metadata-only gate visuals, and currents applied as direct velocity disturbance.
- `holoocean`: attempts to create a HoloOcean BlueROV2 scenario, queue BlueROV2 thruster commands, read simulator state for the referee/debug path, and expose filtered sensor data to controllers.

Run the easy track in fallback first:

```bash
conda run -n ocean python marine_race_arena/scripts/run_marine_race.py --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --controller pygame --adapter fallback --duration 500
```

Then run the diagnostic and try HoloOcean:

```bash
conda run -n ocean python marine_race_arena/scripts/diagnose_holoocean_adapter.py --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --print-gate-bars
conda run -n ocean python marine_race_arena/scripts/diagnose_gate_visual_rotations.py --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --output-dir diagnostics/gate_rotation_tests_selected --selected-only
conda run -n ocean python marine_race_arena/scripts/run_marine_race.py --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --controller pygame --adapter holoocean --duration 500 --dt 0.033
conda run -n ocean python marine_race_arena/scripts/run_marine_race.py --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --controller pygame --adapter holoocean --duration 500 --dt 0.033 --show-front-camera
conda run -n ocean python marine_race_arena/scripts/run_marine_race.py --track marine_race_arena/tracks/marine_race_vertical_serpent.json --controller pygame --adapter holoocean --duration 850 --dt 0.033
conda run -n ocean python marine_race_arena/scripts/run_marine_race.py --track marine_race_arena/tracks/marine_race_mixed_endurance.json --controller pygame --adapter holoocean --duration 1300 --dt 0.033
```

Auto mode is strict by default. It does not silently fall back if HoloOcean is missing or misconfigured:

```bash
conda run -n ocean python marine_race_arena/scripts/run_marine_race.py --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --controller pygame --adapter auto --allow-fallback --duration 500
```

The HoloOcean adapter imports `holoocean` only inside the adapter. It builds a custom scenario dictionary with configured BlueROV2 participants, prefers the configured map such as `OpenWater-Hovering`, and then tries the configured fallback such as `PierHarbor-Hovering`. It fails loudly when `--adapter holoocean` cannot initialize. The diagnostic script also tries prebuilt scenario names so environment-name problems are easier to distinguish from custom scenario problems.

Command mapping:

- `thrusters` mode sends an 8-value clamped BlueROV2 thruster list for HoloOcean control scheme `0`.
- `high_level` mode maps `surge`, `sway`, `heave`, and `yaw` to a conservative 8-thruster baseline.
- Commands are clamped to safe ranges, and heave is limited near configured `z_min` and `z_max` to avoid pushing the simulator into unstable extremes.

Sensor separation:

- Official/student observations include configured non-ground-truth sensors, beacon observations, time, participant id, and race progress.
- The benchmark tracks include an official front RGB camera exposed as `observation["sensors"]["FrontCamera"]`.
- The HoloOcean scenario config mounts this camera as `RGBCamera` named `FrontCamera` on `CameraSocket`, with rotation `[0.0, 0.0, 0.0]`, `Hz=30`, `640x480`, and FOV `90.0`.
- If the installed HoloOcean runtime returns the image under `RGBCamera`, the adapter aliases it to `FrontCamera` before filtering.
- Manual pygame testing can show this feed live with `--show-front-camera`. The viewer uses the filtered controller observation, supports image arrays or nested lists, and does not expose ground truth. In the tested HoloOcean 2.3.0 installation, the Python sensor docstring still says `RGBA`, but the runtime `FrontCamera` buffer behaves as `BGRA/BGR`; the viewer now uses that practical ordering for correct colors. OpenCV is used when installed; if OpenCV is unavailable, the runner prints a clear warning or uses a pygame fallback when no other pygame display is active.
- The adapter filters out `PoseSensor`, `LocationSensor`, `RotationSensor`, `DynamicsSensor`, and explicit ground-truth fields in official mode.
- The referee still uses ground-truth pose internally for gate validation and out-of-bounds checks.
- The oracle controller receives ground truth only when not in official mode.

Obstacles:

- Fixed obstacles are loaded from JSON when `--obstacles fixed` is active. Random obstacles are generated reproducibly from `--seed`.
- Random obstacles are centered at gate-pair midpoints in the local corridor frame, with only small symmetric lateral offsets so they do not always appear on one side of the route.
- In HoloOcean, active box obstacles are spawned with `env.spawn_prop("box", ..., sim_physics=False)` by default. They are visible, collidable props but are not affected by gravity/current unless `--obstacle-physics dynamic` is explicitly selected.
- In fallback, active obstacles remain metadata and collisions are approximated with a simple bounding check along the participant movement segment.
- Obstacle collisions emit `obstacle_collision`, add the obstacle's `penalty_s`, and do not cause DNF by default.

Gate visuals:

- In HoloOcean 2.3.0, gate visuals are spawned at runtime with `env.spawn_prop("box", location, rotation, scale, sim_physics=False, material, tag)`.
- Each logical gate is still built from four bars: top, bottom, left, and right.
- Gate visual orientation is generated from `passage_direction`, which is the source of truth for the gate normal. The factory derives a full 3D frame with local X along the gate normal, local Y along the right/opening width axis, and local Z along the up/opening height axis. This supports yaw-only and pitched gate definitions.
- All sample tracks use the Abu Dhabi-style 1.5 m x 1.5 m internal opening. The side bars are generated as vertical pillars with their height on world Z.
- Runtime HoloOcean spawning uses a hybrid visual mode by default: the left and right pillars are single vertical box props, while the top and bottom bars are dense overlapping micro-blocks. The default micro-blocks are intentionally much smaller than the bar thickness so the top and bottom read as continuous strips instead of visible large cubes.
- The optional `uniform` mode uses one elongated box prop per logical gate bar. The installed HoloOcean 2.3.0 Python API documents prop rotation as `[roll, pitch, yaw]`, but the tested Unreal backend renders box yaw correctly when the generated gate rotation is sent as `[yaw, pitch, roll]`. The adapter keeps the internal math in roll/pitch/yaw and applies that HoloOcean-specific conversion only at the `spawn_prop` boundary.
- HoloOcean `spawn_prop` box `scale` is a multiplier on a one-meter box, so each logical bar passes its meter dimensions directly as `scale`.
- Set `MARINE_RACE_GATE_VISUAL_MODE=uniform` only when intentionally testing the four-long-bar version. Set `MARINE_RACE_GATE_VISUAL_MODE=segmented` to represent every logical bar as a chain of axis-aligned box segments along the same mathematical centerline.
- If `spawn_prop` is unavailable, gate bars remain metadata and can be exported by `HoloOceanVisualSpawner` for manual Unreal placement.
- The referee always uses abstract gate geometry, regardless of visual spawning status.
- Use `--print-gate-bars` with the diagnostic script to print every gate id, source bar id, position, logical rotation, runtime spawn rotation, dimensions, and spawn method. Use `diagnose_gate_visual_rotations.py --single-long-only --fixed-front-camera` to spawn one four-long-bar test gate per yaw/pitch case and save `ViewportCapture` screenshots under `diagnostics/`. Add `--rotation-mapping rpy` only when intentionally reproducing the broken mapping for comparison.
- The box props are physical/collidable in the tested runtime. The no-yaw oracle attempts simple translational gate transits, but it remains a debug feasibility controller rather than a competition-quality controller.

Currents:

- In HoloOcean 2.3.0, physical current coupling is active through `env.set_ocean_currents(agent_name, velocity)`.
- The adapter applies the configured current field at the rover position each simulator tick.
- If this API is not available in another HoloOcean installation, the adapter warns and exposes currents in observations/logs only.
- The fallback adapter physically applies currents to its simple kinematic point vehicle.

Collision sensor:

- The generated BlueROV2 scenario includes `CollisionSensor`.
- The adapter maps `CollisionSensor`/contact values to `adapter.get_collision_state(participant_id)`.
- The referee receives collision status and applies the configured time penalty with cooldown. Collision is not DNF in the final benchmark rules.

Depth safety:

- `z_min` and `z_max` are enforced by the referee using adapter ground truth.
- A vehicle below `z_min` or above `z_max` is marked as out-of-bounds and receives the configured penalty with cooldown.
- Official controllers do not receive this bounds check as privileged navigation ground truth.

Troubleshooting HoloOcean environments:

- Verify that the `holoocean` Python package imports in the same Python environment used to run the script.
- Verify that the Ocean package containing `OpenWater-Hovering` or `PierHarbor-Hovering` is installed.
- To check basic availability, run:

```bash
conda run -n ocean python -c "import holoocean; print(holoocean); print(getattr(holoocean, '__version__', 'unknown')); print(getattr(holoocean, '__file__', None))"
```

- If `OpenWater-Hovering` is unavailable, set `world.map` or `world.fallback_environment` to an installed scenario and revalidate the track.
- If the rover does not move, run the diagnostic first. It prints the generated agent config, raw sensor keys, zero-action test, forward-action test, current-coupling method, and whether the pose changed.
- If expected sensors are missing, check the generated sensor list in the diagnostic output and compare it with the installed HoloOcean package.
- If the live viewport is very choppy during manual/debug runs, try `--disable-front-camera`. The official RGB camera is 640x480 at 30 Hz and can be expensive on some machines.

## Track Examples

`marine_race_horseshoe_bay.json` is a `clean_gate` 12-gate, 1-lap open U-shaped route with standard 1.5 m x 1.5 m gate openings, no currents, safe depth near `z = -4.0`, and clear point-to-point visibility.

`marine_race_vertical_serpent.json` is a `clean_gate` 17-gate, 1-lap slalom with strong depth variation, vertical double-gate elements, no currents, and a longer duration budget.

`marine_race_mixed_endurance.json` is a `current_gate` 22-gate, 1-lap endurance route with diagonals, vertical variation, double gates, a split-S-like section, strong currents, localized jets, a vortex, beacon noise, and dropout.

## Add A Student Controller

Start from:

```text
marine_race_arena/controllers/student_template.py
```

A controller must implement:

```python
def reset(self, race_info): ...
def step(self, observation): ...
def close(self): ...
```

For high-level control return:

```python
return {
    "surge": 0.4,
    "sway": 0.0,
    "heave": 0.0,
    "yaw": 0.0,
}
```

`surge` is forward/backward, `sway` is lateral, `heave` is vertical, and `yaw` is rotation. Values are clamped by the adapter.

Official controllers can read the front camera like this:

```python
image = observation["sensors"].get("FrontCamera")
if image is not None:
    # HoloOcean RGBCamera usually returns a uint8 array shaped (480, 640, 4).
    height = image.shape[0] if hasattr(image, "shape") else len(image)
```

For thruster control return:

```python
{"thrusters": [0.0, 0.0, 0.0, 0.0]}
```

Run an external controller with:

```bash
conda run -n ocean python marine_race_arena/scripts/run_marine_race.py --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --participant-controller path/to/my_controller.py --controller-class MyController --adapter holoocean --duration 500 --dt 0.033
```

You can also use `--participant-controller package.module:MyController` or `--participant-controller package.module.MyController`.

Official observations exclude ground-truth pose, exact gate centers, full track geometry, and referee internals. They may include beacon messages, `FrontCamera`, depth, IMU, DVL/velocity when configured, and current estimates when allowed by the sensor profile. The oracle controller is debug-only and not competition-valid.

## Add A New Track

Copy one of the example JSON files and update:

- Race metadata and lap count.
- Optional `benchmark_task` mode if the track should be validated as a benchmark task.
- Bounds with safe `z_min` and `z_max`.
- Start pose.
- Gate ids, positions, rotations, colors, and passage directions.
- Gate sequence and finish gate.
- Beacon noise/dropout.
- Currents.
- Fixed obstacles or `obstacle_generation` settings for `obstacle_gate` tracks.
- Participant controller settings.
- Declared path length.

Then validate before running:

```bash
conda run -n ocean python marine_race_arena/scripts/validate_track_config.py --track path/to/your_track.json
```

## Logs

The runner writes JSONL race events and a final summary JSON under `results/marine_race` by default. Events include race start, gate passed, lap completed, collision, obstacle collision, out of bounds, stuck, penalty, race finish, DNF, manual stop, controller error, and race summary.

## Known Limitations

- The HoloOcean adapter has been validated against the local `ocean` conda environment with HoloOcean 2.3.0. Other HoloOcean versions may expose different worlds, sensors, or control behavior.
- Runtime gate spawning uses `spawn_prop("box", ...)` with hybrid micro-block top/bottom bars by default; scenario-based static object config was not identified in the installed Python API.
- Visual gate boxes are physical in the tested runtime. The no-yaw oracle is intended for feasibility testing and may still collide if dynamics, currents, or track geometry exceed its simple controller assumptions.
- The referee validates the vehicle center point only. `vehicle_clearance_margin_m` can shrink the accepted aperture, but full-body gate validation is reserved for a future extension.
- The fallback runner is a simple point-vehicle feasibility tool, not a BlueROV2 physics model.
- HoloOcean physical current coupling depends on `env.set_ocean_currents`; the adapter reports inactive coupling if that API is missing.
- Vortex current is a simplified analytic field, not a CFD model.
- HoloOcean obstacle spawning depends on `env.spawn_prop`; fallback obstacle collisions use approximate geometry rather than full rigid-body contact.
