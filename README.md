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
  controllers/     student template, acoustic baseline, debug oracle
  referee/         gate validation, race state, scoring, logger, referee
  tracks/          easy, medium, hard JSON tracks
  scripts/         validation and race runner entry points
tests/             simulator-independent pytest tests
```

## Arena, Beacon, Controller, Referee

The arena owns the static race definition: bounds, gate geometry, debug visual gate bars, acoustic beacons, currents, and optional obstacle metadata.

The beacon system guides the rover toward the next expected gate. It does not validate gate passage. In official mode it returns bearing, elevation, range, signal strength, and metadata, but not exact gate positions.

Controllers receive observations and return commands. Student controllers should use only allowed sensors and beacon observations. The built-in acoustic controller is a simple baseline.

The referee validates gates using simulation ground truth. This is allowed because the referee is not a participant controller. The first implementation validates the vehicle center point and keeps the interface ready for future full-body validation.

## JSON Track Configuration

Track files live in `marine_race_arena/tracks/`. A track JSON contains:

- `race`: name, format, laps, timing mode, duration, official mode.
- `world`: HoloOcean package/map preference, arena origin, and explicit bounds.
- `track`: declared path length, gate size defaults, gate sequence.
- `start`: spawn pose.
- `finish`: final gate id.
- `gates`: gate id, type, position, rotation, size, color, passage direction, optional linked gate and beacon override.
- `beacon`: global acoustic beacon defaults.
- `currents`: constant, localized jet, sinusoidal, or vortex placeholder fields.
- `participants`: vehicle, controller, sensors, spawn, and control mode.
- `referee`: validation, penalties, and scoring settings.

All configured starts and gates must remain inside `world.bounds`. Underwater depth safety is enforced with `z_min` and `z_max`; values below `z_min` are unsafe and cause out-of-bounds/DNF. The example tracks use safe depths around `z = -4.0` to `z = -5.7` and avoid the seabed.

## Gate Validation Rule

A gate crossing is valid only when:

- The vehicle center crosses the expected gate plane between the previous and current pose.
- The crossing direction matches the gate `passage_direction`.
- The segment intersection point is inside the internal opening.
- The gate is the expected id in `track.gate_sequence`.
- The participant is inside arena bounds.
- No invalid collision is reported during that tick.

The referee uses the abstract gate geometry, not visual collision geometry.

## Timing

Two timing modes are supported:

- `green_to_finish`: official time starts at the GREEN/simulation start event and ends at the final gate.
- `first_gate_to_last_gate`: official time starts at the first valid crossing of G01 on lap 1 and ends at the final valid crossing of the finish gate.

The logger also saves green-to-finish time for every finished participant.

## Scoring And Ranking

Default penalties:

- Minor collision: +5 s.
- Gate collision: +10 s.
- Wrong direction: +20 s unless configured as DSQ.
- Out of bounds: DNF.
- Missed gate: DNF by default.

Finished participants rank ahead of unfinished participants. Finished racers rank by lower penalized official time. Non-finished racers rank by more completed gates, fewer collisions, then shorter distance to the next expected gate.

## Official And Debug Modes

Official mode does not expose ground-truth pose, exact gate positions, the full track geometry, or referee internals to participant controllers.

The debug oracle controller is explicitly a cheating feasibility tool:

```text
marine_race_arena.controllers.oracle_gate_follower.OracleGateFollowerController
```

It uses own ground-truth pose and exact target gate geometry. It is blocked by the runner when `--official` is set and is not competition-valid.

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
- `vortex`: placeholder that validates and logs but evaluates as zero until a physical adapter is added.

In the fallback adapter, currents are applied to the simple point-vehicle kinematics and exposed in logs/observations. In the tested HoloOcean 2.3.0 installation, the HoloOcean adapter applies currents through `env.set_ocean_currents(agent_name, velocity)`. If that method is missing in another installation, the adapter reports the limitation and exposes configured currents in observations/logs only.

## Validate Tracks

Run validation with either module or script style:

```bash
conda run -n ocean python -m marine_race_arena.scripts.validate_track_config --track marine_race_arena/tracks/abu_dhabi_marine_easy.json
conda run -n ocean python marine_race_arena/scripts/validate_track_config.py --track marine_race_arena/tracks/abu_dhabi_marine_medium.json
```

Validation checks required fields, unique gate ids, sequence references, finish gate, bounds, depth safety, positive sizes, nonzero passage directions, declared length, linked gates, split-S consistency, beacon ids, participant controller references, and supported current types.

## Run Races

The runner builds the arena, loads controllers, starts the referee/logger, and runs the selected simulator adapter.

```bash
conda run -n ocean python marine_race_arena/scripts/run_marine_race.py --track marine_race_arena/tracks/abu_dhabi_marine_easy.json --controller oracle --adapter fallback --duration 300
conda run -n ocean python marine_race_arena/scripts/run_marine_race.py --track marine_race_arena/tracks/abu_dhabi_marine_medium.json --controller acoustic --adapter fallback --duration 600
conda run -n ocean python marine_race_arena/scripts/run_marine_race.py --track marine_race_arena/tracks/abu_dhabi_marine_hard.json --official --adapter fallback --duration 900
```

Useful flags:

- `--adapter auto`: try HoloOcean first. If it fails, fallback is used only when `--allow-fallback` is also set.
- `--adapter fallback`: run the deterministic point-vehicle adapter.
- `--adapter holoocean`: require the HoloOcean adapter.
- `--allow-fallback`: explicitly allow fallback kinematics after HoloOcean initialization failure.
- `--official`: force official mode and block oracle ground truth.
- `--headless`: request headless HoloOcean mode when supported.
- `--record`: request HoloOcean recording when supported.
- `--participant-controller`: external `module:Class`, fully qualified `module.Class`, or file path.
- `--log-dir`: output directory for JSONL events and summary JSON.
- `--seed`: deterministic beacon noise/dropout seed.

## HoloOcean Adapter

Two simulator adapters are available:

- `fallback`: simple point-vehicle kinematics, no Unreal/HoloOcean process, metadata-only gate visuals, and currents applied as direct velocity disturbance.
- `holoocean`: attempts to create a HoloOcean BlueROV2 scenario, queue BlueROV2 thruster commands, read simulator state for the referee/debug path, and expose filtered sensor data to controllers.

Run the easy track in fallback first:

```bash
conda run -n ocean python marine_race_arena/scripts/run_marine_race.py --track marine_race_arena/tracks/abu_dhabi_marine_easy.json --controller oracle --adapter fallback --duration 300
```

Then run the diagnostic and try HoloOcean:

```bash
conda run -n ocean python marine_race_arena/scripts/diagnose_holoocean_adapter.py --track marine_race_arena/tracks/abu_dhabi_marine_easy.json
conda run -n ocean python marine_race_arena/scripts/run_marine_race.py --track marine_race_arena/tracks/abu_dhabi_marine_easy.json --controller oracle --adapter holoocean --duration 120
conda run -n ocean python marine_race_arena/scripts/run_marine_race.py --track marine_race_arena/tracks/abu_dhabi_marine_medium.json --controller acoustic --adapter holoocean --duration 300
```

Auto mode is strict by default. It does not silently fall back if HoloOcean is missing or misconfigured:

```bash
conda run -n ocean python marine_race_arena/scripts/run_marine_race.py --track marine_race_arena/tracks/abu_dhabi_marine_easy.json --controller oracle --adapter auto --allow-fallback --duration 300
```

The HoloOcean adapter imports `holoocean` only inside the adapter. It builds a custom scenario dictionary with configured BlueROV2 participants, prefers the configured map such as `OpenWater-Hovering`, and then tries the configured fallback such as `PierHarbor-Hovering`. It fails loudly when `--adapter holoocean` cannot initialize. The diagnostic script also tries prebuilt scenario names so environment-name problems are easier to distinguish from custom scenario problems.

Command mapping:

- `thrusters` mode sends an 8-value clamped BlueROV2 thruster list for HoloOcean control scheme `0`.
- `high_level` mode maps `surge`, `sway`, `heave`, and `yaw` to a conservative 8-thruster baseline.
- Commands are clamped to safe ranges, and heave is limited near configured `z_min` and `z_max` to avoid pushing the simulator into unstable extremes.

Sensor separation:

- Official/student observations include configured non-ground-truth sensors, beacon observations, time, participant id, and race progress.
- The adapter filters out `PoseSensor`, `LocationSensor`, `RotationSensor`, `DynamicsSensor`, and explicit ground-truth fields in official mode.
- The referee still uses ground-truth pose internally for gate validation and out-of-bounds checks.
- The oracle controller receives ground truth only when not in official mode.

Gate visuals:

- In HoloOcean 2.3.0, gate bars are spawned at runtime with `env.spawn_prop("box", location, rotation, scale, sim_physics=False, material, tag)`.
- Each gate is built from four box props: top, bottom, left, and right.
- If `spawn_prop` is unavailable, gate bars remain metadata and can be exported by `HoloOceanVisualSpawner` for manual Unreal placement.
- The referee always uses abstract gate geometry, regardless of visual spawning status.
- The box props are physical/collidable in the tested runtime. The debug oracle can clip a bar after valid center-point crossings, so HoloOcean gate visuals are verified as spawned but the oracle is not yet a robust physical racing controller.

Currents:

- In HoloOcean 2.3.0, physical current coupling is active through `env.set_ocean_currents(agent_name, velocity)`.
- The adapter applies the configured current field at the rover position each simulator tick.
- If this API is not available in another HoloOcean installation, the adapter warns and exposes currents in observations/logs only.
- The fallback adapter physically applies currents to its simple kinematic point vehicle.

Collision sensor:

- The generated BlueROV2 scenario includes `CollisionSensor`.
- The adapter maps `CollisionSensor`/contact values to `adapter.get_collision_state(participant_id)`.
- The referee receives collision status and applies the configured penalty/DNF rules.

Depth safety:

- `z_min` and `z_max` are enforced by the referee using adapter ground truth.
- A vehicle below `z_min` or above `z_max` is marked out-of-bounds/DNF.
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

## Track Examples

`abu_dhabi_marine_easy.json` is a 6-gate, 1-lap infrastructure/debug track with wider gates, weak constant current, safe depth around `z = -4.0`, and no split-S.

`abu_dhabi_marine_medium.json` is the first standard race: 11 gates, 2 laps, 1.5 m openings, one double gate, moderate lateral current, and one localized jet.

`abu_dhabi_marine_hard.json` is an advanced benchmark: 11 gates, 2 laps, two double-gate pairs, a marine split-S, vertical maneuvering, stronger localized currents, sinusoidal vertical disturbance, noisy beacons, and dropout.

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
{"surge": 0.3, "sway": 0.0, "heave": 0.0, "yaw": 0.1}
```

For thruster control return:

```python
{"thrusters": [0.0, 0.0, 0.0, 0.0]}
```

Run an external controller with:

```bash
python marine_race_arena/scripts/run_marine_race.py --track marine_race_arena/tracks/abu_dhabi_marine_easy.json --participant-controller path.to.module.ControllerClass
```

If you load from a file path, set `controller_class` in the participant config or use a module/class reference instead.

## Add A New Track

Copy one of the example JSON files and update:

- Race metadata and lap count.
- Bounds with safe `z_min` and `z_max`.
- Start pose.
- Gate ids, positions, rotations, colors, and passage directions.
- Gate sequence and finish gate.
- Beacon noise/dropout.
- Currents.
- Participant controller settings.
- Declared path length.

Then validate before running:

```bash
conda run -n ocean python marine_race_arena/scripts/validate_track_config.py --track path/to/your_track.json
```

## Logs

The runner writes JSONL race events and a final summary JSON under `results/marine_race` by default. Events include race start, gate passed, lap completed, collision, out of bounds, stuck, penalty, race finish, DNF, controller error, and race summary.

## Known Limitations

- The HoloOcean adapter has been validated against the local `ocean` conda environment with HoloOcean 2.3.0. Other HoloOcean versions may expose different worlds, sensors, or control behavior.
- Runtime gate spawning uses `spawn_prop("box", ...)`; scenario-based static object config was not identified in the installed Python API.
- Visual gate boxes are physical in the tested runtime, and the current debug oracle can collide with bars on the easy track. This verifies collision plumbing but also shows that the oracle is not a competition-quality physical controller.
- The referee validates the vehicle center point only; full-body gate validation is reserved for a future extension.
- The fallback runner is a simple point-vehicle feasibility tool, not a BlueROV2 physics model.
- HoloOcean physical current coupling depends on `env.set_ocean_currents`; the adapter reports inactive coupling if that API is missing.
- Vortex current is a clean placeholder.
- Obstacles are preserved in config but require a physical spawning adapter.
