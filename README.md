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
  tracks/          easy, medium, hard JSON tracks
  scripts/         validation and race runner entry points
tests/             simulator-independent pytest tests
```

## Arena, Beacon, Controller, Referee

The arena owns the static race definition: bounds, gate geometry, debug visual gate bars, acoustic beacons, currents, and optional obstacle metadata.

The beacon system guides the rover toward the next expected gate. It does not validate gate passage. In official mode it returns bearing, elevation, range, signal strength, and metadata, but not exact gate positions.

Controllers receive observations and return commands. The default sample-track controller is the manual `pygame` controller. Student controllers should use only allowed sensors and beacon observations. The built-in acoustic controller is a simple baseline.

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
Set `referee.gate_validation.vehicle_clearance_margin_m` to shrink the valid center-point aperture by a safety margin. The default is `0.0` for backward compatibility; the sample tracks use small positive margins to reduce valid-but-too-close center crossings near physical bars.

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
- Space: stop.

The controller opens a small Pygame control window. Keep focus on that Pygame window while driving. It does not use ground truth.

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
conda run -n ocean python -m marine_race_arena.scripts.validate_track_config --track marine_race_arena/tracks/marine_race_horseshoe_bay.json
conda run -n ocean python marine_race_arena/scripts/validate_track_config.py --track marine_race_arena/tracks/marine_race_vertical_serpent.json
conda run -n ocean python marine_race_arena/scripts/validate_track_config.py --track marine_race_arena/tracks/marine_race_mixed_endurance.json
```

Validation checks required fields, unique gate ids, sequence references, finish gate, bounds, depth safety, positive sizes, nonzero passage directions, declared length, linked gates, split-S consistency, beacon ids, participant controller references, and supported current types.

## Run Races

The runner builds the arena, loads controllers, starts the referee/logger, and runs the selected simulator adapter.

```bash
conda run -n ocean python marine_race_arena/scripts/run_marine_race.py --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --controller pygame --adapter fallback --duration 500
conda run -n ocean python marine_race_arena/scripts/run_marine_race.py --track marine_race_arena/tracks/marine_race_vertical_serpent.json --controller pygame --adapter fallback --duration 850
conda run -n ocean python marine_race_arena/scripts/run_marine_race.py --track marine_race_arena/tracks/marine_race_mixed_endurance.json --controller pygame --adapter fallback --duration 1300
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
conda run -n ocean python marine_race_arena/scripts/run_marine_race.py --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --controller pygame --adapter fallback --duration 500
```

Then run the diagnostic and try HoloOcean:

```bash
conda run -n ocean python marine_race_arena/scripts/diagnose_holoocean_adapter.py --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --print-gate-bars
conda run -n ocean python marine_race_arena/scripts/diagnose_gate_visual_rotations.py --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --output-dir diagnostics/gate_rotation_tests_selected --selected-only
conda run -n ocean python marine_race_arena/scripts/run_marine_race.py --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --controller pygame --adapter holoocean --duration 500 --dt 0.033
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
- The adapter filters out `PoseSensor`, `LocationSensor`, `RotationSensor`, `DynamicsSensor`, and explicit ground-truth fields in official mode.
- The referee still uses ground-truth pose internally for gate validation and out-of-bounds checks.
- The oracle controller receives ground truth only when not in official mode.

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

`marine_race_horseshoe_bay.json` is a 12-gate, 1-lap open U-shaped route with standard 1.5 m x 1.5 m gate openings, weak current, safe depth near `z = -4.0`, and clear point-to-point visibility.

`marine_race_vertical_serpent.json` is a 17-gate, 1-lap slalom with strong depth variation, vertical double-gate elements, localized jets, and a longer duration budget.

`marine_race_mixed_endurance.json` is a 22-gate, 1-lap endurance route with diagonals, vertical variation, double gates, a split-S-like section, multiple localized jets, beacon noise, and dropout.

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
python marine_race_arena/scripts/run_marine_race.py --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --participant-controller path.to.module.ControllerClass
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
- Runtime gate spawning uses `spawn_prop("box", ...)` with hybrid micro-block top/bottom bars by default; scenario-based static object config was not identified in the installed Python API.
- Visual gate boxes are physical in the tested runtime. The no-yaw oracle is intended for feasibility testing and may still collide if dynamics, currents, or track geometry exceed its simple controller assumptions.
- The referee validates the vehicle center point only. `vehicle_clearance_margin_m` can shrink the accepted aperture, but full-body gate validation is reserved for a future extension.
- The fallback runner is a simple point-vehicle feasibility tool, not a BlueROV2 physics model.
- HoloOcean physical current coupling depends on `env.set_ocean_currents`; the adapter reports inactive coupling if that API is missing.
- Vortex current is a clean placeholder.
- Obstacles are preserved in config but require a physical spawning adapter.
