# Configuration

A race is one JSON file. This page builds one from scratch, then documents each
block you are likely to change.

A complete, validated three-gate example lives at
[`docs/examples/minimal_track.json`](examples/minimal_track.json) — everything
below is quoted from it or from the official circuits in
[`marine_race_arena/tracks/`](../marine_race_arena/tracks/).

Validate any file you write, without launching the simulator:

```bash
python -m marine_race_arena.scripts.validate_track_config \
  --track docs/examples/minimal_track.json --benchmark-task clean_gate
```

```text
Gates per lap: 3
Laps: 1
Declared path length: 28.00 m
Computed path length: 28.00 m
Validation passed.
```

The validator recomputes the course length from the geometry and compares it to
your declared value within `length_tolerance_m`, so a mistyped coordinate is
caught before you spend engine time on it.

---

## 1. The smallest race that works

Six blocks are required: `race`, `world`, `track`, `start`, `finish`, `gates`.
Three gates in a straight line:

```jsonc
{
  "race": {
    "name": "Example Straight Three",
    "laps": 1,
    "expected_gates_per_lap": 3,
    "timing_mode": "first_gate_to_last_gate",
    "max_duration_s": 180
  },
  "world": {
    "package": "Ocean",
    "map": "OpenWater-Hovering",
    "bounds": { "x_min": -30.0, "x_max": 30.0,
                "y_min": -12.0, "y_max": 12.0,
                "z_min":  -8.0, "z_max":  -1.0 }
  },
  "track": {
    "declared_length_m": 28.0,
    "length_tolerance_m": 2.0,
    "gate_inner_size_m": [1.5, 1.5],
    "gate_sequence": ["G01", "G02", "G03"]
  },
  "start":  { "position": [-18.0, 0.0, -4.0], "rotation_rpy_deg": [0.0, 0.0, 0.0] },
  "finish": { "gate_id": "G03" },
  "gates": [
    { "id": "G01", "type": "single", "position": [-10.0, 0.0, -4.0],
      "rotation_rpy_deg": [0.0, 0.0, 0.0], "color": "#00ff88",
      "passage_direction": [1.0, 0.0, 0.0] }
    // G02 at x = 0, G03 at x = 10, same orientation
  ]
}
```

Run it like any other track:

```bash
python -m marine_race_arena.scripts.run_marine_race \
  --track docs/examples/minimal_track.json \
  --benchmark-task clean_gate --controller rule_gate_center_then_commit \
  --adapter holoocean --official --headless \
  --seed 0 --dt 0.1 --duration 180 \
  --log-dir results/minimal_track
```

---

## 2. World and bounds

```jsonc
"world": {
  "package": "Ocean",
  "map": "OpenWater-Hovering",
  "preferred_environment": "OpenWater-Hovering",
  "fallback_environment": "PierHarbor-Hovering",
  "arena_origin": [0.0, 0.0, 0.0],
  "bounds": { "x_min": -36.0, "x_max": 36.0,
              "y_min": -16.0, "y_max": 20.0,
              "z_min":  -8.0, "z_max":  -1.0 }
}
```

`bounds` is not decoration: leaving it is an out-of-bounds event the referee
charges. `z` is negative downward, so `z_max: -1.0` keeps the vehicle at least
one metre below the surface.

Fog is configured separately and is part of the perception problem:

```jsonc
"water_fog": { "enabled": true, "density": 5.0,
               "start_distance_m": 1.0, "color_rgb": [0.4, 0.6, 1.0] }
```

---

## 3. Gates and the sequence

Every gate is a position, an orientation and a passage direction. The
**order** comes from `track.gate_sequence`, not from the array order:

```jsonc
"track": {
  "declared_length_m": 93.8,
  "length_tolerance_m": 4.0,
  "gate_inner_size_m": [1.5, 1.5],
  "gate_bar_thickness_m": 0.18,
  "gate_depth_m": 0.22,
  "gate_sequence": ["G01", "G02", "..."]
}
```

```jsonc
{
  "id": "G01",
  "type": "single",
  "position": [-30.0, -10.0, -4.0],
  "rotation_rpy_deg": [0.0, 0.0, 33.7],
  "color": "#00ff88",
  "passage_direction": [0.832, 0.555, 0.0]
}
```

`passage_direction` is the direction a **valid** crossing travels through the
aperture. Crossing the same plane the other way is a wrong-direction event.

`finish.gate_id` names the last gate of the sequence; `race.laps` repeats the
sequence, and `race.expected_gates_per_lap` must match its length.

---

## 4. Starting pose

```jsonc
"start": { "position": [-33.33, -12.22, -4.0], "rotation_rpy_deg": [0.0, 0.0, 33.7] }
```

Timing follows `race.timing_mode`. With `first_gate_to_last_gate` the clock
starts at the first valid crossing, so the approach from the start pose is not
charged.

---

## 5. Participants, vehicles and sensors

```jsonc
"participants": [
  {
    "id": "bluerov2_01",
    "vehicle": "BlueROV2",
    "controller": "rule_gate_center_then_commit",
    "controller_class": null,
    "spawn": { "position": [-33.33, -12.22, -4.0], "rotation_rpy_deg": [0.0, 0.0, 33.7] },
    "sensors": {
      "profile": "official_vision_acoustic",
      "allowed_sensors": ["DepthSensor", "IMUSensor", "DVLSensor",
                          "CollisionSensor", "FrontCamera"],
      "holoocean_sensors": [
        { "sensor_type": "DepthSensor", "socket": "DepthSocket", "Hz": 30,
          "configuration": { "Sigma": 0.0 } }
      ]
    }
  }
]
```

`allowed_sensors` is the filter the participant observation is built from: a
sensor that is not listed never reaches the controller, whatever the simulator
produces. `holoocean_sensors` is the engine-side configuration — rates and noise.

`controller` takes a built-in alias; for your own code use `module_or_file` plus
`class` (see [controllers.md](controllers.md)). The command line overrides the
file, which is what the experiment scripts rely on.

---

## 6. Acoustic beacons

One independent transmitter per gate, and the participant's only non-visual way
to know where the course goes:

```jsonc
"beacon": {
  "enabled": true,
  "position_offset": [0.0, 0.0, 0.35],
  "range_m": 90.0,
  "angular_noise_std_deg": 0.2,
  "range_noise_std_m": 0.2,
  "dropout_probability": 0.0,
  "update_rate_hz": 10.0
}
```

Packets are delivered only when physically in range and not dropped. Noise,
scheduling and dropout are seeded by the run seed, the transmitter, the receiver
and the transmission index — so reception does not depend on how many vehicles
are racing or in what order controllers are polled, and a fleet run is
reproducible.

---

## 7. Currents

Profiles are named in the track file and selected at run time:

```jsonc
"current_profiles": {
  "none":   { "layout": "none" },
  "medium": { "layout": "gate_relative", "scale": 0.5 },
  "strong": { "layout": "gate_relative", "scale": 1.0 }
}
```

```bash
--current-profile none | medium | strong
```

`--current-profile none` disables currents whatever the file says, which is how
a `current_gate` circuit is run current-free. A run is only accepted as
current evidence when its metadata records the native HoloOcean backend, no
fallback and active current coupling — the summary field
`physical_current_coupling_active` reports this.

---

## 8. Obstacles

Either explicit obstacles from the file, or seeded-random ones:

```bash
--obstacles none | fixed | random
--obstacle-density low | medium | high      # random only
--obstacle-physics static | dynamic         # suspended, or gravity enabled
```

Random obstacles are generated from the run seed, so the same seed gives the
same field. The summary records `physical_obstacles_requested` and
`physical_obstacles_spawned`; a physical obstacle check is only meaningful when
those agree on the HoloOcean adapter.

---

## 9. Referee: validation, penalties, scoring

```jsonc
"referee": {
  "gate_validation": {
    "vehicle_model": "center_point",
    "vehicle_clearance_margin_m": 0.1,
    "stuck_timeout_s": 45.0,
    "stuck_speed_threshold_m_s": 0.02,
    "timeout_enabled": false,
    "collision_penalty_cooldown_s": 1.0,
    "out_of_bounds_penalty_cooldown_s": 1.0
  },
  "penalties": {
    "minor_collision_s": 5.0,
    "gate_collision_s": 10.0,
    "out_of_bounds_s": 10.0,
    "stuck_s": 15.0,
    "wrong_direction_s": 0.0,
    "missed_gate_dnf": true,
    "severe_collision_dnf": false,
    "out_of_bounds_dnf": false,
    "wrong_direction_dsq": false
  },
  "scoring": {
    "rank_finished_by": "penalized_time",
    "rank_unfinished_by": "completed_gates"
  }
}
```

Penalties are seconds added to the official time. The identity
`penalized_time = official_time + penalties` holds on every finished run and is
checked automatically — see [reproducing-the-paper.md](reproducing-the-paper.md).

The cooldowns prevent a single sustained contact from being charged once per
simulation step.

---

## 10. Fleets and communication

Multi-vehicle settings live in the run configuration rather than the track, so
one circuit can be raced solo or as a team. From
[`configs/fleet.json`](../configs/fleet.json):

```jsonc
"fleet": {
  "num_rovers": 2,
  "start_gap_s": 90.0,
  "lateral_offset_m": 3.0,
  "team_id": "fleet_01",
  "inter_vehicle_collision": {
    "mode": "diagnostic",          // off | diagnostic | penalize
    "xy_threshold_m": 0.8,
    "z_threshold_m": 0.75,
    "cooldown_s": 1.0
  },
  "comms": {
    "enabled": false,
    "sound_speed_m_s": 1500.0,
    "max_range_m": 100.0,
    "processing_delay_s": 0.05,
    "packet_loss_prob": 0.0,
    "max_payload_bytes": 128,
    "min_send_interval_s": 0.5
  }
}
```

Each vehicle gets its own controller instance and its own referee state;
`team_summary` in the run summary aggregates expected and completed gates,
finish status, elapsed and penalized time, contacts and inter-vehicle events.

The comms block is the only channel between vehicles, and it is physical:
range-limited, delayed by the sound speed, rate-limited and lossy. Coordination
policies must tolerate stale and missing messages.

The same knobs exist as command-line flags — `--staggered-start`,
`--num-rovers`, `--start-gap-s`, `--comms-enabled`, `--comms-packet-loss-prob`
and the `--inter-vehicle-collision-*` family. See
[experiments.md](experiments.md) for worked fleet commands.

---

## 11. Run configurations

`run.py` wraps the runners with a single JSON object describing *how* to run,
pointing at a track file describing *what* to run:

```jsonc
{
  "run": { "scenario": "single",        // single | fleet | benchmark
           "adapter": "holoocean", "allow_fallback": false,
           "headless": true, "official": true,
           "seed": 0, "dt": 0.033, "duration_s": 560 },
  "track": "marine_race_arena/tracks/marine_race_horseshoe_bay.json",
  "controller": { "name": "rule_gate_baseline" },
  "benchmark_task": "clean_gate",
  "obstacles": { "mode": "none" },
  "currents": { "profile": "none" },
  "output": { "log_dir": "results/marine_race" }
}
```

```bash
python run.py --dry-run              # resolve and print, launch nothing
python run.py                        # config.json
python run.py configs/fleet.json
python run.py configs/benchmark.json
```

`--dry-run` prints the exact runner invocation a configuration maps to, which is
the quickest way to check a new file.

---

## 12. Benchmark tasks

`--benchmark-task` declares what a track is being used for, and the validator
enforces it:

| Task | The validator requires |
|---|---|
| `clean_gate` | one participant, no configured currents, no active obstacles |
| `current_gate` | one participant, at least one configured current above the minimum speed |
| `obstacle_gate` | one participant, no currents, at least one active static obstacle |
| `multi_rov` | at least two participants |

This is why running a `current_gate` circuit current-free needs an explicit
`--benchmark-task clean_gate` override: the geometry, gates, laps and referee
stay identical, only the declared task changes.
