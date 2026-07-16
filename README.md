# Marine Race Arena

Marine Race Arena is a configurable benchmark layer for autonomous underwater
gate racing. Track configuration, participant autonomy and referee scoring are
separate concerns. Replaceable adapters isolate simulator-specific sensing and
actuation; the reference implementation uses HoloOcean with a BlueROV2-class
vehicle.

In official mode, controllers operate entirely from onboard information. The
referee uses privileged simulator state only to validate crossings, violations,
timing, ranking and team scores. Referee decisions are never returned to vehicle
autonomy.

The whole simulation is launched from one configuration file:

```bash
python run.py
python run.py configs/fleet.json
python run.py configs/benchmark.json
```

The paper sources live under [`article/`](article/). The final validation below
comes from 78 real-HoloOcean runs produced by the current onboard-only
implementation under one frozen source fingerprint.

## 1. Current scope

The current release provides:

- Three official tracks with unchanged `1.5 x 1.5 m` gate apertures.
- A real HoloOcean/BlueROV2 adapter for physical validation.
- An engine-free fallback adapter for unit tests and runner plumbing only.
- Independent acoustic transmitters `B01` through `BN`, one per ordered gate.
- A strict official controller contract based on local time, onboard sensors,
  received beacon packets and optional inter-rover messages.
- `LocalCourseTracker`, which estimates mission progress inside each controller.
- Two camera-assisted official controllers:
  `rule_gate_baseline` (continuous visual servoing) and
  `rule_gate_center_then_commit` (center, then commit through the aperture).
- Staggered multi-rover execution, independent referee state per rover and one
  team-level `team_summary`.
- `leader_follower`, which coordinates a fleet from controller-local progress
  estimates sent through the acoustic communication channel.
- Referee-side inter-vehicle proximity diagnostics and an optional penalty mode.
- `none`, `medium` and `strong` current profiles on every official track.
- Seeded-random obstacles verified on every official track; explicit fixed
  obstacle definitions are also supported by the schema.

Current compensation and obstacle avoidance are not claimed as solved. The
framework supports those scenarios and records their real outcomes.

## 2. Installation

The documented environment is Python 3.9 in a conda environment named `ocean`:

```bash
conda create -n ocean python=3.9 -y
conda activate ocean
pip install -r requirements.txt

# One-time HoloOcean world installation.
python -c "import holoocean; holoocean.install('Ocean')"
python -c "import holoocean; print(holoocean.installed_packages())"
```

Run commands from the repository root. Unit tests can use the fallback adapter;
article-facing and physical validation must use `--adapter holoocean` without
`--allow-fallback`.

## 3. Quick start

```bash
# Default configured run.
python run.py

# Resolve the configuration without launching HoloOcean.
python run.py --dry-run

# Ready-made fleet and benchmark configurations.
python run.py configs/fleet.json
python run.py configs/benchmark.json
```

`run.py` reads one JSON object, resolves its scenario and invokes the appropriate
runner.

## 4. Configuration

`config.json` is the documented default. The main run-level fields are:

```jsonc
{
  "run": {
    "scenario": "single",          // single | fleet | benchmark | smoke
    "adapter": "holoocean",        // holoocean | fallback | auto
    "allow_fallback": false,
    "headless": true,
    "record": false,
    "official": true,
    "seed": 0,
    "dt": 0.033,
    "duration_s": 560,
    "motion_compensation": "none",
    "gate_timeout_s": null
  },
  "track": "marine_race_arena/tracks/marine_race_horseshoe_bay.json",
  "controller": {
    "name": "rule_gate_baseline",
    "module_or_file": null,
    "class": null
  },
  "benchmark_task": "clean_gate",  // clean_gate | obstacle_gate | current_gate | multi_rov
  "obstacles": {
    "mode": "none",                // none | fixed | random
    "density": "medium",           // low | medium | high
    "physics": "static"            // static | dynamic
  },
  "currents": {
    "profile": "none"              // none | medium | strong
  },
  "sensors": {
    "disable_front_camera": false
  },
  "output": {
    "log_dir": "results/marine_race",
    "log_participant_states": false
  },
  "debug": {
    "show_front_camera": false,
    "print_beacons": false
  },
  "fleet": {
    "num_rovers": 2,
    "start_gap_s": 90.0,
    "lateral_offset_m": 3.0,
    "team_id": "fleet_01",
    "inter_vehicle_collision": {
      "mode": "diagnostic",
      "xy_threshold_m": 0.8,
      "z_threshold_m": 0.75,
      "release_threshold_m": 1.05,
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
  },
  "benchmark": {
    "seeds": [0, 1, 2],
    "output_dir": "results/benchmarks/config_run"
  },
  "smoke": {
    "wall_timeout_s": 900,
    "output_dir": "results/benchmarks/staggered_multi_rover_smoke"
  }
}
```

Track physics and geometry remain in the JSON files under
`marine_race_arena/tracks/`. Do not enlarge gates, relax referee margins or
change official layouts to recover a result.

## 5. Official onboard-only controller contract

### Static initialization

`reset(mission_info)` receives only the assigned mission:

```python
{
    "participant_id": "bluerov2_01",
    "initial_beacon_id": "B01",
    "total_beacons": 12,
    "laps": 1,
    "command_limits": {
        "surge": [-0.95, 0.95],
        "sway":  [-0.95, 0.95],
        "heave": [-0.95, 0.95],
        "yaw":   [-0.95, 0.95]
    }
}
```

For a fleet, initialization additionally contains one static block:

```python
{
    "fleet": {
        "participant_order": ["bluerov2_01", "bluerov2_02"],
        "release_index": 0,
        "predecessor_id": None
    }
}
```

### Per-step observation

`step(observation)` receives exactly these top-level fields:

```python
{
    "local_time_s": 2.145,
    "sensors": {
        "FrontCamera": ...,
        "DepthSensor": ...,
        "IMUSensor": ...,
        "DVLSensor": ...
    },
    "beacons": [
        {
            "beacon_id": "B01",
            "bearing_deg": ...,
            "elevation_deg": ...,
            "range_m": ...,
            "signal_strength": ...,
            "received_at_s": ...
        }
    ],
    "comms": {
        "inbox": [
            {
                "from": "bluerov2_01",
                "payload": ...,
                "received_at_s": ...
            }
        ]
    }
}
```

`comms` exists only when the inter-rover channel is enabled. `local_time_s` and
all reception timestamps are relative to that rover's release. The exact sensor
subset follows the official sensor profile; a contact sensor may be present as
an onboard measurement. Simulator pose, world-frame velocity, exact geometry,
configured current vectors and referee state remain outside the controller.

Every gate beacon transmits independently at its configured rate. Packets are
delivered only when physically in range and not dropped. Noise, scheduling and
dropout are seeded. A controller decides which beacon ID it currently expects.

Enable packet diagnostics with either:

```bash
conda run -n ocean python -m marine_race_arena.scripts.run_marine_race ... --print-beacons
```

or set `debug.print_beacons` to `true`. The printer displays only packets that
the controller physically received.

### Controller interface

A controller implements `reset(mission_info)`, `step(observation)` and `close()`.
Commands use normalized body-frame `surge`, `sway`, `heave` and `yaw` values.
External controllers can be loaded from a module or file without modifying the
runner:

```jsonc
"controller": {
  "module_or_file": "path/to/my_controller.py",
  "class": "MyController"
}
```

## 6. Local course progression and official controllers

`LocalCourseTracker` is the reusable controller-side progression component. It
starts from `initial_beacon_id` and advances through:

```text
SEARCH -> APPROACH -> VISUAL_ALIGN -> COMMIT -> VERIFY_EXIT -> ADVANCE
                                                               |
                                                    next beacon or FINISHED
```

It uses only participant-local time, received beacon packets, the forward
camera and DVL velocity.
Passage confirmation requires persistent visual alignment, DVL-integrated
forward displacement, a close beacon-range minimum followed by a range
turnaround, persistent disappearance of the aligned gate and fresh packets that
place the expected beacon behind the rover. Temporary camera loss, a single
dropout or an isolated range jump cannot advance local progress. The referee
scores independently and may disagree with the controller's estimate.

The two official camera-assisted controllers share this tracker:

- `rule_gate_baseline`: continuously corrects visual centering through passage.
- `rule_gate_center_then_commit`: establishes a stable visual lock, then holds a
  commit trajectory through the aperture.

`leader_follower` wraps an official gate-passing controller. Each rover sends
only this controller-local heartbeat:

```python
{
    "local_beacon_index": 4,
    "local_lap": 1,
    "local_status": "RUNNING"
}
```

The predecessor is assigned from the static release order. A follower yields
only while a fresh predecessor report shows less than the configured local gate
margin. Missing or stale reports are fail-open, so the wrapped base controller
continues. Latency, range, payload size, per-sender transmit interval, seeded
loss and stale-message handling remain active. With communication disabled, the
wrapper runs its base controller without coordination.

Manual `keyboard`, `pygame` and `pygame_keyboard` controllers remain available
for inspection and data collection. Debug controllers that use privileged state
are rejected in official mode.

## 7. Official tracks, currents and obstacles

| Track | File | Gates | Nominal duration |
| --- | --- | ---: | ---: |
| Horseshoe Bay | `marine_race_arena/tracks/marine_race_horseshoe_bay.json` | 12 | 560 s |
| Vertical Serpent | `marine_race_arena/tracks/marine_race_vertical_serpent.json` | 17 | 900 s |
| Mixed Endurance | `marine_race_arena/tracks/marine_race_mixed_endurance.json` | 22 | 1300 s |

All three tracks expose `none`, `medium` and `strong` current profiles. Current
runs are accepted as physical evidence only when metadata records the real
HoloOcean adapter, no fallback and active physical current coupling.

All three tracks can also use fixed obstacles from the track file or deterministic
random obstacles generated from the run seed. Random obstacle mode supports
`low`, `medium` and `high` density. A physical obstacle check is accepted only
with the effective HoloOcean adapter, no fallback, a positive requested count,
`physical_obstacle_spawn_complete=true` and a spawned count equal to the
requested count. These checks validate scenario construction; they do not claim
that a controller avoids the obstacles.

Configuration-only examples:

```bash
python -m marine_race_arena.scripts.validate_track_config --track marine_race_arena/tracks/marine_race_vertical_serpent.json --benchmark-task current_gate --current-profile medium
python -m marine_race_arena.scripts.validate_track_config --track marine_race_arena/tracks/marine_race_mixed_endurance.json --benchmark-task obstacle_gate --obstacles random --obstacle-density medium --obstacle-physics static --current-profile none --seed 0
```

## 8. Referee, fleet scoring and outputs

The referee checks ordered gate crossings from privileged simulator state,
applies penalties and produces official status, timing and ranking. None of that
state is used for controller navigation.

Fleet mode runs one cooperative team. Each rover has independent controller and
referee state. `team_summary` aggregates expected and completed gates, finish
status, elapsed and penalized time, gate/obstacle contacts and inter-vehicle
events. Inter-vehicle modes are `off`, `diagnostic` and `penalize`; diagnostic is
the validation default.

Every race execution writes:

- `*_summary.json` with referee results and, in fleet mode, `team_summary`;
- `*.jsonl` structured events.

The official tracker-based controllers additionally write controller-local
progression diagnostics, kept separate from referee truth. Controllers without
a tracker, such as manual inputs, need not emit those diagnostics.

The repeated benchmark and coordination wrappers additionally write per-run
metadata recording adapter, source fingerprint, command, seed, current coupling
and obstacle-spawn status. A direct `run_marine_race` invocation does not create
that wrapper metadata.

## 9. Validation lifecycle

The release workflow is:

1. Compile and run the simulator-independent test suite.
2. Run focused real-HoloOcean progression smoke tests.
3. Freeze the implementation and start a fresh result directory.
4. Execute the complete HoloOcean matrix without fallback.
5. Validate provenance, artifact completeness and local-versus-referee progress.
6. Report every result, including collisions, timeouts, DNFs and mismatches.

Standard checks:

```bash
python -m compileall -q marine_race_arena tests run.py validate_final_matrix.py summarize_final_results.py
conda run -n ocean python -m pytest -q
conda run -n ocean python -m marine_race_arena.scripts.run_staggered_multi_rover_smoke
```

Example clean and current sweeps:

```bash
conda run -n ocean python -m marine_race_arena.scripts.run_benchmark --benchmark-task clean_gate --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --controller rule_gate_baseline --adapter holoocean --seeds 0 1 2 3 4 --duration 560 --dt 0.033 --obstacles none --current-profile none --motion-compensation none --output-dir results/onboard_only_validation/final_20260715/clean/horseshoe/rule_gate_baseline --official

conda run -n ocean python -m marine_race_arena.scripts.run_benchmark --benchmark-task current_gate --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --controller rule_gate_center_then_commit --adapter holoocean --seeds 0 1 2 3 4 --duration 560 --dt 0.033 --obstacles none --current-profile medium --motion-compensation none --output-dir results/onboard_only_validation/final_20260715/currents/horseshoe/rule_gate_center_then_commit/medium --official
```

Example coordination sweep for one start gap:

```bash
conda run -n ocean python -m marine_race_arena.scripts.run_holoocean_coordination_validation --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --team-size 3 --start-gap-s 8 --lateral-offset-m 1.5 --seeds 0 1 2 --inter-vehicle-modes diagnostic --conditions no_coordination leader_follower --min-gate-gap 2 --comms-packet-loss-prob 0 --duration-s 560 --dt 0.033 --headless --log-participant-states --output-dir results/onboard_only_validation/final_20260715/coordination/main/gap_8
```

Audit a completed fresh result root with:

```bash
python -m marine_race_arena.scripts.validate_onboard_only_results results/onboard_only_validation/final_20260715
python validate_final_matrix.py results/onboard_only_validation/final_20260715
python summarize_final_results.py results/onboard_only_validation/final_20260715
```

The first command validates every discovered artifact and local/referee
progress pair. The second enforces the exact 78-run manifest, seed coverage,
matched condition fields, absence of duplicate/unexpected runs and one frozen
source fingerprint. The third generates the 78-row run manifest, the 124-row
participant manifest, aggregate metrics and the local-versus-referee timing
analysis directly from each run's referee summary and event log.

## 10. Current 78-run protocol

The matrix is complete. Exact coverage is **78/78 runs** and **124 participant
rows** under source fingerprint
`e7d3107784ea53056febcc3966b267ef59ee6d0f24d523c5c9b9446efca044b8`.
There are no execution failures or artifact-contract failures. The independent
progress audit nevertheless fails scientifically in 15 runs: it records 11
false local advancements and 7 missed local advancements rather than repairing
them. Across the matrix, 58 runs are referee-finished and 46 are clean.

| Family | Conditions | Runs |
| --- | --- | ---: |
| Clean single rover | 3 tracks x 2 official controllers x seeds 0-4 | 30 |
| Current single rover | Horseshoe x medium/strong x 2 controllers x seeds 0-4 | 20 |
| Two-rover fleet | Horseshoe, 90 s gap x 2 homogeneous controller choices x seeds 0-4 | 10 |
| Coordination | 8/0 s gap x coordinated/uncoordinated x seeds 0-2 | 12 |
| Yield-margin ablation | 8/0 s gap x `min_gate_gap=1` x seeds 0-2 | 6 |
| **Total** |  | **78** |

The principal measured outcomes are:

| Case | Continuous servo | Center-then-commit |
| --- | --- | --- |
| Horseshoe clean | 5/5 finished, 12.0 gates, 194.456 +/- 4.286 s | 5/5 finished, 12.0 gates, 197.683 +/- 6.312 s |
| Vertical clean | 5/5 finished, 17.0 gates, 236.947 +/- 5.745 s | 4/5 finished, 14.8 gates, 241.139 +/- 1.567 s |
| Mixed clean | 2/5 finished, 17.0 gates, 394.746 +/- 2.772 s | 5/5 finished, 22.0 gates, 410.659 +/- 7.850 s |
| Horseshoe medium current | 2/5 finished, 8.4 gates, 43.4 GW/run | 3/5 finished, 8.8 gates, 25.2 GW/run |
| Horseshoe strong current | 0/5 finished, 3.0 gates | 0/5 finished, 3.2 gates |
| Homogeneous fleet, 90 s gap | 5/5 teams, 303.092 +/- 2.874 s | 5/5 teams, 307.976 +/- 5.227 s |

Times are mean +/- population standard deviation over finished runs only; fleet
times are team-level. All clean and 90 s fleet runs have zero gate/world (GW),
inter-vehicle, out-of-bounds and stuck events. Under medium current, finished-only
official/penalized times are 337.887 +/- 13.563 s / 692.887 +/- 53.563 s for
continuous servo and 217.261 +/- 1.229 s / 223.928 +/- 1.504 s for
center-then-commit. Neither controller finishes a strong-current seed.

Primary three-rover coordination (`min_gate_gap=2`) finishes all three coordinated
runs at each gap with zero GW and inter-vehicle events. At an 8 s gap, the
uncoordinated condition finishes 2/3 teams, averages 34.3/36 gates, 127.3 GW and
2.0 inter-vehicle events per run; the coordinated condition finishes 3/3 at
285.725 +/- 3.129 s. At simultaneous release, both conditions finish 3/3;
uncoordinated runs average 9.3 GW and 2.3 inter-vehicle events, while coordinated
runs eliminate both but record one stuck event per run and 303.310 +/- 1.076 s
penalized team time. The `min_gate_gap=1` ablation finishes 3/3 cleanly at both
gaps, with mean team times 266.024 +/- 6.765 s (8 s) and
262.339 +/- 6.287 s (0 s).

Separate physical capability checks pass both named current profiles on all three
official tracks and seeded static-obstacle spawning on all three tracks. They do
not claim obstacle-avoidance performance.

Coordination uses a continuous-servo leader and two center-then-commit followers,
with no currents or obstacles, `min_gate_gap=2` for the primary condition and
diagnostic inter-vehicle mode. The yield-margin ablation runs only the coordinated
condition. No additional communication-loss sweep is assigned a numerical claim.

## 11. Project layout

```text
run.py                 configuration-driven entry point
validate_final_matrix.py exact 78-run coverage and artifact audit
summarize_final_results.py manifest, aggregates and progression timing report
config.json            default configuration
configs/               fleet, benchmark and viewing examples
marine_race_arena/
  adapters/            HoloOcean and engine-free test adapters
  arena/               gates, beacons, currents, obstacles and comms
  config/              schema, loader, validation and benchmark tasks
  controllers/         local tracker, official controllers and coordination
  participants/        controller loading and sensor filtering
  referee/             crossing validation, penalties, logs and team score
  scripts/             runners, smoke checks and artifact validation
  tracks/              three official tracks and focused test tracks
tests/                 simulator-independent tests
article/               paper sources
docs/                  release notes
```

## 12. Known limitations

- Current rejection is not solved and must be evaluated from the real outcomes.
- Random-obstacle construction is supported; obstacle avoidance is not validated.
- Dense uncoordinated fleets may collide or fail, and those outcomes are valid.
- Inter-vehicle penalty calibration remains experimental; diagnostic mode is the
  default for validation.
- The fallback adapter is not physical evidence.
- HoloOcean startup and repeated long-track sweeps can be slow.

## 13. Paper

Build the paper with:

```bash
cd article
latexmk -pdf -interaction=nonstopmode main.tex
```
