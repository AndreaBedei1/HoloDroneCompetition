# Marine Race Arena

Marine Race Arena is a HoloOcean-based underwater gate-racing benchmark for
BlueROV2-class vehicles. A race — world bounds, ordered gates, currents,
obstacles, participants, sensors, and referee rules — is described declaratively,
and a controller is plugged in by implementing three methods. The autonomy stack
is kept strictly separate from the evaluation: a controller observes only a
documented **official observation**, while an independent referee uses privileged
state to validate gate crossings, collisions, timeouts, ranking, and team scoring.

The whole simulation is launched from a **single configuration file**:

```bash
python run.py                 # uses ./config.json
python run.py configs/fleet.json
```

A companion paper describing the architecture, evaluation, and results lives in
[`article/`](article/) (`article/main.pdf`).

---

## 1. v0.1 scope

Claimed and reproducible in this release:

- Official single-rover clean-gate benchmark on three official tracks
  (1.5 m × 1.5 m gate apertures).
- HoloOcean BlueROV2 integration plus a simulator-independent kinematic fallback
  adapter for unit tests and plumbing.
- `rule_gate_baseline`, a deterministic acoustic-beacon + front-camera controller.
- Custom controller loading by alias, module, or file path.
- Staggered multi-rover evaluation of one cooperative team (coordination and course-completion speed), aggregated into a single `team_summary`.
- Inter-vehicle collision **diagnostics** with an optional penalty mode.

Identified as open future work, **not** part of the v0.1 results: current
compensation, close-proximity fleet racing, and fully calibrated rover-rover
collision penalties.

---

## 2. Installation

The framework runs in a Python 3.9 conda environment (named `ocean` throughout
this README). From scratch:

```bash
# 1. Create and activate the environment
conda create -n ocean python=3.9 -y
conda activate ocean

# 2. Install the Python dependencies
pip install -r requirements.txt

# 3. Download the HoloOcean "Ocean" world package (one-time, large download)
python -c "import holoocean; holoocean.install('Ocean')"

# 4. Verify the engine is available
python -c "import holoocean; print(holoocean.installed_packages())"   # expect ['Ocean']
```

Run all commands from the repository root. The kinematic fallback adapter has no
HoloOcean dependency, so the unit tests and config plumbing run without the
engine (set `adapter: fallback`); only real HoloOcean runs need step 3. Building
the paper additionally needs a LaTeX toolchain (see Section 12).

---

## 3. Quick start

```bash
# Default official single-rover run on Horseshoe Bay (reads config.json)
python run.py

# Preview the resolved scenario and arguments without launching
python run.py --dry-run

# Pick a different configuration
python run.py configs/fleet.json
python run.py configs/benchmark.json
```

`run.py` reads one JSON config, selects a scenario, and dispatches to the
appropriate runner. When the configured `adapter` is `holoocean`, a default
`python run.py` launches the real simulator run.

---

## 4. Configuration file

The configuration is a single JSON object. `config.json` is the documented
default; `configs/` holds ready-made fleet and benchmark examples. The
`run.scenario` field selects what is launched:

| Scenario | Launches | Purpose |
| --- | --- | --- |
| `single` | one rover on a track | official single-rover benchmark |
| `fleet` | staggered rovers, one team | fleet/team evaluation |
| `benchmark` | repeated single-rover trials | multi-seed sweep |
| `smoke` | staggered multi-rover smoke | release plumbing check |

Fields (all optional fall back to runner defaults):

```jsonc
{
  "run": {
    "scenario": "single",          // single | fleet | benchmark | smoke
    "adapter": "holoocean",        // holoocean | fallback | auto
    "allow_fallback": false,       // permit fallback if HoloOcean fails (auto)
    "headless": true,
    "record": false,
    "official": true,              // official sensor/timing mode (no ground truth)
    "seed": 0,
    "dt": 0.033,                   // control timestep (~30 Hz with HoloOcean)
    "duration_s": 560,             // max race duration
    "motion_compensation": "none", // only "none" ships; current compensation is future work
    "gate_timeout_s": null
  },
  "track": "marine_race_arena/tracks/marine_race_horseshoe_bay.json",
  "controller": {
    "name": "rule_gate_baseline",  // built-in alias, OR:
    "module_or_file": null,        // "path/to/ctrl.py" or "pkg.mod:Class"
    "class": null                  // class name when loading from a file
  },
  "benchmark_task": "clean_gate",  // clean_gate | obstacle_gate | current_gate | multi_rov
  "obstacles": { "mode": "none", "density": "medium", "physics": "static" },
  "currents":  { "profile": "none" },           // none | medium | strong
  "sensors":   { "disable_front_camera": false },
  "output":    { "log_dir": "results/marine_race", "log_participant_states": false },
  "debug":     { "show_front_camera": false, "print_beacon_targets": false },
  "fleet": {                                     // used when scenario == fleet/smoke
    "num_rovers": 2,
    "start_gap_s": 90.0,
    "lateral_offset_m": 3.0,
    "team_id": "fleet_01",
    "inter_vehicle_collision": {
      "mode": "diagnostic",                      // off | diagnostic | penalize
      "xy_threshold_m": 0.8, "z_threshold_m": 0.75,
      "release_threshold_m": null, "cooldown_s": 1.0
    },
    "comms": {                                   // optional inter-rover acoustic channel
      "enabled": false,                          // off by default
      "sound_speed_m_s": 1500.0, "max_range_m": 100.0,
      "processing_delay_s": 0.05, "packet_loss_prob": 0.0,
      "max_payload_bytes": 128, "min_send_interval_s": 0.5
    }
  },
  "benchmark": { "seeds": [0, 1, 2], "output_dir": "results/benchmarks/config_run" },
  "smoke":     { "wall_timeout_s": 900, "output_dir": "results/benchmarks/staggered_multi_rover_smoke" }
}
```

Do not enlarge official gate sizes or change official track geometry when
reporting official benchmark results.

### Track files

Per-track physics and geometry remain in the track JSON under
`marine_race_arena/tracks/`. The run-level configuration above only selects which
track to run and how. Track JSON sections: `race`, `world`, `start`/`finish`,
`track`/`gates`, `beacon`, `currents`/`current_profiles`,
`obstacle_generation`/`obstacles`, `participants` (incl. `start_delay_s`), and
`referee` (gate validation, penalties, scoring). `create_best_tracks.py`
regenerates the three official tracks and is kept for provenance.

---

## 5. Writing a controller

A controller implements three methods. Subclassing a base class is optional; any
object exposing these callables is accepted.

```python
class MyController:
    def reset(self, race_info):
        self.target = race_info.get("initial_target_gate_id")

    def step(self, observation):
        beacon = observation.get("beacon", {})
        race   = observation.get("race", {})
        sensors = observation.get("sensors", {})
        yaw = 0.0
        if beacon.get("valid") and beacon.get("bearing_deg") is not None:
            yaw = max(-0.2, min(0.2, beacon["bearing_deg"] / 90.0))
        return {"surge": 0.25, "sway": 0.0, "heave": 0.0, "yaw": yaw}

    def close(self):
        pass
```

Load it from the config without touching the package:

```jsonc
"controller": { "module_or_file": "path/to/my_controller.py", "class": "MyController" }
// or
"controller": { "module_or_file": "my_pkg.my_controller:MyController" }
```

**Command** (`step` return), each clamped to `[-1, 1]`:

| Field | Meaning |
| --- | --- |
| `surge` | forward/back body-frame thrust |
| `sway` | lateral body-frame thrust |
| `heave` | vertical thrust (depth-safety constrained near bounds) |
| `yaw` | yaw-rate command |

**Observation** (`step` argument): a dict with `time_s`, `participant_id`,
`sensors`, `beacon`, and `race`. The `sensors` dict follows the participant
profile (front camera, depth, IMU, DVL/velocity, collision, derived heading and
depth). The `beacon` dict carries `valid`, `target_gate_id`, `bearing_deg`,
`elevation_deg`, `range_m`, `signal_strength`, and `mode`. The `race` dict carries
`status`, `lap`, `completed_gates`, `target_gate_id`, `target_sequence_index`, and
`official_time_started`. In official mode, ground-truth pose/location/rotation
sensors are filtered out, and the true environment current vector
(`environment_current_m_s`) is stripped from the observation entirely (it is
available only as non-official diagnostic telemetry). A controller must infer
current effects from onboard sensing (e.g. the DVL/velocity residual).

When the optional inter-rover acoustic channel is enabled (`fleet.comms`), a
controller may broadcast by returning a small `"message"` payload alongside its
command, and receives an `observation["comms"]["inbox"]` of teammates' messages.
The channel models the underwater acoustic medium (range-dependent latency,
limited range, packet loss, tiny payloads, half-duplex rate limit), not a perfect
link, and is off by default. Message content/handling is up to your controller;
because payloads are authored by controllers, they carry only legally observable
information (the channel never injects ground-truth state).

Built-in controller aliases: `rule_gate_baseline`, `acoustic_baseline`,
`acoustic_vision_baseline`, `vision_gate_baseline`, `student_template`,
`keyboard`/`manual`, `pygame`, and the debug-only `oracle` (blocked in official
mode).

---

## 6. Official tracks

All tracks use `gate_inner_size_m = [1.5, 1.5]` and
`timing_mode = first_gate_to_last_gate`.

| Track | File | Gates | Length | Purpose |
| --- | --- | ---: | ---: | --- |
| Horseshoe Bay | `marine_race_arena/tracks/marine_race_horseshoe_bay.json` | 12 | 93.8 m | Clean-gate baseline, fleet demo |
| Vertical Serpent | `marine_race_arena/tracks/marine_race_vertical_serpent.json` | 17 | 118.3 m | Vertical/slalom sequencing |
| Mixed Endurance | `marine_race_arena/tracks/marine_race_mixed_endurance.json` | 22 | 206.3 m | Endurance / current-oriented |

Run another track by editing `"track"` in the config (set `duration_s` to roughly
`560` / `900` / `1300` for the three tracks). Mixed Endurance also defines
`medium`/`strong` current profiles, but current robustness is experimental in v0.1.

---

## 7. Referee and scoring

The referee uses privileged simulator state and is independent of the controller.
A gate crossing is valid when the segment between consecutive positions crosses
the gate plane in the direction of the normal and the intersection lies inside the
aperture (shrunk by a safety margin). Gates must be passed in sequence; crossing a
non-target gate is a missed-gate event (DNF by default). Collisions, out-of-bounds,
and stuck conditions accrue time penalties without terminating. The single-rover
score is `penalized_time = official_time + penalties`; finished rovers rank by
penalized time, unfinished rovers by progress.

Fleet mode runs several rovers as a single cooperative team (not competitors); the goal is to coordinate and complete the course quickly, and there is one team only. Each rover keeps independent state; the official result is the
`team_summary` (total gates, total penalties, and elapsed time from first release
to last finish). Inter-vehicle proximity is detected on the referee side
(`off`/`diagnostic`/`penalize`); `diagnostic` is the recommended default until the
contact thresholds are empirically calibrated.

---

## 8. Outputs

Each run writes to the configured `log_dir`:

- `*_summary.json` — per-rover summaries, ranking, and `team_summary` in fleet mode.
- `*.jsonl` — one structured event per line (`gate_passed`, `collision`,
  `out_of_bounds`, `stuck`, `race_finish`, `dnf`, `inter_vehicle_collision`, …).
- Benchmark and smoke runners additionally write CSV/markdown aggregates.

`results/`, `diagnostics/`, logs, and recordings are git-ignored; do not commit
generated artifacts.

---

## 9. Project layout

```text
run.py                 single config-driven entry point
config.json            default run configuration
configs/               example fleet and benchmark configurations
marine_race_arena/
  adapters/            HoloOcean and fallback simulator adapters
  arena/               gates, bounds, beacons, currents, obstacles
  config/              dataclasses, JSON loader, validation, benchmark tasks
  controllers/         baselines, manual controllers, student template
  participants/        participant model, sensor filtering, controller loader
  referee/             gate validation, race state, scoring, logging, team summary
  scripts/             run_marine_race, run_benchmark, smoke, release checks
    diagnostics/       one-off diagnostic and calibration tools
  tracks/              official tracks and small validation fixtures
tests/                 simulator-independent pytest suite
article/               IEEE paper sources and compiled main.pdf
docs/                  release notes
```

---

## 10. Tests and release checks

```bash
python -m compileall -q marine_race_arena tests run.py
conda run -n ocean python -m pytest -q
conda run -n ocean python -m marine_race_arena.scripts.run_staggered_multi_rover_smoke
```

The release-check helper prints or runs the standard checks:

```bash
python -m marine_race_arena.scripts.run_release_v0_1_checks        # print
python -m marine_race_arena.scripts.run_release_v0_1_checks --run  # execute
```

The inter-vehicle collision calibration tool now lives under the diagnostics
subpackage:

```bash
conda run -n ocean python -m marine_race_arena.scripts.diagnostics.calibrate_inter_vehicle_collision_threshold --quick
```

---

## 11. Known limitations

- Inter-vehicle collision thresholds are conservative defaults
  (`xy=0.8 m`, `z=0.75 m`, `release=1.05 m`, `cooldown=1.0 s`) and not yet
  empirically calibrated; `diagnostic` mode is recommended, `penalize` is experimental.
- Close-proximity fleet racing is not validated; the fleet demo uses a 90 s gap.
- Current compensation is an open problem. Under disturbance currents the rule
  baseline degrades: on Horseshoe Bay it still finishes the `medium` profile but
  with gate contacts (12/12, 8 contacts, +40 s), and it does not finish the
  `strong` profile (3/12). Designing a controller that rejects the current from
  the legal observation is left to future work.
- The fallback adapter is kinematic, not a physical simulator.
- HoloOcean loading can be slow; the debug `oracle` controller is blocked in official mode.

---

## 12. Paper

The architecture, evaluation protocol, gate-validation and scoring formalism,
fleet aggregation, and results are described in the paper under
[`article/`](article/). Build it with:

```bash
cd article && latexmk -pdf -interaction=nonstopmode main.tex
```
