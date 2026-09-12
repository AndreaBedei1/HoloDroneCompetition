# Marine Race Arena

Marine Race Arena is a configurable, reproducible benchmark for autonomous
underwater gate racing. Track configuration, participant autonomy and referee
scoring are separate concerns: a controller operates entirely from onboard
information, while an independent referee uses privileged simulator state only
to validate ordered gate crossings, enforce the rules and compute the official
result. Referee state is never returned to vehicle autonomy.

The reference backend is HoloOcean 2.3.0 with a BlueROV2-class vehicle.

This repository is the release that accompanies the Marine Race Arena
manuscript in [`article_journal/`](article_journal/). Every number reported in
that paper is recomputed from the evidence package in
[`artifacts/paper/`](artifacts/paper/) by a single command
([Verifying the paper](#11-verifying-the-paper)).

## 1. Features

- Declarative race configuration: world bounds, ordered gate sequence with
  centres, passage directions and aperture sizes, acoustic beacon network and
  its noise model, current profiles, obstacle generation, participants with
  their vehicles, sensors, controllers and release delays, and the referee's
  validation margins, penalties and scoring rules.
- Three official circuits with unchanged `1.5 x 1.5 m` gate apertures.
- A strict participant-level information boundary: local time, onboard sensors,
  received acoustic packets and optional inter-vehicle messages.
- Onboard visual-acoustic gate perception and a controller-local course tracker.
- Two interpretable rule-based reference controllers.
- Staggered multi-vehicle execution, independent per-vehicle referee state, and
  team-level aggregation and ranking.
- Distributed leader--follower coordination over the acoustic channel.
- A frozen recurrent-PPO reference controller with its validation episodes.
- Structured per-run logging and automated artifact validation.

Current rejection and obstacle avoidance are not claimed as solved. The
framework supports those scenarios and records their real outcomes.

## 2. Installation

The documented environment is Python 3.9 in a conda environment named `ocean`.
HoloOcean 2.3.0 is **not on PyPI** (`pip install holoocean==2.3.0` fails; PyPI
only ships 0.5.8), so it is installed from its official source before the
pinned Python dependencies.

```bash
conda create -n ocean python=3.9 -y
conda activate ocean

# 1. HoloOcean 2.3.0 client, from the official source
#    (https://github.com/byu-holoocean/HoloOcean). This also pulls numpy,
#    scipy and matplotlib.
cd <HoloOcean-2.3.0 source>/client && pip install .

# 2. Pinned runtime dependencies (run from this repository root).
pip install -r requirements.txt

# 3. One-time world installation. Worlds are stored per HoloOcean version
#    under the user profile and shared across environments.
python -c "import holoocean; holoocean.install('Ocean')"
python -c "import holoocean; print(holoocean.installed_packages())"   # -> ['Ocean']
```

Optional dependency sets:

```bash
pip install -r requirements-rl.txt    # load and evaluate the learned controller
pip install -r requirements-dev.txt   # test suite and figure regeneration
```

Run every command from the repository root. The test suite and the engine-free
fallback adapter need only step 2; simulator evidence requires step 1 and must
use `--adapter holoocean` without `--allow-fallback`.

## 3. Quick start

```bash
python run.py                      # default single-vehicle official run
python run.py --dry-run            # resolve the configuration, launch nothing
python run.py configs/fleet.json   # two-vehicle staggered fleet
python run.py configs/benchmark.json
```

`run.py` reads one JSON object, resolves its scenario (`single`, `fleet` or
`benchmark`) and invokes the matching runner. `config.json` is the documented
default; `configs/` holds ready-made fleet and benchmark configurations.

A direct invocation is equivalent:

```bash
python -m marine_race_arena.scripts.run_marine_race \
  --track marine_race_arena/tracks/marine_race_horseshoe_bay.json \
  --benchmark-task clean_gate \
  --controller rule_gate_center_then_commit \
  --adapter holoocean --seed 0 \
  --duration 560.0 --current-profile none --official
```

## 4. Official circuits

| Circuit | File | Gates | Length | Nominal duration |
| --- | --- | ---: | ---: | ---: |
| Horseshoe Bay | `marine_race_arena/tracks/marine_race_horseshoe_bay.json` | 12 | 93.8 m | 560 s |
| Vertical Serpent | `marine_race_arena/tracks/marine_race_vertical_serpent.json` | 17 | 118.3 m | 900 s |
| Mixed Endurance | `marine_race_arena/tracks/marine_race_mixed_endurance.json` | 22 | 206.3 m | 1300 s |

All three expose `none`, `medium` and `strong` current profiles, and support
fixed obstacles from the track file or deterministic random obstacles generated
from the run seed. A configuration can be validated without launching the
simulator:

```bash
python -m marine_race_arena.scripts.validate_track_config \
  --track marine_race_arena/tracks/marine_race_vertical_serpent.json \
  --benchmark-task current_gate --current-profile medium
```

## 5. Observation and information boundary

`reset(mission_info)` gives a controller only its assigned mission:

```python
{
    "participant_id": "bluerov2_01",
    "initial_beacon_id": "B01",
    "total_beacons": 12,
    "laps": 1,
    "command_limits": {"surge": [-0.95, 0.95], "sway": [-0.95, 0.95],
                       "heave": [-0.95, 0.95], "yaw": [-0.95, 0.95]},
    # fleet runs add a static block:
    "fleet": {"participant_order": ["bluerov2_01", "bluerov2_02"],
              "release_index": 0, "predecessor_id": None},
}
```

`step(observation)` receives exactly these top-level fields:

```python
{
    "local_time_s": 2.145,
    "sensors": {"FrontCamera": ..., "DepthSensor": ..., "IMUSensor": ..., "DVLSensor": ...},
    "beacons": [{"beacon_id": "B01", "bearing_deg": ..., "elevation_deg": ...,
                 "range_m": ..., "signal_strength": ..., "received_at_s": ...}],
    "comms": {"inbox": [{"from": "bluerov2_01", "payload": ..., "received_at_s": ...}]},
}
```

`comms` exists only when the inter-vehicle channel is enabled. `local_time_s`
and all reception timestamps are relative to that vehicle's release. Simulator
pose, world-frame velocity, exact gate geometry, configured current vectors and
referee state stay outside the controller. Every gate beacon transmits
independently; packets arrive only when physically in range and not dropped,
and noise, scheduling and dropout are seeded by the run seed, the transmitter,
the receiver and the transmission index.

## 6. Controller interface

A controller implements `reset(mission_info)`, `step(observation)` and
`close()`, and returns normalized body-frame `surge`, `sway`, `heave` and `yaw`
commands. It is selected by built-in alias, by dotted module path with an
explicit class, or by file path plus class name -- no change to the package:

```jsonc
"controller": { "module_or_file": "path/to/my_controller.py", "class": "MyController" }
```

Built-in aliases: `rule_gate_baseline`, `rule_gate_center_then_commit`,
`leader_follower`, `student_template`, the manual `keyboard` / `pygame`
controllers, and the debug-only `oracle` (rejected in official mode).

`LocalCourseTracker` is the reusable controller-side progression component. It
starts from `initial_beacon_id` and advances through

```text
SEARCH -> APPROACH -> VISUAL_ALIGN -> COMMIT -> VERIFY_EXIT -> ADVANCE
                                                               |
                                                    next beacon or FINISHED
```

using only participant-local time, received beacon packets, the forward camera
and DVL velocity. Passage confirmation needs persistent visual alignment,
DVL-integrated forward displacement, a close beacon-range minimum followed by a
range turnaround, persistent disappearance of the aligned gate, and fresh
packets placing the expected beacon behind the vehicle. The referee scores
independently and may disagree with this estimate.

### Continuous Servo

`rule_gate_baseline` keeps correcting visual centring all the way through the
passage. It is the stronger controller where approaches are well conditioned.

### Center-then-Commit

`rule_gate_center_then_commit` establishes a stable visual lock first and then
holds a commit trajectory through the aperture, so a late image-centroid jump
caused by partial near-field contours cannot deflect the vehicle when the
aperture margin is smallest. The two controllers differ in exactly this one
respect and share observations, guidance, tracker and confirmation logic.

## 7. Fleet and team evaluation

Fleet mode runs one cooperative team: each vehicle has independent controller
and referee state, and `team_summary` aggregates expected and completed gates,
finish status, elapsed and penalized time, gate and obstacle contacts and
inter-vehicle events. Inter-vehicle proximity modes are `off`, `diagnostic`
(the validation default) and `penalize`.

`leader_follower` wraps a gate-passing controller and coordinates from
controller-local progress alone. Each vehicle broadcasts only

```python
{"local_beacon_index": 4, "local_lap": 1, "local_status": "RUNNING"}
```

over the acoustic channel, with its range-dependent latency and seeded loss.
The predecessor comes from the static release order; a follower yields only
while a fresh predecessor report shows less than the configured local gate
margin, and missing or stale reports are fail-open. `LF(1)` -- a one-gate
margin -- is the recommended default.

```bash
python -m marine_race_arena.scripts.run_holoocean_coordination_validation --min-gate-gap 1
```

## 8. Learned reference controller

The released policy is a recurrent PPO controller over the 27-D onboard
observation encoding, with the 4-D body-frame action interface, evaluated
through the same information boundary and referee as the rule-based
controllers.

```text
artifacts/paper/ppo/model/policy_recurrent_ppo_27d.zip   frozen checkpoint
artifacts/paper/ppo/provenance.json                      identity, hashes, contracts
artifacts/paper/ppo/campaign_config.json                 training configuration
artifacts/paper/ppo/validation/                          the validation episodes
```

Load and evaluate it (needs `requirements-rl.txt`):

```bash
python -m marine_race_arena.learning.gen2.evaluate_speed_robust \
  --controller retry6_50k --track horseshoe_bay \
  --checkpoint artifacts/paper/ppo/model/policy_recurrent_ppo_27d.zip \
  --seed 20370909 --out results/ppo_eval
```

Training code is not part of this release; the checkpoint is frozen.

## 9. Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

The suite is simulator-independent: it uses the engine-free fallback adapter
and the released artifacts, and never launches HoloOcean. Tests that need the
RL stack are skipped unless `requirements-rl.txt` is installed.

## 10. Released evidence

```text
artifacts/paper/
  manifest.json       one row per file: purpose, size, SHA-256, what it supports
  benchmark/          per-run rows of the reported benchmark matrix
  current_free/       the nine current-free onboard-only runs
  ppo/                the frozen policy, its configuration and validation
  perception/         perception audit metrics and figure source captures
```

`manifest.json` is the index; every file in the package is listed with its
SHA-256, and `verify_claims.py` fails if any of them changes.

## 11. Verifying the paper

```bash
python article_journal/scripts/verify_claims.py        # every reported quantity
python article_journal/scripts/regenerate_tables.py    # the data-driven tables
python article_journal/scripts/generate_figures.py     # track layouts, controller plot
python article_journal/scripts/make_perception_figure.py
```

All four are post-processing only: they read `artifacts/paper/`, launch no
simulator and modify no artifact. `verify_claims.py` exits non-zero on any
mismatch; `regenerate_tables.py` rewrites the tables byte-identically and
re-checks the penalty identity on every finished run.

## 12. Building the paper

```bash
cd article_journal
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

`article_journal/main.tex` is the only manuscript source and
`article_journal/main.pdf` the compiled manuscript.

## 13. Known limitations

- Current rejection is not solved; the reported outcomes are the real ones.
- Random-obstacle construction is supported; obstacle avoidance is not validated.
- Dense uncoordinated fleets may collide or fail, and those outcomes are valid.
- Inter-vehicle penalty calibration remains experimental.
- The fallback adapter is plumbing for tests, not physical evidence.
- Results are simulation results; physical validation is not claimed.

## 14. Citation

```bibtex
@article{marine_race_arena,
  title   = {Marine Race Arena: A Configurable HoloOcean Benchmark for
             Underwater Gate Racing and Team-Level Fleet Evaluation},
  author  = {Bedei, Andrea and Bacchiani, Lorenzo and Pau, Giovanni and Girau, Roberto},
  journal = {Robotics and Autonomous Systems},
  note    = {Under review},
  year    = {2026}
}
```
