# Getting started

This guide takes you from an empty machine to a finished race.

Everything runs from the repository root. Commands are shown for a POSIX-style
shell; on Windows they work unchanged in Git Bash, and in PowerShell you only
need to replace the trailing `\` line continuations with backticks.

---

## 1. Prerequisites

| | |
|---|---|
| **Python** | 3.9 — the version the benchmark and the reported results were verified with. |
| **HoloOcean** | 2.3.0, installed from source (see step 3). Needed for any real run. |
| **Disk** | ~6 GB for the HoloOcean `Ocean` world package. |
| **OS** | Developed and validated on Windows; the Python package itself is platform-neutral. |

Only the HoloOcean client is unusual. The rest is a normal pinned `pip` install.

> You can run the full test suite, validate track configurations and reproduce
> every number in the paper **without HoloOcean**. You need it only to fly races.

---

## 2. Clone and create the environment

```bash
git clone https://github.com/AndreaBedei1/HoloDroneCompetition.git
cd HoloDroneCompetition

conda create -n ocean python=3.9 -y
conda activate ocean
```

---

## 3. Install the HoloOcean 2.3.0 client

**HoloOcean 2.3.0 is not on PyPI.** `pip install holoocean==2.3.0` fails, because
PyPI only ships 0.5.8. Get the 2.3.0 source from the
[official repository](https://github.com/byu-holoocean/HoloOcean) (the full
instructions live in the [HoloOcean docs](https://byu-holoocean.github.io/holoocean-docs))
and install its `client` package:

```bash
cd <HoloOcean-2.3.0 source>/client && pip install .
```

This pulls in numpy, scipy and matplotlib unpinned. Step 4 fixes the numpy
version afterwards, which is why the order matters.

---

## 4. Install the pinned dependencies

Back in the repository root:

```bash
pip install -r requirements.txt
```

Two optional sets:

```bash
pip install -r requirements-rl.txt    # to load and evaluate the learned controller
pip install -r requirements-dev.txt   # to run the tests and regenerate paper figures
```

| File | Contains |
|---|---|
| `requirements.txt` | numpy, OpenCV, pygame, psutil — the benchmark runtime. |
| `requirements-rl.txt` | gymnasium, torch, stable-baselines3, sb3-contrib. |
| `requirements-dev.txt` | pytest, matplotlib, Pillow. |

The rule-based and fleet benchmark experiments do not require the RL
dependencies; the recurrent-PPO evaluation does. Install that set only when you
intend to run the learned controller.

---

## 5. Install the simulator world

HoloOcean downloads its worlds once, stores them per HoloOcean version under your
user profile, and shares them across environments:

```bash
python -c "import holoocean; holoocean.install('Ocean')"
python -c "import holoocean; print(holoocean.installed_packages())"
```

The second command must print `['Ocean']`.

---

## 6. Check the installation without launching the simulator

```bash
python -m marine_race_arena.scripts.validate_track_config \
  --track marine_race_arena/tracks/marine_race_horseshoe_bay.json \
  --benchmark-task clean_gate
```

Expected tail:

```text
Gates per lap: 12
Laps: 1
Declared path length: 93.80 m
Computed path length: 93.81 m
Validation passed.
```

If that works, the package, the track files and the configuration schema are all
fine — independently of HoloOcean.

With `requirements-dev.txt` installed you can also run the whole suite, which is
simulator-independent and takes roughly four minutes:

```bash
python -m pytest tests -q
```

---

## 7. Your first race

```bash
python -m marine_race_arena.scripts.run_marine_race \
  --track marine_race_arena/tracks/marine_race_horseshoe_bay.json \
  --benchmark-task clean_gate --controller rule_gate_center_then_commit \
  --adapter holoocean --official --headless \
  --seed 0 --dt 0.033 --duration 560 \
  --log-dir results/first_race
```

HoloOcean takes about a minute to start, then the race runs. When it ends the
referee prints the official summary and two files appear in `results/first_race/`:

| File | Contents |
|---|---|
| `<race>_<timestamp>.jsonl` | line-delimited event log |
| `<race>_<timestamp>_summary.json` | referee result: status, gates, official and penalized time, penalties, events |

`--official` enforces the participant sensor profile and the official timing mode.
`--headless` asks HoloOcean not to open a window. Drop it if you want to watch.

More races — currents, fleets, coordination, the learned controller — are in
**[experiments.md](experiments.md)**, ready to copy.

---

## 8. Running without HoloOcean

The engine-free fallback adapter exercises the runner, the referee, the logging
and your controller's plumbing. It is **not** physical evidence and must never be
used for reported results:

```bash
python -m marine_race_arena.scripts.run_marine_race \
  --track marine_race_arena/tracks/marine_race_horseshoe_bay.json \
  --benchmark-task clean_gate --controller rule_gate_center_then_commit \
  --adapter fallback --allow-fallback \
  --seed 0 --dt 0.1 --duration 20 \
  --log-dir results/fallback_check
```

Camera-gated controllers do not make real progress under the fallback adapter —
it has no rendered camera, so the visual alignment stage never confirms. A short
run that produces a summary is the expected outcome.

---

## 9. Configuration-driven runs

For repeatable setups, describe the run in JSON instead of flags:

```bash
python run.py --dry-run              # resolve the config, print the command, launch nothing
python run.py                        # config.json: single official run
python run.py configs/fleet.json     # two-vehicle staggered fleet
python run.py configs/benchmark.json # repeated seeds on one circuit
```

`run.py` maps the JSON onto the matching runner. See
**[configuration.md](configuration.md)** for every field.

---

## 10. If something goes wrong

| Symptom | Cause and fix |
|---|---|
| `pip install holoocean==2.3.0` fails | Expected — 2.3.0 is not on PyPI. Install the client from source (step 3). |
| `installed_packages()` returns `[]` | The world was never downloaded. Re-run step 5. |
| The run reports `adapter: fallback` when you asked for HoloOcean | The engine could not start. `--adapter holoocean` without `--allow-fallback` makes this a hard failure instead of a silent downgrade — use it for anything you intend to report. |
| A race ends with `status: RUNNING` | The `--duration` budget ran out before the last gate. Give the circuit its nominal duration (560 / 900 / 1300 s). |
| A HoloOcean process outlives the run on Windows | Install `psutil` (it is in `requirements.txt`); the adapter uses it to verify engine shutdown by UUID. |
| Camera-gated controller makes no progress | You are on the fallback adapter. Only the HoloOcean adapter renders the forward camera. |
