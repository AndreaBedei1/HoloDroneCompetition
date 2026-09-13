# Ready-to-run experiments

Eight experiments, each a single command you can paste. Together they cover the
whole benchmark: both reference controllers, three circuits, an environmental
disturbance, a two-vehicle fleet, distributed coordination, the learned
controller and the evaluation harness.

**Prerequisites for everything on this page:** the `ocean` environment from
[getting-started.md](getting-started.md), with HoloOcean 2.3.0 and the `Ocean`
world installed. Experiment 7 additionally needs `requirements-rl.txt`. Run every
command from the repository root.

---

## What was verified, and how

Every command below was executed on real HoloOcean while writing this page.

| # | Experiment | Verified |
|---|---|---|
| 1 | Center-then-Commit, Horseshoe Bay | **complete race** — finished 12/12 |
| 2 | Continuous Servo, Horseshoe Bay | **complete race** — finished 12/12 |
| 3 | Center-then-Commit, Vertical Serpent | time-boxed — reached gate 11 of 17 |
| 4 | Medium current, Horseshoe Bay | time-boxed — reached gate 10 of 12, contacts charged |
| 5 | Two-vehicle fleet | time-boxed — both vehicles released and scored |
| 6 | Leader–follower coordination | time-boxed — three vehicles, no contact |
| 7 | Recurrent PPO reference controller | **complete episode** — finished 12/12 |
| 8 | Current-free evaluation harness | time-boxed — full artifact set written |

*Time-boxed* means the command was run with a reduced `--duration` to confirm
that the scenario builds, the simulator starts, the controllers run, the referee
scores and the expected files appear — not that the vehicle reached the finish.
Each of those experiments shows both forms below.

> **On timing.** A re-run reproduces the referee's *outcome* — gates completed,
> finished or not, events charged — but not the recorded official time to the
> decimal. Engine stepping depends on machine state: the two complete races below
> came in around 19 % faster than the values in the frozen matrix for the same
> seed, with identical outcomes and identical controller ordering. Use the
> released artifacts, not a re-run, when you need the published numbers.

**How long things take.** On the machine these were run on, wall-clock time is
roughly **1.5–2× the simulated time, plus about a minute of engine start-up**.
A race ends when the vehicle finishes, so a controller that completes a circuit
costs far less than one that burns the whole `--duration` budget. The measured
figures below are what those runs actually took.

---

## 1. Center-then-Commit on Horseshoe Bay

The reference clean-water race: 12 gates, onboard sensing only, independent
referee.

```bash
python -m marine_race_arena.scripts.run_marine_race \
  --track marine_race_arena/tracks/marine_race_horseshoe_bay.json \
  --benchmark-task clean_gate --controller rule_gate_center_then_commit \
  --adapter holoocean --official --headless \
  --seed 0 --dt 0.033 --duration 560 --current-profile none \
  --log-dir results/e1_ctc_horseshoe
```

| | |
|---|---|
| **Expect** | `status: FINISHED`, `completed_gates: 12`, zero collisions, zero out-of-bounds, zero stuck events, `penalized_time == official_time` |
| **Observed** | finished 12/12 in 159.5 s of race time, no events |
| **Writes** | `results/e1_ctc_horseshoe/` — one `.jsonl` event log, one `_summary.json` |
| **Takes** | about 6–7 minutes wall clock, of which roughly a minute is engine start-up |

---

## 2. Continuous Servo on the same circuit

The other reference controller, same circuit, same seed. The two differ in
exactly one respect — what happens during the commit through the aperture — so
this is the cleanest A/B the benchmark offers.

```bash
python -m marine_race_arena.scripts.run_marine_race \
  --track marine_race_arena/tracks/marine_race_horseshoe_bay.json \
  --benchmark-task clean_gate --controller rule_gate_baseline \
  --adapter holoocean --official --headless \
  --seed 0 --dt 0.033 --duration 560 --current-profile none \
  --log-dir results/e2_servo_horseshoe
```

| | |
|---|---|
| **Expect** | `status: FINISHED`, `completed_gates: 12`, zero events — on this short, largely planar circuit the two controllers are effectively tied |
| **Observed** | finished 12/12 in 156.8 s, no events — 2.7 s ahead of Center-then-Commit, the same ordering the frozen matrix records for this seed |
| **Writes** | `results/e2_servo_horseshoe/` |
| **Takes** | about 6 minutes |

Compare the two summaries directly:

```bash
python -c "import glob,json,os; [print(os.path.basename(os.path.dirname(p)).ljust(22), (d:=json.load(open(p)))['participants'][0]['status'], d['participants'][0]['completed_gates'], round(d['participants'][0]['official_time_s'],1)) for p in sorted(glob.glob('results/e*_*horseshoe/*_summary.json'))]"
```

```text
e1_ctc_horseshoe       FINISHED 12 159.5
e2_servo_horseshoe     FINISHED 12 156.8
```

---

## 3. A harder circuit: Vertical Serpent

17 gates, and unlike Horseshoe Bay it changes depth continuously. This is where
the two reference controllers stop agreeing.

```bash
python -m marine_race_arena.scripts.run_marine_race \
  --track marine_race_arena/tracks/marine_race_vertical_serpent.json \
  --benchmark-task clean_gate --controller rule_gate_center_then_commit \
  --adapter holoocean --official --headless \
  --seed 0 --dt 0.033 --duration 900 --current-profile none \
  --log-dir results/e3_vertical_ctc
```

| | |
|---|---|
| **Expect** | a longer race; give it the circuit's nominal 900 s budget or it will end with `status: RUNNING` |
| **Observed** | time-boxed at `--duration 130`: reached gate 11 of 17 with zero events in about 5 minutes |
| **Writes** | `results/e3_vertical_ctc/` |
| **Takes** | 5 minutes for the time-boxed check shown above. A full run costs whatever the vehicle needs: under ten minutes if it completes the circuit, closer to half an hour if it consumes the whole 900 s budget |

Swap the track file for `marine_race_mixed_endurance.json` (22 gates,
`--duration 1300`) for the long circuit.

---

## 4. The same circuit under a medium current

Clean-track success is not robustness. Identical circuit, identical controller,
identical seed — only the water moves.

```bash
python -m marine_race_arena.scripts.run_marine_race \
  --track marine_race_arena/tracks/marine_race_horseshoe_bay.json \
  --benchmark-task current_gate --controller rule_gate_center_then_commit \
  --adapter holoocean --official --headless \
  --seed 0 --dt 0.033 --duration 560 --current-profile medium \
  --log-dir results/e4_medium_current
```

| | |
|---|---|
| **Expect** | contacts where experiment 1 had none, and `penalized_time` above `official_time`. In the frozen matrix this condition drops Center-then-Commit to 3 finishes in 5 seeds |
| **Observed** | time-boxed at `--duration 150`: gate 10 of 12 with **2 gate/world collisions**, against zero in the clean run |
| **Writes** | `results/e4_medium_current/` |
| **Takes** | 6 minutes time-boxed. A full run is under ten minutes if it finishes and around twenty if it burns the 560 s budget — which under this current is a real possibility |
| **Note** | `--benchmark-task current_gate` is what makes the validator *require* an active current. Forgetting it is the usual reason a "current" run silently runs clean. |

---

## 5. A two-vehicle fleet

Two vehicles, staggered release, independent per-vehicle referee state, one team
score.

```bash
python -m marine_race_arena.scripts.run_marine_race \
  --track marine_race_arena/tracks/marine_race_horseshoe_bay.json \
  --benchmark-task clean_gate --controller rule_gate_center_then_commit \
  --adapter holoocean --official --headless \
  --staggered-start --num-rovers 2 --start-gap-s 90 \
  --staggered-lateral-offset-m 3.0 --team-id fleet_demo \
  --inter-vehicle-collision-mode diagnostic \
  --seed 0 --dt 0.033 --duration 560 --current-profile none \
  --log-dir results/e5_fleet_two
```

| | |
|---|---|
| **Expect** | a `team_summary` block alongside the two participants: total completed gates, team elapsed and penalized time, gate/world contacts and inter-vehicle proximity events |
| **Observed** | time-boxed at `--start-gap-s 30 --duration 150`: both vehicles released and scored independently, 17 team gates, zero inter-vehicle events |
| **Writes** | `results/e5_fleet_two/` |
| **Takes** | 8 minutes time-boxed. At a 90 s release gap the team is on the course for roughly 300 simulated seconds, so budget somewhere over ten minutes |
| **Note** | `diagnostic` records proximity events without charging them; `penalize` charges them. |

Or drive the same thing from a file:

```bash
python run.py configs/fleet.json
```

---

## 6. Leader–follower coordination

A deliberately heterogeneous three-vehicle convoy — a slower Continuous Servo
leader followed by two faster Center-then-Commit vehicles — so followers catch
the vehicle ahead unless they coordinate. Coordination uses nothing but locally
estimated progress sent over the acoustic channel.

```bash
python -m marine_race_arena.scripts.run_holoocean_coordination_validation \
  --track marine_race_arena/tracks/marine_race_horseshoe_bay.json \
  --output-dir results/e6_coordination \
  --duration-s 560 --dt 0.033 --team-size 3 --start-gap-s 8 \
  --seeds 0 --conditions leader_follower --min-gate-gap 1 --headless
```

| | |
|---|---|
| **Expect** | one run directory per condition and seed, plus `validation.json` and `validation.md` summarising the comparison |
| **Observed** | time-boxed at `--duration-s 130`: three vehicles released and coordinating, 23 team gates, zero gate/world collisions and zero inter-vehicle proximity events |
| **Writes** | `results/e6_coordination/<mode>/seed_<n>/<condition>/` plus `validation.json`, `validation.md` |
| **Takes** | 9 minutes time-boxed with three vehicles. Each extra condition and seed is another run of the same size |

`validation.md` is the part worth reading — one row per seed and condition:

```text
| Seed | Condition       | OK  | All finished | Team gates | Inter-vehicle events | Gate/world collisions | ... | Comms delivered/dropped |
|    0 | leader follower | yes | False        |      23/36 |                    0 |                     0 | ... |                  1325/0 |
```

The comms column is the honest part: 1325 heartbeats actually crossed the
acoustic channel, and coordination was computed from those alone.

Add `--conditions no_coordination leader_follower` to run the matched
uncoordinated baseline in the same invocation — that is the comparison worth
looking at. `--min-gate-gap 2` is the conservative margin, `--comms-packet-loss-prob`
injects seeded acoustic loss.

---

## 7. The learned reference controller

The frozen recurrent-PPO policy released with the repository, evaluated through
the same observation boundary and the same referee as the rule-based
controllers. **Needs `requirements-rl.txt`.**

```bash
python -m marine_race_arena.learning.gen2.evaluate_speed_robust \
  --controller recurrent_ppo --track horseshoe_bay \
  --checkpoint artifacts/paper/ppo/model/policy_recurrent_ppo_27d.zip \
  --seed 20370909 --out results/e7_ppo \
  --wall-timeout-s 1800 --stall-timeout-s 180 --gate-stall-timeout-s 120
```

| | |
|---|---|
| **Expect** | one `*.result.json` per episode with `technical_status: VALID`, the gates completed, and `observation_contract: onboard_local_transition_27d_v1` |
| **Observed** | **finished 12/12 gates in 199.2 s** with 2 collisions, `FINISHED` / `VALID` |
| **Writes** | `results/e7_ppo/` — one result and one progress file per episode |
| **Takes** | about 17 minutes for one full episode at 10 Hz |
| **Note** | The runner launches one isolated process per episode with watchdogs, so a hung simulator is detected rather than silently draining the budget. Replace `--controller recurrent_ppo` with `--controller rules` to run the deterministic reference through the identical harness. |

Swap `--track` for `vertical_serpent` or `mixed_endurance` for the other
circuits. Training code is not part of this release; the checkpoint is frozen.

---

## 8. The evaluation harness: current-free validation

The multi-seed harness behind the manuscript's current-free demonstration. It
writes per-episode results, a summary and an evaluation manifest recording the
adapter, the seeds, the model hashes and whether currents were genuinely zero.

```bash
python -m marine_race_arena.learning.closed_loop_eval \
  --track marine_race_arena/tracks/marine_race_horseshoe_bay.json \
  --controller rule_gate_center_then_commit \
  --seeds 1800-1802 --out results/e8_current_free \
  --adapter holoocean --current-profile none --dt 0.1 --duration 560
```

| | |
|---|---|
| **Expect** | `eval_results.csv`, `eval_results.json`, `eval_summary.json` and `evaluation_manifest.json`; the summary reports `completions` out of `n_eval` |
| **Observed** | time-boxed at one seed and `--duration 130`: all four artifacts written, manifest recording the HoloOcean adapter and zero currents. The vehicle did not reach the finish inside that budget, so `completions: 0` — give it the full duration for a completion |
| **Writes** | `results/e8_current_free/` |
| **Takes** | a few minutes for the time-boxed check. Budget several minutes per seed at full duration, and multiply by the number of seeds |

Each seed is saved as it completes, so a long evaluation is crash-safe and can be
resumed rather than restarted.

The same three circuits at three seeds each are what the manuscript reports as
the 9-of-9 current-free demonstration; the released artifacts are in
[`artifacts/paper/current_free/`](../artifacts/paper/current_free/).

---

## Reading a result

Every race writes two files. The summary is the referee's word:

```bash
python -c "import glob,json; p=sorted(glob.glob('results/e1_ctc_horseshoe/*_summary.json'))[-1]; d=json.load(open(p)); a=d['participants'][0]; print(a['status'], a['completed_gates'], round(a['official_time_s'],1), 'penalties', a['penalties_s'])"
```

| Field | Meaning |
|---|---|
| `status` | `FINISHED`, `DNF` or `RUNNING` (the duration budget ran out) |
| `completed_gates` | gates the **referee** validated, not what your controller believes |
| `official_time_s` | race time under the track's timing mode |
| `penalized_time_s` | `official_time_s` plus charged penalties |
| `collisions`, `out_of_bounds_events`, `stuck_events` | charged events |
| `adapter`, `fallback_used` | provenance — for anything you report these must be `holoocean` and `false` |
| `team_summary` | present in fleet runs: team gates, team times, inter-vehicle events |

The `.jsonl` alongside it is the full event log — one line per event, including
every `gate_passed` and every controller command.

---

## Before you report a number

* `--adapter holoocean` **without** `--allow-fallback`, so a failed engine is an
  error rather than a silent downgrade to kinematics.
* `--official`, so the participant sensor profile and the official timing apply.
* Give the circuit its nominal duration: 560 s, 900 s, 1300 s.
* Record the seed. Beacon reception, packet loss and obstacle generation are all
  seeded from it.
* Quote the referee's `_summary.json`, not your controller's own bookkeeping.
