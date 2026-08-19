# RL Generation 2 — progress and how to resume

Branch: `feature/rl-gen2-recurrent-dagger` (from `fdbd8c6`, the Gen-1 endgame tip).
Read [`docs/rl_generations.md`](rl_generations.md) first — it defines the Gen-1/Gen-2
boundary and lists the Gen-1 artifacts that must never be overwritten.

Goal: a fully learned recurrent onboard controller that learns multi-gate
navigation from the onboard rule expert, corrects compounding error with
DAgger, and only then uses recurrent PPO as fine-tuning.

Inference contract, non-negotiable:

```
action = learned_policy(observation, recurrent_state)
```

No rules fallback, no blending, no hybrid, no privileged information.

---

## Status

| Phase | State |
|---|---|
| Gen-1 preservation and boundary | done |
| Gen-2 seed bands | done |
| Sealed final holdout (5 circuits) | done |
| Pre-registered readiness gate | done (committed before any Gen-2 measurement) |
| Recurrent policy infrastructure | done |
| Expert demonstration pipeline | done |
| Expert data correctness proven | done — 28 episodes over all 7 lengths, completion 0.929 |
| First expert corpus | done — 400 episodes, 321 796 transitions |
| Recurrent BC | first run done; re-running cleanly (see below) |
| DAgger rounds 1–3 | round 1 done; re-running cleanly |
| Recurrent PPO | **not started** — blocked on DAgger readiness |
| Ablation A/B/C/D | implemented, waiting for arms C and D |

## Contracts (unchanged from Gen-1, deliberately)

* Observation `onboard_local_transition_v1`, exactly 35 features.
* Action `surge_sway_heave_yaw_pm1_v1`, 4 continuous axes in `[-1, 1]`.
* Currents disabled.
* Expert: `rule_gate_center_then_commit` (`RuleGateCenterThenCommitController`),
  `uses_ground_truth = False`. Verified at runtime to read only `local_time_s`,
  `sensors` and `beacons`.

## Seed bands

| Band | Range | Purpose |
|---|---|---|
| TRAIN | 40 000 – 49 999 | the only band that may produce gradient-bearing samples |
| VALIDATION | 50 000 – 50 999 | evaluate learned policies; never supplies expert labels |
| TEST | 51 000 – 51 999 | untouched until model selection |
| FINAL_HOLDOUT | 52 000 – 52 999 | sealed until one Gen-2 policy is frozen |

TRAIN sub-roles: expert demos 40 000–43 999, DAgger 44 000–47 999, PPO
48 000–49 499, smoke 49 500–49 999.
VALIDATION sub-roles: BC stages 50 000–50 299, DAgger rounds 50 300–50 599,
ablation 50 600–50 899, PPO validation 50 900–50 999.

`assert_expert_labelling_seed` raises on anything outside TRAIN, so the expert
physically cannot label a validation, test or holdout state.

## Sealed final holdout

Five circuits under `marine_race_arena/tracks/gen2_final_holdout/`, generated
once at git `fdbd8c6`:

| Circuit | Gates | Seed | SHA-256 (first 16) |
|---|---|---|---|
| gen2_final_holdout_01 | 12 | 52000 | `3a2614747d22c07b` |
| gen2_final_holdout_02 | 15 | 52001 | `cafc4f81a63d2bbf` |
| gen2_final_holdout_03 | 17 | 52002 | `d7691b4744aa569d` |
| gen2_final_holdout_04 | 20 | 52003 | `8660a08e21c0da26` |
| gen2_final_holdout_05 | 22 | 52004 | `0b8f610f09eb5650` |

Manifest SHA-256 `00c7db38fea15a97b47d7588f11adb392a8c6bef144e4131980379c13ca00a6b`.

The seal is enforced in the two episode drivers every Gen-2 rollout goes
through (`expert_rollout.run_gen2_episode`, `evaluation.run_policy_episode`),
default-deny. Opening requires a once-only unseal record bound to the frozen
policy's own SHA-256.

## Pre-registered readiness gate

`gen2_readiness_gate_v1`, thresholds SHA-256
`917dcd3486f6404be36797db7a7527a2094b8a5ed8b419e479554774ccfedfef`,
published at `results/rl_public/gen2_readiness/preregistration.json` **before**
any Gen-2 measurement existed.

16 criteria: universal transition ≥ 0.95, gate1→gate2 ≥ 0.95, completion
3/5/8/12/17/22 ≥ 0.95/0.90/0.85/0.70/0.50/0.45, OOB ≤ 0.005, collisions ≤ 0.10,
wrong direction ≤ 0.01, plus evidence criteria (≥ 500 transition cases, ≥ 40 per
length, ≥ 200 gate1→gate2 cases, expert-free inference, contract match).
Safety and evidence are non-overridable.

## Architecture

```
35-D observation -> frozen normalization -> MLP encoder [35-256-128]
                 -> LSTM [128->128] -> policy head [128-128-4]
                                    -> value head  [128-128-1]
```

339,977 parameters total; 191,236 on the actor path. Behaviour cloning
optimizes the **real** `RecurrentPPO` policy, so BC→PPO transfer is save/load
rather than a hand-folded weight copy — the parity test asserts exact equality.

## Measured so far

Expert competence on the Gen-2 course family (HoloOcean, 6 workers, 27
episodes covering every length):

| Gates | Episodes | Completed |
|---|---|---|
| 2 | 4 | 4 |
| 3 | 4 | 4 |
| 5 | 4 | 4 |
| 8 | 3 | 2 |
| 12 | 4 | 3 |
| 17 | 4 | 4 |
| 22 | 4 | 4 |

Overall completion **0.926**, gate success **0.959**, 1 episode with a
collision, 0 out-of-bounds, 39 009 transitions.

The expert is therefore **very good but not perfect** on this family — it
misses roughly one episode in fourteen at the mid lengths. Two consequences:

* the BC corpus will contain some failed expert episodes, so
  `--completed-only` exists as an option and the corpus statistics record the
  completion rate per length;
* the expert's own rate is the practical ceiling for a pure imitator, which is
  another reason DAgger and then PPO are needed rather than more cloning.

Throughput: 6.7 steps/s single worker, 15.1 steps/s at six workers (2.25×).
Roughly 150 simulator steps per gate. Engine ceiling is 10 processes and the
box is shared, so collection runs at 5–6 workers.

### A defect the validation batch caught

An early 17-gate `turn_right` course failed at 2/17. The cause was the course
family, not the expert: a monotone 45° turn at ~4 m spacing closes a circle of
radius ~5 m, so long turning courses wrapped onto themselves and put a later
gate beside an earlier one. The onboard front end has no legal way to
disambiguate that — it is an *ambiguous* course, not a hard one. Courses are
now relaxed (turn deltas attenuated) until non-adjacent gates clear 4 m. 84 of
1500 sampled courses needed relaxation, max 4 passes, and pattern/length/
direction diversity is unchanged.

## Expert corpus v1

400 episodes, **321 796 transitions**, 400 unique courses.
`corpus_sha256 = 04258c69ffaca9c48f1716e61883d26e3184d302814a3bb8935bb4a7b5f6ddf3`
(manifest re-verified: 400/400 shards present, no hash drift, no partials).

| Gates | Episodes | Transitions | Expert completion |
|---|---|---|---|
| 2 | 118 | 33 687 | 1.000 |
| 3 | 90 | 40 212 | 1.000 |
| 5 | 92 | 76 765 | 0.978 |
| 8 | 67 | 92 210 | 0.955 |
| 12 | 23 | 47 491 | 0.957 |
| 17 | 8 | 22 658 | 1.000 |
| 22 | 2 | 8 773 | 1.000 |

Overall expert completion 0.985, gate success 0.9908, 0 out-of-bounds,
11 episodes with a collision. Collection took 7.4 h at 5 workers.

## First BC run, and why it is being repeated

The first pipeline produced a working recurrent BC policy:

* validation action MSE **9.2e-05**, per-axis correlation 0.969–0.994
* stage A (2 gates) completion **1.000**
* stage C (5 gates) completion **0.733**
* **gate1→gate2 transition 1.000 at every stage measured**

That last line is the point of the whole experiment: Gen-1 recorded a first
gate of 1.00 against a complete gate1→gate2 transition of 0.7333, and recurrent
BC alone already closes that gap on 2–5 gate courses.

The run is nevertheless being repeated, because two pipeline instances ran
concurrently. Stopping the first chain's shell did not stop its detached
python child, and a second pipeline started beside it. Both wrote the same BC
checkpoint and the same DAgger round directory. They graded stage B at 0.90 and
0.7667 on identical seeds — a four-episode difference at n=30, comfortably
inside binomial noise, and HoloOcean is not bit-reproducible
(`exact_observation_trace_reproducibility: false` in the Gen-1 parallel
benchmark), so the two numbers cannot be attributed.

Nothing was lost: the corpus verified intact, and the contended artifacts are
kept under `results/rl/gen2/_contended_20260819/` with the interleaved log as
evidence. Two changes followed:

1. `Pipeline` now takes an exclusive lock on its base directory and refuses to
   start beside a live holder (a lock from a dead PID is ignored).
2. Stage evaluation now runs at 60 episodes rather than 30, since n=30 cannot
   separate 0.77 from 0.90.

DAgger round 1 (learner-driven, expert-labelled, audited as a pure learner
rollout with zero takeovers) reached validation completion 0.60–0.625 on mixed
3–5 gate courses with gate1→gate2 0.875–0.975, below the 0.80 advance bar, so
the campaign correctly declined to lengthen the courses.

## How to resume

Status is a file read, never a poll:

```bash
cat results/rl/gen2/pipeline_state.json
```

Corpus progress:

```bash
ls results/rl/gen2/expert_corpus/episodes/*.npz | wc -l
```

Run the whole pipeline (resumable; skips shards that exist):

```bash
python -m marine_race_arena.learning.gen2.pipeline --base results/rl/gen2 --episodes 400 --workers 5
```

Individual phases:

```bash
python -m marine_race_arena.learning.gen2.collect_expert --out results/rl/gen2/expert_corpus --episodes 400 --workers 5
```

```bash
python -m marine_race_arena.learning.gen2.train_bc --corpus results/rl/gen2/expert_corpus --out results/rl/gen2/bc_v1 --stages A,B,C
```

```bash
python -m marine_race_arena.learning.gen2.run_dagger --base results/rl/gen2 --bc results/rl/gen2/bc_v1/bc_policy.zip --rounds 1,2,3
```

Tests:

```bash
python -m pytest tests/learning/gen2 -q
```

## Gates that must be satisfied, in order

1. **BC progression** — stage A (2 gates) ≥ 0.95, B (3) ≥ 0.90, C (5) ≥ 0.80.
   If MSE is tiny but rollouts are poor, do **not** keep cloning; go to DAgger.
   `train_bc.recommend_next_phase` encodes this.
2. **DAgger round advance** — validation completion at the round's own lengths
   ≥ 0.80 / 0.75 / 0.65 / 0.55 / 0.40 for rounds 1–5. Advance on competence,
   never on iteration count.
3. **Pre-PPO readiness** — 3 gate ≥ 0.95, 5 ≥ 0.90, 8 ≥ 0.80, 12 ≥ 0.60–0.70,
   17/22 substantial, OOB ≈ 0, collisions controlled. **PPO must not start
   before this.** PPO is fine-tuning, not discovery; it must not be used to
   rescue a poor imitation controller.
4. **Gen-2 readiness gate** — the 16 pre-registered criteria above.
5. **Freeze exactly one policy**, then open the sealed holdout once.
6. **Retrospective only after that** — the three old circuits are diagnostics,
   not held-out generalization.

## Known constraints

* `torch` is the CPU wheel (`2.8.0+cpu`) and `device="cpu"` is hardcoded in the
  Gen-1 learning modules. Two RTX 6000 Ada GPUs are present but unused by
  training. The Gen-2 LSTM is small enough that CPU training is not the
  bottleneck — HoloOcean rollout is.
* The box is shared with another HoloOcean project and ComfyUI. Re-measure
  capacity per session with `holoocean_capacity.capacity_snapshot()`; never
  assume a previous session's worker count.
* Killing a collector's shell does **not** kill its detached workers. Check for
  live `spawn_main` children before launching a second collector on the same
  corpus directory.
