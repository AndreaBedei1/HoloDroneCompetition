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
| Expert data correctness proven | done — 17/17 HoloOcean episodes completed, 0 collisions |
| First expert corpus | collecting |
| Recurrent BC | queued behind collection |
| DAgger rounds 1–3 | queued behind BC |
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

Expert competence on the Gen-2 course family (HoloOcean, 6 workers):

| Gates | Episodes | Completed | Collisions |
|---|---|---|---|
| 2 | 4 | 4 | 0 |
| 3 | 4 | 4 | 0 |
| 5 | 3 | 3 | 0 |
| 8 | 2 | 2 | 0 |
| 12 | 1 | 1 | 0 |
| 17 | 1 | 1 | 0 |
| 22 | 2 | 2 | 0 |

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
