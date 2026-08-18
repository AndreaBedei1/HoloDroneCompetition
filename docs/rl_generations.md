# RL generations: the Gen-1 / Gen-2 boundary

This document draws a hard line between two experiments so neither can be
mistaken for the other, and so nothing produced by Generation 1 is overwritten,
re-run, or silently reinterpreted by Generation 2.

| | **RL Generation 1** | **RL Generation 2** |
|---|---|---|
| Name | feed-forward pure PPO | expert-bootstrapped recurrent learned controller |
| Branch | `feature/rl-universal-gate-transition` -> `feature/rl-multistep-sac-transition` | `feature/rl-gen2-recurrent-dagger` |
| Final commit | `fdbd8c6` *Record final PPO holdout verdict* | (in progress) |
| Method | PPO from scratch on procedural geometry | expert bootstrap -> recurrent BC -> DAgger -> recurrent PPO fine-tune |
| Architecture | `MlpPolicy`, `pi=[256,256] vf=[256,256]`, feed-forward, frame stack 1 | MLP encoder -> LSTM -> policy/value heads (`gen2_recurrent_lstm_v1`) |
| Observation contract | `onboard_local_transition_v1` (35 features) | `onboard_local_transition_v1` (35 features) — **unchanged** |
| Action contract | `surge_sway_heave_yaw_pm1_v1` (4 continuous) | `surge_sway_heave_yaw_pm1_v1` (4 continuous) — **unchanged** |
| Currents | disabled | disabled |
| Seed space | <= 33 999 (see `learning/seed_registry.py`) and the `rl_holdout_policy` bands from 1 000 000 | 40 000 – 52 999 (see `learning/gen2/seeds.py`) |
| Readiness gate | `rl_readiness_gate_v1`, thresholds sha256 `b33f1e16…9b35` | `gen2_readiness_gate_v1` (separate object, separate hash) |
| Status | **CLOSED. Preserved as experimental evidence.** | active |

Generation 2 is a *new experiment*, not a continuation. It shares the
observation and action contracts on purpose, so the two generations remain
directly comparable on identical seeds; it shares nothing else.

---

## 1. Generation 1: final recorded result

These numbers are the Gen-1 result of record. They are **not to be modified,
re-derived, or re-run.**

* Procedural readiness: **FAIL** — 4 of 19 pre-registered criteria missed
  (`universal_transition_success` 0.82 vs >= 0.90; `completion_3_gate` 0.80 vs
  >= 0.95; `completion_5_gate` 0.85 vs >= 0.90; `completion_8_gate` 0.55 vs
  >= 0.75). No override was requested or applied.
* 500-case universal transition success: **0.82**
* First gate crossing: **1.00**
* Complete gate1 -> gate2 survival: **0.7333** (88/120)
* Final official circuits (Horseshoe Bay, Vertical Serpent, Mixed Endurance),
  30 paired trials: **PPO 0/30 completed, rules 30/30 completed.** PPO crossed
  87 of 510 gates; every PPO episode ended in a `crossing_missed_gate` DNF at
  the gate *after* its last crossing.
* SAC: 7 runs (v1..v7), no deployable policy. v7 ended in
  `baseline_collapse_rollback` at 300 032 transitions with `do_not_resume: true`.

The diagnostic that defines the Gen-2 hypothesis: **first-gate crossing is
perfect while the full gate1 -> gate2 transition is only 0.73.** Gen-1 could
learn to cross a gate but not to re-acquire the next one. Survivor-conditioned
rates hid this, which is why Gen-2 instruments unconditional survival from
episode start.

## 2. Where the Gen-1 artifacts physically live

`results/` is gitignored except `results/rl_public/**` (minus `*.zip`, `*.npz`,
`*.npy`, `*.mp4`). Two consequences matter:

**Git-tracked and present in both worktrees** — the compact audit trail:

| Artifact | Path |
|---|---|
| Freeze manifest for the final PPO | `results/rl_public/ppo_final_generic_929792/artifact_freeze.json` |
| Procedural readiness verdict | `results/rl_public/ppo_final_readiness_929792/readiness_verdict.json` / `.md` |
| 500-case benchmark + survival curve | `results/rl_public/ppo_final_readiness_929792/ppo_final_generic_929792/evaluation.json` |
| Old final holdout, PPO 0/30 vs rules 30/30 | `results/rl_public/ppo_final_holdout_929792/final_experiment_report.md` / `.json` |
| Hash-chained holdout access ledger (60 entries) | `results/rl_public/ppo_final_holdout_929792/final_circuit_evaluations.jsonl` |
| Rule-controller baseline | `results/rl_public/final_benchmark/` |

**Disk-only, and present in exactly ONE worktree** — the heavy evidence:

| Artifact | Path (only copy) |
|---|---|
| Frozen final Gen-1 PPO weights | `C:\Users\andrea.bedei3\Desktop\HoloDroneCompetition-sac\results\rl\universal_transition\final\ppo_final_generic_929792.zip` |
| Entire SAC campaign, 7 runs, ~13 GB | `…-sac\results\rl\universal_transition\sac\` |
| PPO ancestry long-runs that produced 929792 | `…-sac\results\rl\universal_transition\longrun\` |

> **Do not remove the `HoloDroneCompetition-sac` worktree.** It holds the only
> copy of the Gen-1 PPO policy weights and the whole SAC campaign (14 GiB in
> `results/rl`). The `*.zip` exclusion means the git-tracked `rl_public` mirror
> carries the checkpoint's manifest and hashes but **not** the weights.

The frozen checkpoint is marked read-only on disk (`-r--r--r--`,
`filesystem_read_only: true`, `training_permanently_closed: true`) and its
freeze record was written with an exclusive create, so it cannot be re-frozen.

## 3. What Generation 2 must not do

1. **Never overwrite a Gen-1 artifact.** Gen-2 writes only under
   `results/rl/gen2/` and `results/rl_public/gen2_*/`.
2. **Never re-run the Gen-1 evaluations.** The frozen 78-run matrix, the
   readiness verdict and the 0/30 holdout stand as recorded.
3. **Never mutate `ReadinessThresholds`** in `learning/rl_readiness_gate.py`.
   Its hash is pinned inside the Gen-1 freeze record; editing the defaults
   would retroactively invalidate a completed experiment. Gen-2 defines its own
   thresholds object with its own hash.
4. **Never repoint `BENCHMARK_ROLE`** or the `rl_holdout_policy` seed bands.
   Gen-2 seeds live in a separate module and a separate numeric range.
5. **Never touch** `results/rl_public/ppo_final_holdout_929792/exploratory_holdout_decision.json`
   or its ledger. Both are content-hashed and hash-chained, and both currently
   verify; any edit, including reformatting, breaks the chain.

## 4. The three old official circuits are no longer pristine

Horseshoe Bay, Vertical Serpent and Mixed Endurance **have been opened.** The
Gen-1 exploratory holdout ran 30 trials across them and the ledger records all
60 accesses.

Therefore, for the remainder of the project:

* they must **not** be called unseen holdouts;
* Gen-2 must **not** train on their geometry;
* Gen-2 must **not** use their individual failure locations to construct
  matching courses;
* they may be used **only** as a *retrospective diagnostic benchmark*, run
  after the Gen-2 policy is frozen and the new holdout protocol is fixed, and
  reported with that label.

A Gen-2 result on those circuits — even 30/30 — is a retrospective diagnostic,
not a held-out generalization claim.

## 5. Generation 2 in one line

> Produce a fully learned recurrent onboard controller that first learns
> reliable multi-gate navigation from a perfect onboard expert, corrects
> compounding error through DAgger, and only then uses PPO for reinforcement
> fine-tuning.

At inference the action is exactly

```
action = learned_policy(observation, recurrent_state)
```

and nothing else: no rules fallback, no action blending, no hybrid controller,
no expert correction, no circuit-specific logic, no privileged coordinates. The
rule controller appears only as a teacher during dataset generation and DAgger
labelling.

## 6. Generation 2 layout

| Concern | Module |
|---|---|
| Seed bands TRAIN/VALIDATION/TEST/FINAL_HOLDOUT | `learning/gen2/seeds.py` |
| Procedural course family | `learning/gen2/course_family.py` |
| Sealed final holdout + access guard | `learning/gen2/holdout_seal.py` |
| Expert / DAgger episode driver | `learning/gen2/expert_rollout.py` |
| Versioned corpus storage | `learning/gen2/dataset.py` |
| Corpus collection CLI | `learning/gen2/collect_expert.py` |
| Recurrent actor-critic | `learning/gen2/recurrent_policy.py` |
| Recurrent behaviour cloning | `learning/gen2/bc_recurrent.py` |
| DAgger aggregation loop | `learning/gen2/dagger.py` |
| Closed-loop evaluation + unconditional survival | `learning/gen2/evaluation.py` |
| Pre-registered readiness gate | `learning/gen2/readiness.py` |

Tests live under `tests/learning/gen2/`.
