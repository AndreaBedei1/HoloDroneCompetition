# Stage-1 Learned-Controller — Public Audit Package

This directory holds **compact, externally inspectable evidence** for the Stage-1
behavioral-cloning (BC) controller of Marine Race Arena. The heavy artifacts
(raw `.npz` datasets, SB3 checkpoints, logs) stay under the git-ignored
`results/rl/`; only audit-relevant JSON/CSV plus the small BC model (~0.3 MB) are
committed here. Everything is **derived from real result files** by
`marine_race_arena/learning/build_public_package.py` — nothing is fabricated.

All results below use the **real HoloOcean adapter** (no fallback), the **unchanged
Stage-1 track and referee** (official `1.5×1.5 m` aperture, `0.10 m` clearance
margin), and **onboard-only controller observations** (36-dim, `onboard_only_v1`;
feature names in `result_manifest.json`). The controller integrates through the
normal runner via `--controller rl_gate_controller --controller-model-path <model>`.

## What was evaluated (read this carefully)

The single-gate task was demonstrated with the rule expert
`rule_gate_center_then_commit` and cloned. Three development evaluations
(`evaluation/dev_history.json`) tell the real story — **open-loop imitation error
does not predict closed-loop success**:

| Development evaluation | Demonstrations | Held-out eval condition | Completion |
| --- | --- | --- | --- |
| Fixed-start | 21 fixed-start demos | **fixed start**, no randomization | **0 %** |
| First randomized | 18 randomized demos | randomized start | 69 % |
| Combined randomized | 34 randomized demos | randomized start | **100 % (20/20)** |

**Classification of the headline 20/20:** it is a **randomized-start (Stage-2
condition)** held-out evaluation of the combined model (`--randomize` was applied
to disjoint seeds 400–419). It is **not** a fixed-start Stage-1 evaluation. The
fixed-start demonstrations gave 0 % closed-loop despite a tiny open-loop MSE — a
real covariate-shift failure — which the Stage-2 start randomization fixes.

Because a randomized-start pass does not by itself establish the **fixed-start
Stage-1** criterion, two independent **frozen 50-seed** evaluations were run on
unused seeds with the corrected metric/randomization code:

- `evaluation_fixed_50/`  — **Evaluation A**: fixed start (Stage-1 condition), seeds 1000–1049.
- `evaluation_randomized_50/` — **Evaluation B**: randomized start (Stage-2 condition), seeds 1100–1149.

Stage 1 passes **only** if Evaluation A ≥ 90 %; Stage 2 passes **only** if
Evaluation B ≥ 90 %. See each directory's `eval_summary.json` for the point
estimate and Wilson 95 % interval, and `frozen_evaluations.md` / `result_manifest.json`
for the verdict.

## Layout

```
result_manifest.json           full provenance (model+dataset hashes, track, referee, obs contract, verdicts)
dataset/                       demonstration episode manifest, hashes, summary (raw .npz stays in results/rl/)
bc/                            BC training report, dataset inspection, per-epoch log, model hash, and the model
evaluation/                    development combined-randomized 20/20 (per-seed results.json/.csv), dev_history,
                               seed_split, randomization_manifest
evaluation_fixed_50/           Evaluation A — frozen fixed-start Stage-1, 50 held-out seeds
evaluation_randomized_50/      Evaluation B — frozen randomized-start Stage-2, 50 held-out seeds
smoke/                         real-HoloOcean rule smoke + PPO plumbing smoke (300 steps; not a trained policy)
reproduction/                  exact commands to reproduce collection, training and both evaluations; environment
```

## Reproduce

Set up the `marine_race_rl` environment (Python 3.9, HoloOcean 2.3.0 from source,
`requirements.txt`, `requirements-rl.txt`), then follow
`reproduction/reproduce_bc_training.txt` and
`reproduction/reproduce_bc_evaluation.txt`. Model and dataset SHA-256 hashes in
`bc/model_hash.json` and `dataset/dataset_hashes.json` let you verify the exact
artifacts even though the heavy files remain local.
