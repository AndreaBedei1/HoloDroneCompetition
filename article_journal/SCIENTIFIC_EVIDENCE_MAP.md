# Scientific evidence map — Marine Race Arena journal manuscript

Target venue: *Robotics and Autonomous Systems* (Elsevier).
Compiled on branch `paper/ras-journal`, worktree
`C:\Users\andrea.bedei3\Desktop\HoloDroneCompetition-paper`, base commit `3cfca76`.

Every row below was checked against a file that is present in this worktree, or —
where the raw artifact is git-ignored — against the read-only copy in the main
worktree. Nothing in this table was taken from prose documentation without
opening the underlying artifact.

## Status vocabulary

| Status | Meaning | May appear as |
| --- | --- | --- |
| **VALIDATED** | Complete, frozen, provenance-carrying evidence. | Headline claim, main table |
| **PRELIMINARY** | Real measurement, but limited scope, small `n`, or externally stored weights. | Explicitly hedged text; supporting table |
| **DIAGNOSTIC** | Instrumentation or audit output; informative about the method, not a performance claim. | Methods / limitations only |
| **DEVELOPMENT ONLY** | Intermediate, superseded, or still running. | Not in the manuscript |

---

## A. VALIDATED evidence

| Claim | Evidence file | Commit / provenance | Status | Paper destination |
| --- | --- | --- | --- | --- |
| 78-run HoloOcean benchmark matrix is complete and internally consistent (`run_count=78`, `complete_matrix=true`, 0 missing metadata) | `results/onboard_only_validation/final_20260715/complete_experiment_manifest.json` (main worktree; git-ignored here) | experiment commit `df098ef4…`; source-tree sha256 `e7d31077…`; HoloOcean 2.3.0; Python 3.9.25; `--dt 0.033` | VALIDATED | §10.1 protocol, Tables 5–7 |
| Clean-track completion and timing for both rule controllers on the three official circuits | same matrix, per-run `*_summary.json`; recomputed with `article/regenerate_tables.py --check` | reproduces the committed `article/tables/validation_results.tex` byte-for-byte | VALIDATED | Table 5 (RQ2) |
| Degradation under `medium` and `strong` Horseshoe Bay currents, incl. the non-monotonic collision count | same matrix, `currents/` subtree | same | VALIDATED | Table 6 (RQ3) |
| Homogeneous two-vehicle fleet: 24/24 team gates, 0 events, both controllers | same matrix, `fleet_gap90/` subtree | same | VALIDATED | Table 7 (RQ4) |
| Three-vehicle leader–follower coordination, LF(1) / LF(2) / uncoordinated at gaps 0 s and 8 s | same matrix, `coordination/` subtree | reproduces `article/tables/holoocean_coordination.tex` byte-for-byte | VALIDATED | Table 7 (RQ4) |
| Controller-local progression is close to, but never identical with, referee progression: 1474 referee vs 1472 local advancements, 1467 matched, 11 false-local, 7 missed-local, 0 event-consistency errors, 0 finish-order inversions, median local lag 0.858 s | `complete_experiment_manifest.json` → `local_vs_referee` | same matrix | VALIDATED | Table 4, §7.5 (information boundary) — **new for the journal** |
| Current-free autonomous completion of all three official circuits: 3/3 + 3/3 + 3/3 = 9/9, 0 collisions, 0 out-of-bounds, 0 wrong-direction crossings | `results/rl_public/visual_pose_v2/official_no_current/summary.json`, `summary.csv`, three `eval_results.csv`, three `evaluation_manifest.json` | `git_sha 8ab0a063…`; `adapter_actual=holoocean`; `fallback_allowed=false`; `currents_actual=[0,0,0]`; `dt=0.1`; seeds 1800–1802; track sha256 recorded | VALIDATED | Table 3 (RQ1) — **new for the journal** |
| Matched six-controller benchmark, 474/474 episodes, identical seeds and geometries: learned policies reach 90.6–100 % on short generated geometries but 0–40 % on the full official circuits, against 93.3 % for the rule baseline | `results/rl_public/final_benchmark/final_benchmark_report.md`, `aggregate_by_group.csv`, `episodes.csv`, `paired_comparisons.csv`, `package_manifest.json` | suite commit `67e3975f…`; package commit `a232d448…`; per-model sha256 recorded; `adapter=holoocean`, `current_profile=none`, `dt=0.1` | VALIDATED | Table 8 (RQ5) — **new for the journal** |
| Learned policies are significantly *faster* than the rule baseline on the geometries they solve (paired sign test over 79 shared episodes, p ≤ 0.001 for all three PPO checkpoints) | `results/rl_public/final_benchmark/final_benchmark_report.md` → "Paired controller comparisons" | same | VALIDATED | §10.6 (RQ5) |
| Pre-registered Generation-1 readiness gate FAILED (4 of 19 criteria); the frozen policy then completed 0/30 official-circuit trials while the paired rule controller completed 30/30 | `results/rl_public/ppo_final_holdout_929792/final_experiment_report.{md,json}`, `final_circuit_evaluations.jsonl` (60 hash-chained entries), `results/rl_public/ppo_final_readiness_929792/readiness_verdict.json` | decision commit `c3484bb5…`; protocol commit `bd1f961c…`; policy sha256 `60ffdca1…` | VALIDATED | §9.5, Table 9 (RQ5) — **new for the journal** |
| Unconditional multi-gate survival decays with sequence length while first-gate crossing stays at 1.00 (120/120 → 88/120 at gate 2 → 6/20 at gate 22) | `results/rl_public/ppo_final_holdout_929792/final_experiment_report.md` (readiness block) | same | VALIDATED | §10.6, Fig. 6 — **new for the journal** |
| Both named current profiles and seeded static obstacles instantiate on every circuit (construction checks, not avoidance trials) | `results/capability_checks/` (main worktree) | recorded in the conference manifest | VALIDATED | §10.1 |

## B. PRELIMINARY evidence

| Claim | Evidence file | Commit / provenance | Status | Paper destination |
| --- | --- | --- | --- | --- |
| Perception audit along the controller's own trajectory on all three official circuits: detection available in 88.3–91.7 % of frames, four valid aperture corners in 78.3–83.3 %, metric PnP in 48.3–53.3 %, median plane-orientation error 0.018–1.129°, no gross (>30°) orientation error, no sign error, no yaw jump >20°, multi-candidate association 1.00 | `artifacts_gen2/visual_collection_smoke/visual_smoke_report.json` and three per-track `summary.json` | captured 2026-09-07, `adapter=holoocean`, 60 frames per track, seeds 67000–67002; the report itself declares `diagnostic_only: true` | PRELIMINARY | §5.4, Table 2 — reported as a perception audit, never as a controller performance claim |
| Exploratory pose front end is unreliable under strong obliquity: on a dedicated ±25°/±45° rig, pose availability 11.7 %, median plane-yaw error 48.2°, orientation-class accuracy 0.00, sign accuracy 0.00 | `results/rl_public/visual_pose_v2/vision_pose/pose_metrics.json`, `metrics_by_condition.csv` | `git_sha 8ab0a063…`, `adapter_actual=holoocean`, 60 frames, 2.0 m apertures | PRELIMINARY (a *negative* result, stated as such) | §5.3, §12 limitations |
| 560-case matched candidate selection at difficulty G1: universal-transition success 0.658–0.830, full-sequence completion falls from 0.6–0.8 at 3 gates to 0.2–0.5 at 22 gates | `results/rl_public/ppo_final_matched_benchmark/matched_benchmark.json` | plan fingerprint `c9065d25…`; selection-rule fingerprint `3db79364…`; **checkpoints live in the `-sac` worktree, not in this repository** | PRELIMINARY | §9.6 supporting text only; excluded from the main learning table |
| Learned observation-v3 controller clears a two-gate straight sequence on 9/10 reserved seeds in real HoloOcean, but not a balanced turn | `results/rl_public/multigate_rl_v3/`, `docs/rl_multigate_policy.md` | recorded as "PARTIAL RL SUCCESS" in `docs/final_no_current_demo.md` | PRELIMINARY | not used; superseded by the 474-episode benchmark |
| Hybrid controller (rule backbone + learned visual servo) reaches 86.7 % on the official circuits | `results/rl_public/final_benchmark/final_benchmark_report.md` | same as the 474-episode suite | PRELIMINARY (not policy-only; the suite itself reports it separately) | Table 8, flagged as not fully learned |

## C. DIAGNOSTIC evidence

| Item | Evidence file | Why diagnostic | Paper destination |
| --- | --- | --- | --- |
| Per-difficulty ladder G1–G6 transition rates | `ppo_final_holdout_929792/final_experiment_report.md`, `results/rl_public/ppo_difficulty_ladder/` | Instrumentation of the readiness gate, not a deployment claim | §9.5 one sentence |
| Reward-shaping audit | `results/rl_public/reward_audit/` | Verifies the reward is bounded and non-cyclable | §9.3 one sentence |
| Environment reset / throughput calibration | `results/rl_public/reset_benchmark/`, `ppo_throughput_calibration/` | Engineering measurement of simulator throughput | §13 reproducibility, one sentence |
| Track-distribution histograms of the procedural course family | `results/rl_public/universal_transition_track_distribution/figures/*.png` | Describes the training distribution | §9.4, optional figure |

## D. DEVELOPMENT ONLY — explicitly excluded from the manuscript

| Item | Location | Reason for exclusion |
| --- | --- | --- |
| **Generation-2 recurrent BC → DAgger → PPO campaign** | `docs/rl_gen2_progress.md`, `docs/rl_gen2_track_specific.md`, `artifacts_gen2/**` | **Training was running while this manuscript was written** (`train_ppo_27d_speed`, campaign `B_20260908_retry6`, PID 66308). No frozen checkpoint, no readiness verdict, no holdout. Nothing from Gen-2 is claimed. |
| SAC campaign, 7 runs (v1–v7) | `docs/sac_universal_transition.md`, `results/rl_public/sac_*` | No deployable policy; v7 ended in `baseline_collapse_rollback` with `do_not_resume: true`. Mentioned in one sentence as a negative methodological note only. |
| Intermediate universal-transition selection and capacity runs | `results/rl_public/universal_transition_*` | Superseded by the frozen readiness verdict. |
| Curriculum dry validation, stage1/stage2 packages, multigate longrun and reliability-first intermediates | `results/rl_public/{curriculum_dry_validation,stage1,stage2,multigate_rl_*}` | Development stages of the pipeline that produced the frozen checkpoints. |
| 27-D observation contract and the conservative 27-D smoke evaluation | `marine_race_arena/learning/config_local_transition_27d.py`, `artifacts_gen2/**` | Active Gen-2 work, not frozen. |

---

## Provenance conventions used in the manuscript

1. Every quantitative claim traces to a file listed above, not to a `docs/*.md` narrative.
2. The 78-run matrix (`dt = 0.033` s, 30 Hz control) and the later current-free /
   learning evaluations (`dt = 0.1` s, 10 Hz control) are **never pooled**, and the
   manuscript states the control rate wherever a time is reported.
3. Where the raw artifact is git-ignored (`results/onboard_only_validation/**`),
   the manuscript cites the aggregation entry point
   (`article/regenerate_tables.py`) and the recorded source-tree fingerprint.
4. Model weights that live only in the `HoloDroneCompetition-sac` worktree are
   identified by SHA-256 and never used to support a headline claim.

## Deliberate omissions

* No result is reported from any run started after the manuscript was drafted.
* No Generation-2 number appears anywhere in the manuscript.
* The three official circuits are **not** described as unseen holdouts: the
  Generation-1 exploratory holdout opened them and its access ledger records all
  60 accesses (`docs/rl_generations.md` §4). Any later learned-controller result
  on those circuits is labelled a retrospective diagnostic benchmark.
