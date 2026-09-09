# Claim audit

Every quantitative claim in the manuscript, its source artifact, and its
verification status. The audit is executable:

```bash
python article_journal/scripts/verify_claims.py
```

The script opens the frozen artifacts, recomputes each value, prints a
per-claim verdict and exits non-zero on any mismatch. It launches nothing and
modifies nothing.

**Last run: 95 checks, 95 verified, 0 mismatched, 0 skipped.**

Rows below are grouped by the section that states the claim. "Verified" means the
script recomputes the value from the artifact named in the Source column; a row
marked *derived* is computed in the script from other verified values.

## Editorial compression disposition

The audit remains artifact-based after the hierarchy pass. The main manuscript
retains the overall learning comparison and the unconditional survival figure;
paired timing, the full readiness table and the full holdout table remain
available as supporting artifacts. Detailed PPO hyperparameters, curriculum
mixtures and checkpoint-freeze mechanics are verified below but are no longer
repeated in the main prose.

---

## Section 5 — Onboard gate perception

| Claim | Source | Verified |
| --- | --- | --- |
| Detection availability 0.900 / 0.883 / 0.917 on the three circuits | `artifacts_gen2/visual_collection_smoke/visual_smoke_report.json` | **YES** |
| Four-corner aperture availability 0.817 / 0.783 / 0.833 | same | **YES** |
| Metric PnP availability 0.517 / 0.483 / 0.533 | same | **YES** |
| Median plane-orientation error 0.018 / 1.129 / 0.064 deg | same | **YES** |
| p90 orientation error 1.204 / 4.287 / 0.064 deg | same | **YES** |
| Gross orientation error (>30 deg) rate 0.000 on all three | same | **YES** |
| Multi-candidate frames 7 / 4 / 11 | same | **YES** |
| Multi-candidate association correct = 1.000 on all three | same | **YES** |
| Online target switches = 0 on all three | same | **YES** |
| 60 frames per circuit, seeds 67000–67002, adapter `holoocean` | same | **YES** |
| Mixed Endurance in-view rates 0.912 / 0.825 over 57 in-view frames | same | **YES** |
| Oblique rig: pose availability 11.7 % | `results/rl_public/visual_pose_v2/vision_pose/pose_metrics.json` | **YES** |
| Oblique rig: median plane-yaw error 48.2 deg, p90 50.3 deg | same | **YES** |
| Oblique rig: orientation-class accuracy 0.00, sign accuracy 0.00 | same | **YES** |
| Oblique rig: median distance error 1.73 m, 60 frames | same | **YES** |
| Detector thresholds: Canny (30, 80); size ≥ 5 %; aspect [0.45, 2.40]; perimeter ratio [0.50, 2.20]; area ≥ 0.008; confidence weights 0.38 / 0.24 / 0.20 / 0.18; accept ≥ 0.38; dedupe 0.04; top 8 | `marine_race_arena/controllers/vision.py` | **YES** (read from source) |
| Association gates: FOV 70 deg; mismatch 32 deg; near-field 1.6 m with area ≥ 0.05 or fraction ≥ 0.30; side offsets 0.10 / 0.18; bearing penalty 0.050 | same | **YES** (read from source) |
| Camera intrinsics f = 320, principal point (320, 240) from 640×480 at 90 deg FOV | `marine_race_arena/controllers/gate_pose.py`, `docs/visual_gate_pose.md` | **YES** |

## Section 7 — Referee and the information boundary

| Claim | Source | Verified |
| --- | --- | --- |
| 1474 referee advancements | 78-run matrix `complete_experiment_manifest.json`, `local_vs_referee.overall` | **YES** |
| 1472 local advancements, 1467 matched | same | **YES** |
| 11 false and 7 missed local advancements | same | **YES** |
| Median local delay 0.858 s, p95 1.353 s | same | **YES** |
| Mean delay 0.722 s, sd 2.83 s, minimum −90.849 s | same | **YES** |
| 0 event-consistency errors | same | **YES** |
| 0 finish-order inversions over 61 pairwise comparisons | same | **YES** |
| 0 cases of local finish before referee; 103 referee-before-local | same | **YES** |
| Per-circuit split 1120 / 159 / 195 referee advancements | same, `local_vs_referee.by_track` | **YES** |
| False and missed shares 0.75 % and 0.47 %; matched share 99.5 % | *derived* | **YES** |
| Referee penalties: 5.0 s collision, 5.0 s obstacle, 10.0 s out of bounds, 15.0 s stuck; 1.0 s cooldowns | `marine_race_arena/` referee configuration and track JSON | **YES** (read from source) |
| Stuck threshold 0.02 m/s; 45 s on Horseshoe Bay, 65 s on the other two | same | **YES** (read from source) |

## Section 10.1 — Protocol

| Claim | Source | Verified |
| --- | --- | --- |
| 78 runs, matrix complete, no missing metadata | `complete_experiment_manifest.json` | **YES** |
| Source-tree fingerprint `e7d31077…`, experiment commit `df098ef4…` | same, `article/REVISION_MANIFEST.md` | **YES** |
| Matrix control timestep 0.033 s | per-run `reproduction_command` (`--dt 0.033`) | **YES** |
| HoloOcean 2.3.0, Python 3.9.25, Windows 10 build 26200 | `complete_experiment_manifest.json`, `REVISION_MANIFEST.md` | **YES** |

## Section 10.2 — RQ1, current-free completion

| Claim | Source | Verified |
| --- | --- | --- |
| 3/3 + 3/3 + 3/3 = 9/9 completions | `results/rl_public/visual_pose_v2/official_no_current/summary.json` | **YES** |
| 51 of 51 gates | same | **YES** |
| 0 collisions, 0 out-of-bounds, 0 wrong-direction crossings | same | **YES** |
| Times 236.0 ± 6.9, 308.6 ± 6.5, 480.4 ± 15.7 s | three `eval_results.csv`, recomputed | **YES** |
| Penalized time equals official time in every run | same | **YES** |
| `adapter_actual = holoocean`, `fallback_allowed = false`, `currents_actual = [0,0,0]` per circuit | same + `evaluation_manifest.json` | **YES** |
| Control timestep 0.1 s, seeds 1800–1802 | `evaluation_manifest.json` | **YES** |
| Course progress 0.40 / 0.38 / 0.43 m/s | *derived* from track lengths and times | **YES** |
| Mixed Endurance uses a `clean_gate` validation override | `summary.json` `benchmark_task_override` | **YES** |

## Section 10.3 — RQ2, controller comparison

| Claim | Source | Verified |
| --- | --- | --- |
| Horseshoe Bay 5/5 both, 12.0/12 gates, 194.5 ± 4.8 vs 197.7 ± 7.1 s | 78-run matrix via `article/regenerate_tables.py --check` | **YES** (byte-identical to committed table) |
| Vertical Serpent 5/5 vs 4/5, 17.0 vs 14.8 gates, 236.9 ± 6.4 vs 241.1 ± 1.8 s | same | **YES** |
| Mixed Endurance 2/5 vs 5/5, 17.0 vs 22.0 gates, 394.7 ± 3.9 vs 410.7 ± 8.8 s | same | **YES** |
| Zero collisions, out-of-bounds and stuck on every clean run | same | **YES** |
| Staged controller 3.2 s (1.7 %) slower on Horseshoe Bay | *derived* | **YES** |

## Section 10.4 — RQ3, currents

| Claim | Source | Verified |
| --- | --- | --- |
| Medium: 2/5 vs 3/5, 8.4 vs 8.8 gates, 43.4 vs 25.2 collisions | 78-run matrix via `regenerate_tables.py --check` | **YES** |
| Medium finished-only times 337.9 ± 19.2 / 692.9 ± 75.7 and 217.3 ± 1.5 / 223.9 ± 1.8 s | same | **YES** |
| Strong: 0/5 both, 3.0 vs 3.2 gates, 0.0 vs 0.8 collisions | same | **YES** |
| Strong set-flow $[0.75, 1.05, 0]$ m/s, magnitude 1.29 m/s | track JSON `current_profiles`; *derived* magnitude | **YES** |
| Medium constant component 0.65 m/s (0.5× strong) | same | **YES** |
| Clean Horseshoe mean course speed ≈ 0.48 m/s (93.8 m in 194.5 s) | *derived* | **YES** |
| Strong is ≈ 2.7× the clean course speed | *derived* | **YES** |
| Medium-current time gap 120.6 s | *derived* | **YES** |

## Section 10.5 — RQ4, fleet and coordination

| Claim | Source | Verified |
| --- | --- | --- |
| Two-vehicle fleet 24/24 team gates, 5/5, zero events, 303.1 ± 3.2 and 308.0 ± 5.8 s | 78-run matrix via `regenerate_tables.py --check` | **YES** |
| Uncoordinated gap 8 s: 2/3, 34.3/36, 127.3 collisions, 2.0 proximity | same | **YES** |
| Uncoordinated gap 0 s: 3/3, 36.0/36, 9.3 collisions, 2.3 proximity | same | **YES** |
| LF(1): 3/3 and zero events at both gaps, 266.0 ± 8.3 and 262.3 ± 7.7 s | same | **YES** |
| LF(2): 285.7 ± 3.8 and 288.3 ± 1.3 s; one stuck event per run at gap 0 s | same | **YES** |
| LF(2) penalized 303.3 ± 1.3 s at gap 0 s; uncoordinated 266.4 ± 64.1 and 564.0 ± 463.0 s | same | **YES** |
| Leader runs Continuous Servo, both followers Center-then-Commit | matrix `fleet_configuration` per run | **YES** |
| Freshness window 2.5 s; three transmitted fields | `marine_race_arena/controllers/leader_follower.py` | **YES** (read from source) |

## Section 10.6 — RQ5, learned controllers

| Claim | Source | Verified |
| --- | --- | --- |
| 474 of 474 episodes planned and run | `results/rl_public/final_benchmark/package_manifest.json` | **YES** |
| Adapter `holoocean`, currents `none`, dt 0.1 s | same | **YES** |
| Generated-geometry completion 100.0 / 100.0 / 100.0 / 96.9 / 90.6 / 89.1 % | `aggregate_by_group.csv`, group `ALL_non_official` | **YES** |
| Official-circuit completion 93.3 / 86.7 / 40.0 / 26.7 / 20.0 / 0.0 % | same, group `ALL_official` | **YES** |
| Wilson 95 % intervals as tabulated | same (columns `success_rate_wilson95_*`) | **YES** |
| Safety-episode and collision-event counts as tabulated | same | **YES** |
| Paired: 79 shared episodes per controller | `final_benchmark_report.md`, paired comparisons | **YES** |
| Paired Δt −6.77 / −7.02 / −7.05 s with CIs; sign-test p ≤ 0.001 | same | **YES** |
| Paired reliability p 0.0005 / 0.0010 / 0.0001 | same | **YES** |
| 32 PPO official failures; 29 missed-gate DNF (90.6 %); 3 time limits | `episodes.csv`, recomputed | **YES** |
| Median PPO failure at 23.9 % of the course | same, recomputed | **YES** |
| Readiness gate FAIL on 4 of 19 criteria, no override | ppo_final_readiness_929792/readiness_verdict.json (supporting artifact; summarized in Section 10.6) | **YES** |
| Readiness observed values 0.820 / 1.000 / 0.882 / 0.852 and per-length completions | same (supporting artifact) | **YES** |
| Survival 1.000, 0.733, 0.717, 0.670, 0.575, 0.533, 0.375, 0.300 | same; main text retains the headline points and Fig. 6 | **YES** |
| Holdout PPO 0/30, rules 30/30, 87 of 510 gates | ppo_final_holdout_929792/final_experiment_report.md and .json (supporting artifact) | **YES** |
| Hash-chained holdout ledger of 60 entries | `final_circuit_evaluations.jsonl` | **YES** |
| Policy SHA-256 `60ffdca1…` | `final_experiment_report.md`, `artifact_freeze.json` | **YES** |

## Sections 9 and 13 — methodology and environment

| Claim | Source | Verified |
| --- | --- | --- |
| 35-feature observation; group sizes 7 / 5 / 4 / 6 / 4 / 9 | `marine_race_arena/learning/config_local_transition.py` (asserted `== 35`) | **YES** |
| PPO hyperparameters: lr 3e-5 → 5e-6 linear, n_steps 2048, batch 256, 4 epochs, γ 0.995, λ 0.95, clip 0.10, ent 1e-3, vf 0.5, grad-norm 0.5, initial σ 0.12 | `marine_race_arena/learning/longrun_config.py` | **YES** (read from source) |
| Network 256×256, tanh, `MlpPolicy`, frame stack 1 | `longrun_config.py`, `rl_train.py`, `docs/rl_generations.md` | **YES** |
| KL guard 0.01 / 0.02 / 0.03, at most one automatic change | `longrun_config.py` | **YES** |
| Curriculum buckets 2 / 3–5 / 6–12 / 13–24 and four stage mixtures | `marine_race_arena/learning/generic_sequence_curriculum.py` | **YES** |
| Promotion thresholds, e.g. S1 at 0.87 and 0.82 over two evaluations | same | **YES** |
| Stable-Baselines3 2.7.1, Gymnasium 1.0.0, PyTorch 2.8.0 | `requirements-rl.txt`, artifact provenance blocks | **YES** |
| Gate apertures 1.5 m × 1.5 m on all three circuits | track JSON `gate_inner_size_m` | **YES** |
| Track lengths 93.8 / 118.3 / 206.3 m; gates 12 / 17 / 22 | track JSON, `article/tables/official_tracks.tex` | **YES** |
| Thruster mixing gain κ = 0.35 | `marine_race_arena/` HoloOcean adapter | **YES** (read from source) |
| Beacon noise levels {0.2, 0.45, 0.6} and dropout ≤ 0.04 | track JSON `beacon` blocks | **YES** |
| Beacon offset default (0, 0, 0.35) m along the gate-up axis | beacon configuration | **YES** |

## Explicitly excluded from the manuscript

These were checked and deliberately **not** used. Each would be a claim the
evidence does not support at the strength a journal reader would assume.

| Excluded claim | Reason |
| --- | --- |
| Any Generation-2 recurrent BC / DAgger / PPO number | Training was in progress while the manuscript was written; no frozen checkpoint, no readiness verdict, no holdout. |
| 560-case matched candidate selection (`ppo_final_matched_benchmark`) as a headline result | Difficulty G1 only, and the checkpoints live outside this repository. Not cited as a main result. |
| Any SAC result | Seven runs, no deployable policy; v7 ended in `baseline_collapse_rollback`. |
| "The first underwater racing benchmark" | Not supportable from the literature review; the manuscript says "to the best of our knowledge" only where the claim is narrow. |
| Obstacle-avoidance performance | Only construction checks were run; no avoidance trial exists. |
| Acoustic channel characterization | Parameters were never swept; the manuscript says so explicitly. |
| Pose-aware perception as an accurate pose estimator | Measured and reported as a negative result instead. |
| Any comparison of a 30 Hz time against a 10 Hz time | Control rate differs by 3×; the manuscript states the rate in every table and never pools them. |
