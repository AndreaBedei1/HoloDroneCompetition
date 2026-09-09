# Final common benchmark: reliability-first PPO vs rule, hybrid and BC-v3

Commit `67e3975f82a1140e1ec69d60fc6cc443c29a4b9a` | adapter `holoocean` | currents `none` | dt 0.1 s | 474/474 episodes.

Every controller ran the identical suite: same test groups, same generated and official geometries, same seeds, same adapter and the same current-free simulator configuration. Timing figures are reported per test group only, and every timing comparison is a paired per-seed difference within one group.

## Controllers under test

| key | kind | policy-only | model | note |
|---|---|---|---|---|
| ppo_525678 | ppo | yes | `results\rl\multigate_reliability_first\r2_reliability_first_seed23001_c3_recovery_476525\checkpoints\ppo_525678_steps.zip` | training-metadata best_reliable and best_fast_reliable alias |
| ppo_900462 | ppo | yes | `results\rl\multigate_reliability_first\r2_reliability_first_seed23001_c3_recovery_476525\checkpoints\ppo_900462_steps.zip` | latest safe checkpoint recorded while the run was in stage C4 |
| ppo_1000814 | ppo | yes | `results\rl\multigate_reliability_first\r2_reliability_first_seed23001_c3_recovery_476525\checkpoints\ppo_1000814_steps.zip` | final checkpoint of the completed reliability-first run |
| bc_v3 | bc | yes | `results/rl/multigate_longrun/bc_v3_balanced_v2_20260728/bc_v3.pt` | behaviour-cloning reference the PPO retention term regularises towards |
| rule_gate_center_then_commit | rule | no | - | no learned component |
| hybrid | hybrid | no | `results/rl_public/stage1/bc/model/best_model.pt` | reported separately: deterministic backbone blended with a learned servo |

## Aggregate comparison (all non-official test groups)

| controller | episodes | success | 95% CI | full-circuit | safety ep | coll ev | OOB ev | wrong-dir | prev-gate ret | timeout | mean jerk | mean pen. t (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bc_v3 | 64 | 89.1% | [0.79, 0.95] | 89.1% | 3 | 36 | 0 | 0 | 0 | 1.6% | 0.0290 | 24.55 |
| hybrid | 64 | 100.0% | [0.94, 1.00] | 100.0% | 2 | 35 | 0 | 0 | 0 | 0.0% | 0.0213 | 25.40 |
| ppo_1000814 | 64 | 96.9% | [0.89, 0.99] | 96.9% | 13 | 156 | 0 | 0 | 0 | 0.0% | 0.0430 | 28.68 |
| ppo_525678 | 64 | 100.0% | [0.94, 1.00] | 100.0% | 4 | 16 | 0 | 0 | 3 | 0.0% | 0.0451 | 16.93 |
| ppo_900462 | 64 | 90.6% | [0.81, 0.96] | 90.6% | 6 | 14 | 0 | 0 | 2 | 0.0% | 0.0407 | 14.77 |
| rule_gate_center_then_commit | 64 | 100.0% | [0.94, 1.00] | 100.0% | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.0211 | 20.13 |

## Aggregate comparison (three official current-free circuits)

| controller | episodes | success | 95% CI | full-circuit | safety ep | coll ev | OOB ev | wrong-dir | prev-gate ret | timeout | mean jerk | mean pen. t (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bc_v3 | 15 | 0.0% | [0.00, 0.20] | 0.0% | 10 | 1200 | 468 | 10 | 2 | 46.7% | 0.0475 | n/a |
| hybrid | 15 | 86.7% | [0.62, 0.96] | 86.7% | 2 | 31 | 0 | 0 | 0 | 6.7% | 0.0317 | 371.35 |
| ppo_1000814 | 15 | 26.7% | [0.11, 0.52] | 26.7% | 10 | 1516 | 33 | 2 | 2 | 6.7% | 0.0503 | 162.53 |
| ppo_525678 | 15 | 20.0% | [0.07, 0.45] | 20.0% | 13 | 1434 | 92 | 3 | 1 | 13.3% | 0.0515 | 195.53 |
| ppo_900462 | 15 | 40.0% | [0.20, 0.64] | 40.0% | 8 | 345 | 0 | 4 | 1 | 0.0% | 0.0491 | 259.93 |
| rule_gate_center_then_commit | 15 | 93.3% | [0.70, 0.99] | 93.3% | 1 | 3 | 0 | 0 | 0 | 0.0% | 0.0323 | 308.99 |

## Per-group results

### single_gate_retention

_Timing columns are `n/a`: the official clock is first-gate-to-last-gate, so a single-gate case has a definitionally zero completion time._

| controller | succ/tot | success | 95% CI | full-circuit | gates | mean t (s) | median t (s) | t/gate (s) | pen. t (s) | coll ev/fr/ep | OOB ev/ep | wrong-dir | prev-gate ret | jerk | path (m) | timeout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bc_v3 | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 8/8 | n/a | n/a | n/a | n/a | 0/0/0 | 0/0 | 0 | 0 | 0.0374 | 5.9 | 0.0% |
| hybrid | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 8/8 | n/a | n/a | n/a | n/a | 0/0/0 | 0/0 | 0 | 0 | 0.0239 | 5.6 | 0.0% |
| ppo_1000814 | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 8/8 | n/a | n/a | n/a | n/a | 0/0/0 | 0/0 | 0 | 0 | 0.0462 | 5.9 | 0.0% |
| ppo_525678 | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 8/8 | n/a | n/a | n/a | n/a | 0/0/0 | 0/0 | 0 | 0 | 0.0398 | 5.9 | 0.0% |
| ppo_900462 | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 8/8 | n/a | n/a | n/a | n/a | 0/0/0 | 0/0 | 0 | 0 | 0.0360 | 5.9 | 0.0% |
| rule_gate_center_then_commit | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 8/8 | n/a | n/a | n/a | n/a | 0/0/0 | 0/0 | 0 | 0 | 0.0291 | 5.7 | 0.0% |

Paired vs the rule baseline on identical seeds:

| A | paired eps | only A finished | only rule finished | mean dt A-rule (s) | A faster | rule faster | time sign p |
|---|---|---|---|---|---|---|---|
| bc_v3 | 8 | 0 | 0 | n/a | 0 | 0 | n/a |
| hybrid | 8 | 0 | 0 | n/a | 0 | 0 | n/a |
| ppo_1000814 | 8 | 0 | 0 | n/a | 0 | 0 | n/a |
| ppo_525678 | 8 | 0 | 0 | n/a | 0 | 0 | n/a |
| ppo_900462 | 8 | 0 | 0 | n/a | 0 | 0 | n/a |

### two_gate_straight

| controller | succ/tot | success | 95% CI | full-circuit | gates | mean t (s) | median t (s) | t/gate (s) | pen. t (s) | coll ev/fr/ep | OOB ev/ep | wrong-dir | prev-gate ret | jerk | path (m) | timeout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bc_v3 | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 16/16 | 9.06 | 9.10 | 4.53 | 9.06 | 0/0/0 | 0/0 | 0 | 0 | 0.0202 | 9.7 | 0.0% |
| hybrid | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 16/16 | 8.54 | 8.55 | 4.27 | 8.54 | 0/0/0 | 0/0 | 0 | 0 | 0.0111 | 9.6 | 0.0% |
| ppo_1000814 | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 16/16 | 6.58 | 6.55 | 3.29 | 6.58 | 0/0/0 | 0/0 | 0 | 0 | 0.0341 | 10.0 | 0.0% |
| ppo_525678 | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 16/16 | 8.06 | 7.20 | 4.03 | 8.06 | 0/0/0 | 0/0 | 0 | 0 | 0.0411 | 10.6 | 0.0% |
| ppo_900462 | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 16/16 | 7.01 | 7.05 | 3.51 | 7.01 | 0/0/0 | 0/0 | 0 | 0 | 0.0297 | 9.9 | 0.0% |
| rule_gate_center_then_commit | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 16/16 | 8.62 | 8.70 | 4.31 | 8.62 | 0/0/0 | 0/0 | 0 | 0 | 0.0086 | 9.6 | 0.0% |

Paired vs the rule baseline on identical seeds:

| A | paired eps | only A finished | only rule finished | mean dt A-rule (s) | A faster | rule faster | time sign p |
|---|---|---|---|---|---|---|---|
| bc_v3 | 8 | 0 | 0 | 0.44 [-0.07, 0.94] | 1 | 7 | 0.0703 |
| hybrid | 8 | 0 | 0 | -0.09 [-0.21, 0.03] | 6 | 1 | 0.1250 |
| ppo_1000814 | 8 | 0 | 0 | -2.05 [-2.26, -1.84] | 8 | 0 | 0.0078 |
| ppo_525678 | 8 | 0 | 0 | -0.56 [-2.79, 1.66] | 7 | 1 | 0.0703 |
| ppo_900462 | 8 | 0 | 0 | -1.61 [-1.76, -1.46] | 8 | 0 | 0.0078 |

### two_gate_left

| controller | succ/tot | success | 95% CI | full-circuit | gates | mean t (s) | median t (s) | t/gate (s) | pen. t (s) | coll ev/fr/ep | OOB ev/ep | wrong-dir | prev-gate ret | jerk | path (m) | timeout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bc_v3 | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 16/16 | 22.66 | 20.10 | 11.33 | 22.66 | 0/0/0 | 0/0 | 0 | 0 | 0.0301 | 11.9 | 0.0% |
| hybrid | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 16/16 | 17.94 | 17.45 | 8.97 | 17.94 | 0/0/0 | 0/0 | 0 | 0 | 0.0271 | 11.0 | 0.0% |
| ppo_1000814 | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 16/16 | 8.96 | 8.25 | 4.48 | 8.96 | 0/0/0 | 0/0 | 0 | 0 | 0.0459 | 10.9 | 0.0% |
| ppo_525678 | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 16/16 | 7.95 | 7.30 | 3.98 | 7.95 | 0/7/0 | 0/0 | 0 | 0 | 0.0477 | 10.9 | 0.0% |
| ppo_900462 | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 16/16 | 9.76 | 8.95 | 4.88 | 9.76 | 0/0/0 | 0/0 | 0 | 0 | 0.0469 | 11.6 | 0.0% |
| rule_gate_center_then_commit | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 16/16 | 18.52 | 17.80 | 9.26 | 18.52 | 0/0/0 | 0/0 | 0 | 0 | 0.0250 | 11.3 | 0.0% |

Paired vs the rule baseline on identical seeds:

| A | paired eps | only A finished | only rule finished | mean dt A-rule (s) | A faster | rule faster | time sign p |
|---|---|---|---|---|---|---|---|
| bc_v3 | 8 | 0 | 0 | 4.14 [-5.70, 13.97] | 2 | 5 | 0.4531 |
| hybrid | 8 | 0 | 0 | -0.59 [-4.64, 3.47] | 5 | 3 | 0.7266 |
| ppo_1000814 | 8 | 0 | 0 | -9.56 [-15.97, -3.15] | 7 | 1 | 0.0703 |
| ppo_525678 | 8 | 0 | 0 | -10.57 [-16.39, -4.76] | 8 | 0 | 0.0078 |
| ppo_900462 | 8 | 0 | 0 | -8.76 [-14.20, -3.32] | 7 | 1 | 0.0703 |

### two_gate_right

| controller | succ/tot | success | 95% CI | full-circuit | gates | mean t (s) | median t (s) | t/gate (s) | pen. t (s) | coll ev/fr/ep | OOB ev/ep | wrong-dir | prev-gate ret | jerk | path (m) | timeout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bc_v3 | 7/8 | 87.5% | [0.53, 0.98] | 87.5% | 15/16 | 16.34 | 17.20 | 8.17 | 16.34 | 0/0/0 | 0/0 | 0 | 0 | 0.0277 | 12.2 | 12.5% |
| hybrid | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 16/16 | 17.52 | 14.85 | 8.76 | 17.52 | 0/0/0 | 0/0 | 0 | 0 | 0.0235 | 11.2 | 0.0% |
| ppo_1000814 | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 16/16 | 11.03 | 10.00 | 5.51 | 11.03 | 0/0/0 | 0/0 | 0 | 0 | 0.0452 | 11.6 | 0.0% |
| ppo_525678 | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 16/16 | 13.53 | 12.60 | 6.76 | 13.53 | 0/47/0 | 0/0 | 0 | 0 | 0.0497 | 13.0 | 0.0% |
| ppo_900462 | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 16/16 | 10.51 | 10.45 | 5.26 | 10.51 | 0/0/0 | 0/0 | 0 | 0 | 0.0414 | 11.1 | 0.0% |
| rule_gate_center_then_commit | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 16/16 | 14.51 | 14.40 | 7.26 | 14.51 | 0/0/0 | 0/0 | 0 | 0 | 0.0222 | 11.1 | 0.0% |

Paired vs the rule baseline on identical seeds:

| A | paired eps | only A finished | only rule finished | mean dt A-rule (s) | A faster | rule faster | time sign p |
|---|---|---|---|---|---|---|---|
| bc_v3 | 8 | 0 | 1 | 2.29 [-1.37, 5.94] | 2 | 5 | 0.4531 |
| hybrid | 8 | 0 | 0 | 3.01 [-2.88, 8.90] | 3 | 5 | 0.7266 |
| ppo_1000814 | 8 | 0 | 0 | -3.49 [-6.09, -0.88] | 7 | 0 | 0.0156 |
| ppo_525678 | 8 | 0 | 0 | -0.99 [-3.09, 1.12] | 5 | 3 | 0.7266 |
| ppo_900462 | 8 | 0 | 0 | -4.00 [-6.98, -1.02] | 7 | 1 | 0.0703 |

### vertical_low_to_high

| controller | succ/tot | success | 95% CI | full-circuit | gates | mean t (s) | median t (s) | t/gate (s) | pen. t (s) | coll ev/fr/ep | OOB ev/ep | wrong-dir | prev-gate ret | jerk | path (m) | timeout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bc_v3 | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 16/16 | 18.23 | 14.60 | 9.11 | 18.23 | 0/1/0 | 0/0 | 0 | 0 | 0.0261 | 12.3 | 0.0% |
| hybrid | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 16/16 | 30.79 | 18.15 | 15.39 | 50.16 | 31/288/1 | 0/0 | 0 | 0 | 0.0266 | 12.4 | 0.0% |
| ppo_1000814 | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 16/16 | 10.90 | 10.50 | 5.45 | 10.90 | 0/0/0 | 0/0 | 0 | 0 | 0.0478 | 12.2 | 0.0% |
| ppo_525678 | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 16/16 | 10.74 | 9.40 | 5.37 | 10.74 | 0/0/0 | 0/0 | 0 | 0 | 0.0492 | 12.3 | 0.0% |
| ppo_900462 | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 16/16 | 11.20 | 10.45 | 5.60 | 11.20 | 0/0/0 | 0/0 | 0 | 1 | 0.0484 | 12.5 | 0.0% |
| rule_gate_center_then_commit | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 16/16 | 17.66 | 16.35 | 8.83 | 17.66 | 0/0/0 | 0/0 | 0 | 0 | 0.0244 | 11.4 | 0.0% |

Paired vs the rule baseline on identical seeds:

| A | paired eps | only A finished | only rule finished | mean dt A-rule (s) | A faster | rule faster | time sign p |
|---|---|---|---|---|---|---|---|
| bc_v3 | 8 | 0 | 0 | 0.56 [-5.78, 6.90] | 2 | 6 | 0.2891 |
| hybrid | 8 | 0 | 0 | 13.12 [-9.89, 36.14] | 1 | 7 | 0.0703 |
| ppo_1000814 | 8 | 0 | 0 | -6.76 [-13.52, -0.01] | 7 | 1 | 0.0703 |
| ppo_525678 | 8 | 0 | 0 | -6.92 [-12.25, -1.60] | 8 | 0 | 0.0078 |
| ppo_900462 | 8 | 0 | 0 | -6.46 [-11.62, -1.30] | 8 | 0 | 0.0078 |

### vertical_high_to_low

| controller | succ/tot | success | 95% CI | full-circuit | gates | mean t (s) | median t (s) | t/gate (s) | pen. t (s) | coll ev/fr/ep | OOB ev/ep | wrong-dir | prev-gate ret | jerk | path (m) | timeout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bc_v3 | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 16/16 | 14.04 | 13.85 | 7.02 | 14.04 | 0/0/0 | 0/0 | 0 | 0 | 0.0252 | 10.6 | 0.0% |
| hybrid | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 16/16 | 15.64 | 11.75 | 7.82 | 15.64 | 0/0/0 | 0/0 | 0 | 0 | 0.0176 | 10.6 | 0.0% |
| ppo_1000814 | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 16/16 | 12.32 | 10.95 | 6.16 | 12.32 | 0/0/0 | 0/0 | 0 | 0 | 0.0403 | 11.3 | 0.0% |
| ppo_525678 | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 16/16 | 9.31 | 8.85 | 4.66 | 9.31 | 0/0/0 | 0/0 | 0 | 0 | 0.0403 | 10.7 | 0.0% |
| ppo_900462 | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 16/16 | 9.59 | 9.20 | 4.79 | 9.59 | 0/0/0 | 0/0 | 0 | 0 | 0.0372 | 10.4 | 0.0% |
| rule_gate_center_then_commit | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 16/16 | 10.94 | 10.90 | 5.47 | 10.94 | 0/0/0 | 0/0 | 0 | 0 | 0.0152 | 10.5 | 0.0% |

Paired vs the rule baseline on identical seeds:

| A | paired eps | only A finished | only rule finished | mean dt A-rule (s) | A faster | rule faster | time sign p |
|---|---|---|---|---|---|---|---|
| bc_v3 | 8 | 0 | 0 | 3.10 [0.91, 5.29] | 1 | 7 | 0.0703 |
| hybrid | 8 | 0 | 0 | 4.70 [-4.46, 13.86] | 2 | 6 | 0.2891 |
| ppo_1000814 | 8 | 0 | 0 | 1.39 [-4.21, 6.98] | 4 | 4 | 1.0000 |
| ppo_525678 | 8 | 0 | 0 | -1.62 [-3.43, 0.18] | 7 | 1 | 0.0703 |
| ppo_900462 | 8 | 0 | 0 | -1.35 [-3.04, 0.34] | 7 | 1 | 0.0703 |

### three_gate_sequence

| controller | succ/tot | success | 95% CI | full-circuit | gates | mean t (s) | median t (s) | t/gate (s) | pen. t (s) | coll ev/fr/ep | OOB ev/ep | wrong-dir | prev-gate ret | jerk | path (m) | timeout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bc_v3 | 6/8 | 75.0% | [0.41, 0.93] | 75.0% | 22/24 | 36.03 | 35.20 | 12.01 | 46.03 | 12/102/1 | 0/0 | 0 | 0 | 0.0311 | 16.6 | 0.0% |
| hybrid | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 24/24 | 24.04 | 24.00 | 8.01 | 26.54 | 4/41/1 | 0/0 | 0 | 0 | 0.0175 | 17.1 | 0.0% |
| ppo_1000814 | 6/8 | 75.0% | [0.41, 0.93] | 75.0% | 22/24 | 28.13 | 28.75 | 9.38 | 42.30 | 46/420/5 | 0/0 | 0 | 0 | 0.0408 | 18.9 | 0.0% |
| ppo_525678 | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 24/24 | 25.20 | 26.05 | 8.40 | 25.20 | 0/2/0 | 0/0 | 0 | 2 | 0.0427 | 21.5 | 0.0% |
| ppo_900462 | 3/8 | 37.5% | [0.14, 0.69] | 37.5% | 19/24 | 24.90 | 24.80 | 8.30 | 26.57 | 1/5/1 | 0/0 | 0 | 0 | 0.0408 | 14.5 | 0.0% |
| rule_gate_center_then_commit | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 24/24 | 25.74 | 26.35 | 8.58 | 25.74 | 0/0/0 | 0/0 | 0 | 0 | 0.0193 | 17.3 | 0.0% |

Paired vs the rule baseline on identical seeds:

| A | paired eps | only A finished | only rule finished | mean dt A-rule (s) | A faster | rule faster | time sign p |
|---|---|---|---|---|---|---|---|
| bc_v3 | 8 | 0 | 2 | 9.68 [-1.53, 20.90] | 1 | 5 | 0.2188 |
| hybrid | 8 | 0 | 0 | -1.70 [-3.74, 0.34] | 5 | 3 | 0.7266 |
| ppo_1000814 | 8 | 0 | 2 | 2.42 [-0.65, 5.48] | 1 | 5 | 0.2188 |
| ppo_525678 | 8 | 0 | 0 | -0.54 [-4.11, 3.04] | 3 | 5 | 0.7266 |
| ppo_900462 | 8 | 0 | 5 | 0.00 [-4.99, 4.99] | 1 | 2 | 1.0000 |

### three_gate_s_shape

| controller | succ/tot | success | 95% CI | full-circuit | gates | mean t (s) | median t (s) | t/gate (s) | pen. t (s) | coll ev/fr/ep | OOB ev/ep | wrong-dir | prev-gate ret | jerk | path (m) | timeout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bc_v3 | 4/8 | 50.0% | [0.22, 0.78] | 50.0% | 16/24 | 45.15 | 44.95 | 15.05 | 75.15 | 24/201/2 | 0/0 | 0 | 0 | 0.0340 | 16.0 | 0.0% |
| hybrid | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 24/24 | 41.45 | 39.30 | 13.82 | 41.45 | 0/0/0 | 0/0 | 0 | 0 | 0.0229 | 20.3 | 0.0% |
| ppo_1000814 | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 24/24 | 43.33 | 41.00 | 14.44 | 112.08 | 110/1028/8 | 0/0 | 0 | 0 | 0.0437 | 21.7 | 0.0% |
| ppo_525678 | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 24/24 | 33.69 | 33.00 | 11.23 | 43.69 | 16/115/4 | 0/0 | 0 | 1 | 0.0503 | 24.0 | 0.0% |
| ppo_900462 | 7/8 | 87.5% | [0.53, 0.98] | 87.5% | 23/24 | 29.86 | 30.00 | 9.95 | 39.14 | 13/65/5 | 0/0 | 0 | 1 | 0.0450 | 19.2 | 0.0% |
| rule_gate_center_then_commit | 8/8 | 100.0% | [0.68, 1.00] | 100.0% | 24/24 | 44.92 | 45.20 | 14.97 | 44.92 | 0/0/0 | 0/0 | 0 | 0 | 0.0250 | 20.9 | 0.0% |

Paired vs the rule baseline on identical seeds:

| A | paired eps | only A finished | only rule finished | mean dt A-rule (s) | A faster | rule faster | time sign p |
|---|---|---|---|---|---|---|---|
| bc_v3 | 8 | 0 | 4 | -0.17 [-7.62, 7.27] | 2 | 2 | 1.0000 |
| hybrid | 8 | 0 | 0 | -3.48 [-8.54, 1.59] | 6 | 2 | 0.2891 |
| ppo_1000814 | 8 | 0 | 0 | -1.60 [-12.92, 9.72] | 6 | 2 | 0.2891 |
| ppo_525678 | 8 | 0 | 0 | -11.24 [-16.23, -6.24] | 7 | 1 | 0.0703 |
| ppo_900462 | 8 | 0 | 1 | -15.20 [-18.41, -11.99] | 7 | 0 | 0.0156 |

### official_horseshoe_bay

| controller | succ/tot | success | 95% CI | full-circuit | gates | mean t (s) | median t (s) | t/gate (s) | pen. t (s) | coll ev/fr/ep | OOB ev/ep | wrong-dir | prev-gate ret | jerk | path (m) | timeout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bc_v3 | 0/5 | 0.0% | [0.00, 0.43] | 0.0% | 24/60 | n/a | n/a | n/a | n/a | 11/99/2 | 142/3 | 0 | 1 | 0.0455 | 136.1 | 80.0% |
| hybrid | 5/5 | 100.0% | [0.57, 1.00] | 100.0% | 60/60 | 225.90 | 231.00 | 18.82 | 225.90 | 0/0/0 | 0/0 | 0 | 0 | 0.0299 | 100.1 | 0.0% |
| ppo_1000814 | 4/5 | 80.0% | [0.38, 0.96] | 80.0% | 53/60 | 161.28 | 161.30 | 13.44 | 162.53 | 1/3/1 | 33/1 | 0 | 0 | 0.0526 | 100.2 | 0.0% |
| ppo_525678 | 3/5 | 60.0% | [0.23, 0.88] | 60.0% | 46/60 | 165.53 | 168.90 | 13.79 | 195.53 | 466/4647/2 | 17/2 | 0 | 0 | 0.0510 | 89.8 | 20.0% |
| ppo_900462 | 5/5 | 100.0% | [0.57, 1.00] | 100.0% | 60/60 | 179.82 | 180.90 | 14.98 | 179.82 | 0/0/0 | 0/0 | 0 | 0 | 0.0430 | 103.9 | 0.0% |
| rule_gate_center_then_commit | 5/5 | 100.0% | [0.57, 1.00] | 100.0% | 60/60 | 218.64 | 217.40 | 18.22 | 218.64 | 0/0/0 | 0/0 | 0 | 0 | 0.0291 | 99.6 | 0.0% |

Paired vs the rule baseline on identical seeds:

| A | paired eps | only A finished | only rule finished | mean dt A-rule (s) | A faster | rule faster | time sign p |
|---|---|---|---|---|---|---|---|
| bc_v3 | 5 | 0 | 5 | n/a | 0 | 0 | n/a |
| hybrid | 5 | 0 | 0 | 7.26 [-10.63, 25.15] | 2 | 3 | 1.0000 |
| ppo_1000814 | 5 | 0 | 1 | -57.67 [-71.89, -43.46] | 4 | 0 | 0.1250 |
| ppo_525678 | 5 | 0 | 2 | -51.43 [-66.22, -36.65] | 3 | 0 | 0.2500 |
| ppo_900462 | 5 | 0 | 0 | -38.82 [-61.44, -16.20] | 5 | 0 | 0.0625 |

### official_vertical_serpent

| controller | succ/tot | success | 95% CI | full-circuit | gates | mean t (s) | median t (s) | t/gate (s) | pen. t (s) | coll ev/fr/ep | OOB ev/ep | wrong-dir | prev-gate ret | jerk | path (m) | timeout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bc_v3 | 0/5 | 0.0% | [0.00, 0.43] | 0.0% | 34/85 | n/a | n/a | n/a | n/a | 302/2776/5 | 326/1 | 8 | 1 | 0.0495 | 172.5 | 40.0% |
| hybrid | 4/5 | 80.0% | [0.38, 0.96] | 80.0% | 81/85 | 319.77 | 294.05 | 18.81 | 341.02 | 17/154/1 | 0/0 | 0 | 0 | 0.0316 | 133.1 | 20.0% |
| ppo_1000814 | 0/5 | 0.0% | [0.00, 0.43] | 0.0% | 17/85 | n/a | n/a | n/a | n/a | 101/848/4 | 0/0 | 1 | 1 | 0.0513 | 35.4 | 0.0% |
| ppo_525678 | 0/5 | 0.0% | [0.00, 0.43] | 0.0% | 20/85 | n/a | n/a | n/a | n/a | 849/8396/5 | 0/0 | 0 | 1 | 0.0461 | 36.9 | 20.0% |
| ppo_900462 | 0/5 | 0.0% | [0.00, 0.43] | 0.0% | 30/85 | n/a | n/a | n/a | n/a | 92/611/4 | 0/0 | 1 | 1 | 0.0500 | 62.0 | 0.0% |
| rule_gate_center_then_commit | 5/5 | 100.0% | [0.57, 1.00] | 100.0% | 85/85 | 276.80 | 273.40 | 16.28 | 276.80 | 0/0/0 | 0/0 | 0 | 0 | 0.0326 | 128.2 | 0.0% |

Paired vs the rule baseline on identical seeds:

| A | paired eps | only A finished | only rule finished | mean dt A-rule (s) | A faster | rule faster | time sign p |
|---|---|---|---|---|---|---|---|
| bc_v3 | 5 | 0 | 5 | n/a | 0 | 0 | n/a |
| hybrid | 5 | 0 | 1 | 39.45 [-63.62, 142.52] | 1 | 3 | 0.6250 |
| ppo_1000814 | 5 | 0 | 5 | n/a | 0 | 0 | n/a |
| ppo_525678 | 5 | 0 | 5 | n/a | 0 | 0 | n/a |
| ppo_900462 | 5 | 0 | 5 | n/a | 0 | 0 | n/a |

### official_mixed_endurance

| controller | succ/tot | success | 95% CI | full-circuit | gates | mean t (s) | median t (s) | t/gate (s) | pen. t (s) | coll ev/fr/ep | OOB ev/ep | wrong-dir | prev-gate ret | jerk | path (m) | timeout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bc_v3 | 0/5 | 0.0% | [0.00, 0.43] | 0.0% | 23/110 | n/a | n/a | n/a | n/a | 887/8830/2 | 0/0 | 2 | 0 | 0.0475 | 88.3 | 20.0% |
| hybrid | 4/5 | 80.0% | [0.38, 0.96] | 80.0% | 109/110 | 565.98 | 587.80 | 25.73 | 583.48 | 14/86/1 | 0/0 | 0 | 0 | 0.0336 | 226.8 | 0.0% |
| ppo_1000814 | 0/5 | 0.0% | [0.00, 0.43] | 0.0% | 25/110 | n/a | n/a | n/a | n/a | 1414/13945/4 | 0/0 | 1 | 1 | 0.0469 | 64.4 | 20.0% |
| ppo_525678 | 0/5 | 0.0% | [0.00, 0.43] | 0.0% | 19/110 | n/a | n/a | n/a | n/a | 119/1148/4 | 75/1 | 3 | 0 | 0.0573 | 73.4 | 0.0% |
| ppo_900462 | 1/5 | 20.0% | [0.04, 0.62] | 20.0% | 59/110 | 520.50 | 520.50 | 23.66 | 660.50 | 253/1974/3 | 0/0 | 3 | 0 | 0.0544 | 149.2 | 0.0% |
| rule_gate_center_then_commit | 4/5 | 80.0% | [0.38, 0.96] | 80.0% | 109/110 | 462.15 | 461.40 | 21.01 | 462.15 | 3/18/1 | 0/0 | 0 | 0 | 0.0352 | 216.3 | 0.0% |

Paired vs the rule baseline on identical seeds:

| A | paired eps | only A finished | only rule finished | mean dt A-rule (s) | A faster | rule faster | time sign p |
|---|---|---|---|---|---|---|---|
| bc_v3 | 5 | 0 | 4 | n/a | 0 | 0 | n/a |
| hybrid | 5 | 1 | 1 | 88.23 [-158.63, 335.10] | 1 | 2 | 1.0000 |
| ppo_1000814 | 5 | 0 | 4 | n/a | 0 | 0 | n/a |
| ppo_525678 | 5 | 0 | 4 | n/a | 0 | 0 | n/a |
| ppo_900462 | 5 | 0 | 3 | 83.20 | 0 | 1 | 1.0000 |

## Paired controller comparisons (identical seeds and geometries)

| A | B | paired eps | A succ | B succ | only A | only B | reliability sign p | mean dt A-B (s) | A faster | B faster | time sign p | mean jerk delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bc_v3 | rule_gate_center_then_commit | 79 | 57 | 78 | 0 | 21 | 0.0000 | 2.84 [0.79, 4.89] | 11 | 37 | 0.0002 | 0.0093 [0.0077, 0.0109] |
| hybrid | rule_gate_center_then_commit | 79 | 77 | 78 | 1 | 2 | 1.0000 | 8.51 [1.01, 16.02] | 32 | 35 | 0.8072 | 0.0000 [-0.0011, 0.0011] |
| ppo_1000814 | rule_gate_center_then_commit | 79 | 66 | 78 | 0 | 12 | 0.0005 | -6.77 [-10.91, -2.63] | 44 | 13 | 0.0001 | 0.0211 [0.0196, 0.0227] |
| ppo_525678 | rule_gate_center_then_commit | 79 | 67 | 78 | 0 | 11 | 0.0010 | -7.02 [-10.11, -3.92] | 48 | 11 | 0.0000 | 0.0231 [0.0211, 0.0250] |
| ppo_900462 | rule_gate_center_then_commit | 79 | 64 | 78 | 0 | 14 | 0.0001 | -7.05 [-11.59, -2.51] | 50 | 6 | 0.0000 | 0.0190 [0.0175, 0.0206] |
| ppo_1000814 | ppo_525678 | 79 | 66 | 67 | 1 | 2 | 1.0000 | 1.49 [-0.14, 3.13] | 26 | 31 | 0.5966 | -0.0019 [-0.0039, 0.0001] |
| ppo_1000814 | ppo_900462 | 79 | 66 | 64 | 5 | 3 | 0.7266 | 1.00 [-1.79, 3.79] | 27 | 25 | 0.8899 | 0.0021 [0.0007, 0.0036] |
| ppo_525678 | ppo_900462 | 79 | 67 | 64 | 6 | 3 | 0.5078 | 0.70 [-0.53, 1.93] | 22 | 31 | 0.2717 | 0.0040 [0.0025, 0.0056] |
| ppo_1000814 | hybrid | 79 | 66 | 77 | 0 | 11 | 0.0010 | -9.13 [-14.41, -3.84] | 45 | 13 | 0.0000 | 0.0211 [0.0197, 0.0225] |
| ppo_525678 | hybrid | 79 | 67 | 77 | 0 | 10 | 0.0019 | -9.48 [-14.11, -4.84] | 46 | 13 | 0.0000 | 0.0230 [0.0213, 0.0248] |
| ppo_900462 | hybrid | 79 | 64 | 77 | 0 | 13 | 0.0002 | -12.32 [-17.03, -7.61] | 53 | 3 | 0.0000 | 0.0190 [0.0178, 0.0203] |

## Failure modes

Where each unfinished episode stopped, and why. `missed_gate_dnf` means the referee ended the run because the participant bypassed its expected gate; the episode therefore stops at the point where the policy lost the sequence, not at the deadline.

### Official circuits

| controller | failures | modes | stopped after gates | median gates |
|---|---|---|---|---|
| bc_v3 | 15 | missed_gate_dnf=8, stuck=1, timeout_without_dnf=6 | 0/12x1, 1/12x1, 1/22x1, 3/22x1, 4/22x1, 5/17x2, 6/17x1, 7/12x1, 7/22x1, 8/12x2, 8/22x1, 9/17x2 | 6 |
| hybrid | 2 | missed_gate_dnf=1, timeout_without_dnf=1 | 13/17x1, 21/22x1 | 17.0 |
| ppo_1000814 | 11 | missed_gate_dnf=10, stuck=1 | 1/17x2, 11/22x1, 2/22x2, 3/17x1, 5/12x1, 5/17x1, 5/22x2, 7/17x1 | 5 |
| ppo_525678 | 12 | missed_gate_dnf=10, stuck=2 | 1/17x1, 1/22x2, 10/22x1, 2/22x1, 3/12x1, 3/17x2, 5/22x1, 6/17x1, 7/12x1, 7/17x1 | 3.0 |
| ppo_900462 | 9 | missed_gate_dnf=9 | 1/17x1, 13/17x1, 14/22x1, 15/22x1, 2/22x1, 5/17x2, 6/17x1, 6/22x1 | 6 |
| rule_gate_center_then_commit | 1 | missed_gate_dnf=1 | 21/22x1 | 21 |

### All non-official groups

| controller | failures | modes | stopped after gates | median gates |
|---|---|---|---|---|
| bc_v3 | 7 | missed_gate_dnf=6, timeout_without_dnf=1 | 0/3x2, 1/2x1, 2/3x4 | 2 |
| hybrid | 0 | - | - | - |
| ppo_1000814 | 2 | missed_gate_dnf=2 | 2/3x2 | 2.0 |
| ppo_525678 | 0 | - | - | - |
| ppo_900462 | 6 | missed_gate_dnf=6 | 2/3x6 | 2.0 |
| rule_gate_center_then_commit | 0 | - | - | - |

## Recommendation

Selection priority: full-circuit reliability; zero safety events; three-gate and vertical-transition reliability; completion time; smoothness. Computed from this benchmark only, not from the training metadata.

| rank | controller | official full-circuit | safety ep | 3-gate | vertical | non-official success | time rank | jerk rank |
|---|---|---|---|---|---|---|---|---|
| 1 | rule_gate_center_then_commit | 93.3% | 1 | 100.0% | 100.0% | 100.0% | 3.60 | 1.46 |
| 2 | ppo_900462 | 40.0% | 14 | 62.5% | 100.0% | 90.6% | 2.11 | 4.36 |
| 3 | ppo_1000814 | 26.7% | 23 | 87.5% | 100.0% | 96.9% | 2.62 | 4.73 |
| 4 | ppo_525678 | 20.0% | 17 | 100.0% | 100.0% | 100.0% | 2.00 | 5.54 |
| 5 | bc_v3 | 0.0% | 13 | 62.5% | 100.0% | 89.1% | 5.57 | 3.27 |

`time rank` and `jerk rank` are mean within-group ranks (1 = best in that group). Ranking inside each group and averaging the ranks keeps every timing judgement inside a single suite; no second from one geometry is ever weighed against a second from another. Groups excluded from timing: single_gate_retention (the official clock runs first-gate-to-last-gate, so a single-gate case is definitionally 0 s).

**Best PPO checkpoint on this benchmark: `ppo_900462`** (PPO order: ppo_900462, ppo_1000814, ppo_525678).

Excluded from selection (reported separately): `hybrid` (not fully policy-based)

## What a further training stage would have to fix

Per-capability breakdown for the PPO checkpoints, so the scope of any follow-up run is argued from measurements rather than from step count.

| controller | single_gate_retention | two_gate_turns | vertical_transitions | three_gate_sequences | official_circuits |
|---|---|---|---|---|---|
| ppo_525678 | 8/8 | 24/24 | 16/16 | 16/16 (4 safety) | 3/15 (13 safety) |
| ppo_900462 | 8/8 | 24/24 | 16/16 | 10/16 (6 safety) | 6/15 (8 safety) |
| ppo_1000814 | 8/8 | 24/24 | 16/16 | 14/16 (13 safety) | 4/15 (10 safety) |

- `ppo_525678` failure modes: official_circuits: missed_gate_dnf=10, stuck=2
- `ppo_900462` failure modes: three_gate_sequences: missed_gate_dnf=6; official_circuits: missed_gate_dnf=9
- `ppo_1000814` failure modes: three_gate_sequences: missed_gate_dnf=2; official_circuits: missed_gate_dnf=10, stuck=1

## Trajectory plots

- `plots/official_mixed_endurance__mixed_endurance__rule_gate_center_then_commit__seed27039__success.png`
- `plots/official_mixed_endurance__mixed_endurance__bc_v3__seed27040__failure.png`
- `plots/official_mixed_endurance__mixed_endurance__ppo_525678__seed27038__failure.png`
- `plots/official_vertical_serpent__vertical_serpent__rule_gate_center_then_commit__seed27036__success.png`
- `plots/official_vertical_serpent__vertical_serpent__ppo_1000814__seed27036__failure.png`
- `plots/official_vertical_serpent__vertical_serpent__ppo_1000814__seed27037__failure.png`
- `plots/official_horseshoe_bay__horseshoe_bay__ppo_900462__seed27032__success.png`
- `plots/official_horseshoe_bay__horseshoe_bay__bc_v3__seed1800__failure.png`
- `plots/official_horseshoe_bay__horseshoe_bay__bc_v3__seed27033__failure.png`
- `plots/three_gate_s_shape__three_gate_s_curve__ppo_900462__seed27030__success.png`
- `plots/three_gate_s_shape__three_gate_s_curve__bc_v3__seed21305__failure.png`
- `plots/three_gate_s_shape__three_gate_s_curve__bc_v3__seed21306__failure.png`
- `plots/three_gate_sequence__three_gate_training__ppo_525678__seed27025__success.png`
- `plots/three_gate_sequence__three_gate_training__bc_v3__seed21302__failure.png`
- `plots/three_gate_sequence__three_gate_training__bc_v3__seed27026__failure.png`
- `plots/vertical_high_to_low__down_0p7m_straight__ppo_1000814__seed21221__success.png`
- `plots/vertical_low_to_high__up_0p7m_straight__ppo_1000814__seed21217__success.png`
- `plots/two_gate_right__right_20deg_generated__ppo_1000814__seed21214__success.png`
- `plots/two_gate_right__right_37deg_fixed__bc_v3__seed21212__failure.png`
- `plots/two_gate_left__left_20deg_generated__ppo_525678__seed21210__success.png`
- `plots/two_gate_straight__straight__ppo_1000814__seed27007__success.png`
- `plots/single_gate_retention__single_gate__bc_v3__seed21200__success.png`

## Official circuit videos

- `artifacts/bc_v3__official_horseshoe_bay__horseshoe_bay__seed1800.mp4`
- `artifacts/bc_v3__official_mixed_endurance__mixed_endurance__seed1800.mp4`
- `artifacts/bc_v3__official_vertical_serpent__vertical_serpent__seed1800.mp4`
- `artifacts/hybrid__official_horseshoe_bay__horseshoe_bay__seed1800.mp4`
- `artifacts/hybrid__official_mixed_endurance__mixed_endurance__seed1800.mp4`
- `artifacts/hybrid__official_vertical_serpent__vertical_serpent__seed1800.mp4`
- `artifacts/ppo_1000814__official_horseshoe_bay__horseshoe_bay__seed1800.mp4`
- `artifacts/ppo_1000814__official_mixed_endurance__mixed_endurance__seed1800.mp4`
- `artifacts/ppo_1000814__official_vertical_serpent__vertical_serpent__seed1800.mp4`
- `artifacts/ppo_525678__official_horseshoe_bay__horseshoe_bay__seed1800.mp4`
- `artifacts/ppo_525678__official_mixed_endurance__mixed_endurance__seed1800.mp4`
- `artifacts/ppo_525678__official_vertical_serpent__vertical_serpent__seed1800.mp4`
- `artifacts/ppo_900462__official_horseshoe_bay__horseshoe_bay__seed1800.mp4`
- `artifacts/ppo_900462__official_mixed_endurance__mixed_endurance__seed1800.mp4`
- `artifacts/ppo_900462__official_vertical_serpent__vertical_serpent__seed1800.mp4`
- `artifacts/rule_gate_center_then_commit__official_horseshoe_bay__horseshoe_bay__seed1800.mp4`
- `artifacts/rule_gate_center_then_commit__official_mixed_endurance__mixed_endurance__seed1800.mp4`
- `artifacts/rule_gate_center_then_commit__official_vertical_serpent__vertical_serpent__seed1800.mp4`

