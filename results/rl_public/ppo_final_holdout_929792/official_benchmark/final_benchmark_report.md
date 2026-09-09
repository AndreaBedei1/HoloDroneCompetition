# Final common benchmark: reliability-first PPO vs rule, hybrid and BC-v3

Commit `bd1f961c04c0a20c35d0cd60cd724c98cee1e255` | adapter `holoocean` | currents `none` | dt 0.1 s | 60/60 episodes.

Every controller ran the identical suite: same test groups, same generated and official geometries, same seeds, same adapter and the same current-free simulator configuration. Timing figures are reported per test group only, and every timing comparison is a paired per-seed difference within one group.

## Controllers under test

| key | kind | policy-only | model | note |
|---|---|---|---|---|
| ppo_final_generic_929792 | ppo | yes | `results/rl/universal_transition/final/ppo_final_generic_929792.zip` | immutable final PPO pinned by the final-circuit freeze record |
| rule_gate_center_then_commit | rule | no | - | no learned component |

## Aggregate comparison (all non-official test groups)

| controller | episodes | success | 95% CI | full-circuit | safety ep | coll ev | OOB ev | wrong-dir | prev-gate ret | timeout | mean jerk | mean pen. t (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo_final_generic_929792 | 0 | n/a | [n/a, n/a] | n/a | 0 | 0 | 0 | 0 | 0 | n/a | n/a | n/a |
| rule_gate_center_then_commit | 0 | n/a | [n/a, n/a] | n/a | 0 | 0 | 0 | 0 | 0 | n/a | n/a | n/a |

## Aggregate comparison (three official current-free circuits)

| controller | episodes | success | 95% CI | full-circuit | safety ep | coll ev | OOB ev | wrong-dir | prev-gate ret | timeout | mean jerk | mean pen. t (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo_final_generic_929792 | 30 | 0.0% | [0.00, 0.11] | 0.0% | 4 | 82 | 0 | 2 | 0 | 0.0% | 0.0807 | n/a |
| rule_gate_center_then_commit | 30 | 100.0% | [0.89, 1.00] | 100.0% | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.0333 | 329.77 |

## Per-group results

### official_horseshoe_bay

| controller | succ/tot | success | 95% CI | full-circuit | gates | mean t (s) | median t (s) | t/gate (s) | pen. t (s) | coll ev/fr/ep | OOB ev/ep | wrong-dir | prev-gate ret | jerk | path (m) | timeout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo_final_generic_929792 | 0/10 | 0.0% | [0.00, 0.28] | 0.0% | 35/120 | n/a | n/a | n/a | n/a | 0/0/0 | 0/0 | 0 | 0 | 0.0738 | 27.8 | 0.0% |
| rule_gate_center_then_commit | 10/10 | 100.0% | [0.72, 1.00] | 100.0% | 120/120 | 225.87 | 228.10 | 18.82 | 225.87 | 0/0/0 | 0/0 | 0 | 0 | 0.0303 | 100.4 | 0.0% |

Paired vs the rule baseline on identical seeds:

| A | paired eps | only A finished | only rule finished | mean dt A-rule (s) | A faster | rule faster | time sign p |
|---|---|---|---|---|---|---|---|
| ppo_final_generic_929792 | 10 | 0 | 10 | n/a | 0 | 0 | n/a |

### official_vertical_serpent

| controller | succ/tot | success | 95% CI | full-circuit | gates | mean t (s) | median t (s) | t/gate (s) | pen. t (s) | coll ev/fr/ep | OOB ev/ep | wrong-dir | prev-gate ret | jerk | path (m) | timeout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo_final_generic_929792 | 0/10 | 0.0% | [0.00, 0.28] | 0.0% | 29/170 | n/a | n/a | n/a | n/a | 82/726/4 | 0/0 | 2 | 0 | 0.0804 | 30.6 | 0.0% |
| rule_gate_center_then_commit | 10/10 | 100.0% | [0.72, 1.00] | 100.0% | 170/170 | 290.55 | 293.35 | 17.09 | 290.55 | 0/0/0 | 0/0 | 0 | 0 | 0.0342 | 129.2 | 0.0% |

Paired vs the rule baseline on identical seeds:

| A | paired eps | only A finished | only rule finished | mean dt A-rule (s) | A faster | rule faster | time sign p |
|---|---|---|---|---|---|---|---|
| ppo_final_generic_929792 | 10 | 0 | 10 | n/a | 0 | 0 | n/a |

### official_mixed_endurance

| controller | succ/tot | success | 95% CI | full-circuit | gates | mean t (s) | median t (s) | t/gate (s) | pen. t (s) | coll ev/fr/ep | OOB ev/ep | wrong-dir | prev-gate ret | jerk | path (m) | timeout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo_final_generic_929792 | 0/10 | 0.0% | [0.00, 0.28] | 0.0% | 23/220 | n/a | n/a | n/a | n/a | 0/0/0 | 0/0 | 0 | 0 | 0.0879 | 19.5 | 0.0% |
| rule_gate_center_then_commit | 10/10 | 100.0% | [0.72, 1.00] | 100.0% | 220/220 | 472.90 | 471.35 | 21.50 | 472.90 | 0/0/0 | 0/0 | 0 | 0 | 0.0355 | 216.6 | 0.0% |

Paired vs the rule baseline on identical seeds:

| A | paired eps | only A finished | only rule finished | mean dt A-rule (s) | A faster | rule faster | time sign p |
|---|---|---|---|---|---|---|---|
| ppo_final_generic_929792 | 10 | 0 | 10 | n/a | 0 | 0 | n/a |

## Paired controller comparisons (identical seeds and geometries)

| A | B | paired eps | A succ | B succ | only A | only B | reliability sign p | mean dt A-B (s) | A faster | B faster | time sign p | mean jerk delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo_final_generic_929792 | rule_gate_center_then_commit | 30 | 0 | 30 | 0 | 30 | 0.0000 | n/a | 0 | 0 | n/a | 0.0474 [0.0450, 0.0498] |

## Failure modes

Where each unfinished episode stopped, and why. `missed_gate_dnf` means the referee ended the run because the participant bypassed its expected gate; the episode therefore stops at the point where the policy lost the sequence, not at the deadline.

### Official circuits

| controller | failures | modes | stopped after gates | median gates |
|---|---|---|---|---|
| ppo_final_generic_929792 | 30 | missed_gate_dnf=30 | 1/12x2, 1/17x3, 1/22x1, 2/12x2, 2/17x3, 2/22x6, 3/12x1, 3/22x2, 4/12x1, 4/17x2, 4/22x1, 5/12x2, 6/12x2, 6/17x2 | 2.0 |
| rule_gate_center_then_commit | 0 | - | - | - |

### All non-official groups

| controller | failures | modes | stopped after gates | median gates |
|---|---|---|---|---|
| ppo_final_generic_929792 | 0 | - | - | - |
| rule_gate_center_then_commit | 0 | - | - | - |

## Recommendation

Selection priority: full-circuit reliability; zero safety events; three-gate and vertical-transition reliability; completion time; smoothness. Computed from this benchmark only, not from the training metadata.

| rank | controller | official full-circuit | safety ep | 3-gate | vertical | non-official success | time rank | jerk rank |
|---|---|---|---|---|---|---|---|---|
| 1 | rule_gate_center_then_commit | 100.0% | 0 | 0.0% | 0.0% | n/a | n/a | 1.00 |
| 2 | ppo_final_generic_929792 | 0.0% | 4 | 0.0% | 0.0% | n/a | n/a | 2.00 |

`time rank` and `jerk rank` are mean within-group ranks (1 = best in that group). Ranking inside each group and averaging the ranks keeps every timing judgement inside a single suite; no second from one geometry is ever weighed against a second from another. Groups excluded from timing: single_gate_retention (the official clock runs first-gate-to-last-gate, so a single-gate case is definitionally 0 s).

**Best PPO checkpoint on this benchmark: `ppo_final_generic_929792`** (PPO order: ppo_final_generic_929792).

## What a further training stage would have to fix

Per-capability breakdown for the PPO checkpoints, so the scope of any follow-up run is argued from measurements rather than from step count.

| controller | single_gate_retention | two_gate_turns | vertical_transitions | three_gate_sequences | official_circuits |
|---|---|---|---|---|---|
| ppo_final_generic_929792 | 0/0 | 0/0 | 0/0 | 0/0 | 0/30 (4 safety) |

- `ppo_final_generic_929792` failure modes: official_circuits: missed_gate_dnf=30

