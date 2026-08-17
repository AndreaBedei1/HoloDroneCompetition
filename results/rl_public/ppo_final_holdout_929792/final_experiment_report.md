# Final PPO experiment closure

### FROZEN PPO

- Checkpoint: `C:\Users\andrea.bedei3\Desktop\HoloDroneCompetition-sac\results\rl\universal_transition\final\ppo_final_generic_929792.zip`
- SHA-256: `60ffdca1a6afc0260487d90cc7282a285cbafd46e76d32a316348c13f5fdfe32`
- Git SHA at freeze: `1f481b83f2e5735be75ee1322dd234f67902cf0a`
- Total transitions: 929792
- Atomic checkpoint validation: PASS (manifest status before readiness: `unverified`).
- Contracts: `{"action": "surge_sway_heave_yaw_pm1_v1", "curriculum": "generic_sequence_curriculum_v1", "dataset_split": "rl_holdout_policy_v1", "hard_family": "rl_hard_families_v1", "observation": "onboard_local_transition_v1", "observation_dim": 35, "reward": "local_transition_reward_v2_collision_entry"}`

### PROCEDURAL READINESS

Transition benchmark (500 VALIDATION cases):

| metric | count | rate | Wilson 95% CI |
| --- | --- | --- | --- |
| universal_transition_success | 410/500 | 0.8200 | [0.7839, 0.8512] |
| first_gate_crossing | 500/500 | 1.0000 | [0.9924, 1.0000] |
| target_switch | 441/500 | 0.8820 | [0.8508, 0.9074] |
| new_target_alignment | 426/500 | 0.8520 | [0.8182, 0.8804] |
| new_target_range_decrease | 440/500 | 0.8800 | [0.8486, 0.9056] |

Safety/events: collisions 41/620 episodes (513 events); missed-gate DNF 69; wrong-direction events 1; OOB episodes 0.

Sequence completion (20 independent VALIDATION sequences per length):

| gates | completed | rate | Wilson 95% CI |
| --- | --- | --- | --- |
| 3 | 16/20 | 0.8000 | [0.5840, 0.9193] |
| 5 | 17/20 | 0.8500 | [0.6396, 0.9476] |
| 8 | 11/20 | 0.5500 | [0.3421, 0.7418] |
| 12 | 14/20 | 0.7000 | [0.4810, 0.8545] |
| 17 | 7/20 | 0.3500 | [0.1812, 0.5671] |
| 22 | 6/20 | 0.3000 | [0.1455, 0.5190] |

Unconditional survival from episode start:

| cross gate | count/eligible-from-start | probability |
| --- | --- | --- |
| 1 | 120/120 | 1.0000 |
| 2 | 88/120 | 0.7333 |
| 3 | 86/120 | 0.7167 |
| 5 | 67/100 | 0.6700 |
| 8 | 46/80 | 0.5750 |
| 12 | 32/60 | 0.5333 |
| 17 | 15/40 | 0.3750 |
| 22 | 6/20 | 0.3000 |

Metric clarification: `first_gate_crossing` means crossing gate 1. The full gate-1→gate-2 transition was 88/120 (0.7333), including switch, reacquisition/alignment and range decrease.

Difficulty ladder:

| rung | n | transition | 95% CI | first gate | switch | alignment | collisions | OOB | sequence score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G1 | 60 | 0.8333 | [0.7197, 0.9069] | 1.0000 | 0.9000 | 0.8667 | 5 | 0 | 0.6667 |
| G2 | 60 | 0.8000 | [0.6822, 0.8817] | 1.0000 | 0.8500 | 0.8333 | 6 | 0 | 0.5000 |
| G3 | 60 | 0.7500 | [0.6277, 0.8422] | 1.0000 | 0.8500 | 0.7833 | 6 | 0 | 0.5833 |
| G4 | 60 | 0.7500 | [0.6277, 0.8422] | 1.0000 | 0.8500 | 0.7667 | 6 | 0 | 0.5000 |
| G5 | 60 | 0.6833 | [0.5577, 0.7869] | 1.0000 | 0.9167 | 0.7333 | 13 | 0 | 0.1667 |
| G6 | 60 | 0.5333 | [0.4089, 0.6537] | 0.9833 | 0.9000 | 0.6333 | 17 | 0 | 0.0833 |

Readiness criteria (unchanged gate, no override):

| criterion | observed | bound | verdict |
| --- | --- | --- | --- |
| universal_transition_success | 0.8200 | >=0.9000 | FAIL |
| first_gate_crossing | 1.0000 | >=0.9700 | PASS |
| target_switch | 0.8820 | >=0.8500 | PASS |
| new_target_alignment | 0.8520 | >=0.8500 | PASS |
| ladder_transition_success | 0.7500 | >=0.5000 | PASS |
| ladder_first_gate_crossing | 1.0000 | >=0.8000 | PASS |
| completion_3_gate | 0.8000 | >=0.9500 | FAIL |
| completion_5_gate | 0.8500 | >=0.9000 | FAIL |
| completion_8_gate | 0.5500 | >=0.7500 | FAIL |
| completion_12_gate | 0.7000 | >=0.5000 | PASS |
| completion_17_gate | 0.3500 | >=0.2500 | PASS |
| completion_22_gate | 0.3000 | >=0.2500 | PASS |
| out_of_bounds_rate | 0.0000 | <=0.0100 | PASS |
| collision_episode_rate | 0.0661 | <=0.2000 | PASS |
| matched_validation_split | 1.0000 | >=1.0000 | PASS |
| matched_difficulty | 1.0000 | >=1.0000 | PASS |
| ladder_mid_rung_coverage | 4.0000 | >=1.0000 | PASS |
| transition_cases | 500.0000 | >=500.0000 | PASS |
| full_sequence_cases_per_length | 20.0000 | >=20.0000 | PASS |

**PROCEDURAL READINESS: FAIL.** Failures: universal_transition_success, completion_3_gate, completion_5_gate, completion_8_gate.

### HOLDOUT PROTOCOL

- Decision record commit: `c3484bb55f12439b23f3af296a9d12dedb165813`
- Benchmark protocol commit: `bd1f961c04c0a20c35d0cd60cd724c98cee1e255`
- Training and checkpoint were frozen before the circuits were opened.
- One exploratory paired execution: seeds 1800–1809, HoloOcean, current-free, no fallback.
- No post-holdout parameter, reward, curriculum, controller tuning or training is permitted.

### FINAL CIRCUIT 1

**Horseshoe Bay**

| controller | completed | gates | collisions (ep/events) | OOB | wrong dir | missed | time mean | path mean | jerk mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | 0/10 | 35/120 (mean 3.5) | 0/0 | 0 | 0 | 10 | n/a | 27.7723 | 0.0738 |
| rules | 10/10 | 120/120 (mean 12.0) | 0/0 | 0 | 0 | 0 | 225.8700 | 100.3924 | 0.0303 |

Per-trial evidence:

| controller | seed | result | gates | failure gate | mechanism | coll | OOB | missed | wrong | time | path | jerk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | 1800 | failed | 1/12 | 2 | crossing_missed_gate | 0 | 0 | 1 | 0 | n/a | 8.8293 | 0.0753 |
| PPO | 1801 | failed | 3/12 | 4 | crossing_missed_gate | 0 | 0 | 1 | 0 | n/a | 23.5562 | 0.0720 |
| PPO | 1802 | failed | 5/12 | 6 | crossing_missed_gate | 0 | 0 | 1 | 0 | n/a | 38.9853 | 0.0744 |
| PPO | 1803 | failed | 6/12 | 7 | crossing_missed_gate | 0 | 0 | 1 | 0 | n/a | 48.3917 | 0.0712 |
| PPO | 1804 | failed | 4/12 | 5 | crossing_missed_gate | 0 | 0 | 1 | 0 | n/a | 30.3586 | 0.0780 |
| PPO | 1805 | failed | 2/12 | 3 | crossing_missed_gate | 0 | 0 | 1 | 0 | n/a | 15.6050 | 0.0733 |
| PPO | 1806 | failed | 6/12 | 7 | crossing_missed_gate | 0 | 0 | 1 | 0 | n/a | 48.0268 | 0.0766 |
| PPO | 1807 | failed | 1/12 | 2 | crossing_missed_gate | 0 | 0 | 1 | 0 | n/a | 8.9495 | 0.0750 |
| PPO | 1808 | failed | 2/12 | 3 | crossing_missed_gate | 0 | 0 | 1 | 0 | n/a | 16.1758 | 0.0686 |
| PPO | 1809 | failed | 5/12 | 6 | crossing_missed_gate | 0 | 0 | 1 | 0 | n/a | 38.8443 | 0.0735 |
| rules | 1800 | completed | 12/12 | n/a | n/a | 0 | 0 | 0 | 0 | 241.6000 | 101.8052 | 0.0329 |
| rules | 1801 | completed | 12/12 | n/a | n/a | 0 | 0 | 0 | 0 | 228.3000 | 99.6973 | 0.0313 |
| rules | 1802 | completed | 12/12 | n/a | n/a | 0 | 0 | 0 | 0 | 226.3000 | 100.2318 | 0.0310 |
| rules | 1803 | completed | 12/12 | n/a | n/a | 0 | 0 | 0 | 0 | 228.4000 | 100.1907 | 0.0316 |
| rules | 1804 | completed | 12/12 | n/a | n/a | 0 | 0 | 0 | 0 | 213.5000 | 99.5788 | 0.0296 |
| rules | 1805 | completed | 12/12 | n/a | n/a | 0 | 0 | 0 | 0 | 240.4000 | 102.0898 | 0.0322 |
| rules | 1806 | completed | 12/12 | n/a | n/a | 0 | 0 | 0 | 0 | 230.0000 | 100.8602 | 0.0300 |
| rules | 1807 | completed | 12/12 | n/a | n/a | 0 | 0 | 0 | 0 | 216.6000 | 100.1003 | 0.0279 |
| rules | 1808 | completed | 12/12 | n/a | n/a | 0 | 0 | 0 | 0 | 205.7000 | 98.6187 | 0.0274 |
| rules | 1809 | completed | 12/12 | n/a | n/a | 0 | 0 | 0 | 0 | 227.9000 | 100.7510 | 0.0293 |

Every PPO run crossed at least gate 1 and ended when the referee recorded one missed-gate DNF at the next attempted gate. The observable terminal mechanism is therefore crossing/missed gate; the referee evidence does not support relabelling it as acquisition, target-switch or reacquisition failure.

### FINAL CIRCUIT 2

**Vertical Serpent**

| controller | completed | gates | collisions (ep/events) | OOB | wrong dir | missed | time mean | path mean | jerk mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | 0/10 | 29/170 (mean 2.9) | 4/82 | 0 | 2 | 10 | n/a | 30.6269 | 0.0804 |
| rules | 10/10 | 170/170 (mean 17.0) | 0/0 | 0 | 0 | 0 | 290.5500 | 129.2207 | 0.0342 |

Per-trial evidence:

| controller | seed | result | gates | failure gate | mechanism | coll | OOB | missed | wrong | time | path | jerk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | 1800 | failed | 4/17 | 5 | crossing_missed_gate | 1 | 0 | 1 | 0 | n/a | 41.6899 | 0.0882 |
| PPO | 1801 | failed | 2/17 | 3 | crossing_missed_gate | 0 | 0 | 1 | 0 | n/a | 24.4646 | 0.0900 |
| PPO | 1802 | failed | 6/17 | 7 | crossing_missed_gate | 65 | 0 | 1 | 1 | n/a | 63.3213 | 0.0702 |
| PPO | 1803 | failed | 1/17 | 2 | crossing_missed_gate | 0 | 0 | 1 | 0 | n/a | 9.2865 | 0.0799 |
| PPO | 1804 | failed | 2/17 | 3 | crossing_missed_gate | 3 | 0 | 1 | 1 | n/a | 35.3018 | 0.0883 |
| PPO | 1805 | failed | 6/17 | 7 | crossing_missed_gate | 13 | 0 | 1 | 0 | n/a | 57.6740 | 0.0761 |
| PPO | 1806 | failed | 1/17 | 2 | crossing_missed_gate | 0 | 0 | 1 | 0 | n/a | 9.1598 | 0.0798 |
| PPO | 1807 | failed | 1/17 | 2 | crossing_missed_gate | 0 | 0 | 1 | 0 | n/a | 9.1091 | 0.0770 |
| PPO | 1808 | failed | 2/17 | 3 | crossing_missed_gate | 0 | 0 | 1 | 0 | n/a | 16.7670 | 0.0780 |
| PPO | 1809 | failed | 4/17 | 5 | crossing_missed_gate | 0 | 0 | 1 | 0 | n/a | 39.4945 | 0.0768 |
| rules | 1800 | completed | 17/17 | n/a | n/a | 0 | 0 | 0 | 0 | 288.6000 | 128.9554 | 0.0345 |
| rules | 1801 | completed | 17/17 | n/a | n/a | 0 | 0 | 0 | 0 | 290.4000 | 129.0011 | 0.0338 |
| rules | 1802 | completed | 17/17 | n/a | n/a | 0 | 0 | 0 | 0 | 301.0000 | 129.9179 | 0.0356 |
| rules | 1803 | completed | 17/17 | n/a | n/a | 0 | 0 | 0 | 0 | 296.3000 | 129.1189 | 0.0352 |
| rules | 1804 | completed | 17/17 | n/a | n/a | 0 | 0 | 0 | 0 | 296.7000 | 129.0808 | 0.0355 |
| rules | 1805 | completed | 17/17 | n/a | n/a | 0 | 0 | 0 | 0 | 274.5000 | 128.2566 | 0.0319 |
| rules | 1806 | completed | 17/17 | n/a | n/a | 0 | 0 | 0 | 0 | 279.0000 | 127.9046 | 0.0322 |
| rules | 1807 | completed | 17/17 | n/a | n/a | 0 | 0 | 0 | 0 | 300.6000 | 131.4463 | 0.0339 |
| rules | 1808 | completed | 17/17 | n/a | n/a | 0 | 0 | 0 | 0 | 276.4000 | 128.3313 | 0.0331 |
| rules | 1809 | completed | 17/17 | n/a | n/a | 0 | 0 | 0 | 0 | 302.0000 | 130.1940 | 0.0359 |

Every PPO run crossed at least gate 1 and ended when the referee recorded one missed-gate DNF at the next attempted gate. The observable terminal mechanism is therefore crossing/missed gate; the referee evidence does not support relabelling it as acquisition, target-switch or reacquisition failure.

### FINAL CIRCUIT 3

**Mixed Endurance**

| controller | completed | gates | collisions (ep/events) | OOB | wrong dir | missed | time mean | path mean | jerk mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | 0/10 | 23/220 (mean 2.3) | 0/0 | 0 | 0 | 10 | n/a | 19.4718 | 0.0879 |
| rules | 10/10 | 220/220 (mean 22.0) | 0/0 | 0 | 0 | 0 | 472.9000 | 216.5504 | 0.0355 |

Per-trial evidence:

| controller | seed | result | gates | failure gate | mechanism | coll | OOB | missed | wrong | time | path | jerk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | 1800 | failed | 2/22 | 3 | crossing_missed_gate | 0 | 0 | 1 | 0 | n/a | 16.8895 | 0.0834 |
| PPO | 1801 | failed | 4/22 | 5 | crossing_missed_gate | 0 | 0 | 1 | 0 | n/a | 36.2978 | 0.0890 |
| PPO | 1802 | failed | 3/22 | 4 | crossing_missed_gate | 0 | 0 | 1 | 0 | n/a | 25.2570 | 0.0877 |
| PPO | 1803 | failed | 2/22 | 3 | crossing_missed_gate | 0 | 0 | 1 | 0 | n/a | 16.2680 | 0.0901 |
| PPO | 1804 | failed | 3/22 | 4 | crossing_missed_gate | 0 | 0 | 1 | 0 | n/a | 25.6944 | 0.0957 |
| PPO | 1805 | failed | 2/22 | 3 | crossing_missed_gate | 0 | 0 | 1 | 0 | n/a | 16.1614 | 0.0883 |
| PPO | 1806 | failed | 2/22 | 3 | crossing_missed_gate | 0 | 0 | 1 | 0 | n/a | 16.4010 | 0.0815 |
| PPO | 1807 | failed | 1/22 | 2 | crossing_missed_gate | 0 | 0 | 1 | 0 | n/a | 8.8328 | 0.0820 |
| PPO | 1808 | failed | 2/22 | 3 | crossing_missed_gate | 0 | 0 | 1 | 0 | n/a | 16.8046 | 0.0833 |
| PPO | 1809 | failed | 2/22 | 3 | crossing_missed_gate | 0 | 0 | 1 | 0 | n/a | 16.1119 | 0.0979 |
| rules | 1800 | completed | 22/22 | n/a | n/a | 0 | 0 | 0 | 0 | 440.8000 | 214.7137 | 0.0337 |
| rules | 1801 | completed | 22/22 | n/a | n/a | 0 | 0 | 0 | 0 | 508.0000 | 219.6977 | 0.0380 |
| rules | 1802 | completed | 22/22 | n/a | n/a | 0 | 0 | 0 | 0 | 448.8000 | 215.3147 | 0.0351 |
| rules | 1803 | completed | 22/22 | n/a | n/a | 0 | 0 | 0 | 0 | 470.8000 | 215.9776 | 0.0356 |
| rules | 1804 | completed | 22/22 | n/a | n/a | 0 | 0 | 0 | 0 | 488.5000 | 217.9938 | 0.0354 |
| rules | 1805 | completed | 22/22 | n/a | n/a | 0 | 0 | 0 | 0 | 468.3000 | 217.0748 | 0.0342 |
| rules | 1806 | completed | 22/22 | n/a | n/a | 0 | 0 | 0 | 0 | 483.8000 | 217.2139 | 0.0360 |
| rules | 1807 | completed | 22/22 | n/a | n/a | 0 | 0 | 0 | 0 | 483.3000 | 217.3404 | 0.0365 |
| rules | 1808 | completed | 22/22 | n/a | n/a | 0 | 0 | 0 | 0 | 471.9000 | 215.4171 | 0.0352 |
| rules | 1809 | completed | 22/22 | n/a | n/a | 0 | 0 | 0 | 0 | 464.8000 | 214.7599 | 0.0352 |

Every PPO run crossed at least gate 1 and ended when the referee recorded one missed-gate DNF at the next attempted gate. The observable terminal mechanism is therefore crossing/missed gate; the referee evidence does not support relabelling it as acquisition, target-switch or reacquisition failure.

### AGGREGATE

| controller | completion | gates | collision ep/events | OOB | wrong dir | missed | path mean | jerk mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | 0/30 (0.0000) | 87/510 | 4/82 | 0 | 2 | 30 | 25.9570 | 0.0807 |
| rules | 30/30 (1.0000) | 510/510 | 0/0 | 0 | 0 | 0 | 148.7211 | 0.0333 |

PPO path length and jerk are measured through early DNF and are not completion-efficiency evidence. Reliability and safety govern: PPO completed 0/30 while rules completed 30/30.

### FINAL VERDICT

**PPO FAILED FINAL HOLDOUT**

The generic procedural readiness target was not achieved, and the frozen PPO then failed all 30 unseen official-circuit trials. The paired frozen rules controller completed all 30. No further RL training or tuning is authorized in this campaign.

### REPRODUCIBILITY

- Tests: 78 passed; final full suite: 1644 passed, 16 skipped, 7 warnings.
- Decision commit: `c3484bb55f12439b23f3af296a9d12dedb165813`
- Artifact freeze/readiness preregistration commit: `f455b764231dc0fdff1d2179376a97aa4f76e353`
- Benchmark protocol commit: `bd1f961c04c0a20c35d0cd60cd724c98cee1e255`
- PPO SHA-256: `60ffdca1a6afc0260487d90cc7282a285cbafd46e76d32a316348c13f5fdfe32`
- Episodes SHA-256: `d7043ee4123531e4549ba5722d7642266b43ef037e0ba67a5e5d3d1b5933f085`
- Aggregate SHA-256: `b9121baf8ea74d1492351ea5612259918f139b7bc80bdce0cd69260acb83079e`
- Ledger SHA-256: `7b6a1a40cd53360199389218bd044e12049fb34d67d2f696569c47090dce7d6e` (60 chained entries).
- Rules source SHA-256: `200b547e56c01b8c9756ce0e7cb0deb316faa3b53f13495c372a392c0d8804ec`
