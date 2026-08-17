# Final PPO procedural readiness

Overall: **FAIL**

## Criteria

| Criterion | Observed | Requirement | Result |
|---|---:|---:|---|
| universal_transition_success | 0.82 | >= 0.9 | FAIL |
| first_gate_crossing | 1.0 | >= 0.97 | PASS |
| target_switch | 0.882 | >= 0.85 | PASS |
| new_target_alignment | 0.852 | >= 0.85 | PASS |
| ladder_transition_success | 0.75 | >= 0.5 | PASS |
| ladder_first_gate_crossing | 1.0 | >= 0.8 | PASS |
| completion_3_gate | 0.8 | >= 0.95 | FAIL |
| completion_5_gate | 0.85 | >= 0.9 | FAIL |
| completion_8_gate | 0.55 | >= 0.75 | FAIL |
| completion_12_gate | 0.7 | >= 0.5 | PASS |
| completion_17_gate | 0.35 | >= 0.25 | PASS |
| completion_22_gate | 0.3 | >= 0.25 | PASS |
| out_of_bounds_rate | 0.0 | <= 0.01 | PASS |
| collision_episode_rate | 0.06612903225806452 | <= 0.2 | PASS |
| matched_validation_split | 1.0 | >= 1.0 | PASS |
| matched_difficulty | 1.0 | >= 1.0 | PASS |
| ladder_mid_rung_coverage | 4.0 | >= 1.0 | PASS |
| transition_cases | 500.0 | >= 500.0 | PASS |
| full_sequence_cases_per_length | 20.0 | >= 20.0 | PASS |

No readiness override was requested or applied. Training remains closed.
