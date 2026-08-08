# Local-transition reward audit

Generated: 2026-08-08T08:54:13Z
Reward contract: `local_transition_reward_v2_collision_entry`

**Verdict: reward_contract_coherent_freeze_it**

## Structural checks

| Check | Result | Detail |
|---|:---:|---|
| every_component_is_clipped | PASS | component |x|<=60.0, total |x|<=100.0 |
| collision_is_event_based_not_per_frame | PASS | entry 50.0 vs per-contact-frame 0.5 |
| collision_shaping_is_capped | PASS | entry cap 150.0, contact cap 10.0 |
| sustained_contact_cannot_outweigh_the_impact | PASS | contact cap 10.0 < entry 50.0 |
| collision_dominates_efficiency_shaping | PASS | collision 50.0 vs efficiency terms |
| leaving_the_course_is_never_profitable | PASS | out of bounds 60.0 > crossing 25.0 |
| wrong_direction_is_never_profitable | PASS | wrong direction 60.0 > crossing 25.0 |
| missed_gate_is_never_profitable | PASS | missed gate 60.0 > crossing 25.0 |
| circling_a_gate_is_penalized | PASS | post-crossing orbit penalty present |
| stalling_is_not_rewarded | PASS | time cost 0.0005 per step is strictly negative |
| completion_is_length_independent | PASS | single constant completion bonus |
| efficiency_disabled_during_reliability_phase | PASS | jerk/energy/action-change are zero until efficiency unlocks |
| progress_shaping_cannot_be_farmed_by_oscillation | PASS | a forward/backward pair sums to 0.0 |
| crossing_reward_outweighs_one_step_of_dense_shaping | PASS | crossing/dense-per-step ratio 2.23 |

Every component is clipped to +-60.0 and the total step reward to +-100.0, so no term is unbounded.

## Per-episode event magnitudes

| Component | Magnitude |
|---|---:|
| acquisition_timeout_penalty | 45.0 |
| collision_contact_penalty | 10.0 |
| collision_penalty | 150.0 |
| completion | 2.0 |
| gate_crossing | 25.0 |
| missed_gate_penalty | 60.0 |
| moving_away_penalty | 12.0 |
| out_of_bounds_penalty | 60.0 |
| post_cross_orbit_penalty | 12.0 |
| previous_gate_return_penalty | 50.0 |
| terminal_failure_penalty | 50.0 |
| wrong_direction_penalty | 60.0 |

## Conclusion

No unbounded term, no duplicated penalty, no per-frame accumulation of an event penalty, and no incentive to stall, leave the course, circle a gate or farm progress without crossing. The contract is coherent and is frozen for the long experiments.
