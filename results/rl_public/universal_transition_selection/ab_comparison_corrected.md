# Corrected selective warm-start versus scratch

Generated: 2026-08-04T18:52:08Z

This report supersedes the original ranking without modifying it. The
original evidence, checkpoints and episode artifacts are unchanged.

## 1. Competence qualification (mandatory)

A policy that avoids safety events by remaining almost stationary is not
safe, it is inactive. Every candidate must first clear these minimums:

| Criterion | Minimum |
|---|---:|
| first gate crossing rate | 0.8000 |
| target switch rate | 0.7000 |
| universal transition success | 0.2000 |
| completed gates | 10.0000 |
| completed gates per episode | 0.5000 |
| episodes reaching first gate | 0.8000 |
| mean distance travelled (m) | 1.0000 |
| mean absolute action | 0.0200 |
| non-trivial action fraction | 0.2500 |
| evaluated transition cases | 50.0000 |

| Initialization | Classification | Qualified | Failed criteria |
|---|---|:---:|---|
| scratch | `degenerate_inactive_policy` | False | first_gate_crossing, target_switch, universal_transition_success, completed_gates_per_episode, episodes_reaching_first_gate, mean_absolute_action, nontrivial_action |
| selective_warm_start | `competent` | True | none |

### Anti-inactivity evidence

| Initialization | Completed gates | Gates/episode | Reaching first gate | Mean abs action | Non-trivial action fraction | Action sample (episodes/steps) |
|---|---:|---:|---:|---:|---:|---|
| scratch | 15 | 0.0148 | 0.0148 | 0.0034 | 0.0000 | 14/39000 |
| selective_warm_start | 946 | 0.9348 | 0.9130 | 0.0889 | 0.9876 | 14/17317 |

Action magnitudes are measured from the recorded trajectories only, so
the sample size is stated with them. Motion is never treated as success:
these metrics exist solely to detect degenerate inactivity.

## 2. Safety ranking (competent policies only)

Priority: collision_episode_rate, collision_entries_per_episode, missed_gate_dnf_rate, wrong_direction_rate, previous_gate_return_rate, acquisition_timeout_rate, universal_transition_success_rate, long_sequence_completion_score, mean_action_jerk.

Collision *episodes* rank first and collision *entries* second; sustained
contact frames are reported but never ranked, because one prolonged
contact would otherwise outweigh several distinct impacts.

| Initialization | Ranked | Collision episodes | Collision entries | Missed gates | Wrong direction | Previous-gate returns | Acquisition timeouts | Transition success | Long completion | Mean jerk |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scratch | False | 103 | 539 (events) | 9 | 4 | 0 | 6 | 0.0000 | 0.0000 | 0.0057 |
| selective_warm_start | True | 417 | 2591 (events) | 78 | 51 | 1 | 436 | 0.3140 | 0.2500 | 0.1099 |

## 3. Final selection

Selected initialization: **selective_warm_start** (highest safety rank among competent candidates).

The selected policy is **competent but not yet reliable**: it clears
the competence gate (first crossing 0.9120, target switch 0.8190, transition success 0.3140) while still
producing 417 collision episodes, 78 missed gates, 51 wrong-direction events and 436 acquisition timeouts.

It is selected because it is the only initialization that learned
useful gate-crossing and beacon-switch behaviour, not because it is
safe. The long-term requirement remains universal transition success
>= 99% on repeated unseen evaluations with strong safety.

`scratch` is excluded as `degenerate_inactive_policy`: it failed first_gate_crossing, target_switch, universal_transition_success, completed_gates_per_episode, episodes_reaching_first_gate, mean_absolute_action, nontrivial_action. Its lower safety-event count does not compensate for absent task competence.
