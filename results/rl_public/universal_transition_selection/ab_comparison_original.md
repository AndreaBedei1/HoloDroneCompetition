# Selective warm-start versus scratch

Generated: 2026-08-04T12:27:27Z

Selection order: safety, unseen transition success, long-sequence completion, action jerk, acquisition time.

| Initialization | Safety clean | Safety events | Transition success | Long completion | Mean jerk | Mean acquisition (s) |
|---|:---:|---:|---:|---:|---:|---:|
| selective_warm_start | False | 983 | 0.3140 | 0.2500 | 0.1098715662055336 | 1.3711267605633803 |
| scratch | False | 122 | 0.0000 | 0.0000 | 0.005676414031620554 | None |

Selected initialization: **scratch**.
