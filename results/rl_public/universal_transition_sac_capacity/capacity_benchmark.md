# Concurrent PPO/SAC HoloOcean capacity benchmark

| SAC workers | Total engines | SAC transitions/s | PPO degradation | Sensors | Orphans | Stable |
|---:|---:|---:|---:|:---:|---:|:---:|
| 1 | 3 | 6.921 | n/a | valid | 0 | yes |
| 2 | 4 | 7.867 | 31.1% | valid | 0 | no |

Selected layout: one SAC rollout worker and one SAC evaluator worker. The
benchmark stopped at the first unstable layout; 4, 6, and 8 SAC workers were
not launched. The failure criterion was PPO throughput degradation above 25%,
not a sensor, launch, memory, or cleanup failure.
