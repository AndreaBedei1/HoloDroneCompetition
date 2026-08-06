# HoloOcean parallel capacity (v2)

Generated: 2026-08-06T22:57:46Z

Layouts are ranked by aggregate valid environment transitions per
second. A larger engine count is never selected if it measures slower,
and any layout breaching the headroom or reliability limits is rejected
outright.

| Total engines | PPO | SAC | PPO tx/s | SAC tx/s | Aggregate tx/s | tx/s per worker | CPU% | RAM% | GPU% | VRAM% | Accepted |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4 | 4 | 0 | 10.78 | 0.00 | 10.78 | 2.69 | 80.8 | 34.5 | 60.0 | 58.2 | no |
| 8 | 8 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 56.2 | 32.7 | 37.4 | 51.6 | no |

Rejected 4 engines (4 PPO + 0 SAC): cpu

Rejected 8 engines (8 PPO + 0 SAC): engine_startup_failures, worker_crashes

## Selected layout

No layout satisfied the headroom and reliability limits.

## Stopping rule

- minimum improvement to keep increasing: 8%
- consecutive non-improvements before stopping: 2
- stopped early: True
- engine counts measured: [4, 8]
