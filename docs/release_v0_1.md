# Marine Race Arena v0.1 Release Notes

Release name: Marine Race Arena v0.1.0

Date: 2026-07-15

## Summary

Marine Race Arena v0.1.0 is a configurable underwater gate-racing benchmark
with replaceable simulator and vehicle adapters. The reference physical backend
uses HoloOcean with a BlueROV2-class vehicle. This release uses one official
onboard-only controller contract for single-rover and cooperative-fleet runs.

The numerical validation matrix contains 78 real-HoloOcean runs and 124
participant rows from the current source. Exact coverage, provenance and the
artifact contract pass; scientific controller/referee disagreements remain
reported as measured.

## Official-mode integrity

Controller initialization contains exactly:

- `participant_id`;
- `initial_beacon_id`;
- `total_beacons`;
- `laps`;
- normalized `command_limits`;
- for a fleet only: static `participant_order`, `release_index` and
  `predecessor_id` inside `fleet`.

Each official per-step observation contains:

- `local_time_s`, measured from that participant's release;
- `sensors`, restricted to configured onboard sensors;
- `beacons`, the list of physically received acoustic packets;
- optional `comms`, present only when the inter-rover channel is enabled.

Each beacon packet contains exactly `beacon_id`, `bearing_deg`, `elevation_deg`,
`range_m`, `signal_strength` and receiver-local `received_at_s`. Every ordered
gate owns one independent sequential transmitter from `B01` through `BN`.
Transmission scheduling, measurement noise and dropout are seeded.

Simulator pose, world-frame velocity, exact gate geometry, configured current
vectors and referee progress remain evaluation-side. The referee independently
scores the physical trajectory and may disagree with controller-local progress.

## Included features

- Three official track JSON files with unchanged `1.5 x 1.5 m` apertures:
  - `marine_race_arena/tracks/marine_race_horseshoe_bay.json`;
  - `marine_race_arena/tracks/marine_race_vertical_serpent.json`;
  - `marine_race_arena/tracks/marine_race_mixed_endurance.json`.
- HoloOcean BlueROV2 adapter for physical experiments.
- Engine-free fallback adapter for unit tests and runner plumbing only.
- `LocalCourseTracker` with controller-local phases
  `SEARCH`, `APPROACH`, `VISUAL_ALIGN`, `COMMIT`, `VERIFY_EXIT`, `ADVANCE` and
  `FINISHED`.
- Passage confirmation from camera alignment, DVL-integrated displacement,
  beacon-range turnaround, persistent visual disappearance and rear-sector
  acoustic evidence.
- `rule_gate_baseline`, the continuous visual-servo controller.
- `rule_gate_center_then_commit`, the staged center-then-commit controller.
- Custom controller loading from built-in aliases, modules and file paths.
- Staggered fleet/team execution with independent controller and referee state.
- Team-level `team_summary` aggregation.
- `leader_follower`, which broadcasts only `local_beacon_index`, `local_lap` and
  `local_status`, and makes yield decisions from local estimates.
- Acoustic latency, range, payload-size, per-sender transmit-rate limit, seeded loss and
  stale-message handling.
- Referee-side inter-vehicle diagnostic and penalty modes.
- `none`, `medium` and `strong` currents on all three official tracks.
- Seeded static random obstacles physically verified on all three official
  tracks at medium density. The generator also exposes low/high density and
  dynamic-physics options; explicit fixed obstacle definitions are supported by
  the schema.
- Structured referee events, controller-local progression diagnostics,
  environment metadata and automated result auditing.

## HoloOcean acceptance lifecycle

Physical evidence follows this lifecycle:

1. Validate the track and requested overlays.
2. Initialize the real HoloOcean adapter with fallback disabled.
3. Spawn the BlueROV2 vehicle, visual gate bars and any requested obstacles;
   construct the independent beacon network in the arena layer.
4. Verify physical ocean-current coupling for current runs.
5. Reset every controller with the minimal static mission assignment.
6. Release each rover on schedule and start its local clock at zero.
7. Keep controller-local progression diagnostics separate from referee truth.
8. Preserve summaries and event logs for successes, contacts, DNFs and timeouts.
9. Audit each discovered artifact's provenance and local/referee consistency,
   then enforce the exact 78-run manifest, seed coverage, matched fields and
   absence of duplicate or unexpected conditions.

A current run is acceptable only when metadata records HoloOcean, no fallback and
active physical coupling. An obstacle construction check is acceptable only with
the effective HoloOcean adapter, no fallback, a positive requested count,
`physical_obstacle_spawn_complete=true` and a spawned count equal to the requested
count. Obstacle construction does not imply validated obstacle avoidance.

## Validation commands

```bash
python -m compileall -q marine_race_arena tests run.py validate_final_matrix.py summarize_final_results.py
conda run -n ocean python -m pytest -q
conda run -n ocean python -m marine_race_arena.scripts.run_staggered_multi_rover_smoke
```

Single clean run:

```bash
conda run -n ocean python -m marine_race_arena.scripts.run_marine_race --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --benchmark-task clean_gate --controller rule_gate_baseline --adapter holoocean --official --headless --seed 0 --dt 0.033 --duration 560 --obstacles none --current-profile none --motion-compensation none --log-dir results/onboard_only_validation/smoke/single_clean
```

Received-packet diagnostics use the current plural form:

```bash
conda run -n ocean python -m marine_race_arena.scripts.run_marine_race --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --benchmark-task clean_gate --controller rule_gate_baseline --adapter holoocean --official --headless --seed 0 --dt 0.033 --duration 560 --obstacles none --current-profile none --motion-compensation none --print-beacons --log-dir results/onboard_only_validation/smoke/print_beacons
```

Configuration checks for the reusable overlays:

```bash
python -m marine_race_arena.scripts.validate_track_config --track marine_race_arena/tracks/marine_race_vertical_serpent.json --benchmark-task current_gate --current-profile strong
python -m marine_race_arena.scripts.validate_track_config --track marine_race_arena/tracks/marine_race_mixed_endurance.json --benchmark-task obstacle_gate --obstacles random --obstacle-density high --obstacle-physics static --current-profile none --seed 0
```

Audit a completed matrix:

```bash
python -m marine_race_arena.scripts.validate_onboard_only_results results/onboard_only_validation/final_20260715
python validate_final_matrix.py results/onboard_only_validation/final_20260715
python summarize_final_results.py results/onboard_only_validation/final_20260715
```

The onboard-only validator accepts any non-empty result set and audits the runs
it discovers. `validate_final_matrix.py` adds the release-specific count, seed,
condition and duplicate checks for the complete matrix.
`summarize_final_results.py` then reads each run-level referee summary and event
log to produce the complete 78-run manifest, 124 participant rows, aggregates
and local-versus-referee timing analysis. Team proximity events come from the
unique team counter and are not double-counted from per-rover involvement.

## Current experiment protocol

The final protocol contains 78 HoloOcean runs under source fingerprint
`e7d3107784ea53056febcc3966b267ef59ee6d0f24d523c5c9b9446efca044b8`.
All 78 expected runs and 124 participant rows are present. Execution failures
and artifact-contract failures are both zero. The progress audit reports 15
mismatched runs, comprising 11 false and 7 missed local advancements; 58 runs
are referee-finished and 46 are clean.

| Family | Conditions | Runs |
| --- | --- | ---: |
| Clean single rover | 3 tracks x 2 controllers x seeds 0-4 | 30 |
| Current single rover | Horseshoe x medium/strong x 2 controllers x seeds 0-4 | 20 |
| Two-rover fleet | Horseshoe, 90 s gap x 2 homogeneous controllers x seeds 0-4 | 10 |
| Coordination | 8/0 s gap x coordinated/uncoordinated x seeds 0-2 | 12 |
| Yield-margin ablation | 8/0 s gap x `min_gate_gap=1` x seeds 0-2 | 6 |
| **Total** |  | **78** |

Measured completion by five-seed case is:

| Case | Continuous servo | Center-then-commit |
| --- | --- | --- |
| Horseshoe clean | 5/5 | 5/5 |
| Vertical clean | 5/5 | 4/5 |
| Mixed clean | 2/5 | 5/5 |
| Horseshoe medium current | 2/5 | 3/5 |
| Horseshoe strong current | 0/5 | 0/5 |
| Homogeneous fleet, 90 s gap | 5/5 teams | 5/5 teams |

All clean and homogeneous-fleet runs have zero gate/world, inter-vehicle,
out-of-bounds and stuck events. Mean clean Horseshoe times are
194.456 +/- 4.286 s and 197.683 +/- 6.312 s respectively. Mean fleet team
times are 303.092 +/- 2.874 s and 307.976 +/- 5.227 s. Under medium current,
the controllers average 8.4 and 8.8 completed gates with 43.4 and 25.2
gate/world collisions per run; neither finishes a strong-current seed.

For primary coordination at an 8 s gap, leader-follower finishes 3/3 cleanly,
whereas no coordination finishes 2/3 and averages 127.3 gate/world collisions
and 2.0 inter-vehicle events. At simultaneous release, both conditions finish
3/3; leader-follower eliminates gate/world and inter-vehicle events but records
one stuck event per run. The `min_gate_gap=1` ablation finishes all six runs
cleanly and reduces mean team time to 266.024 s at gap 8 and 262.339 s at gap 0.

Nominal single-rover durations are 560 s for Horseshoe Bay, 900 s for Vertical
Serpent and 1300 s for Mixed Endurance. Current, fleet and coordination runs on
Horseshoe use 560 s and `dt=0.033`.

The primary coordination condition uses a continuous-servo leader, two
center-then-commit followers, diagnostic inter-vehicle mode, no currents, no
obstacles, zero configured packet loss and `min_gate_gap=2`. The ablation uses
`min_gate_gap=1` and runs only the coordinated condition.

The homogeneous two-rover fleet uses a `3.0 m` lateral offset and diagnostic
inter-vehicle thresholds `0.8 m` in the horizontal plane, `0.75 m` vertically
and `1.05 m` for release, with a `1.0 s` cooldown.

## Fleet team summary

Fleet mode evaluates one cooperative team. Per-rover rows remain diagnostic;
`team_summary` is the official aggregate and includes:

- rover count and expected gates;
- completed gates and all-rovers-finished status;
- team start, finish, elapsed and penalized time;
- gate, obstacle and inter-vehicle events;
- aggregate penalties.

An inter-vehicle event counts once at team level. Diagnostic mode records these
events without modifying the score; penalty mode remains experimental.

## Known limitations

- Current compensation is not solved; outcomes under medium and strong currents
  must be reported as measured.
- Random obstacles can be generated and physically spawned, but obstacle
  avoidance is not validated.
- Dense uncoordinated fleets may collide or fail.
- Inter-vehicle penalty calibration remains experimental.
- The fallback adapter is not physical evidence.
- HoloOcean initialization and repeated long-track sweeps can be slow.

## Explicitly not included

- Modified official gate geometry, track layout or referee margins to hide a
  controller failure.
- A guaranteed current-rejection controller.
- A claim that the official controllers avoid generated obstacles.
- Hidden simulator or referee feedback inside vehicle autonomy.

## Release checklist

- [x] `python -m compileall -q marine_race_arena tests run.py validate_final_matrix.py summarize_final_results.py` passes.
- [x] `conda run -n ocean python -m pytest -q` passes.
- [x] Focused real-HoloOcean progression smoke tests pass.
- [x] Current coupling and obstacle-spawn lifecycle checks pass.
- [x] All 78 fresh HoloOcean runs are present.
- [x] Onboard-only execution and artifact-contract validation pass; the
  separate scientific progress audit records 15 referee/local mismatches.
- [x] README and paper values match only the accepted fresh artifacts.
- [x] No generated results, caches or stale summaries are committed.
- [x] Git status and final source fingerprint are reviewed.

The release tag is intentionally outside this local commit and was not created.
