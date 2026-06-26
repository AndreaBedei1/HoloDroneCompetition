# Marine Race Arena v0.1 Release Notes

Release name: Marine Race Arena v0.1.0

Date: TBD

## Summary

Marine Race Arena v0.1.0 is the first release candidate for a HoloOcean-based underwater drone racing benchmark using BlueROV2-style vehicles. The release focuses on clean official single-rover gate racing and a stable staggered fleet/team demonstration. It does not claim solved current compensation, close-proximity multi-rover racing, or fully calibrated rover-rover collision penalties.

## Included Features

- Official single-rover clean-gate benchmark mode.
- HoloOcean BlueROV2 adapter.
- Simulator-independent fallback adapter for tests and runner plumbing.
- Three official track JSON files:
  - `marine_race_arena/tracks/marine_race_horseshoe_bay.json`
  - `marine_race_arena/tracks/marine_race_vertical_serpent.json`
  - `marine_race_arena/tracks/marine_race_mixed_endurance.json`
- Official 1.5 m x 1.5 m gate openings in official tracks.
- Configurable race JSON covering race metadata, world bounds, gates, beacon, currents, obstacles, participants, sensors, referee, penalties, and scoring.
- `rule_gate_baseline` controller for official clean-gate evaluation.
- Custom controller interface and loader for built-in aliases, Python modules, `module:Class`, fully qualified classes, and file-path controllers.
- Staggered fleet/team evaluation mode.
- Team-level `team_summary` aggregation for multi-rover fleet runs.
- Referee-side inter-vehicle collision diagnostics.
- Optional `inter_vehicle_collision_mode=penalize`, kept experimental until calibration is complete.
- Inter-vehicle collision calibration script.

## Validation Commands

```bash
python -m compileall -q marine_race_arena tests
conda run -n ocean python -m pytest -q
conda run -n ocean python -m marine_race_arena.scripts.run_staggered_multi_rover_smoke
```

Optional single-rover clean HoloOcean validation:

```bash
conda run -n ocean python -m marine_race_arena.scripts.run_marine_race --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --benchmark-task clean_gate --controller rule_gate_baseline --adapter holoocean --official --headless --seed 0 --dt 0.033 --duration 560 --obstacles none --current-profile none --motion-compensation none --log-dir results/benchmarks/single_rover_clean_manual_run
```

## Stable Fleet Smoke Snapshot

Latest saved local stable smoke output during this release-candidate cleanup:

| participant_id | start_delay_s | release_time_s | status | gates | official_time_s | collisions | inter_vehicle | stuck | out_of_bounds |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bluerov2_01` | 0.0 | 0.0 | FINISHED | 12/12 | 225.324 | 0 | 0 | 0 | 0 |
| `bluerov2_02` | 90.0 | 90.024 | FINISHED | 12/12 | 240.042 | 3 | 0 | 0 | 0 |

Fleet `team_summary` snapshot:

- `team_id`: `fleet_01`
- `rover_count`: 2
- `total_completed_gates`: 24
- `expected_total_gates`: 24
- `all_rovers_finished`: true
- `team_elapsed_time_s`: 344.223
- `total_gate_collisions`: 3
- `total_obstacle_collisions`: 0
- `total_inter_vehicle_collisions`: 0
- `total_collisions`: 3
- `total_penalties_s`: 15.0
- `team_penalized_time_s`: 359.223
- `inter_vehicle_collision_mode`: `diagnostic`

If the smoke is rerun before tagging, use the newest generated `results/benchmarks/staggered_multi_rover_smoke/multi_rover_smoke_summary.json` values in release communication.

## Single-Rover Clean Snapshot

Latest local single-rover official clean Horseshoe validation:

- Track: `marine_race_arena/tracks/marine_race_horseshoe_bay.json`
- Controller: `rule_gate_baseline`
- Adapter: `holoocean`
- Current profile: `none`
- Obstacles: `none`
- Motion compensation: `none`
- Status: `FINISHED`
- Gates: 12/12
- Official time: 228.393 s
- Penalized time: 228.393 s
- Collisions: 0
- Obstacle collisions: 0
- Out-of-bounds events: 0
- Stuck events: 0

## Fleet Team Summary

Fleet mode is not a race between independent teams. All generated rovers belong to one participant/team. Per-rover rows remain available for diagnostics, but the official fleet-level score is `team_summary`.

`team_summary` aggregates:

- rover count
- expected total gates
- total completed gates
- all-rovers-finished flag
- team start time
- team finish time
- team elapsed time
- gate collisions
- obstacle collisions
- inter-vehicle collisions
- total collisions
- penalties
- team penalized time

An inter-vehicle event counts once at team level, not once per rover. Per-rover `involved_inter_vehicle_collisions` is diagnostic only.

## Known Limitations

- Inter-vehicle collision calibration is incomplete.
- `inter_vehicle_collision_mode=diagnostic` is recommended for v0.1.
- `inter_vehicle_collision_mode=penalize` is implemented but experimental.
- Close-proximity fleet racing is not fully validated.
- DVL/current compensation is experimental.
- Current-profile robustness is not an official v0.1 success claim.
- The fallback adapter is not a physical simulator.
- HoloOcean startup can be slow or timeout during repeated calibration sweeps.
- Calibration thresholds currently use conservative defaults until a full successful sweep is available.

## Explicitly Not Included

- A solved current-compensation controller.
- A validated DVL current observer baseline.
- Fully calibrated rover-rover collision penalties.
- Close-formation or head-to-head multi-rover racing.
- Changed official track geometry, gate sizes, or referee rules to hide failures.

## Recommended Description / Citation Placeholder

Use this placeholder until a formal paper citation exists:

> Marine Race Arena v0.1.0 is a HoloOcean-based BlueROV2 underwater gate-racing benchmark with configurable tracks, official clean-gate evaluation, referee scoring, and staggered fleet/team diagnostics.

## Release Checklist

- [ ] `python -m compileall -q marine_race_arena tests` passes.
- [ ] `conda run -n ocean python -m pytest -q` passes.
- [ ] `conda run -n ocean python -m marine_race_arena.scripts.run_staggered_multi_rover_smoke` passes.
- [ ] Optional single-rover clean HoloOcean command passes.
- [ ] README updated.
- [ ] No large generated results or caches are committed.
- [ ] Git status reviewed.
- [ ] Tag created only after validation is complete.
