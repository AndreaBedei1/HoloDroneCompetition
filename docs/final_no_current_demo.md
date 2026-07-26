# Current-free official circuits — autonomous onboard completion

This is the practical engineering deliverable: **a fully autonomous onboard-only controller
completes all three official race circuits in real HoloOcean with currents explicitly
disabled**, no manual intervention, no simulator fallback, and no privileged controller input.

## Current-free mode (runtime override, official JSON unchanged)

`closed_loop_eval` (and `multigate_diagnostic`) accept `--current-profile none`, which resolves
the track with **all currents removed** and records the applied currents in the run manifest,
asserting `currents_actual == [0, 0, 0]` before any episode runs. The official circuit JSON files
are **not** edited — this is a runtime experiment override only.

Two of the circuits (`horseshoe_bay`, `vertical_serpent`) are already `clean_gate` tasks with no
configured current, so `--current-profile none` is a no-op there. `mixed_endurance` is a
`current_gate` task whose validation *requires* a current, so a current-free run additionally
passes `--benchmark-task clean_gate`. That override changes only the validation mode — the track
geometry, gate positions, gate order, laps and the referee (margins, scoring) are unchanged. The
manifest records `current_profile`, `benchmark_task_override` and `currents_actual` for audit.

## Controller

The best onboard controller for the circuits is the deterministic **`rule_gate_center_then_commit`**
(`controllers/official_baselines.py`): onboard-only (received beacons + FrontCamera + depth / IMU
/ DVL + its own `LocalCourseTracker`), beacon homing → visual align → center-then-commit passage
→ exit → next-beacon turn. It reads no referee, world pose or gate pose. The learned single-gate
BC-v1 does **not** complete multi-gate on its own (see `docs/multigate_curriculum.md`); the
`hybrid_gate_controller` integrates BC-v1's visual servo on top of this same backbone.

## Results

Fresh, self-contained verification from this branch's committed SHA is published under
`results/rl_public/visual_pose_v2/official_no_current/`:

```
official_no_current/
  summary.json / summary.csv        # per-circuit completion, currents_actual, referee status
  circuit_horseshoe/                # eval_results.json + evaluation_manifest.json (per seed)
  circuit_vertical/
  circuit_mixed/
```

Each `evaluation_manifest.json` records `adapter_actual = holoocean`, `fallback_allowed = false`,
`current_profile = none`, `currents.currents_actual = [0,0,0]`, the model/controller, the track
sha256 and the code `git_sha`. Per-seed rows record the referee status, completed vs expected
gates, and collision / out-of-bounds / wrong-direction counts.

The frozen benchmark's independent clean (current-free) matrix corroborates this:
`rule_gate_center_then_commit` completes Horseshoe 5/5, Vertical 4/5, Mixed 5/5.

## Reproduce

```bash
# All three circuits, current-free, sequentially, with a final summary:
scripts\run_all_official_no_current.bat

# Or one at a time:
scripts\run_official_horseshoe_no_current.bat
scripts\run_official_vertical_no_current.bat
scripts\run_official_mixed_no_current.bat    # adds --benchmark-task clean_gate
```

Every run uses real HoloOcean (`--adapter holoocean`, no `--allow-fallback`), disables currents,
and writes a manifest proving `currents_actual == [0,0,0]`. Use the dedicated `marine_race_rl`
conda environment.
