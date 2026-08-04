# Universal local gate-transition PPO

This training path replaces the sequence-length curriculum with one feed-forward
PPO behavior that repeats the same local operation: center the current gate,
cross it in order, acquire the next beacon, and continue. Sequence length and
transition position are neither curriculum variables nor policy inputs.

## Observation and action contract

`onboard_local_transition_v1` has exactly 35 finite, clipped features in this
order:

1. `beacon_present`
2. `beacon_bearing_sin`
3. `beacon_bearing_cos`
4. `beacon_elevation_norm`
5. `beacon_range_norm`
6. `beacon_signal_strength`
7. `beacon_age_norm`
8. `vision_present`
9. `vision_center_x`
10. `vision_center_y`
11. `vision_area_fraction`
12. `vision_confidence`
13. `depth_norm`
14. `depth_present`
15. `depth_error_norm`
16. `depth_reference_present`
17. `dvl_surge_norm`
18. `dvl_sway_norm`
19. `dvl_heave_norm`
20. `dvl_present`
21. `imu_yaw_rate_norm`
22. `imu_present`
23. `previous_surge`
24. `previous_sway`
25. `previous_heave`
26. `previous_yaw`
27. `vision_center_x_rate`
28. `vision_center_y_rate`
29. `vision_rate_valid`
30. `beacon_bearing_rate`
31. `beacon_elevation_rate`
32. `beacon_range_rate`
33. `beacon_rate_valid`
34. `target_changed_recently`
35. `time_since_target_change_norm`

The two derivative masks distinguish a valid measured zero from missing or
newly reacquired data. Beacon angular differences wrap at ±180 degrees, and all
rates use the real environment `dt`. Target changes reset beacon derivatives;
visual loss and reacquisition reset visual derivatives.

The policy never sees simulator pose, true gate geometry, future gates, referee
state, gate index, remaining/total gate count, sequence progress, lap number, or
tracker phases. It directly generates normalized surge, sway, heave, and yaw.
Deterministic code is limited to ordered-crossing confirmation, expected-beacon
selection, legal local temporal state, reward, and termination.

The local transition tracker has a close-passage path for real HoloOcean images
whose gate bars become cropped before a stable visual centroid is available.
Two consecutive close camera detections and forward expected-beacon packets may
start `COMMIT`; advancement still requires DVL displacement, a tight 0.60 m
beacon-range envelope, range turnaround, camera-confirmed disappearance, and
two fresh rear-sector packets. This path is disabled for existing controllers,
uses no referee state, and is covered by cropped-passage and near-miss tests.

## Initialization and reward

Two initialization modes are supported:

- `selective_warm_start` creates a fresh 35-input network and maps only the 26
  retained input columns by semantic feature name from the preserved 65-input
  sequence checkpoint. New temporal columns start at zero. Compatible biases,
  deeper layers, action/value heads, and log standard deviation are copied;
  optimizer state is not copied and no parameter is frozen.
- `scratch` uses normal seeded PPO initialization.

The reward uses bounded signed deltas for current-beacon bearing, elevation and
range, visual centering, centered visual-area growth, and post-crossing
alignment/range progress. It never rewards the deterministic target-change flag.
Every gate has the same crossing reward and completion has one small constant
bonus, independent of sequence length. Missed-gate DNF, wrong-direction crossing,
collision, out of bounds, previous-gate return, and acquisition timeout are
strong one-time penalties. Previous-gate geometry is training/evaluation-only.
Efficiency costs remain locked until three qualifying evaluations at maximum
geometric difficulty; reliability always ranks ahead of speed or smoothness.

## Curriculum and evaluation

Each worker samples focused two-gate transitions and full sequences of 3, 5, 8,
12, 17, or 22 gates. Every G1–G6 level contains varied sequence lengths. Only
geometry becomes harder: alignment, yaw, elevation, combined changes, spacing,
orientation, then official-like patterns. Promotion requires two consecutive
safety-clean evaluations with at least 99% unseen transition success and the
minimum transition count for the current level.

`universal_transition_success_rate` requires the first gate crossing, correct
target switch, next-target alignment and decreasing range, with no return,
missed gate, collision, out-of-bounds event, wrong-direction event, or acquisition
timeout. The dedicated comparison uses at least 1,000 unseen two-gate cases and
also reports full-sequence completion by length and transition success by
position. Checkpoints rank by safety, transition success, long-sequence
completion, jerk, then acquisition time.

Dedicated benchmark cases are written atomically one episode at a time and are
resumed by deterministic case identity after interruption. Two isolated
evaluator processes are used after the two training arms finish. Each loads the
same immutable checkpoint and writes disjoint case directories.

## Parallel HoloOcean layout

The default long run uses one learner and two `SubprocVecEnv` workers with the
Windows `spawn` method. Each worker owns a distinct HoloOcean engine/UUID,
sampler seed, generated-track directory, active track, tracker, and log. The
rollout is 1,024 steps per worker (2,048 total), with batch size 256. Evaluation
uses separate deterministic simulator instances. Training workers are closed
before evaluation begins, then recreated with sampler, curriculum and learner
RNG state restored before learning resumes; rollout and evaluator engines never
overlap.

HoloOcean shutdown is verified against both the exact `_world_process` handle
and the generated HoloOcean UUID owned by each adapter. Normal context-manager
cleanup runs first; if HoloOcean 2.3.0 leaves that child alive (including the
case where its process handle has already reported exit), the adapter reaps only
the same-UUID process before the next engine is launched. This prevents idle
evaluator engines from accumulating across the 1,000-case benchmark.

At every evaluation boundary, the trainer first writes a complete atomic
checkpoint and marks status `evaluating`, then releases rollout workers. If the
machine stops during evaluation, resume recognizes that the boundary evaluation
is still absent from checkpoint history, completes it without adding another
rollout, and atomically replaces the same-timestep checkpoint with the retained
evaluation and curriculum history.

The real benchmark is written to
`results/rl/universal_transition/parallel_benchmark/parallel_benchmark.{json,md}`.
It compares one/two workers and `frames_per_sec=true/false`, records throughput,
wall time, CPU/GPU/RAM, launch failures, sensor cadence/freshness, and exact trace
reproducibility. DVL is configured at 15 Hz in a 30 Hz world, so an emission on
alternate ticks is correct. The selected default is headless,
`frames_per_sec=false`, two workers.

## Commands

Run the controlled 40,960-transition A/B comparison and dedicated unseen suite:

```powershell
conda run -n marine_race_rl --no-capture-output python -m marine_race_arena.learning.compare_transition_initializations --warm-config configs/rl/ppo_universal_transition_ab_warm.json --scratch-config configs/rl/ppo_universal_transition_ab_scratch.json --output-dir results/rl/universal_transition/ab_comparison
```

Run the real parallel benchmark:

```powershell
conda run -n marine_race_rl --no-capture-output python -m marine_race_arena.learning.benchmark_transition_parallelism --output-dir results/rl/universal_transition/parallel_benchmark --transitions 128 --repeats 2
```

Start, inspect, stop, and atomically resume the long run:

```powershell
conda run -n marine_race_rl --no-capture-output python -m marine_race_arena.learning.train_ppo_transition train --config configs/rl/ppo_universal_transition_long.json
conda run -n marine_race_rl --no-capture-output python -m marine_race_arena.learning.train_ppo_transition status results/rl/universal_transition/longrun/universal_transition_seed23001
conda run -n marine_race_rl --no-capture-output python -m marine_race_arena.learning.train_ppo_transition stop results/rl/universal_transition/longrun/universal_transition_seed23001
conda run -n marine_race_rl --no-capture-output python -m marine_race_arena.learning.train_ppo_transition train --config configs/rl/ppo_universal_transition_long.json --run-dir results/rl/universal_transition/longrun/universal_transition_seed23001 --resume
```

Checkpoint manifests are replaced last and hash both the SB3 ZIP and JSON
sidecar. The sidecar preserves optimizer/update counters, absolute learning-rate
schedule, Python/NumPy/Torch RNG, central curriculum/evaluation history, aliases,
and every worker sampler state. `num_timesteps`, evaluation frequency, and
checkpoint frequency always count total transitions across workers.
