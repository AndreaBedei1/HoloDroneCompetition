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

Three initialization modes are supported:

- `selective_warm_start` creates a fresh 35-input network and maps only the 26
  retained input columns by semantic feature name from the preserved 65-input
  sequence checkpoint. New temporal columns start at zero. Compatible biases,
  deeper layers, action/value heads, and log standard deviation are copied;
  optimizer state is not copied and no parameter is frozen.
- `warm_start_transition_checkpoint` continues an already-competent 35-feature
  policy. Policy and value networks transfer verbatim; the optimizer,
  learning-rate schedule, PPO rollout state and total-transition counter all
  start fresh, so the run inherits the behavior without inheriting the source
  experiment's optimizer momentum or schedule progress. Every transferred
  parameter stays trainable. The source path and SHA-256 are recorded in the
  run manifest and verified during preflight.
- `scratch` uses normal seeded PPO initialization.

The reward uses bounded signed deltas for current-beacon bearing, elevation and
range, visual centering, centered visual-area growth, and post-crossing
alignment/range progress. It never rewards the deterministic target-change flag.
Every gate has the same crossing reward and completion has one small constant
bonus, independent of sequence length. Missed-gate DNF, wrong-direction crossing,
out of bounds, previous-gate return, and acquisition timeout are strong one-time
penalties. Previous-gate geometry is training/evaluation-only. Efficiency costs
remain locked until three qualifying evaluations at maximum geometric
difficulty; reliability always ranks ahead of speed or smoothness.

### Collision accounting

Collision shaping is charged per *entry* into contact, never per contact frame:

- entering contact costs the full `collision_penalty` (50.0), and a distinct
  second impact after a confirmed separation is charged again in full;
- remaining in contact costs `collision_contact_frame_penalty` (0.5) per frame,
  capped at `collision_contact_penalty_episode_cap` (10.0) for the whole
  episode, so hundreds of consecutive contact frames can never accumulate
  without limit and can never outweigh the impact that caused them;
- cumulative entry shaping is capped at
  `collision_entry_penalty_episode_cap` (150.0). Both caps are per-episode
  constants, so collision cost does not depend on sequence length;
- a single dropped contact frame does not end a contact.
  `collision_separation_frames` (3) clear frames are required before the next
  contact counts as a new impact.

One collision therefore stays far more expensive than any slow but valid
correction: the whole efficiency budget over a 200-step episode is under a tenth
of a single collision entry. `collision_episode`, `collision_entry` and
`collision_contact_frames` are counted and reported separately.

## Curriculum and evaluation

Each worker samples focused two-gate transitions and full sequences of 3, 5, 8,
12, 17, or 22 gates. Every G1–G6 level contains varied sequence lengths. Only
geometry becomes harder: alignment, yaw, elevation, combined changes, spacing,
orientation, then official-like patterns. Promotion requires two consecutive
evaluations that pass the competence gate, are safety-clean, and reach at least
99% unseen transition success, plus the minimum transition count for the current
level.

`universal_transition_success_rate` requires the first gate crossing, correct
target switch, next-target alignment and decreasing range, with no return,
missed gate, collision, out-of-bounds event, wrong-direction event, or acquisition
timeout. The dedicated comparison uses at least 1,000 unseen two-gate cases and
also reports full-sequence completion by length and transition success by
position.

### Competence qualification, safety ranking, final selection

Selection is three explicitly separated stages
(`marine_race_arena/learning/transition_selection.py`).

**1. Competence qualification (mandatory).** A policy that produces few safety
events by refusing to move is not safe, it is inactive. Before any safety
comparison a candidate must clear every minimum below. Metrics absent from an
older report are recorded as not evaluated and cannot fail the gate.

| Criterion | Minimum | Group |
|---|---:|---|
| `first_gate_crossing_rate` | 0.80 | task competence |
| `target_switch_rate` | 0.70 | task competence |
| `universal_transition_success_rate` | 0.20 | task competence |
| `completed_gate_count` | 10 | task participation |
| `mean_completed_gates_per_episode` | 0.50 | task participation |
| `fraction_of_episodes_reaching_first_gate` | 0.80 | task participation |
| `mean_distance_travelled_m` | 1.0 | activity |
| `mean_absolute_action` | 0.02 | activity |
| `nontrivial_action_fraction` | 0.25 | activity |
| `transition_n` | 50 | evidence |

The activity minimums come from the measured arms: scratch commanded a mean
absolute action of 0.0034 with no step above 0.05, while warm commanded 0.0848
with 97.9% of steps non-trivial. Every threshold therefore sits far above the
inactive policy and far below the competent one. Failing an activity criterion
classifies a policy `degenerate_inactive_policy`; failing only competence or
participation classifies it `insufficient_task_competence`; too few evaluated
cases classifies it `insufficient_evaluation_evidence`. Only qualified policies
are ranked.

**2. Safety ranking (qualified policies only),** in this fixed priority, with
every counter normalized per evaluated episode:

1. collision episodes
2. collision entries
3. missed-gate DNF
4. wrong-direction events
5. previous-gate returns
6. acquisition timeouts
7. universal transition success
8. long-sequence completion
9. jerk and efficiency

Collision *episodes* rank first and collision *entries* second.
`collision_contact_frames` is reported but never ranked, because prolonged
contact generates many repeated events from a single physical collision and
would otherwise outweigh several distinct impacts.

**3. Final selection** is the deterministic maximum over candidates visited in
sorted name order, so equal keys always resolve identically.

### Anti-inactivity metrics

Transition and full-sequence evaluations report first-gate crossing rate,
target-switch rate, universal transition success, completed gates (total and per
episode), fraction of episodes reaching the first gate, mean distance travelled,
mean absolute action per axis (`surge`, `sway`, `heave`, `yaw`) and overall,
fraction of steps with a non-trivial action, acquisition-timeout rate and count,
collision episodes, collision entries, sustained collision contact frames,
missed gates, wrong-direction events, previous-gate returns and jerk.

Motion is never treated as success. These metrics exist only to detect
degenerate inactivity, and none of them is part of the policy observation: the
encoder still sees exactly the 35 onboard features listed above. Step
displacement is exposed through the environment `info` dictionary only, which
the observation encoder never reads.

### Two evaluation levels

Waiting for a distant evaluation risks discovering a lost warm-start behavior
far too late, so the long run evaluates on an explicit early schedule of 25,000,
50,000, 75,000, 100,000 and 150,000 transitions, then every 50,000.

- **Intermediate** (every scheduled point): 100 unseen two-gate transitions plus
  two full sequences at each of lengths 3, 5, 8, 12, 17 and 22, at fixed
  evaluation seeds that are never used for training.
- **Dedicated** (~1,000 unseen transitions): only when the first competent
  checkpoint appears, when competence materially improves (at least +0.05
  transition success over the best validated checkpoint, and at least 100,000
  transitions since the previous dedicated run), when a checkpoint becomes the
  new candidate best, or when the final model is validated.

Both levels evaluate the atomic checkpoint written immediately before the
boundary rather than the live learner object, write cases atomically one episode
at a time, and resume by deterministic case identity after interruption. Both
run across two isolated evaluator processes, each loading the same immutable
checkpoint and writing disjoint case directories, after the rollout workers have
been closed; rollout and evaluator engines never overlap.

### Checkpoint aliases and rollback

Aliases are `last`, `latest_competent`, `latest_safe_competent`,
`best_universal_transition` and `best_long_sequence`. Every alias except `last`
requires the competence gate, so `latest_safe_competent` can never be assigned
to an inactive policy, and `latest_safe_competent` additionally requires a
safety-clean evaluation at 99% transition success. Promotion of geometric
difficulty likewise requires competent evaluations; sequence length is never a
curriculum variable and never a policy input.

The warm-start source is copied once into `<run>/baseline/` with its SHA-256 and
is never rewritten, so an immutable rollback target survives even if the source
experiment directory moves. If two consecutive evaluations fall below the
initialization baseline (`first_gate_crossing_rate` < 0.80,
`target_switch_rate` < 0.70, `universal_transition_success_rate` < 0.20) the run
records a rollback pointing at `latest_competent` (or the immutable baseline)
and stops with reason `baseline_collapse_rollback` rather than spending the rest
of the budget on a collapsed policy. A single recovered evaluation clears the
counter.

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
the same-UUID process before the next engine is launched. A final ownership
check enforces zero `Holodeck.exe` children under that sequential worker after
close; it cannot affect the other evaluator worker. This prevents idle evaluator
engines from accumulating across the 1,000-case benchmark. The same exact-owner
cleanup runs between HoloOcean world-candidate retries because `holoocean.make`
can raise after spawning Unreal but before returning an environment context.

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

## Completed validation

The real parallel benchmark selected two unthrottled workers. One worker with
`frames_per_sec=false` averaged 7.695 transitions/s and 16.634 s per 128-step
rollout; two workers averaged 12.316 transitions/s and 10.393 s, a 1.601x
speedup. The corresponding `frames_per_sec=true` results were 7.413 and 12.381
transitions/s. Every launch succeeded, camera/IMU/depth frames were valid, and
the DVL emitted on its expected alternate ticks. Deterministic case generation
was reproducible; noisy simulator observation traces were not bit-identical.

Both A/B arms trained for exactly 40,960 environment transitions with identical
seeds, geometries, PPO settings, and evaluation cases. Their final dedicated
holdouts each contain 1,000 unseen two-gate transitions plus two unseen full
sequences at every length 3, 5, 8, 12, 17, and 22:

| Initialization | Safety events | Transition success | First crossing | Target switch | Alignment | Range decrease | Long score | Mean jerk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Selective warm start | 983 | 31.4% | 91.2% | 81.9% | 42.6% | 81.5% | 0.25 | 0.109872 |
| Scratch | 122 | 0.0% | 1.5% | 0.2% | 0.0% | 0.2% | 0.00 | 0.005676 |

Warm start recorded 417 collision episodes (2,591 collision events), 78 missed
gates, one previous-gate return, 51 wrong-direction events, and 436 acquisition
timeouts. Its full-sequence completion rates were 100% at length 3, 50% at
length 5, and 0% at lengths 8, 12, 17, and 22. Scratch recorded 103 collision
episodes (539 events), nine missed gates, no previous-gate returns, four
wrong-direction events, and six acquisition timeouts; it completed none of the
full sequences.

### Corrected selection

The original ranking summed safety events and compared them directly. It
selected `scratch` (122 events versus 983) even though that policy crossed 15
gates in 1,012 episodes and commanded a mean absolute action of 0.0034 — it
avoided events by barely moving. A long run initialized from it stalled: at
100,352 transitions its holdout showed 0.0% transition success, 45% first
crossing and 0% target switch.

The competence gate corrects this. `scratch` fails seven criteria and is
classified `degenerate_inactive_policy`; `selective_warm_start` passes and is
selected. The corrected report is written to
`results/rl/universal_transition/ab_comparison/ab_comparison_corrected.{json,md}`;
the original `ab_comparison.{json,md}`, both arms' checkpoints and every episode
artifact are left unmodified as evidence.

| Initialization | Classification | Completed gates | Gates/episode | Reaching first gate | Mean abs action | Non-trivial action fraction |
|---|---|---:|---:|---:|---:|---:|
| Selective warm start | `competent` | 946 | 0.9348 | 0.9130 | 0.0889 | 0.9876 |
| Scratch | `degenerate_inactive_policy` | 15 | 0.0148 | 0.0148 | 0.0034 | 0.0000 |

Action magnitudes are measured from the recorded trajectories (14 episodes per
arm), and the sample size is stated with them in the corrected report.

The warm start is **competent but not yet reliable**. It is selected because it
is the only initialization that learned useful gate-crossing and beacon-switch
behavior, not because it is safe: it still produced 417 collision episodes, 78
missed gates, 51 wrong-direction events and 436 acquisition timeouts, and 31.4%
is far below the 99% requirement. The immediate objective is to preserve its
crossing and beacon-switch behavior while reducing collisions, missed gates,
wrong direction, previous-gate returns and acquisition timeouts.

## Commands

Run the controlled 40,960-transition A/B comparison and dedicated unseen suite:

```powershell
conda run -n marine_race_rl --no-capture-output python -m marine_race_arena.learning.compare_transition_initializations --warm-config configs/rl/ppo_universal_transition_ab_warm.json --scratch-config configs/rl/ppo_universal_transition_ab_scratch.json --output-dir results/rl/universal_transition/ab_comparison
```

Run the real parallel benchmark:

```powershell
conda run -n marine_race_rl --no-capture-output python -m marine_race_arena.learning.benchmark_transition_parallelism --output-dir results/rl/universal_transition/parallel_benchmark --transitions 128 --repeats 2
```

Regenerate the corrected A/B report from the frozen artifacts without touching
the original evidence:

```powershell
conda run -n marine_race_rl --no-capture-output python -m marine_race_arena.learning.transition_ab_report --arm selective_warm_start=results/rl/universal_transition/ab_comparison/ab_selective_warm_atomic_v3_seed23001 --arm scratch=results/rl/universal_transition/ab_comparison/ab_scratch_atomic_v3_seed23001 --output-dir results/rl/universal_transition/ab_comparison --output-name ab_comparison_corrected --original-report results/rl/universal_transition/ab_comparison/ab_comparison.json
```

Start, inspect, stop, and atomically resume the reliability long run initialized
from the competent warm-start weights:

```powershell
conda run -n marine_race_rl --no-capture-output python -m marine_race_arena.learning.train_ppo_transition train --config configs/rl/ppo_universal_transition_warm_reliability.json
conda run -n marine_race_rl --no-capture-output python -m marine_race_arena.learning.train_ppo_transition status results/rl/universal_transition/longrun/universal_transition_warm_reliability_seed23001
conda run -n marine_race_rl --no-capture-output python -m marine_race_arena.learning.train_ppo_transition stop results/rl/universal_transition/longrun/universal_transition_warm_reliability_seed23001
conda run -n marine_race_rl --no-capture-output python -m marine_race_arena.learning.train_ppo_transition train --config configs/rl/ppo_universal_transition_warm_reliability.json --run-dir results/rl/universal_transition/longrun/universal_transition_warm_reliability_seed23001 --resume
```

The superseded scratch long run
(`results/rl/universal_transition/longrun/universal_transition_seed23001`) and
all of its checkpoints are preserved unchanged; it is never reused or resumed.

Checkpoint manifests are replaced last and hash both the SB3 ZIP and JSON
sidecar. The sidecar preserves optimizer/update counters, absolute learning-rate
schedule, Python/NumPy/Torch RNG, central curriculum/evaluation history, aliases,
and every worker sampler state. `num_timesteps`, evaluation frequency, and
checkpoint frequency always count total transitions across workers.
