# Learned multi-gate policy (observation v3)

> Long-run continuation: the crash-safe, resumable C0-C7 training workflow is
> documented in [`rl_multigate_longrun.md`](rl_multigate_longrun.md). This page
> remains the frozen 5,000-step R0-R2 result and is not rewritten as a long-run
> success.

## Result

**PARTIAL RL SUCCESS.** The new feed-forward policy completed the straight
two-gate task on **9/10 reserved seeds** in real HoloOcean, current-free, with
no simulator fallback and no runtime rule-generated action. It did not meet the
balanced-turn R2 criterion, so the curriculum did not advance to three-gate or
official-circuit RL evaluation.

The compact evidence is in `results/rl_public/multigate_rl_v3/`. Heavy datasets
and SB3 ZIPs remain git-ignored; the package records their exact paths, sizes,
SHA-256 hashes, configurations, and reproduction commands.

## Runtime architecture

The accurate description is:

> An onboard multi-sensor RL controller with minimal deterministic gate-index
> and passage-confirmation logic.

`rl_multigate_controller` receives the official camera, acoustic beacons,
depth, DVL, IMU, and its previous applied action. `LocalCourseTracker` may:

- maintain the expected beacon, local gate index, and lap;
- retain recent onboard visual/range/DVL history;
- confirm passage from multiple onboard signals;
- advance to the next expected beacon;
- declare the mission complete.

It may not generate surge, sway, heave, yaw, commit, exit, homing, centring, or
next-gate-turn actions. All four continuous runtime commands come directly from
the learned policy. The final constraints are code- and manifest-stamped:

```text
rule_action_weight = 0
hybrid_blending = false
rule_controller_instantiated = false
deterministic_runtime_intervention_count = 0
```

Passage confirmation is deterministic state estimation, not deterministic
action control. Disappearance alone never advances a gate: the tracker combines
prior visibility/centering/area, decreasing then receding range, several absent
camera frames, forward DVL displacement, forward action, and the previous beacon
moving rearward. Camera dropout is not treated as a crossing.

## Observation v3

`onboard_multigate_rl_v3` is a fixed `float32[59]`. The first 36 columns are the
unchanged `onboard_only_v1` contract. The appended 23 normalized/masked columns
are:

```text
camera_present
vision_recently_seen
vision_recently_lost
steps_since_gate_seen_norm
steps_since_gate_seen_present
last_seen_center_x
last_seen_center_y
last_seen_area_fraction
last_seen_confidence
last_seen_present
gate_was_recently_centered
gate_was_recently_large
forward_displacement_since_visual_loss_norm
forward_displacement_since_visual_loss_present
beacon_range_delta_norm
beacon_range_delta_present
beacon_range_min_recent_norm
beacon_range_min_recent_present
beacon_now_receding
expected_beacon_changed
steps_since_beacon_change_norm
previous_gate_in_rear_sector
previous_gate_bearing_present
```

No true pose, gate world coordinate, referee target/state, hidden route geometry,
current vector, metric visual gate pose, or training reward is present. A v1
model cannot load silently: the controller checks version `v3`, dimension 59,
and the four-axis action contract.

## Initialization and training

The selected policy is a feed-forward PPO MLP with `[256, 256]` hidden layers.
Recurrence was not needed for R1 and was not introduced.

1. The frozen BC-v1 input layer was expanded from 36 to 59 columns. Existing
   weights and all downstream layers were copied; the 23 new columns were zeroed.
   Neutral-feature output parity was verified.
2. A training-only DAgger expert collected transition labels. The learned policy
   induced the approach/first crossing; `rule_gate_center_then_commit` supplied
   labels only after the local gate switch. Failed episodes were preserved in
   raw collection logs but excluded from the three-episode, 597-step warm-start.
3. The selected `bc_v3_dagger_neutral.pt` initialized PPO. The expert is absent
   at runtime.
4. PPO ran for 5,000 real-HoloOcean steps on `two_gate_straight.json` with:

```text
learning_rate = 1e-5
n_steps = 500
batch_size = 100
n_epochs = 1
clip_range = 0.05
target_kl = 0.01
max_acceptable_kl = 0.02
action_std = 0.10
```

The selected R1 run reached maximum approximate KL `0.001928`, 0% action
saturation, and 1.0 held-out completion at 5,000 steps. Checkpoint selection
used completion, gates, safety, then finished time—not training return.

Every reward component is bounded and separately persisted. Simulator/referee
truth is used only inside training reward for authoritative crossings and
safety/terminal events; it never enters the policy observation. An R2 audit
found that a non-`FINISHED` referee terminal could retain a positive return.
That run was stopped, the terminal accounting was fixed to apply a −30 failure
penalty, and a repeated failed audit returned −28.74.

## Measured curriculum

| Stage | Real-HoloOcean result | Decision |
| --- | --- | --- |
| R0, transferred v3 single gate | 10/10, zero safety events | pass |
| R1 checkpoint selection | 10/10, zero safety events | select |
| R1 reserved final | **9/10**, mean 1.9/2 gates, zero collision/OOB/wrong direction/interventions | pass (≥8/10) |
| R2 left probe | 0/1; failure varied from post-gate timeout to frame collision under engine nondeterminism | hold |
| R2 right probe | 1/1 but 69.6 s, two wrong-direction crossings, three return indications | hold |
| R2 first 5k continuation | best at 2,500: 1/2 gates, one collision; max KL 0.00067 | reject |
| R2 corrected-reward 5k | no improvement at 2,500 or 5,000; max KL 0.000219 | stop PPO |
| R3 / official circuits | not run because R2 criterion was unmet | no claim |

The selected final model is:

```text
path: results/rl/multigate_v3/r1/ppo_multigate_v3/20260727_133031/best_model/best_model.zip
sha256: de7e835132ee57fbe94ee3a2388f0f7554fd6bf41228c8e048000c61fd5b0b6b
```

## Paired comparison

Same track, seeds `21200–21204`, real HoloOcean, currents disabled:

| Controller | Success | Finished mean time | Penalized mean (150 s DNF) | Jerk | Inference |
| --- | ---: | ---: | ---: | ---: | ---: |
| rule | 5/5 | 8.50 s | 8.50 s | 0.0079 | 20.11 ms |
| hybrid | 5/5 | 8.48 s | 8.48 s | 0.0095 | 60.55 ms |
| learned RL | 4/5 | 11.25 s | 39.00 s | 0.0395 | 42.41 ms |

All three had zero collision/OOB/wrong-direction events on this paired subset.
RL had zero deterministic runtime interventions, but did not preserve the
baselines' reliability, speed, or smoothness. The experiment therefore does not
support the preferred superiority claim.

## Reproduce

```bat
scripts\train_rl_multigate_two_gate.bat
scripts\resume_rl_multigate_two_gate.bat <run-dir> 10000
scripts\eval_rl_multigate_two_gate.bat
scripts\compare_rule_hybrid_rl.bat
```

Three-gate and official learned-policy launchers are intentionally not created:
no selected model exists for those stages. Continue R2 only after adding balanced
left/right transition coverage and obtaining at least 8/10 held-out success in
both directions.
