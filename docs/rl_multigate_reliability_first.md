# Reliability-first multi-gate PPO

Status: **RELIABILITY-FIRST LONG-RUN READY**. The one-million-step run is
prepared but has not been started. The operator must launch it manually.

## Scope and protected initialization

The profile starts from the frozen selected BC-v3:

```text
results/rl/multigate_longrun/bc_v3_balanced_v2_20260728/bc_v3.pt
SHA-256 c9a889125278122b57307e21b7c2390baa61429f8952093bcc2760d9b3a49bf2
```

All rollout actions come from the PPO policy. The optional retention step uses
only previously recorded offline observations/actions. It never constructs a
rule or expert controller in the training environment.

The committed configuration is
`configs/rl_multigate_longrun_reliability_first.json`: BC-v3 initialization,
1,000,000 steps, LR `1e-5 -> 2e-6`, 2 PPO epochs, 2,048 rollout steps, batch
256, clip 0.05, target/high/absolute KL `0.005/0.012/0.020`, entropy 0.0005,
and initial action std 0.08.

## Reliability gates

Mission completion and gate order dominate every speed metric. Only a full
20-case evaluation can promote. Promotion requires two consecutive full
evaluations with:

- overall completion at least 90%;
- single-gate retention at least 95%;
- straight retention at least 90%;
- left and right completion at least 85%;
- zero collision, out-of-bounds, and wrong-direction affected episodes;
- at most two previous-gate returns.

C0/C1/C2/C3/C4 must remain active for at least
25k/50k/75k/100k/100k steps respectively. Later stages increase the
left/right floor by two percentage points per stage after C1, capped at 95%.
Light five-case suites never update the promotion streak.

The first 50k steps use replay weights 30% single retention, 25% straight,
25% current stage, 15% previous stage, and 5% targeted failures. The normal
mixture is 20/20/30/20/10. Turn sampling actively corrects left/right count
imbalance from C2 onward.

## Safety metrics

Evaluation and status keep these values distinct:

```text
collision_event_count          collision_frame_count
out_of_bounds_event_count      out_of_bounds_frame_count
wrong_direction_crossing_count
safety_warning_event_count     safety_warning_frame_count
episodes_with_any_safety_event
```

Event counters come from the unchanged referee. Contact/out-of-bounds frames
come from the physical per-tick episode state. Warning frames indicate a
vehicle within 0.5 m of a world boundary without already being outside.
Promotion, rollback, and checkpoint rejection use events and affected
episodes, never repeated frame persistence.

## BC retention and KL

After each configured PPO update, a bounded auxiliary optimizer step samples
the committed offline datasets. Its weight starts at 0.10, is capped at 0.20,
decays through 150k steps, and is disabled after C2. A frozen BC-v3 mean and
std 0.08 define the initial-policy KL anchor.

Every update logs:

```text
ppo_policy_loss
bc_retention_loss
initial_policy_kl
combined_policy_loss
retention_weight
```

No auxiliary data are collected online and no runtime expert query occurs.

## Reward phases

The initial `reliability` phase has zero time cost. Completion, correct order,
safety, post-gate clearance, and next-gate acquisition dominate. After two
consecutive reliable full suites, `efficiency` activates a capped time,
detour, jerk, and energy penalty.

The efficiency time term is at most:

```text
0.002 * 1800 = 3.6 reward units per episode
```

The failed-episode timeout penalty is 30.0, so a faster failure cannot recover
the failure penalty through the time term. Checkpoint selection independently
puts completion before every speed metric.

## Checkpoints and rollback

Checkpoint ordering is completion, minimum left/right success, single
retention, straight retention, affected safety episodes, previous-gate
returns, gates, penalized time, successful time, jerk, then inference time.
Mean training reward is never a selector.

The pipeline maintains:

```text
initial_bc_v3
best_reliable
best_fast_reliable
best_left
best_right
best_three_gate
best_overall
latest_safe
last
```

`best_fast_reliable` is ineligible unless the full reliability threshold
passes. `best_three_gate` is created only from a suite containing three-gate
cases.

A full-suite regression triggers when completion falls by more than ten
percentage points, single retention falls below 90%, straight below 85%,
either direction below 70%, a safety-episode trend appears, or
previous-gate returns exceed the configured limit. The regressed checkpoint
is retained; the pipeline restores `best_reliable` policy, optimizer, and
compatible scheduler state, halves LR, can raise retention weight, demotes
one stage, resets to a new episode, and increments `rollback_count`. More than
two automatic attempts stops safely.

## Bounded real-HoloOcean validation

All runs below used real HoloOcean, no currents, no fallback, and code SHA
`c76c23f565d3906360f9d31723d371c7fc506e6c`.

The frozen BC-v3 baseline on 20 validation cases (seeds 25500–25519) was
20/20. Single, straight, left, and right were all 100%; all safety
event/frame counts and previous-gate returns were zero. Mean successful and
penalized time was 19.235 s, jerk 0.023388.

The conservative PPO smoke requested 25,600 steps and completed one SB3
rollout boundary later at 26,624. It stayed in C0 with one reliable full pass,
reward phase `reliability`, zero rollback, zero simulator restart, zero NaN,
max approx-KL 0.001942, max initial-policy KL 0.002930, and zero action
saturation. Its 12,288-step light suite was 5/5 with zero safety. Its
24,576-step full suite was 20/20, all four categories 100%, zero safety
events/frames, zero returns, time 18.05 s, and jerk 0.022504. One full pass
did not promote.

A paired straight two-gate check used identical seeds 25400–25404:

| Controller | Completion | Safety | Returns | Mean time (s) | Jerk |
| --- | ---: | ---: | ---: | ---: | ---: |
| BC-v3 | 5/5 | 0 | 1 | 9.18 | 0.0218 |
| best_reliable PPO | 5/5 | 0 | 0 | 8.26 | 0.0198 |
| rule | 5/5 | 0 | 0 | 8.46 | 0.0080 |
| hybrid | 5/5 | 0 | 0 | 8.48 | 0.0097 |

The paired result measures a small straight-track sample only. It does not
establish general PPO improvement, three-gate success, or official-circuit
success. Those claims remain locked.

## Manual commands

Preparation:

```bat
conda run -n marine_race_rl python -m marine_race_arena.learning.longrun_tools prepare --config configs\rl_multigate_longrun_reliability_first.json
```

Start exactly one million steps (manual only):

```bat
scripts\start_rl_multigate_reliability_first.bat 1000000 r2_reliability_first_seed23001
```

Status and TensorBoard:

```bat
scripts\status_rl_multigate_reliability_first.bat results\rl\multigate_reliability_first\r2_reliability_first_seed23001
conda run -n marine_race_rl tensorboard --logdir results\rl\multigate_reliability_first\r2_reliability_first_seed23001\tensorboard
```

Graceful stop and resume:

```bat
scripts\stop_rl_multigate_reliability_first.bat results\rl\multigate_reliability_first\r2_reliability_first_seed23001
scripts\resume_rl_multigate_reliability_first.bat results\rl\multigate_reliability_first\r2_reliability_first_seed23001 500000
```

Selection and R2 evaluation:

```bat
scripts\select_best_rl_multigate_reliability_first.bat results\rl\multigate_reliability_first\r2_reliability_first_seed23001
scripts\eval_rl_multigate_reliability_first.bat results\rl\multigate_reliability_first\r2_reliability_first_seed23001 best_reliable
```

The same eval command accepts `initial_bc_v3`, `best_fast_reliable`,
`latest_safe`, and `last`.

Three-gate evaluation, only after the stored R2 evidence passes:

```bat
conda run -n marine_race_rl python -m marine_race_arena.learning.closed_loop_eval --track marine_race_arena\tracks\tests\three_gate_s_curve.json --controller rl_multigate_controller --model results\rl\multigate_reliability_first\r2_reliability_first_seed23001\best_models\best_reliable.zip --seeds 25400-25419 --out results\rl\multigate_reliability_first\r2_reliability_first_seed23001\evaluations\three_gate --adapter holoocean --current-profile none
```

Paired BC/PPO/rule/hybrid comparison:

```bat
scripts\compare_rl_multigate_reliability_first.bat results\rl\multigate_reliability_first\r2_reliability_first_seed23001 best_reliable
```

Official current-free circuits remain locked until reliable three-gate
success. The full one-million-step command above was deliberately not run.
