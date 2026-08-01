# PPO-only sequence curriculum

This is the active learning workflow after the final common benchmark. Hybrid,
rule-controller, and BC-v3 files remain untouched as historical reproducibility
artifacts, but they are not imported by `train_ppo_sequence`, are not sampled as
action sources, are not used as losses, and are not eligible for checkpoint
selection or primary sequence reports.

The initialization policy is exclusively:

`ppo_900462_steps.zip`

SHA-256:

`a1227836242487bab37eedd33162273ef3c5109f7ef90714cbb1afdd8432a490`

The selected architecture is feed-forward PPO. A matched 1,024-step real
HoloOcean smoke retained the transferred policy and completed both known
three-gate failures; the recurrent PPO/LSTM smoke completed 0/3 gates in both
400-step bounded replicas. Recurrent support remains tested and resumable, but
it is not active in this run.

## Legal deployed state

`onboard_ppo_sequence_v4` preserves all 59 v3 inputs and appends six values
maintained by the onboard local course tracker: normalized current index,
remaining count, completed fraction, previous-gate-cleared state, new-target
acquisition state, and time since the last locally confirmed crossing. The
policy never receives simulator pose, referee progress, future gate coordinates,
or the full circuit layout.

The local tracker may identify the expected beacon, confirm a crossing from
camera/beacon/DVL evidence, and advance the target. It has no action API. Every
surge, sway, heave, and yaw value comes directly from PPO.

## Curriculum

| Stage | Gates | Procedural content | Promotion |
|---|---:|---|---|
| S1 | 3 | straight, left/right S-curves, horizontal/vertical offsets, randomized start | 95%, two safety-clean full evaluations, 50k steps |
| S2 | 4–5 | alternating turns, larger yaw/elevation changes, spacing variation | 95%, two safety-clean full evaluations, 75k steps |
| S3 | 6–8 | consecutive turns, S-curves, vertical serpents, recovery straights | 90%, two safety-clean full evaluations, 125k steps |
| S4 | 9–12 | long mixed Horseshoe-like paths without official geometry | 90%, two safety-clean full evaluations, 175k steps |
| S5 | 13–22 | endurance combinations without official seeds/layouts | 90%, two safety-clean full evaluations, 250k steps |

Every full evaluation retains three-gate cases and checks collisions, OOB,
wrong direction, missed-gate DNF, previous-gate return, next-target acquisition,
time, path length, and jerk. An unmeasured category is `null`, never `0.0`.
Official current-free layouts use a separate periodic holdout seed role and are
never replayed into training.

Checkpoint order is: longest safely completed sequence, completion rate, safety,
missed/return cleanliness, completion time, then smoothness. `latest_safe` is
written only after all required safety and sequence-integrity counters are zero.

## Commands

```powershell
conda run -n marine_race_rl --no-capture-output python -m marine_race_arena.learning.train_ppo_sequence train --config configs/rl/ppo_sequence_curriculum.json
conda run -n marine_race_rl python -m marine_race_arena.learning.train_ppo_sequence status results/rl/ppo_sequence_curriculum/ppo900462_long_sequences_seed23001
conda run -n marine_race_rl python -m marine_race_arena.learning.train_ppo_sequence stop results/rl/ppo_sequence_curriculum/ppo900462_long_sequences_seed23001
conda run -n marine_race_rl --no-capture-output python -m marine_race_arena.learning.train_ppo_sequence train --config configs/rl/ppo_sequence_curriculum.json --resume
```

The stop command creates a graceful stop request. The trainer finishes the
current PPO rollout and atomically saves policy, optimizer, learning-rate
schedule, RNG, new-step count, curriculum/replay/reward phase, evaluation
history, and checkpoint-selection aliases before exiting.
