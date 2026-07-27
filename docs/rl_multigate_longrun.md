# Multi-gate RL long-run operations

Status: **LONG-RUN TRAINING PIPELINE READY**.

This page is the operator manual for the manually launched, current-free
observation-v3 PPO run. It describes infrastructure and bounded validation,
not a completed long-run result. The selected R1 model remains frozen at
`results/rl/multigate_v3/r1/ppo_multigate_v3/20260727_133031/best_model/best_model.zip`
with SHA-256
`de7e835132ee57fbe94ee3a2388f0f7554fd6bf41228c8e048000c61fd5b0b6b`.

## Why a longer run is needed

The earlier 5,000-environment-step experiment established that the complete
real-HoloOcean PPO path worked and retained useful straight two-gate
competence. It was not enough experience to learn a balanced distribution of
post-crossing turns. PPO environment timesteps are simulator interactions;
they are not supervised epochs. A BC epoch may revisit every stored label,
whereas each PPO timestep provides one new on-policy transition and only a
small number of bounded optimizer passes over a rollout.

The default plan therefore exposes 100k, 250k, 500k, 1M, and 2M step runs and
defaults to one million steps. Progress is judged by completed held-out
evaluations and environment steps, never by a promised wall-clock duration.

## Data and BC-v3 preparation

The versioned data layout is:

```text
datasets/multigate_v3/
    r1_original/
    r2_balanced_turns_v1/
    r2_dagger_corrections_v1/
```

The balanced plan contains 30 straight, 50 left, and 50 right transition
episodes spanning 0 to +/-5, +/-10, +/-15, +/-20, +/-30, +/-40, and +/-45
degree turns, variable separation, offsets, starting yaw, gate loss, noise,
and recovery difficulty. Raw episodes and generated tracks are ignored by
Git; only plans and manifests are committed.

Collect expert trajectories from clean committed code:

```bat
conda run -n marine_race_rl python -m marine_race_arena.learning.multigate_longrun_data collect --out datasets\multigate_v3\r2_balanced_turns_v1 --straight 30 --left 50 --right 50 --first-seed 26000
```

Collect genuine DAgger corrections by rolling out the learned R1 policy,
querying the deterministic controller only in the shadow expert, and retaining
the pre-crossing-to-acquisition window and difficult visited states:

```bat
conda run -n marine_race_rl python -m marine_race_arena.learning.multigate_longrun_data collect --out datasets\multigate_v3\r2_dagger_corrections_v1 --straight 10 --left 20 --right 20 --first-seed 26200 --dagger-model results\rl\multigate_v3\r1\ppo_multigate_v3\20260727_133031\best_model\best_model.zip
```

Train candidate BC-v3 models with geometry/track/seed-disjoint splits:

```bat
conda run -n marine_race_rl python -m marine_race_arena.learning.bc_longrun_v3 --single-gate results\rl\stage1\demos_rand2\stage1_demos.npz --straight results\rl\multigate_v3\demos\two_gate_dagger\multigate_v3_demos.npz --turns datasets\multigate_v3\r2_balanced_turns_v1\dataset.npz --corrections datasets\multigate_v3\r2_dagger_corrections_v1\dataset.npz --out results\rl\multigate_longrun_bc_v3\bc_v3.pt --epochs 200 --patience 25 --seed 28000 --closed-loop
```

BC-v3 starts from the selected R1 policy function and mixes single-gate,
straight-transition, turning-transition, and corrective samples with
retention anchors. Early stopping uses validation loss, but final candidate
selection is lexicographic: single gate, straight, left, right, safety, then
time. It is not selected by frame-level MSE alone.

## Curriculum

The runtime generator makes a new, current-free training track at every reset
without changing referee rules:

| Stage | Distribution |
| --- | --- |
| C0 | Single-gate retention, straight two-gate, small perturbations |
| C1 | Signed turn uniformly sampled in `[-10, +10]` degrees |
| C2 | Signed turn uniformly sampled in `[-20, +20]` degrees |
| C3 | Signed turn uniformly sampled in `[-30, +30]` degrees |
| C4 | Signed turn uniformly sampled in `[-45, +45]` degrees |
| C5 | Three-gate left/right, S-curves, vertical steps |
| C6 | Six-gate and progressively longer sequences |
| C7 | Official current-free circuits, gated behind prior success |

The nominal replay mixture is 20% retention, 20% straight, 30% current stage,
20% previous difficult stage, and 10% targeted failures. Promotion requires
multi-episode balanced evidence. Retention collapse can demote a stage, and
all transitions are appended to `curriculum_history.jsonl`.

The default remains the 59-feature feed-forward observation-v3 policy. An
explicit four-frame experimental configuration is available at
`configs/rl_multigate_longrun_frame_stack.json`. It expands the R1 input layers
so the newest frame initially reproduces the frozen policy, stamps the stack in
the policy/checkpoint contract, resets history correctly at episode boundaries,
and evaluates with the same stack. It is not the default and should be launched
in a separate run directory only after a measured feed-forward plateau.
Recurrence is deliberately not switched on automatically; it would require a
separate policy family, hidden-state reset/evaluation contract, and evidence
that stacking is insufficient.

## PPO and safety policy

The committed default is MLP `[256, 256]`, learning rate `3e-5` with linear
decay to `5e-6`, rollout 2,048, batch 256, four epochs, gamma 0.995, GAE 0.95,
clip 0.10, target KL 0.01, entropy coefficient 0.001, value coefficient 0.5,
gradient clipping 0.5, and initial action standard deviation 0.12.

Each update records approximate KL, clip fraction, policy/value losses,
entropy, explained variance, action standard deviation, measured gradient
norm, action saturation, completion/gates, directional counts, and every
reward component. Adaptation is bounded:

- sustained KL below 0.001 plus flat evaluation allows one 1.25x learning-rate
  adjustment;
- KL above 0.02 halves the schedule and reduces epochs once; repetition stops
  safely;
- KL above 0.03 saves an unsafe diagnostic checkpoint and requires an explicit
  resume;
- NaN, low disk, a plateau, or `stop.requested` stops at a checkpoint boundary.

Plateau detection uses held-out completion and gates across at least 100,000
steps. It writes a recommendation rather than silently changing architecture.

The reward records authoritative crossing separately from tracker state. Only
the referee can award a gate/completion bonus. Post-crossing shaping rewards
the signed turn, leaving the previous gate behind, next-beacon progress,
stable visual reacquisition, centring, and useful forward progress. It
penalizes return, stationary behavior, wrong-direction events, action jerk,
collisions, OOB, and timeout.

## Run layout and recovery

Every run is isolated under:

```text
results/rl/multigate_longrun/<run_name>/
    config.json
    status.json
    pid.txt
    checkpoints/
    best_models/
    evaluations/
    tensorboard/
    logs/
    curriculum_history.jsonl
    crash_reports/
```

Checkpoint ZIP, state, and manifest are written atomically; the manifest is
the commit marker. Resume verifies hashes, ZIP integrity, code/config
contract, observation/action versions, then restores the SB3 optimizer,
scheduler, timestep count, RNG, curriculum, evaluation history, and selection
state. Simulator physical state is intentionally not restored: HoloOcean
starts a fresh episode. Recoverable simulator failures close the episode and
restart up to five times. Logs rotate at 20 MiB with five backups; preflight
checks disk, RAM, GPU visibility, Ocean installation, model hash, clean Git,
and conflicting live training PIDs.

## Exact operating commands

Prepare and validate the machine:

```bat
scripts\prepare_rl_multigate_longrun.bat
```

Start the recommended 1,000,000-step run in a hidden background process:

```bat
scripts\start_rl_multigate_longrun.bat 1000000 r2_longrun_seed22001
```

Inspect status:

```bat
scripts\status_rl_multigate_longrun.bat results\rl\multigate_longrun\r2_longrun_seed22001
```

Request a graceful stop and wait for a final checkpoint:

```bat
scripts\stop_rl_multigate_longrun.bat results\rl\multigate_longrun\r2_longrun_seed22001
```

Resume for 500,000 additional steps:

```bat
scripts\resume_rl_multigate_longrun.bat results\rl\multigate_longrun\r2_longrun_seed22001 500000
```

Open TensorBoard:

```bat
conda run -n marine_race_rl tensorboard --logdir results\rl\multigate_longrun\r2_longrun_seed22001\tensorboard
```

Select the best valid safe checkpoint:

```bat
scripts\select_best_rl_multigate_checkpoint.bat results\rl\multigate_longrun\r2_longrun_seed22001
```

Evaluate held-out straight, left, and right transitions plus retention:

```bat
scripts\eval_rl_multigate_longrun.bat results\rl\multigate_longrun\r2_longrun_seed22001 r2
```

Compare rule, hybrid, and learned controllers on identical held-out seeds:

```bat
scripts\compare_rl_multigate_longrun.bat results\rl\multigate_longrun\r2_longrun_seed22001
```

After the configured R2 gate passes, evaluate the three-gate S-curve:

```bat
conda run -n marine_race_rl python -m marine_race_arena.learning.closed_loop_eval --track marine_race_arena\tracks\tests\three_gate_s_curve.json --controller rl_multigate_controller --model results\rl\multigate_longrun\r2_longrun_seed22001\best_models\best_overall.zip --seeds 25100-25119 --out results\rl\multigate_longrun\r2_longrun_seed22001\evaluations\three_gate --adapter holoocean --current-profile none
```

Only after the R2 and R3 gates pass may the operator explicitly request the
official suite:

```bat
scripts\eval_rl_multigate_longrun.bat results\rl\multigate_longrun\r2_longrun_seed22001 official
```

The official command refuses to run unless the stored evaluation evidence
unlocks it. All final evaluations use real HoloOcean, currents disabled, no
fallback, no runtime rule action, no hybrid blending, and held-out seeds.

## Success criteria

Light evaluation runs every 10,000 steps on fixed development cases. Full
evaluation runs every 50,000 steps on at least 20 balanced held-out cases,
including retention and prior stages. Checkpoints are written every 10,000
steps. Selection is lexicographic: completion, worst left/right performance,
gates, safety, previous-gate returns, penalized time, then jerk. The pipeline
maintains `best_overall`, `best_left`, `best_right`, `best_three_gate`,
`latest_safe`, and `last`.

The first manual run should cover C0 through C4 with automatic promotion, seed
22001 for training, the committed development/selection ranges for tuning,
and the reserved final ranges untouched. Completing the engineering pipeline
is not itself an RL success claim; that claim requires the later held-out
long-run evidence.
