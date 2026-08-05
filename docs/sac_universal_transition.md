# Independent multi-step SAC universal-transition arm

This branch adds a SAC arm that shares only the task contract and the initial
deterministic actor behaviour with the active PPO experiment. It does not share
rollouts, replay, critics, optimizers, losses, entropy state, or actions.

## Contract and architecture

- observation: `onboard_local_transition_v1`, exactly 35 legal onboard values;
- action: `surge_sway_heave_yaw_pm1_v1`, all four axes emitted by SAC;
- unchanged sampler, tracker, reward, termination, curriculum, competence gate,
  unseen seeds, and evaluation aggregation;
- stochastic tanh-squashed Gaussian actor with two 256-unit tanh layers;
- PPO policy-mean warm transfer only; all critics, targets, optimizers, entropy,
  counters, n-step queues, and replay start independently;
- twin Q critics and Polyak targets (`tau=0.005`);
- three-step targets, `gamma=0.995`, with the actual legal bootstrap discount
  persisted per replay entry;
- automatic entropy tuning with target entropy `-4`;
- one-million-entry stratified replay: 35% general, 30% crossing/target-switch,
  20% successful transitions, 15% safety failures/timeouts. Event categories
  are training-only metadata and never actor observations.

The competent actor source SHA-256 is
`74b1f83c6f9c740dcc307b271ff1393dd86de9e210e3a8788036f7c91a708747`.
The source ZIP and initialized SAC actor are copied once into an immutable
per-run baseline.

Every SAC atomic checkpoint commits actor, critics, targets, all optimizers,
entropy, replay and event indices, n-step queues, counters, Python/NumPy/Torch
RNG, worker samplers, evaluations, aliases, initialization provenance, and the
configuration hash. The hash manifest is replaced last.

## Operational commands

Run PPO commands from
`C:\Users\andrea.bedei3\Desktop\HoloDroneCompetition` and SAC commands from
`C:\Users\andrea.bedei3\Desktop\HoloDroneCompetition-sac`.

PPO status:

```bat
conda run -n marine_race_rl --no-capture-output python -m marine_race_arena.learning.train_ppo_transition status C:\Users\andrea.bedei3\Desktop\HoloDroneCompetition\results\rl\universal_transition\longrun\universal_transition_warm_reliability_seed23001
```

SAC status:

```bat
conda run -n marine_race_rl --no-capture-output python -m marine_race_arena.learning.train_sac_transition status C:\Users\andrea.bedei3\Desktop\HoloDroneCompetition-sac\results\rl\universal_transition\sac\universal_transition_multistep_sac_seed23001
```

PPO safe stop:

```bat
conda run -n marine_race_rl --no-capture-output python -m marine_race_arena.learning.train_ppo_transition stop C:\Users\andrea.bedei3\Desktop\HoloDroneCompetition\results\rl\universal_transition\longrun\universal_transition_warm_reliability_seed23001
```

SAC safe stop:

```bat
conda run -n marine_race_rl --no-capture-output python -m marine_race_arena.learning.train_sac_transition stop C:\Users\andrea.bedei3\Desktop\HoloDroneCompetition-sac\results\rl\universal_transition\sac\universal_transition_multistep_sac_seed23001
```

Fast looping geometry preview (no HoloOcean):

```bat
conda run -n marine_race_rl --no-capture-output python -m marine_race_arena.learning.preview_transition_tracks sampler --config configs\rl\sac_universal_transition_warm_long.json --backend geometry --difficulty all --episode-type all --lengths 3,5,8,12,17,22 --seed-start 63001 --num-tracks 100 --speed 10 --show-gate-frames --show-camera-frustum --loop
```

Exact active PPO tracks:

```bat
conda run -n marine_race_rl --no-capture-output python -m marine_race_arena.learning.preview_transition_tracks active --run C:\Users\andrea.bedei3\Desktop\HoloDroneCompetition\results\rl\universal_transition\longrun\universal_transition_warm_reliability_seed23001 --workers all --backend geometry --speed 5
```

Exact active SAC tracks:

```bat
conda run -n marine_race_rl --no-capture-output python -m marine_race_arena.learning.preview_transition_tracks active --run C:\Users\andrea.bedei3\Desktop\HoloDroneCompetition-sac\results\rl\universal_transition\sac\universal_transition_multistep_sac_seed23001 --workers all --backend geometry --speed 5
```

Rendered PPO long-sequence evaluation:

```bat
conda run -n marine_race_rl --no-capture-output python -m marine_race_arena.learning.preview_transition_tracks evaluate --algorithm ppo --run C:\Users\andrea.bedei3\Desktop\HoloDroneCompetition\results\rl\universal_transition\longrun\universal_transition_warm_reliability_seed23001 --checkpoint best_universal_transition --lengths 3,5,8,12,17,22 --episodes-per-length 2 --backend holoocean --camera chase --live --record --output-dir results\rl\universal_transition\visual_evaluations\ppo
```

Rendered SAC long-sequence evaluation:

```bat
conda run -n marine_race_rl --no-capture-output python -m marine_race_arena.learning.preview_transition_tracks evaluate --algorithm sac --run C:\Users\andrea.bedei3\Desktop\HoloDroneCompetition-sac\results\rl\universal_transition\sac\universal_transition_multistep_sac_seed23001 --checkpoint best_universal_transition --lengths 3,5,8,12,17,22 --episodes-per-length 2 --backend holoocean --camera chase --live --record --output-dir results\rl\universal_transition\visual_evaluations\sac
```

Rendering only observes camera frames; the normal deterministic policy action,
tracker, reward, referee, and unseen sampler determine the result.

Equal-budget comparison:

```bat
conda run -n marine_race_rl --no-capture-output python -m marine_race_arena.learning.compare_ppo_sac_transition --ppo-run C:\Users\andrea.bedei3\Desktop\HoloDroneCompetition\results\rl\universal_transition\longrun\universal_transition_warm_reliability_seed23001 --sac-run C:\Users\andrea.bedei3\Desktop\HoloDroneCompetition-sac\results\rl\universal_transition\sac\universal_transition_multistep_sac_seed23001
```

Incremental capacity benchmark:

```bat
conda run -n marine_race_rl --no-capture-output python -m marine_race_arena.learning.benchmark_sac_parallelism --config configs\rl\sac_universal_transition_warm_long.json --output-dir results\rl\universal_transition\sac\capacity_benchmark --ppo-run C:\Users\andrea.bedei3\Desktop\HoloDroneCompetition\results\rl\universal_transition\longrun\universal_transition_warm_reliability_seed23001 --workers 1,2,4,6,8 --transitions 128
```

The capacity guard counts all live `Holodeck.exe` processes plus atomic launch
reservations and refuses any launch beyond ten total engines. SAC closes its
rollout engines before evaluators and cleanup remains scoped to worker-owned
UUIDs.

PPO and SAC may only be ranked after unseen evaluations at equal new-environment
budgets. Neither is reliable until repeated unseen transition success reaches
at least 99% with strong safety and long-sequence chaining.

