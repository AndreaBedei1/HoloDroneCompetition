# `track_specific_gen2_v1` — completing the three official circuits

Branch: `feature/rl-gen2-track-specific` (from `feature/rl-gen2-recurrent-dagger`).

**This controller is trained using fragments of the three evaluation tracks and
therefore does NOT claim unseen-track generalization.** That is a deliberate,
recorded property of the protocol, not an oversight. The generalization result
lives in the separate procedural experiment and its five sealed holdouts, which
are untouched.

## Objective, in order

1. Make the learned recurrent controller **complete** Horseshoe Bay, Vertical
   Serpent and Mixed Endurance.
2. Once completion is reliable, **reduce completion time** without giving up
   reliability or safety.

Milestones: at least one complete run on each circuit → ≥ 8/10 on each →
ideally 10/10 on each. Only then does speed optimization begin.

## Why the old circuits may be used

The Gen-1 exploratory holdout ran 30 paired trials across all three and logged
60 ledger entries. They are open. Using them for development is legitimate; the
only thing that would be illegitimate is claiming they were unseen.

## What a fragment is

A contiguous window of the real gate sequence, **copied byte-for-byte**:

* gate positions, rotations, passage directions, types and links unchanged;
* no rotation, no translation, no perturbation, no resampling;
* world bounds inherited, so an out-of-bounds on a fragment means exactly what
  it means on the full circuit.

150 windows of length 2–5 across the three circuits (38 / 52 / 60), every one
verified identical to source.

Two rules make a window legal.

**Linked pairs are never split.** Vertical Serpent links G08–G09; Mixed
Endurance links G08–G09, G14–G15 and G18–G19. Cutting between a pair leaves a
dangling reference and changes what the gate *is*, so the window grows to
contain both. 28 of the 150 grew, which is why lengths 6 appear without being
requested.

**The inbound state is reconstructed, not invented.** A fragment starting at
gate *k > 1* places the vehicle 1.5 m past gate *k-1*, on that gate's centre
depth, heading along its passage direction — the pose it actually holds the
instant it finishes crossing there. Spawning it in front of gate *k* instead
would delete the very transition the fragment exists to teach. A prefix
fragment keeps the circuit's own start pose.

Verified on HoloOcean with the expert: `horseshoe_bay_G01-G03` 3/3,
`vertical_serpent_G05-G07` 3/3 (internal, reconstructed inbound),
`vertical_serpent_G07-G09` 3/3 (linked pair).

## A trap worth recording

Mixed Endurance declares `benchmark_task: current_gate` and carries five
currents. The official 0/30-vs-30/30 comparison was run current-free, so
fragments drop the currents — and **must drop that task with them**, or the
loader rejects the config with *"benchmark_task current_gate requires at least
one marine current"*. The Gen-1 sequence generator documents the same trap in a
comment; a fragment smoke test walked straight into it anyway. Both now move
together.

## Seeds

The campaign has its own band so it can never be confused with the procedural
bookkeeping:

| Role | Range |
|---|---|
| fragment expert demonstrations | 60 000 – 63 999 |
| fragment DAgger rollouts | 64 000 – 65 999 |
| fragment evaluation | 66 000 – 66 999 |
| full-circuit trials | 67 000 – 67 999 |
| recurrent PPO speed tuning | 68 000 – 69 499 |
| smoke | 69 500 – 69 999 |

`TRACK_SPECIFIC` is a trainable band alongside `TRAIN`. The `VALIDATION`,
`TEST` and `FINAL_HOLDOUT` guards are unchanged, and the sealed holdout
remains default-deny.

## Starting policy

The clean matched ablation picked the parent. On 60 identical courses:

| Arm | Completion | gate 3 survival |
|---|---|---|
| Gen-1 feed-forward PPO | 0.417 | 0.289 |
| feed-forward BC (capacity-matched) | 0.017 | 0.022 |
| **recurrent BC** | **0.750** | **0.911** |
| recurrent BC + generic DAgger r1 | 0.717 | 0.778 |

Generic DAgger round 1 did **not** improve the matched result, so it is not
continued unchanged. The parent is `results/rl/gen2/bc_v1/bc_policy.zip`.

## Cycle

Short iterations, each answering one question:

```
fragment corpus -> BC fine-tune -> fragment matrix -> full circuits
      ^                                                    |
      +------------------ failure map ---------------------+
```

The failure map ranks the exact gates where circuits die. `targeted_fragments`
turns those into real windows containing the failing transition with run-up.
No geometry is invented at any point.

## Model selection is lexicographic

1. completion rate
2. safety
3. gates completed
4. completion time
5. path length
6. jerk

10/10 at 230 s beats 8/10 at 190 s. A faster DNF is worthless.

## Speed targets (rule baseline)

| Circuit | Rule controller |
|---|---|
| Horseshoe Bay | ~225.9 s |
| Vertical Serpent | ~290.6 s |
| Mixed Endurance | ~472.9 s |

Recurrent PPO is reserved for this phase, initialized from the reliable
imitation policy. Completion must dominate the reward; speed must never be able
to buy a missed gate. `best_completion_policy` is preserved permanently and
never overwritten by `best_speed_policy`.

## Preserved, untouched

Gen-1 PPO 0/30, the procedural readiness verdict, the recurrent BC matched
ablation, generic DAgger r1, the five sealed Gen-2 holdouts, expert corpus v1
(`04258c69…`, 400 episodes / 321 796 transitions), and every earlier report.
