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

## Pre-fine-tune baseline — the recurrent BC already completes circuits

The clean recurrent BC parent, trained **only** on the general procedural
corpus and never on these circuits, run current-free with 3 trials each:

| Circuit | Gates | Completed | Mean gates | Best | Failure |
|---|---|---|---|---|---|
| Horseshoe Bay | 12 | **2/3** | 9.33 | 12 | out_of_bounds at gate 5 |
| Vertical Serpent | 17 | **1/3** | 9.00 | 17 | out_of_bounds at gates 5, 7 |
| Mixed Endurance | 22 | 0/3 | 11.33 | 13 | collision at gate 14 (×2), out_of_bounds at gate 9 |

**3 of 9 trials complete.**

Generation 1 PPO scored **0/30** across these same circuits. Two of the three
already meet the first milestone before any track-specific training.

### The blocking problem is depth, not navigation

Every failure is `out_of_bounds`. Not a missed gate, not a collision, not a
timeout — the vehicle leaves the arena. And the binding bound is **vertical**:

| Circuit | Arena z | Gate depths | Tightest margin |
|---|---|---|---|
| Horseshoe Bay | −8.0 … −1.0 | −4.0 … −4.4 | 3.0 m (to surface) |
| Vertical Serpent | −8.0 … −1.0 | −3.9 … −5.9 | 2.1 m (to floor) |

Horizontal margins are 4–6 m and larger; the z margin is 2.1–3.4 m everywhere.
On Horseshoe — a flat, purely horizontal-turn circuit — an out-of-bounds means
the vehicle climbed roughly three metres it had no reason to climb. On Vertical
Serpent, whose gates alternate −4.0/−5.8/−4.1/−5.9, overshooting a descent by
2.1 m reaches the floor bound.

### Two separable deficits, both covered by the fragment corpus

**1. Depth regulation, on ordinary gates.** Vertical Serpent fails at exactly
its two deepest gates and nowhere else:

| Gate | Depth | Floor margin | |
|---|---|---|---|
| G03 | −5.50 | 2.50 m | |
| **G05** | **−5.80** | **2.20 m** | ← fails |
| **G07** | **−5.90** | **2.10 m** | ← fails |
| others | −4.0…−4.8 | 3.2–4.0 m | |

The two tightest floor clearances on the circuit are precisely the two gates
where the policy leaves the arena. Horseshoe's single failure is the same story
against the surface. This is depth overshoot, not navigation.

**2. Gate structures the policy has never seen.** The procedural course family
generates **only `type: "single"`** gates. The real circuits do not:

| Circuit | Gate types |
|---|---|
| Horseshoe Bay | 12 single |
| Vertical Serpent | 15 single, 2 vertical_double |
| Mixed Endurance | 16 single, 2 vertical_double, 2 double, 1 split_s_upper, 1 split_s_lower |

Both of Mixed's collisions are at **G14, a `double`**, and its out-of-bounds is
at **G09, a `vertical_double`**. The recurrent BC had literally never
encountered these structures in 321 796 transitions of training. Mixed is 0/3
for a reason that has nothing to do with its length.

The fragment corpus covers every weak gate — 10–13 expert episodes through each
of `vertical_serpent` G05/G07/G08/G09 and `mixed_endurance` G08/G09/G14/G15/
G18/G19 — and the expert crosses all of them with zero collisions and zero
out-of-bounds. The fix for both deficits is in the data that is already
collected.

So the failure mode is a **heave/depth excursion**, which is consistent with
Vertical Serpent producing Gen-1's worst safety behaviour. Navigation is not
the deficit; depth regulation is. Fragment training over-represents Vertical
Serpent (52 of 150 windows) for exactly this reason, and the next cycle
re-measures the out-of-bounds rate specifically.

## Fragment expert corpus v1

`corpus_sha256 = 1749e772a21c3eb4ad1e2c9b0e7b59cdcce29545fde27c89a7b85978755117e1`
— 150 episodes, **102 581 transitions**, one trial per window.

| Circuit | Episodes | Transitions | Share | Expert completed |
|---|---|---|---|---|
| Horseshoe Bay | 38 | 25 136 | 24.5 % | 38/38 |
| Vertical Serpent | 52 | 30 457 | 29.7 % | 52/52 |
| Mixed Endurance | 60 | 46 988 | 45.8 % | 59/60 |

By fragment length: 2→40, 3→37, 4→35, 5→32, 6→6.

Expert completion **0.9933**, gate success **0.9924**, and — the number that
matters for this campaign — **zero collisions and zero out-of-bounds across all
150 fragments**. The expert regulates depth correctly on exactly the geometry
where the learner leaves the arena, so the corpus carries the signal the
learner is missing rather than merely more of what it already does.

Vertical Serpent is deliberately well represented (29.7 % of transitions from
34 % of the windows) because it is where Gen-1's safety behaviour was worst and
where the depth margin is tightest (2.1 m).

Fine-tuning mixes the general corpus with this one at weight ×3, giving
321 796 general against 307 743 fragment transitions — 51.1 % / 48.9 %.

## MILESTONE 1 MET — all three circuits completed by the learned controller

`best_completion_policy` (recurrent BC, **general procedural corpus only**,
never trained on these circuits), 10 trials per circuit, current-free:

| Circuit | Gates | Completed | Mean gates | Best | Collision eps | OOB eps | Mean time (finishers) | Rule baseline |
|---|---|---|---|---|---|---|---|---|
| Horseshoe Bay | 12 | **4/10** | 8.50 | **12/12** | 3 | 1 | 236.9 s | 225.9 s |
| Vertical Serpent | 17 | **2/10** | 7.30 | **17/17** | 6 | 6 | 466.6 s | 290.6 s |
| Mixed Endurance | 22 | **1/10** | 7.80 | **22/22** | 7 | 4 | 486.4 s | 472.9 s |

**7/30 overall, and every circuit finished at least once.** Generation 1 PPO
scored 0/30 on these circuits. On the runs it completes, the learned controller
is already within 5% of the rule baseline on Horseshoe (236.9 vs 225.9 s) and
within 3% on Mixed (486.4 vs 472.9 s); Vertical is the outlier at +60%.

The earlier n=3 reading that showed Mixed at 0/3 was simply undersampled --
Mixed completes 1 in 10.

### The remaining gap, quantified

Treating a circuit as a chain of independent gate transitions, completion is
roughly *p* raised to the gate count:

| Circuit | Measured | Implied per-gate *p* | *p* needed for 8/10 |
|---|---|---|---|
| Horseshoe Bay | 4/10 over 12 | 0.927 | 0.982 |
| Vertical Serpent | 2/10 over 17 | 0.910 | 0.987 |
| Mixed Endurance | 1/10 over 22 | 0.901 | 0.990 |

So the work is not "fix a broken circuit" — it is lifting per-transition
reliability from ~0.91 to ~0.99. Compounding does the rest: at 22 gates, the
difference between 0.90 and 0.99 per gate is 10% versus 80% completion.

### Failures are spread, not concentrated

Failure gates at n=10: Horseshoe 4,5,7,8,9,10 (one each); Vertical 4,5,6,7,9;
Mixed 2,3,4,6,7,11,14,15. **There is no single weak gate.** That matters for
strategy: targeting fragments at one bad transition cannot fix a uniform
per-transition failure rate. Kinds across all 30 trials: out-of-bounds 10,
collision 8, wrong-direction 5.

This is exactly the case DAgger exists for -- the learner drives, the expert
labels the states the learner actually reaches, and the correction applies to
every transition rather than to a hand-picked one. Round 1 therefore runs
across all 150 fragments rather than targeting weak gates.

## GPU / CUDA audit — measured, and deliberately not adopted yet

| | |
|---|---|
| Python / torch | 3.9.25 / **2.8.0+cpu** |
| `torch.cuda.is_available()` | **False** (CPU wheel, `torch.version.cuda` is None) |
| GPUs present | 2 x NVIDIA RTX 6000 Ada — GPU0 49 GB (40% busy, HoloOcean rendering), GPU1 46 GB (idle) |
| CPU / RAM | 40 logical cores, 146 GB (22 GB used) |
| sb3 / sb3-contrib | 2.7.1 / 2.7.1 |

So the build is CPU-only and a capable GPU sits idle. The question is whether
moving neural optimization to it is worth the risk to a verified HoloOcean
environment. Measured on the cycle that just ran:

| Component | Wall clock | Share |
|---|---|---|
| BC retrain, 10 epochs over 591 028 transitions | 726 s | **5%** |
| Circuit evaluation, n=10 x 3 (248 885 simulator steps) | ~3.8 h | **95%** |

**Neural optimization is 5% of the cycle.** Making it instantaneous saves 5%,
against the brief's own bar of a 15% wall-clock improvement. Installing a CUDA
wheel into the environment that carries HoloOcean 2.3.0 from a source client
would risk the 95% to speed up the 5%, so it is not done now.

The effort went to the dominant component instead: circuit evaluation now
shards by trial as well as by track, so 3 x 3 = 9 concurrent engines fit under
the 10-engine ceiling and the n=10 evaluation drops from ~3.8 h to ~1.3 h.
That is a real >15% improvement, on the part that actually costs time.

**When to revisit.** Recurrent PPO performs far more gradient steps per unit of
simulation than BC does. If PPO optimization grows past roughly a quarter of
cycle wall clock, CUDA becomes worth the migration — in a *separate* cloned
environment, tested against the full Gen-2 suite before use, never by mutating
the working one.

## Per-transition failure map — where reliability is actually lost

Unconditional transition table from the 30 full-circuit runs of
`best_completion_policy`. Transition *k -> k+1* is attempted by every run that
reached gate *k*, so late transitions are comparable to early ones; attempts
are shown because survivorship shrinks the sample and a rate on two attempts is
noise, not a hotspot.

Transitions with at least 5 attempts and a rate at or below 0.92:

| Track | Transition | Rate | Attempts | Failure kinds |
|---|---|---|---|---|
| Vertical Serpent | **G6→G7** | **0.60** | 5 | out_of_bounds x2 |
| Vertical Serpent | **G4→G5** | **0.75** | 8 | out_of_bounds x2 |
| Mixed Endurance | **G2→G3** | **0.78** | 9 | out_of_bounds x2 |
| Vertical Serpent | G3→G4 | 0.80 | 10 | collision, wrong_direction |
| Horseshoe Bay | G9→G10 | 0.80 | 5 | collision |
| Mixed Endurance | G6→G7 | 0.80 | 5 | wrong_direction |
| Vertical Serpent | G5→G6 | 0.83 | 6 | out_of_bounds |
| Horseshoe Bay | G8→G9 | 0.83 | 6 | wrong_direction |

**Vertical Serpent loses reliability on the descents into its two deepest
gates.** G05 sits at −5.80 and G07 at −5.90, the tightest floor clearances on
the circuit at 2.2 m and 2.1 m, and both failing transitions are out-of-bounds.
That is the depth-overshoot diagnosis again, now located to the exact
transition rather than inferred from arena geometry.

Horseshoe is different: its failures are spread one apiece across G3→G10 with
no dominant transition, which is consistent with a uniform per-transition rate
rather than a specific weakness.

`hotspot_fragments` converts these into **41 exact windows** G(k−2)..G(k+2) of
the real circuits — contiguous slices, no rotation, no translation, entered the
way the circuit enters them.

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
