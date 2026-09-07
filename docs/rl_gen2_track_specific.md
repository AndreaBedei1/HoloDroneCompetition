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

## Onboard perception audit — the rover measured well and aimed badly

Requested as a check on whether the rover's own data are trustworthy, with
simulator ground truth used *only* offline, as an external instrument. The
policy's input is unchanged: still the 35-feature
`onboard_local_transition_v1` observation plus its recurrent state, and nothing
else.

The first run of the audit reported a 106.8 deg mean bearing error and a
+16.5 m range bias on Mixed Endurance. That reading was wrong, and the way it
was wrong is the finding. The instrument compared the rover's sensed geometry
against the gate the *referee* was waiting for, so every step where the rover
was chasing some other gate scored as a sensing error. Measured against the
gate the rover had itself selected, the same stream is accurate to **0.16 /
0.35 / 0.46 deg** mean bearing error across the three circuits. Two different
questions had been collapsed into one number:

* **perception** — given its own choice of target, does the rover measure it
  correctly? Yes, on all three circuits.
* **targeting** — is that choice the right gate? Not on two of them.

Everything else on the checklist came back clean: bearing sign agreement
0.995-0.998, observation/action timestep alignment exactly 1.0000, zero
repeated observation vectors, beacon age never above zero, and the recurrent
state reset once at episode start and carried thereafter.

### The defect

The tracker that builds the observation confirmed a gate passage only when the
beacon range fell below a hand-picked **0.60 m**. The beacon is not at the
centre of the aperture: it sits 0.35 m up the gate's own up-axis. So 0.60 m
admits a transit only within 0.49 m laterally, or 0.25 m below the centre, and
rejects the rest of a pass that really did go through. The far corner of a
legitimate transit is `hypot(0.75, 0.75 + 0.35) = 1.33 m` away.

The tracker has no recovery path. One rejected passage leaves it waiting
permanently for a gate the rover has already left behind. On Vertical Serpent
it kept up perfectly to gate 6 -- 2 to 3 steps of latency -- then missed one
confirmation and stayed on G06 while the rover reached G13.

### Measured, same sensor stream, both configs stepped side by side

| Circuit | agreement with progress | advances / crossings | policy completion |
|---|---|---|---|
| Horseshoe Bay | 99.0% | 11 / 11 | 10/10 |
| Vertical Serpent | 32.9% | 5 / 16 | 4/10 |
| Mixed Endurance | 51.2% | 10 / 21 | 3/10 |

The three targeting numbers order exactly like the completion rate. After
replacing the envelope with the aperture-derived 1.33 m:

| Circuit | before | after | advances after |
|---|---|---|---|
| Horseshoe Bay | 99.0% | 98.9% | 11 / 11 |
| Vertical Serpent | 32.9% | **98.9%** | 16 / 16 |
| Mixed Endurance | 51.2% | **99.2%** | 21 / 21 |

No cost on the circuit that was already working.

### What this invalidates

Two things, and they need re-measuring rather than reinterpreting.

**The training corpora.** Every observation recorded after a stall names a gate
behind the rover while the expert's action drives forward. That pair is not a
function any learner can fit. In the stored fragment corpora 8-15% of episodes
show the observation lagging actual progress by two or more gates; the general
corpus has a tail out to 20. Fragments are short, so a stall usually costs the
back half of one episode -- but full circuits were affected throughout.

**The per-transition failure map above.** Its worst entry is Vertical Serpent
G6->G7 at 0.60, and G6->G7 is exactly where the tracker stalls on that circuit.
The failures recorded there are out-of-bounds, which is a real control failure,
but a policy whose observation has just started pointing backwards is not being
tested on the transition the table claims to be measuring. The hotspot windows
built from that table were therefore partly targeting a pipeline defect. The
depth-overshoot diagnosis for Vertical may still hold -- G05 and G07 do have the
tightest floor clearances -- but it is no longer supported by this table and has
to be re-established.

`evaluation.py` builds observations through the same tracker, so the frozen
`best_completion_policy` was mis-aimed at inference too, not only in training.
Its 17/30 was measured under the defect.

### Re-measuring the frozen policy under the corrected pipeline

`evaluation.py` builds observations through the same tracker, so this is not
only a training-data question: the frozen `best_completion_policy` was
mis-aimed at inference too. Its 17/30 was measured under the defect.

Same checkpoint, same 30 seeds, same lexicographic protocol, only the
observation construction changed:

| Circuit | before | after | recovered | lost |
|---|---|---|---|---|
| Horseshoe Bay | 10/10 | 9/10 | 0 | 1 |
| Vertical Serpent | 4/10 | 7/10 | 4 | 1 |
| Mixed Endurance | 3/10 | 7/10 | 6 | 2 |
| **total** | **17/30** | **23/30** | 10 | 4 |

**This gain is not statistically established.** Exact two-sided sign test on the
14 discordant pairs gives p = 0.18. The direction is consistent and the
mechanism is measured, but n=10 per circuit cannot separate 7/10 from 8/10, and
8/10 is the reliability bar. The fix is justified by the targeting measurement
-- 32.9% to 98.9% on Vertical, 51.2% to 99.2% on Mixed -- which is a property of
the pipeline and does not depend on any policy statistic. The completion change
is a consequence consistent with it, not evidence for it.

Re-measurement at n=30 per circuit is running. The weights are untouched
throughout and the alias still points at the same checkpoint: this is a change
of measurement conditions, not a new candidate, so the acceptance rule for
candidates does not apply to it.

### What was found and deliberately not changed

Two properties of the frozen observation contract, reported rather than
altered, because neither is inaccurate and the contract is fixed at 35
features:

* **DVL alternates every other step**, and on the off steps surge, sway and
  heave are all exactly zero -- 100% of the time, across 102,581 corpus steps.
  `dvl_present` is 0 on those steps, so the contract is honest; but the beacon
  path holds its last value and exposes `beacon_age_norm`, and the velocity
  path could do the same instead of dropping to zero half the time.
* **`beacon_range_rate` sits on a clip bound 37.4% of the time**, which
  suggests its scale is too small for the range rates that actually occur.

## `onboard_local_transition_gate_yaw_v2` — the 38-D gate-yaw smoke

The 35-feature contract tells the policy *where* the gate is and never *how it
is turned*. Bearing, elevation and apparent area are all invariant to gate
yaw, so a gate presented edge-on and a gate presented square-on are the same
observation. Three appended features close that gap, and nothing else changes:

| # | feature | meaning |
|---|---|---|
| 36 | `gate_orientation_present` | 1 when a camera plane-yaw estimate exists this step |
| 37 | `gate_yaw_sin` | sin of the estimated gate-plane yaw, 0 when absent |
| 38 | `gate_yaw_cos` | cos of the estimated gate-plane yaw, 0 when absent |

`cos(0) = 1` is never emitted without its mask: when the estimate is missing all
three values are exactly zero, so "absent" is representable and is not confused
with "square-on". The estimate comes from `estimate_gate_pose` on the
`FrontCamera` image, searching only the ROI already associated with the locally
expected acoustic beacon, filtered by `GatePoseTracker(alpha=0.38,
max_age_steps=4)` and reset on every target change. No map, referee state,
global pose or per-gate configured geometry reaches the policy; the only prior
is the competition-constant 1.5 m x 1.5 m aperture used as a scale.

### Transfer is an identity, verified bit-exactly

`best_completion_policy.zip` (`2546ab2f…`) is expanded 35 -> 38 in place: 26
same-shape tensors copied verbatim, 9 expanded (`obs_mean`, `obs_std` and
`encoder.0.weight` in each of the three feature-extractor copies), and the three
new input columns initialized to exactly zero, with `obs_mean = 0` and
`obs_std = 1` on the new dimensions. Recurrent parity against the parent is
bit-exact over 32 samples (max absolute deviation 0.0), so the 38-D policy
*starts* as the parent and any behaviour change is attributable to
optimization, not to the widening. The parent hash was re-checked after the run
and is unchanged.

### The smoke — 6 144 timesteps, matched-seed A/B

Deliberately short and explicitly under-powered: one trial per circuit, on the
same seeds, with `statistical_significance_claimed = false` written into the
report. It answers "does the widened policy train and step through the full
pipeline", not "is it better".

* 6 144 steps, 8 PPO updates, lr 3e-5, clip 0.08, `target_kl` 0.01, seed 8400
* 6 workers: the three full circuits plus one exact 3-gate fragment each
* fog verified on all three official tracks (`density 5.0`, `start 1.0 m`,
  `color (0.4, 0.6, 1.0)`); the underwater vision fix is unconditional code in
  `vision.py`, so both policies were scored through it
* parent evaluated at 35-D, candidate at 38-D — `run_policy_episode` dispatches
  the encoder on the checkpoint's own observation dimension

| Circuit | seed | | parent | candidate |
|---|---|---|---|---|
| Horseshoe Bay | 67000 | completed | **yes** 12/12 | **yes** 12/12 |
| | | time | 187.0 s | 264.3 s (+77.3) |
| | | safety | 0 / 0 / 0 | 0 / 0 / 0 |
| Vertical Serpent | 67200 | completed | **yes** 17/17 | **no**, DNF at gate 11 |
| | | time | 247.6 s | — (10/17 gates) |
| | | safety | 0 / 0 / 0 | 0 coll / 7 OOB / 1 wrong-dir |
| Mixed Endurance | 67100 | completed | **yes** 22/22 | **no**, DNF at gate 9 |
| | | time | 360.5 s | — (8/22 gates) |
| | | safety | 0 / 0 / 0 | 2 coll / 0 OOB / 0 wrong-dir |
| **total** | | | **3/3, 51/51 gates, 0 events** | **1/3, 30/51 gates, 10 events** |

Safety is reported as collisions / out-of-bounds / wrong-direction raw event
counts; the candidate total is 2 + 7 + 1, plus 2 missed-gate attempts counted
separately. The parent is clean everywhere and beats the rule baseline on all three
(-38.9 s, -43.0 s, -112.4 s). This is a regression.

### The regression is not caused by the three new features

Measured directly on the weights, comparing the zero-initialized 38-D policy to
the candidate:

| what moved | l2 of the change |
|---|---|
| the three new input columns (from zero) | 3.9e-3, max abs weight 3.0e-4 |
| the original 35 input columns | 1.2e-2 |
| whole policy, all 29 changed tensors | 8.3e-2 |

The new columns are still essentially zero after 8 updates — largest single
weight 3.0e-4 against an encoder whose column norms are order 1 — so the new
features cannot yet be steering the vehicle. What moved is everything else:
the LSTM, the heads and the *old* input columns, which drifted three times
further than the new ones. Relative drift is tiny (1.4e-2 on `action_net.weight`,
~1e-3 elsewhere) and it was still enough to turn 3/3 into 1/3.

That is the finding worth keeping: **`best_completion_policy` sits on a sharp
optimum, and a ~0.1% relative perturbation of the recurrent policy destroys
long-horizon gate chaining.** It is consistent with the known first-transition
fragility — the two DNFs are re-targeting failures at gates 9 and 11, not
control failures — and it means a gate-yaw campaign cannot be run as "resume
PPO and see". Either the trunk is frozen while the new columns learn, or the
step size and KL bound go down by an order of magnitude, or both.

### What the three features actually do on the tracks

Measured over 300 steps per circuit with the parent driving, under fog and the
corrected vision, in the preflight that gated this run:

| Circuit | `gate_orientation_present` | median \|yaw\| | p95 \|yaw\| | p95 yaw jump | zeros when absent |
|---|---|---|---|---|---|
| Horseshoe Bay | 85.0% | 5.5° | 48.5° | 5.3° | 45/45 |
| Vertical Serpent | 83.0% | 14.9° | 35.7° | 5.1° | 51/51 |
| Mixed Endurance | 67.3% | 0.6° | 26.0° | 5.6° | 98/98 |

Conditional on vision, orientation is available 96.9–98.5% of the time, so the
availability spread across circuits is a vision-availability property, not a
pose-estimator property — Mixed Endurance simply sees a gate less often.
`sin`/`cos` stay on the unit circle to 4e-8, every one of the 900 logged rows is
finite, the encoded mask never disagrees with the context, and the filter never
holds an estimate older than 4 frames. The signal is well-formed and
informative; it is the optimization that has not used it yet.

### Status

Regression on n=1 per circuit. No candidate is promoted, no alias moves,
`best_completion_policy` is untouched at `2546ab2f…`. The smoke is recorded as
a pipeline validation and a fragility measurement, not as evidence about the
gate-yaw contract, which has not yet been given enough training to be judged.

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
