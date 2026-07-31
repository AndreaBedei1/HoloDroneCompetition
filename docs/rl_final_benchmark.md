# Final common benchmark

One identical test suite for every deployable controller, used to select the
final recommended checkpoint. Nothing here is taken from the training metadata:
`best_reliable` is treated as just another candidate.

## Reproduce with one command

```bat
scripts\run_final_benchmark.bat
```

or, explicitly:

```bash
conda run -n marine_race_rl python -m marine_race_arena.learning.final_benchmark run \
  --out results/rl/final_benchmark/reliability_first_20260731 --workers 6 \
  --episodes-per-case 8 --official-episodes 5 \
  --adapter holoocean --current-profile none --video --video-stride 5
conda run -n marine_race_rl python -m marine_race_arena.learning.final_benchmark_report \
  --out results/rl/final_benchmark/reliability_first_20260731
```

The run is resumable: completed episodes are keyed by
`controller|group|case|seed` and skipped on re-invocation, so an interrupted
benchmark continues instead of restarting. `--workers N` runs N independent
real-HoloOcean instances; each `holoocean.make` call takes its own UUID, so the
instances do not share memory or state.

## What is held identical

Every controller sees the same test groups, the same generated and official
geometries, the same seeds, the same adapter (`holoocean`, no fallback), the
same `dt = 0.1 s`, and the same current-free configuration. Generated
transition tracks are written once into `generated_tracks/` and reused
byte-for-byte by every controller. Official circuits are unmodified: currents
are disabled through the documented runtime override only, and Mixed Endurance
additionally takes the `clean_gate` benchmark-task override so a `current_gate`
circuit can run current-free. Gate order, gate geometry, laps and the referee
are untouched.

## Controllers

| key | kind | policy-only | model |
|---|---|---|---|
| `ppo_525678` | PPO | yes | training-metadata `best_reliable` / `best_fast_reliable` |
| `ppo_900462` | PPO | yes | latest safe checkpoint recorded during stage C4 |
| `ppo_1000814` | PPO | yes | final checkpoint after the rollback to C3 |
| `bc_v3` | BC | yes | `bc_v3_balanced_v2_20260728/bc_v3.pt`, the retention reference |
| `rule_gate_center_then_commit` | rule | n/a | deterministic baseline |
| `hybrid` | hybrid | **no** | rule backbone + BC-v1 visual servo |

The hybrid controller is run on the identical suite but **reported separately
and excluded from checkpoint selection**, because it is not fully policy-based.

For the four policy-only controllers every runtime command comes from the
learned policy. `_assert_policy_only` fails the episode if the controller
declares any rule weight, hybrid blending or rule-controller instance, or if it
reports a non-zero runtime-intervention count; the check's evidence is stored in
each episode row (`policy_evidence`). There is no runtime rule action, expert
correction, fallback control or blending anywhere in the learned path.

## Test groups

| group | geometry | gates |
|---|---|---|
| `single_gate_retention` | `stage1_single_gate` | 1 |
| `two_gate_straight` | `two_gate_straight` | 2 |
| `two_gate_left` | fixed 36.9 deg + generated 20 deg | 2 |
| `two_gate_right` | fixed -36.9 deg + generated -20 deg | 2 |
| `vertical_low_to_high` | generated +0.7 m straight, +1.2 m with 15 deg left | 2 |
| `vertical_high_to_low` | generated -0.7 m straight, -1.2 m with 15 deg right | 2 |
| `three_gate_sequence` | `stage3_three_gates` | 3 |
| `three_gate_s_shape` | `three_gate_s_curve` | 3 |
| `official_horseshoe_bay` | official circuit, current-free | 12 |
| `official_vertical_serpent` | official circuit, current-free | 17 |
| `official_mixed_endurance` | official circuit, current-free, `clean_gate` | 22 |

The vertical groups move the second gate in depth: positive displacement raises
it (low-to-high), negative lowers it (high-to-low), verified by a test that
reads the generated track back.

## Seeds

Half of each case's seeds are **reused** from the already-allocated final
evaluation ranges, so results stay comparable with earlier official runs; half
come from `MULTIGATE_FINAL_BENCHMARK_HOLDOUT_SEEDS` (27000-27499), a range that
has never been used for training, checkpoint selection, reward design or
hyper-parameter tuning. The three official circuits deliberately share the same
reused prefix (1800, 1801) because the published current-free rule runs used
1800-1804 on each circuit. `tests/learning/test_final_benchmark.py` asserts the
holdout range is disjoint from every training and selection role.

## Metrics

Per episode: referee status, evaluation end reason, gates completed vs expected,
full-circuit completion, official and penalized time, time per gate, collision
events / frames / episode flag, out-of-bounds events / frames / episode flag,
wrong-direction crossings, previous-gate returns, missed-gate attempts, stuck
events, timeout, path length, mean action jerk, action saturation, inference
time, and the policy-only evidence.

`collision_frames` and `out_of_bounds_frames` come from per-step sampling in the
episode probe (collision sensor state, and position against the track bounds);
event counters come from the unchanged referee. Previous-gate returns are
recomputed geometrically from the trajectory for *every* controller (clearing a
passed gate by more than 1 m and then dropping back behind its plane), so the
metric does not depend on a controller's self-report; the learned controller's
own counter is kept alongside it as `previous_gate_returns_controller`.

## Timing rules

Times are never compared across test suites. Timing figures are reported per
group; every timing comparison is a paired per-seed difference within one group,
with an exact binomial sign test on the paired wins. `single_gate_retention` is
excluded from all timing: the official clock runs first-gate-to-last-gate, so a
single-gate case has a definitionally zero completion time.

Selection priority 4 ("completion time") uses the **mean within-group rank**
rather than a pooled mean, so no second from one geometry is ever weighed
against a second from another.

## Selection priority

1. full-circuit reliability (official circuits)
2. zero safety events
3. three-gate and vertical-transition reliability
4. completion time (mean within-group rank)
5. smoothness (mean within-group jerk rank)

## Outputs

| file | content |
|---|---|
| `episodes.json` / `episodes.csv` | every episode, machine-readable |
| `aggregate_by_group.csv` | per controller and group, all metrics |
| `paired_comparisons.csv` | paired comparisons, overall and per group |
| `aggregate_report.json` | the full aggregate, comparisons and recommendation |
| `final_benchmark_report.md` | the concise report |
| `plots/` | trajectory plots for representative successes and failures |
| `artifacts/` | per-episode trajectories and official-circuit videos |

## Why `three_gate_success` read `0.0`

`three_gate_success: 0.0` in the training status did **not** mean the policy
failed three-gate cases. It meant no three-gate case was ever evaluated.

`_evaluation_cases` adds a three-gate case only from stage C5 (`six_gate` from
C6). The reliability-first run peaked at C4 and finished at C3, so no evaluation
ever contained one. `aggregate_evaluation` then divided by an empty subset and
published `0.0`, which is indistinguishable from "ran and failed every episode".

Audit the claim directly:

```bash
conda run -n marine_race_rl python -m marine_race_arena.learning.three_gate_audit \
  results/rl/multigate_reliability_first/r2_reliability_first_seed23001_c3_recovery_476525
```

It reads the stored evaluation **rows** (not the ambiguous summary field) and
reports the verdict. For this run: 81 evaluations scanned, 0 containing a
three-gate case, 0 three-gate episodes ever run, verdict `never_evaluated`.

### The fix

A category with no episodes now reports `None`, never `0.0`:

* `aggregate_evaluation` publishes `completion_rate: None` plus `evaluated` and
  `n` for every category, and lists `evaluated_categories` /
  `not_evaluated_categories`;
* the evaluation report carries `not_evaluated_reasons`, naming the stage that
  would introduce the missing category;
* the run status gains `three_gate_status` (`not_evaluated` / `failed` /
  `partial` / `passed`), `three_gate_evaluated`, `three_gate_n` and
  `three_gate_note`, so a null or a zero can no longer be misread;
* `rate_or` makes every consumer state what an unmeasured category counts as,
  instead of silently reading it as a 0% success rate;
* `official_evaluation_unlocked` requires *measured* three-gate competence, so
  the circuits can no longer be gated on a number that was never produced;
* the rollback gate skips a floor for a category the suite never ran, instead of
  treating the absence as a regression.

Because the category was never measured, three-gate competence was untested for
the whole run. The final benchmark therefore evaluates three-gate sequences and
S-shapes explicitly, for every controller.
