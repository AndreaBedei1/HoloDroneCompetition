# Staged PPO Plan (not yet run)

The 50,000-step PPO job is **deliberately not started** yet. All correctness fixes,
the public audit package and the frozen A/B evaluations are complete first. This plan
sizes PPO from **measured** HoloOcean timings and defines gated stages.

## Measured cost inputs

| Quantity | Measured | Source |
| --- | ---: | --- |
| HoloOcean step (advance one control tick) | ~0.13 s/step | demo/eval episode wall times |
| Fresh reset (relaunch) | 15.6 s | `results/rl_public/reset_benchmark/` |
| Persistent reset (teleport) | 6.8 s | same |
| Eval episode (finished, ~110 steps + launch) | ~30 s | frozen evaluations |
| Inference | ~42 ms/step | frozen evaluations |

An untrained PPO episode runs to `max_steps` (it does not finish early), so with
`max_steps=400` each training episode is ~400 steps ≈ 52 s of stepping plus one reset.

## Wall-clock estimates (real HoloOcean, single env)

Rough estimate = `steps × 0.13 s` (stepping) + `(steps / max_steps) × reset_s` +
`n_evals × eval_seeds × 30 s`. With `max_steps=400`, persistent reset (6.8 s) and eval
every quarter of the run over 5 seeds:

| PPO steps | Stepping | Resets | Eval | **Total (persistent)** | Total (fresh reset) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | ~2 min | ~0.3 min | ~5 min | **~7–8 min** | ~8 min |
| 5,000 | ~11 min | ~1.5 min | ~7 min | **~20 min** | ~23 min |
| 10,000 | ~22 min | ~3 min | ~7 min | **~32 min** | ~38 min |
| 50,000 | ~108 min | ~14 min | ~13 min | **~2.5 h** | ~3 h |

Reset overhead is a small fraction of stepping at `max_steps=400`; the 2.3× reset
speedup helps most with short episodes / frequent resets.

## Staged, gated procedure

Run each stage and inspect before continuing. **Never claim convergence from reward
curves** — advancement is judged only by held-out completion under the unchanged referee.

1. **1,000-step smoke** (~8 min): confirm reward components, action ranges, resets and
   simulator stability. No performance claim. **Done this pass** (both arms; see
   `results/rl_public/stage1/ppo_smoke/`).
2. **5,000-step diagnostic** (~20 min): inspect the reward-component and eval CSVs; check
   the policy is not collapsing to a saturated action. Continue only if held-out
   completion trends upward.
3. **10,000-step pilot** (~32 min): evaluate the best checkpoint on held-out seeds.
   Continue only if it shows meaningful, non-trivial completion.
4. **Longer run (e.g. 50,000)** only if the pilot improves held-out completion.

Run **two arms** and compare on identical held-out seeds:

- **PPO from scratch** — control (SB3-default exploration).
- **BC-initialized PPO** — warm-started from the committed public BC model
  `results/rl_public/stage1/bc/model/best_model.pt` via the verified exact
  normalization-aware transfer, with a **safe per-axis exploration std** derived from the
  BC validation residuals (clamped to `[0.05, 0.15]`; see `bc_ppo_init.py`).

Both arms use the same track, seeds, reward and conservative PPO config; only the BC-init
arm applies the warm-start. Each run writes to
`results/rl/stage1/ppo_<arm>/<timestamp>/` with full provenance (`run_config.json`,
`action_std.json`, `environment.json`, `reproduce.txt`), a **timestep-zero** held-out
evaluation before training, resume support, and best-model selection by held-out
completion (not training reward). **Fresh reset only** — persistent reset stays
experimental (see the reset benchmark).

## Run it (no hand-written Python needed)

The 1,000-step smoke is launched with one command (or the Windows scripts in `scripts/`);
see `docs/rl_quickstart.md`:

```bash
conda run -n marine_race_rl python -m marine_race_arena.learning.launch_stage1_ppo --arm bcinit  --steps 1000
conda run -n marine_race_rl python -m marine_race_arena.learning.launch_stage1_ppo --arm scratch --steps 1000
```

## 1k smoke result (this pass)

Both 1,000-step arms were run on real HoloOcean (no fallback), dev seeds 1200–1204.
Compact results: `results/rl_public/stage1/ppo_smoke/`. These are **plumbing smokes, not
convergence** — no superiority claim.

| | BC-init | Scratch |
| --- | --- | --- |
| Timestep-zero held-out completion | **1.00** (warm-start = BC baseline) | 0.00 (untrained) |
| Completion after 1,000 steps | 1.00 | 0.00 (1k ≪ convergence) |
| Final-policy action saturation (sampled) | **0.00** (std floored at 0.05) | 0.33 (SB3 default std ≈ 1.0) |

The BC-init timestep-zero completion matched the BC baseline (warm-start intact and not
destroyed by the first updates); the near-zero action saturation vs. the scratch arm's
0.33 is exactly the effect the safe warm-start is designed to produce. Resume of the
BC-init run to 1,500 steps was verified (eval history appended, timestep-0 not
duplicated, best model preserved).

## Prepared next-stage commands (NOT run automatically)

Advance only if the gates below hold. Seeds 1200–1204 are **development** seeds; do not
use them as final test seeds.

**5,000-step diagnostics** (run both, compare on the dev seeds):

```bash
conda run -n marine_race_rl python -m marine_race_arena.learning.launch_stage1_ppo --arm bcinit  --steps 5000
conda run -n marine_race_rl python -m marine_race_arena.learning.launch_stage1_ppo --arm scratch --steps 5000
```

**10,000-step pilots** (prepare only; run only if the 5k diagnostic improves held-out
completion or another justified metric):

```bash
conda run -n marine_race_rl python -m marine_race_arena.learning.launch_stage1_ppo --arm bcinit  --steps 10000
conda run -n marine_race_rl python -m marine_race_arena.learning.launch_stage1_ppo --arm scratch --steps 10000
```

Advance from 5k → 10k → longer **only if all** hold: no crashes; no NaN; no manifest
mismatch; action saturation stays low; approx-KL stays acceptable; best-model selection
works; the BC-init warm-start has not catastrophically collapsed. A 50,000-step launcher
is **deliberately not provided** as a default action.

## Final scientific evaluation

The final comparison must use **new, unseen seeds** — e.g. a range such as `1300–1349`
(confirm they are disjoint from every seed used in demos/dev/frozen A-B before using
them) — evaluated with `closed_loop_eval` under the unchanged referee, on both the fixed
and randomized start conditions.
