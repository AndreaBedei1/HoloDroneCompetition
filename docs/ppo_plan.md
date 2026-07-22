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
   simulator stability. No performance claim.
2. **5,000-step diagnostic** (~20 min): inspect the reward-component and eval CSVs; check
   the policy is not collapsing to a saturated action. Continue only if held-out
   completion trends upward.
3. **10,000-step pilot** (~32 min): evaluate the best checkpoint on held-out seeds.
   Continue only if it shows meaningful, non-trivial completion.
4. **Longer run (e.g. 50,000)** only if the pilot improves held-out completion.

Run **two arms** and compare on identical held-out seeds:

- **PPO from scratch** — control.
- **BC-initialized PPO** — warm-started from `results/rl/stage1/bc_rand_combined/best_model.pt`
  via the verified exact normalization-aware transfer (`bc_model_path=...`).

Each run writes to `results/rl/stage1/ppo/<arm>/<timestamp>/` with full provenance
(`run_config.json`, `environment.json`, `reproduce.txt`); resume is supported and the
best model is kept by held-out completion (not training reward). Optionally use
`PersistentRaceSession` for training throughput (experimental; see the reset benchmark).

## Exact next command (BC-initialized PPO, 1k smoke)

```bash
conda activate marine_race_rl
python - <<'PY'
from marine_race_arena.learning.train_workflow import run_ppo_training
run_ppo_training(
    "marine_race_arena/tracks/training/stage1_single_gate.json",
    stage="stage1", algorithm="ppo_bcinit",
    total_timesteps=1000, train_seed=0, eval_seeds=[1200, 1201, 1202, 1203, 1204],
    output_root="results/rl",
    env_kwargs=dict(adapter="holoocean", allow_fallback=False, max_steps=400, dt=0.1),
    hidden_sizes=(256, 256), checkpoint_freq=500, eval_freq=1000,
    ppo_kwargs=dict(n_steps=1000, batch_size=100, n_epochs=5),
    bc_model_path="results/rl/stage1/bc_rand_combined/best_model.pt",
)
PY
```

Use fresh seeds (e.g. 1200+) disjoint from all evaluation seeds. From-scratch control:
drop `bc_model_path` and set `algorithm="ppo_scratch"`.
