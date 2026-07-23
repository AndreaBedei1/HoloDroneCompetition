# Stage-1 PPO Quickstart (real HoloOcean)

This is the practical workflow for running the safe, reproducible Stage-1 PPO smoke
experiments on Windows. It assumes the documented HoloOcean install and the dedicated
`marine_race_rl` Conda environment (see `README.md` / `docs/rl_progress.md`).

Everything below uses **fresh reset** and the **committed public BC model**
(`results/rl_public/stage1/bc/model/best_model.pt`, hash-verified before launch). No
step starts a long run: the 1,000-step smokes verify the workflow, not convergence.

> The scripts only ever start this repository's own HoloOcean process and write under
> `results/rl/`. They never kill Python/Unreal/HoloOcean processes. If another HoloOcean
> project is running on the machine, it is left untouched.

## 1. Verify the installation

```bash
conda run -n marine_race_rl python -m pytest tests/learning -q
```

## 2. Dry run (no HoloOcean launched)

Checks the branch, verifies the model and track hashes, and prints the exact effective
configuration and output directory — without starting the simulator:

```bash
conda run -n marine_race_rl python -m marine_race_arena.learning.launch_stage1_ppo --arm bcinit --steps 1000 --dry-run
```

## 3. Run the BC-initialized 1k smoke

```bat
scripts\run_stage1_ppo_bcinit_1k.bat
```

## 4. Run the from-scratch control 1k smoke

```bat
scripts\run_stage1_ppo_scratch_1k.bat
```

## 5. Resume a run

Pass the run directory printed by the launcher (the resume defaults to 1,500 total steps):

```bat
scripts\resume_stage1_ppo.bat results\rl\stage1\ppo_bcinit\<actual-run-directory> --steps 1500
```

## 6. Inspect the results

Each run writes a timestamped directory under
`results/rl/stage1/<ppo_bcinit|ppo_scratch>/<timestamp>/`:

| What | Where |
| --- | --- |
| Timestep-zero (pre-training) held-out eval | `evaluation/initial_eval.json` |
| Periodic held-out eval history (incl. the timestep-0 row) | `evaluation/eval.csv` |
| Best model (by held-out completion, not reward) | `best_model/best_model.zip` + `best_model/best_metrics.json` |
| Final model | `final_model.zip` |
| Latest checkpoint (for resume) | `checkpoints/ppo_*_steps.zip` |
| Run configuration (hyperparameters, eval seeds) | `run_config.json` |
| BC action-std warm-start provenance (per-axis std, source, BC hashes) | `action_std.json` |
| Environment manifest (packages, adapter actually used, wall-clock) | `environment.json` |
| Exact reproduction command (fresh or resume) | `reproduce.txt` |
| Training logs (CSV: policy/value loss, entropy, approx-KL, clip fraction, …) | `logs/progress.csv` |
| Reward config / seeds / track copy + hash | `reward_config.json`, `seeds.json`, `track.json`, `track_sha256.txt` |

Compact published smoke summaries (safe to share) are under
`results/rl_public/stage1/ppo_smoke/`.

## 7. Stop safely

Press **Ctrl+C** in the terminal running the script to stop only that run; the launcher
closes the HoloOcean environment cleanly. **Do not** kill all Python/Unreal/HoloOcean
processes — another project may be using them.

## Notes

- Development eval seeds are **1200–1204**. These are for checkpoint selection during the
  smokes; the final scientific evaluation must use new, unseen seeds (see
  `docs/ppo_plan.md`).
- Persistent reset is experimental and **not** used here — see
  `results/rl_public/reset_benchmark/`.
- The 5k/10k/50k stages are **not** started automatically; the exact next commands are in
  `docs/ppo_plan.md`.
