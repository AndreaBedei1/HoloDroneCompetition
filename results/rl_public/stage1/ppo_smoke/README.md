# Stage-1 PPO Smoke Results (real HoloOcean)

These are **workflow smoke tests**, not training results.

- **1,000 steps is NOT convergence.** No scientific superiority claim is made.
- Both arms use the same track, seeds, reward and conservative PPO config; only the
  BC-init arm applies the imitation warm-start (safe per-axis exploration std).
- Seeds **1200-1204 are development seeds** (checkpoint selection). The final
  scientific evaluation must use new, unseen seeds (see `docs/ppo_plan.md`).
- Purpose: verify both workflows run, the simulator stays stable, checkpointing and
  resume work, action statistics are logged, and the BC warm-start is not destroyed
  immediately.

See `comparison.json` for the side-by-side, and `bcinit_1k/` / `scratch_1k/` for each
arm's run config, timestep-zero and final evaluation, eval history, action statistics,
environment manifest, model hashes and reproduction command. SB3 model ZIPs are not
committed (only their hashes + reproduce.txt).
