# HoloOcean Episode-Reset Benchmark

Fresh reset relaunches the HoloOcean engine (`holoocean.make`) every episode;
persistent reset (`PersistentRaceSession`, experimental) keeps one engine alive and
teleports the vehicle. Measured on the Stage-1 track, real HoloOcean, 3 seeds
(`reset_benchmark.json` has the raw numbers).

| Mode | Reset wall time | Notes |
| --- | --- | --- |
| Fresh (relaunch per episode) | **15.56 ± 0.35 s** | guaranteed identical to the benchmark |
| Persistent (teleport, one engine) | **6.77 ± 0.04 s** | **2.3× faster** |

## Equivalence check (persistent vs fresh)

- Referee gate progress under a fixed action sequence: **matches**.
- Residual DVL speed after reset: **0.0 m/s** (no stale velocity).
- Initial observation: **NOT bit-identical** — one binary perception feature
  (a vision/beacon `*_present` flag) can differ on the very first frame after
  teleport, giving `max_abs_diff = 1.0`.

## Recommendation

- **Fresh reset is the default everywhere, including the PPO smokes.** Frozen
  correctness evaluations (Evaluation A/B) use fresh reset — the initial observation is
  guaranteed identical to a normal run.
- **Persistent reset is experimental and is NOT the recommended PPO mode.** It is a
  promising 2.3× training speedup but is **not yet bit-equivalent** on the first
  observation frame, and it has two further limitations found on inspection:
  - The trailing engine reset that restarts the clock also **restores the spawn pose**,
    undoing a preceding teleport. `PersistentRaceSession.reset_episode` therefore
    **refuses a non-trivial `start_randomization`** rather than silently dropping it.
  - Per-episode **beacon noise is not re-seeded** in persistent mode (the arena beacon
    manager is reused).
  Use fresh reset for anything randomized, noisy, or that must match the benchmark.
- Validated only on the noise-free, yaw-0 Stage-1 track. On tracks with beacon noise
  or non-zero start yaw, re-validate before use (the persistent path re-uses the arena
  beacon manager). This limitation is recorded in `reset_benchmark.json`.

The normal fresh-reset `RaceEpisode` is unchanged; persistent reset is a separate,
opt-in class.
