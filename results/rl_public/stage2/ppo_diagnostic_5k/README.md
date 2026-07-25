# Stage-2 Randomized PPO Diagnostic (5,000 steps)

**These are 5,000-step diagnostics, not final scientific results.** Development seeds 1410-1419 were used for checkpoint selection; the reserved final ranges (1500-1549, 1550-1599) remain untouched. PPO improvement over BC is not established until a new held-out evaluation shows it.

- **Primary comparison:** `bcinit_controlled` vs `scratch_controlled` (identical exploration std and hyperparameters; only the initial weights differ).
- **`scratch_default`** is an exploration-variance diagnostic (SB3's ~1.0 default std), not a proof about BC weights.
- Completion is split into interior vs **extreme-corner** (|lateral| >= 0.8 m and |yaw| >= 12 deg) -- where the frozen BC evaluation failed.

See `calibration/` (KL-safe config selection), each arm's folder (run config, timestep-zero + best + final eval, KL metrics, action statistics, model hashes, reproduce command) and `comparison/` (overall + interior-vs-extreme + failure analysis). SB3 model ZIPs are not committed (hashes + reproduce.txt only).
