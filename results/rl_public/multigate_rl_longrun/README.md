# Multi-gate RL long-run preparation

Status: **LONG-RUN TRAINING PIPELINE READY**.

This package describes the infrastructure and bounded smoke evidence for a
future manually launched multi-day run. It is not a trained long-run result and
does not claim RL engineering success.

The selected R1 model remains frozen at its original git-ignored path and
SHA-256. The recommended initialization is the versioned balanced BC-v3
derived from it; `bc_v3_manifest.json` records its hash, disjoint splits, test
metrics, and 20-case closed-loop selection result. Raw datasets, generated
tracks, model binaries, PPO checkpoints, TensorBoard events, videos and
per-step logs remain git-ignored. Only compact plans, hashes, configuration,
selection policy and validation summaries are published here.

See `docs/rl_multigate_longrun.md` for the exact prepare, start, status, stop,
resume and evaluation commands.
