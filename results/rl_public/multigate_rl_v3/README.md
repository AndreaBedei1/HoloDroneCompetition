# Learned Multi-gate RL v3

**Honest result category: PARTIAL RL SUCCESS.** The feed-forward, observation-v3 PPO controller completed the current-free straight two-gate task on 9/10 reserved seeds in real HoloOcean with no fallback and no runtime rule-generated action. It did not meet the balanced-turn R2 criterion, so three-gate and official-circuit RL claims were not made.

The selected PPO ZIP and raw datasets remain under git-ignored `results/rl`; this package publishes their SHA-256 hashes, exact paths, compact metrics, provenance, failure evidence, and reproduction commands. See `experiment_manifest.json`, `two_gate/`, `three_gate/failure_analysis.json`, and `comparison/`.
