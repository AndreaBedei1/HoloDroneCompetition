# Multi-gate RL reliability-first package

Status: **RELIABILITY-FIRST LONG-RUN READY**.

This compact package publishes the reliability safeguards and bounded
real-HoloOcean validation for the manual one-million-step PPO pipeline. It
does not publish or claim a completed long run.

The immutable warm start is BC-v3 SHA-256
`c9a889125278122b57307e21b7c2390baa61429f8952093bcc2760d9b3a49bf2`.
The real validation ran on committed code
`c76c23f565d3906360f9d31723d371c7fc506e6c`, currents off and no fallback.

- `preparation_manifest.json`: branch, provenance, protected refs/models, tests.
- `config.json`: public snapshot of the one-million-step configuration.
- `baseline_bc_v3_eval.json`: 20-case frozen BC-v3 baseline.
- `conservative_smoke_results.json`: 26,624-step PPO smoke and paired evidence.
- `checkpoint_policy.json`: reliability-first selector and aliases.
- `rollback_policy.json`: trigger, restoration, and bounded-attempt rules.
- `reward_schedule.json`: reliability and efficiency phases plus failure bound.
- `exact_longrun_command.txt`: the one command the operator may run manually.

The PPO smoke remained in C0 after exactly one reliable full suite. It is not
evidence for C1+, a three-gate policy, or official circuits. The five-seed
paired straight result is descriptive and does not establish general PPO
improvement.

Full documentation: `docs/rl_multigate_reliability_first.md`.
