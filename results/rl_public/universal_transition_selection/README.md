# Universal transition initialization selection

Compact, auditable copy of the initialization selection for the universal local
gate-transition PPO. The heavy evidence (checkpoints, 1,012 episode rows per
arm, trajectories) stays under the ignored `results/rl/universal_transition/`.

| File | Contents |
|---|---|
| `ab_comparison_original.md` | The superseded ranking, unmodified. It summed safety events and selected `scratch`. |
| `ab_comparison_corrected.md` | Corrected three-stage report: competence qualification, safety ranking, final selection. |
| `ab_comparison_corrected.json` | Machine-readable corrected report, including per-criterion gate results and both arms' recomputed metrics. |

## Why the original selection was wrong

`scratch` produced 122 safety events against warm's 983 and was therefore ranked
first. It reached that count by barely moving: 15 gates crossed in 1,012
episodes (1.5% first crossing, 0.2% target switch, 0.0% transition success), a
mean absolute action of 0.0034, and no step above 0.05 on any axis. The
scratch-initialized long run confirmed it: at 100,352 transitions the holdout
still showed 0.0% transition success and 0% target switch.

## Corrected result

Competence qualification is now mandatory and precedes safety ranking.

- `scratch` → `degenerate_inactive_policy` (fails seven criteria), not ranked.
- `selective_warm_start` → `competent`, selected.

The selected policy is **competent but not reliable**: 31.4% transition success
with 417 collision episodes, 78 missed gates, 51 wrong-direction events and 436
acquisition timeouts. It is selected because it is the only initialization that
learned useful gate-crossing and beacon-switch behaviour. The requirement
remains universal transition success >= 99% on repeated unseen evaluations with
strong safety.

Method and thresholds: `docs/ppo_universal_transition.md`.
Implementation: `marine_race_arena/learning/transition_selection.py`.
Regeneration: `marine_race_arena/learning/transition_ab_report.py`.
