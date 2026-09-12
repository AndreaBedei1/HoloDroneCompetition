# Claim audit

The executable audit is read-only and launches no simulator:

```bash
python article_journal/scripts/verify_claims.py
```

The audit covers the headline values retained in the current journal manuscript.
A historical 78-run compact release package is checked when needed for retained
artifact provenance; the raw tree remains immutable and external to this worktree.

| Claim retained in the manuscript | Evidence | Status |
| --- | --- | --- |
| Current-free reference completion is 9/9 across 51 gates, with zero collision, out-of-bounds and wrong-direction events | `results/rl_public/visual_pose_v2/official_no_current/summary.json` and per-track CSV files | verified |
| The historical 78-run release package and referee/local progression audit are retained as repository provenance | `article_journal/scientific_release/matrix_78_20260715/manifest.json`, `runs.json`, `runs.csv` | verified; not a complete current-manuscript evaluation matrix |
| Recurrent PPO validation completes 9/9 episodes, with one collision and zero out-of-bounds events | `artifacts_gen2/ppo_27d_direct_speed_campaign_C_20260909_retry1/final_validation_50k/evaluation_incremental.json` | verified |
| Recurrent PPO uses the participant-level information boundary and action interface described in Section 9 | `marine_race_arena/learning/`, campaign configuration and checkpoint artifacts | verified |
| The current-free rule reference, systematic benchmark evaluation and learned-controller validation use their reported control rates | the corresponding manifests and Section 10 protocol | verified as an experimental-design constraint |
| The perception capture remains diagnostic rather than a general performance claim | `artifacts_gen2/visual_collection_smoke/visual_smoke_report.json` | preliminary/diagnostic |

Superseded learning claims and associated historical artifacts are not part of
the current main manuscript. They remain in the repository as historical
supporting material and are not used as current evidence.
