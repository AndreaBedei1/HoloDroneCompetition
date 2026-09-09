# Claim audit

The executable audit is read-only and launches no simulator:

```bash
python article_journal/scripts/verify_claims.py
```

The audit covers the headline values retained in the journal manuscript. The
78-run matrix is checked from its compact release package when the raw matrix is
not available; the raw tree remains immutable and external to this worktree.

| Claim retained in the manuscript | Evidence | Status |
| --- | --- | --- |
| Current-free reference completion is 9/9 across 51 gates, with zero collision, out-of-bounds and wrong-direction events | `results/rl_public/visual_pose_v2/official_no_current/summary.json` and per-track CSV files | verified |
| The systematic matrix contains 78 runs and the referee/local progression audit is recorded | `article_journal/scientific_release/matrix_78_20260715/manifest.json`, `runs.json`, `runs.csv` | verified; internal audit flags are preserved in the manifest |
| Final Gen-2 validation is 9/9 finished episodes, 153 gates, one collision and zero out-of-bounds events | `artifacts_gen2/ppo_27d_direct_speed_campaign_C_20260909_retry1/final_validation_50k/evaluation_incremental.json` | verified |
| Final PPO uses the 27-D contract and selected 50,688-step checkpoint | `campaign_config.json`, `training_decision_log.json`, checkpoint SHA file | verified |
| The current-free rule reference is separate from both the 30-Hz matrix and the final 10-Hz PPO validation | the corresponding manifests and Section 10 protocol | verified as an experimental-design constraint |
| The perception capture remains diagnostic rather than a general performance claim | `artifacts_gen2/visual_collection_smoke/visual_smoke_report.json` | preliminary/diagnostic |

Superseded Gen-1/35-D PPO claims, the former 474-episode comparison, readiness
survival curve and associated significance claims are not part of the current
main manuscript. Their artifacts remain in the repository as historical
supporting material and are not silently relabelled as Gen-2 evidence.
