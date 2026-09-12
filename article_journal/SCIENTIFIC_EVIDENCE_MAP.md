# Scientific evidence map

This map records the evidence used by the current journal manuscript. “Verified”
means that the value is recomputed by `scripts/verify_claims.py`; “diagnostic”
means that it informs method or limitations but is not a broad performance claim.

| Manuscript claim | Evidence artifact | Status | Destination |
| --- | --- | --- | --- |
| The benchmark protocol contains an onboard-only participant boundary and independent referee | `marine_race_arena/`, referee implementation and local/referee audit | verified | Sections 3, 7 and 10 |
| Reference-controller evaluation covers the three clean circuits and one medium-current condition on Horseshoe Bay | `tables/clean_tracks.tex`, `tables/currents.tex` and the released result artifacts | verified | Section 10 |
| The historical 78-run release package is retained as repository provenance | `scientific_release/matrix_78_20260715/manifest.json` and `runs.json` | verified; not a complete current-manuscript evaluation matrix | Repository provenance |
| Current-free reference controller completes 9/9 official runs | `results/rl_public/visual_pose_v2/official_no_current/summary.json` | verified | Section 10, Table `current_free` |
| Recurrent PPO completes 9/9 validation episodes; one collision and zero OOB events | `artifacts_gen2/ppo_27d_direct_speed_campaign_C_20260909_retry1/final_validation_50k/evaluation_incremental.json` | verified | Section 10, Table `learning` |
| Recurrent PPO architecture, contract ordering and selected checkpoint are recorded in committed artifacts | `marine_race_arena/learning/`, campaign config and decision log | verified | Section 9 |
| Perception captures and pose metrics are limited-scope diagnostics | `artifacts_gen2/visual_collection_smoke/` and `results/rl_public/visual_pose_v2/vision_pose/` | diagnostic/preliminary | Sections 5 and 12 |

Superseded learning materials are historical artifacts only and are not used to
support current manuscript claims. No archival DOI is asserted; a DOI remains
an explicit administrative TODO in the data availability statement.
