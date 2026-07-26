# Multi-gate progression: diagnostic, controllers, and the current-free circuits

This documents the multi-gate work on `feature/visual-pose-multigate`: how the onboard
controllers behave across multiple gates, and the evidence that the three official circuits can
be completed **current-free** by an autonomous onboard-only controller.

## The question

Can the existing onboard stack — a learned single-gate policy (**BC-v1**), the controller-side
`LocalCourseTracker`, and a hierarchical beacon/vision controller — already complete multiple
gates and, ultimately, the three official circuits with currents disabled?

The observation-v2 visual-pose front end was *not* a prerequisite for answering this: the
`LocalCourseTracker` already advances the expected beacon from robust onboard evidence
(camera + DVL + acoustic range), so a controller that can pass one gate and navigate to the next
beacon can chain gates without metric gate-pose.

## Tracks (training-only, current-free by construction)

| track | gates | layout |
| --- | --- | --- |
| `tests/two_gate_straight.json` | 2 | aligned, 5 m apart (M1) |
| `tests/two_gate_left_curve.json` / `two_gate_right_curve.json` | 2 | small transition turn (M2) |
| `tests/three_gate_s_curve.json` | 3 | S-curve, +/-36.9 deg yawed gates (M3) |
| `training/stage3_three_gates.json` | 3 | aligned with a vertical step |

## The diagnostic (`marine_race_arena.learning.multigate_diagnostic`)

Runs any onboard controller through the real runner and records, per control step, the
`LocalCourseTracker` phase / expected beacon / filtered range, the referee's own gate crossings,
out-of-bounds, collisions and wrong-direction events, and the vehicle position (position/referee
events are ground truth used **only** for offline diagnosis — never fed to the controller). It
then classifies the *first* failure (`TRACKER_FALSE_ADVANCE`, `NEXT_GATE_TURN_FAILED`,
`RETURN_TO_PREVIOUS_GATE`, `OUT_OF_BOUNDS`, `COLLISION`, `FAILED_GATE_ALIGNMENT`, ...), so a
corrective iteration targets the earliest root cause. Currents are disabled and asserted.

## Finding 1 — BC-v1 alone does not do multi-gate

On `two_gate_straight` (real HoloOcean, currents off), the frozen **BC-v1** reaches at most
**1 of 2** gates and completes **0**. It wanders between gates: collisions, out-of-bounds, and a
`RETURN_TO_PREVIOUS_GATE` (wrong-direction) crossing; the tracker sometimes *false-advances*
(reports a passage the referee does not validate). The single-gate imitation policy did not learn
to navigate from one gate to the next. See `results/rl_public/visual_pose_v2/multigate_v1/`.

## Finding 2 — a deterministic onboard controller already completes all three circuits current-free

The official deterministic `rule_gate_center_then_commit` (`controllers/official_baselines.py`)
is onboard-only (received beacons + FrontCamera + depth/IMU/DVL + its own `LocalCourseTracker`;
no referee/world/gate pose) and does beacon homing → visual align → center-then-commit passage →
exit → next-beacon turn. In the frozen benchmark's **clean (current-free)** matrix it completes
every official circuit, real HoloOcean, no fallback, referee FINISHED:

| circuit (gates) | `rule_gate_center_then_commit` | `rule_gate_baseline` |
| --- | --- | --- |
| Horseshoe Bay (12) | 5/5 | 5/5 |
| Vertical Serpent (17) | 4/5 | 5/5 |
| Mixed Endurance (22) | 5/5 | 2/5 |

So the practical engineering objective — *a fully autonomous onboard controller completes all
three circuits with currents disabled* — is **achievable, and met by the deterministic
controller**. A fresh re-verification from this branch's committed SHA is published under
`results/rl_public/visual_pose_v2/official_no_current/` (see `docs/final_no_current_demo.md`).

## The hybrid controller (`hybrid_gate_controller`)

To integrate the learned policy without regressing multi-gate reliability,
`controllers/hybrid_gate_controller.py` uses the deterministic rule controller as the backbone
and blends **BC-v1** in **only** during `VISUAL_ALIGN` (weight ramped by visual confidence,
capped at 0.55). The backbone is the sole authority in SEARCH / APPROACH / COMMIT / VERIFY_EXIT /
ADVANCE — the phases where BC-v1 alone fails — so the hybrid cannot do worse than the rule
controller in the critical passage/exit. Command provenance (rule / bc / weight) is logged.

Real-HoloOcean result (current-free): the hybrid completes the **two-gate straight 2/2**, the
**three-gate S-curve 2/2**, and the **Horseshoe Bay circuit 2/2** (12/12 gates, 0 collisions /
out-of-bounds) — where **BC-v1 alone completes 0/8** multi-gate. The learned visual servo is thus
integrated without regressing multi-gate reliability. See
`results/rl_public/visual_pose_v2/hybrid_controller/`. A genuinely pose-aware learned controller
(observation v2) remains blocked on a reliable real-image gate-pose detector (see
`docs/visual_gate_pose.md`).

## Decision-tree outcome

Case A of the task's decision tree applies: an onboard controller completes the multi-gate
objective, so **visual pose is an optional improvement, not a blocker**. The learned/hybrid path
is the open research direction; the circuits themselves are solved current-free.

## Reproduce

```bash
# BC-v1 multi-gate diagnostic (first failure classified):
scripts\run_two_gate_v1_diagnostic.bat
scripts\run_three_gate_v1_diagnostic.bat
# Hybrid multi-gate:
scripts\run_two_gate_hybrid.bat
scripts\run_three_gate_hybrid.bat
# Current-free official circuits (best controller):
scripts\run_all_official_no_current.bat
```
