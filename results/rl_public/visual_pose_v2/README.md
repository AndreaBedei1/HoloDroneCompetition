# Observation-v2 Visual Gate-Pose — Real-HoloOcean Detector Validation

**Status: NOT YET SUCCESSFUL for the multi-gate / three-circuit objective.** This package
records the perception foundation and its honest real-HoloOcean validation.

## What was delivered (built, tested, committed)

- **Stage-2 audit fixes** — penalty metric naming (`penalty_s` vs raw/penalized completion
  time), and a strengthened seed registry (consumed dev allocations moved to used;
  observation-v2 / multi-gate forward ranges; reserved final ranges; pairwise-disjoint roles).
- **`controllers/gate_pose.py`** — an onboard visual gate-pose front end: pinhole intrinsics
  (640×480, FOV 90 → fx=fy=320), inner **aperture-corner** detection, planar **PnP**
  (`SOLVEPNP_IPPE`) with the square-planar sign ambiguity resolved by the bar-height ratio +
  reprojection error, a projective + known-size **fallback**, and a **temporal filter**. 18
  tests on synthetic gates at known poses verify corner detection, distance/lateral/yaw-sign
  recovery, fallbacks and the tracker. It uses only the camera image and the public 1.5 m
  aperture — never simulator pose. See `docs/visual_gate_pose.md`.

## Honest real-HoloOcean result (`vision_pose/pose_metrics.json`)

Validated on the five yaw-rotated single-gate tracks (real HoloOcean, no fallback, currents
disabled). Two bounded corrective iterations:

| iteration | detection rate | pose availability | notes |
| --- | --- | --- | --- |
| 0 (synthetic-tuned mask) | 0.10 | 0.00 | rejected the saturated green gate on real images |
| 1 (v1 pixel classifier) | **0.25** | **0.10** | distance ~0.65 m usable, but **yaw unreliable** (median 139°, sign 0.0) |

**Verdict:** the deterministic geometric detector is **not yet reliable on real images**.
Aperture-corner detection is the bottleneck — real gates (thin green bars, underwater haze,
~3.9 m) rarely yield a clean quadrilateral, so the metric gate-plane **yaw** from PnP is not
trustworthy. Distance and lateral offset are usable when a quad is found.

## What was NOT done (blocked or out of session scope)

Observation-v2 encoder wiring into BC/PPO, the pose-aware expert + hybrid controller, v2
demonstrations + BC-v2, the multi-gate curriculum (M1–M5), and the three official circuits
current-free. All of these depend on a **reliable gate-pose detector**, which was not
achieved here.

## Recommended next step

Replace or augment the deterministic corner detector with a **learned corner/pose model**
(or further CV iteration — e.g. temporal corner tracking, colour-model tuning against the real
gate, sub-pixel bar-edge fitting) until real-image **yaw-sign accuracy** is reliable. Then
wire observation v2, collect pose-aware demonstrations, train BC-v2, and run the multi-gate
curriculum. Until then, the **frozen center-only v1 controller remains the best available**
(single-gate 50/50 fixed-start, 48/50 randomized) — it is unchanged.

Nothing frozen was modified: `main` at f4d6375, the BC-v1 model hash, the 78-run matrix, the
frozen A/B evaluations and the Stage-1/2 public results are all untouched.
