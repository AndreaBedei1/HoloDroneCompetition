# Observation-v2 visual pose + multi-gate + current-free circuits

This package covers three things on `feature/visual-pose-multigate`:

1. a **corrected** onboard visual gate-pose front end and its honest real-HoloOcean validation;
2. a **multi-gate diagnostic** of the frozen learned controller (BC-v1) and an onboard hybrid;
3. the practical engineering deliverable — **all three official circuits completed current-free**
   by an autonomous onboard controller in real HoloOcean.

## Headline

**The current-free three-circuit objective is achievable and is met by an autonomous onboard
controller.** The deterministic onboard `rule_gate_center_then_commit` completes every official
circuit with currents disabled, real HoloOcean, no fallback, referee FINISHED. The learned
single-gate **BC-v1** does *not* generalise to multi-gate on its own; the `hybrid_gate_controller`
integrates BC-v1's visual servo on top of the deterministic backbone. Metric visual gate-pose
(observation v2) is now an **optional** improvement, not a blocker — it remains unreliable on real
images and is documented honestly below.

## Layout

```
visual_pose_v2/
  vision_pose/            # corrected detector validation (canonical yaw, per-track aperture,
                          #   stationary+moving, modulo-180 error) -> pose_metrics.json
  multigate_v1/           # BC-v1 multi-gate diagnostic (first-failure classification)
  hybrid_controller/      # hybrid config + multi-gate + circuit results
  official_no_current/    # THE DELIVERABLE: 3 circuits current-free (summary.json/.csv + per-circuit)
  experiment_manifest.json
```

## 1. Corrected visual-pose validation (`vision_pose/pose_metrics.json`)

Four corrections were applied before judging geometric vision (see `docs/visual_gate_pose.md`):

- **canonical gate-plane yaw** — a frontal gate's normal faces the camera, so the previous
  formula read ~180 deg, not 0. `canonical_gate_plane_yaw` orients the normal toward the camera
  and wraps to `[-90, 90]` (0 = frontal); error uses the modulo-180 `plane_angle_distance_deg`,
  and a separate `orientation_class` gives frontal/left/right accuracy;
- **per-track aperture size** — the yaw validation tracks are **2.0 m**, not the 1.5 m the
  detector hard-coded (which mis-scaled every distance by 0.75); the official gates are 1.5 m;
- **one simulator step per captured frame** (the capture previously stepped twice per frame,
  skewing frame/ground-truth alignment);
- **stationary and moving** capture modes to separate perception from motion.

Even corrected, the deterministic detector is **not yet reliable on real images** (aperture-corner
detection on thin, hazy, distant real gates rarely yields a clean quadrilateral, so metric yaw is
untrustworthy). Distance/lateral are usable when a quad is found. This is why observation v2 is not
wired into a policy — but it is no longer on the critical path (below).

## 2. Multi-gate diagnostic (`multigate_v1/`)

`multigate_diagnostic.py` runs an onboard controller through the real runner and classifies the
first failure from the tracker + referee timeline. **BC-v1 does not do multi-gate**: across the
two-gate straight, two-gate curve and three-gate S-curve it completes **0/8** and reaches at most
1 gate, failing by collisions, out-of-bounds and return-to-previous-gate (it passes gate 1 but
cannot turn to acquire gate 2). The **`hybrid_gate_controller`** (`hybrid_controller/`), which
blends BC-v1's visual servo onto the deterministic backbone, completes the **two-gate 2/2**,
**three-gate 2/2** and the **Horseshoe Bay circuit 2/2** current-free. See
`docs/multigate_curriculum.md`.

## 3. Current-free official circuits (`official_no_current/`)

Fresh verification from this branch's committed SHA (see `summary.json` / `summary.csv` and the
per-circuit `eval_summary.json` + `evaluation_manifest.json`). Every manifest records
`adapter_actual = holoocean`, `fallback_allowed = false`, `current_profile = none`,
`currents.currents_actual = [0,0,0]`, the controller, the track sha256 and the code git_sha.
`mixed_endurance` additionally records `benchmark_task_override = clean_gate` (a `current_gate`
task cannot be run current-free otherwise; geometry/gates/laps/referee unchanged). See
`docs/final_no_current_demo.md`.

The frozen benchmark's independent clean matrix corroborates:
`rule_gate_center_then_commit` = Horseshoe 5/5, Vertical 4/5, Mixed 5/5.

## Nothing frozen was modified

`main` at f4d6375, the BC-v1 model hash, the 78-run matrix, the frozen A/B evaluations, the
Stage-1/2 public results and the onboard-only validation matrix are all untouched. New results are
written only under `results/rl/` (git-ignored) and this `results/rl_public/visual_pose_v2/` package.
