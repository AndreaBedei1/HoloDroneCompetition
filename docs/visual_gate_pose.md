# Visual Gate-Pose Front End (observation v2)

## Why center-only vision is ambiguous

Observation **v1** (`onboard_only_v1`, 36 features) describes the gate visually with only
five numbers: `vision_present`, `vision_center_x`, `vision_center_y`, `vision_area_fraction`,
`vision_confidence`. The deterministic detector internally also computes the apparent width
and height, but the policy never receives them.

Center-only vision cannot distinguish several physically different situations that look
similar:

- a gate that is **centered in the image but approached obliquely** (the gate plane is
  rotated relative to the camera) vs. a gate approached head-on;
- **lateral displacement** (the rover is off to one side, gate plane frontal) vs. **yaw
  misalignment** (the rover points away, but is centered on the gate);
- how far the gate plane is **rotated left or right**.

The frozen BC-v1 model therefore treats "the gate is to the right in the image" and "the
gate plane is rotated" as the same cue, which is exactly where it failed at the extreme
corners of the Stage-2 randomization envelope (large lateral + yaw combined). This is the
motivation for observation v2.

Observation v1 remains **fully supported and frozen** — no v1 model or result changes.

## How gate-plane pose is estimated (`controllers/gate_pose.py`)

Onboard, from the FrontCamera image only (never simulator pose or world geometry):

1. **Aperture corners.** The gate is a square frame with a known **1.5 x 1.5 m** inner
   opening. The detector thresholds the bright bars, finds contours (`RETR_CCOMP`), and
   takes the frame's **inner hole** — the aperture — approximated to a quadrilateral. The
   four corners are validated (convexity, area, aspect, side-length consistency,
   non-self-intersection) and ordered canonically (top-left, top-right, bottom-right,
   bottom-left). If no valid quad is found, the pipeline degrades to the center-only v1
   detector.
2. **Camera intrinsics.** Built from the configured camera (640 x 480, horizontal FOV 90 deg):
   `fx = fy = width / (2 tan(FOV/2)) = 320`, `cx = 320`, `cy = 240`.
3. **Planar PnP.** `cv2.solvePnPGeneric(..., SOLVEPNP_IPPE)` on the four corners with the
   known square model gives up to two planar solutions. The **square-planar sign ambiguity**
   is resolved with the unambiguous **bar-height ratio** (which side bar appears taller) plus
   the **reprojection error** (which cleanly separates the correct solution from its mirror
   when there is perspective). Non-finite / negative-depth solutions are rejected.
4. **Fallbacks.** When PnP is unavailable or unstable, a **projective** stage still reports
   frontal / rotated-left / rotated-right from the bar-height ratio and skew, and a
   **known-size** distance estimate (`z = fy * 1.5 / aperture_pixel_height`) plus the center
   offset gives a metric-ish translation. So the project never blocks on perfect PnP.
5. **Temporal filter** (`GatePoseTracker`). A controller-side EMA over past detections only,
   with angular wrapping for yaw/pitch and staleness expiry, stabilizes the single-frame
   estimate and never keeps a stale pose indefinitely.

Coordinate convention (OpenCV): +x right, +y down, +z forward. Reported translation is
`(lateral_x, vertical_y, forward_z)`; gate-plane yaw is the rotation about the camera
vertical (0 = frontal, sign disambiguated by the bar-height ratio). The minimum useful
output — even without metric PnP — is a stable **frontal / rotated-left / rotated-right**.

## What enters observation v2 (`onboard_visual_pose_v2`)

The 36 v1 features are preserved unchanged; 12 pose features are appended (masked when
absent — ground truth is never substituted):

| feature | meaning |
| --- | --- |
| `pose_present` | 1 if a metric/So-projective pose is available |
| `gate_lateral_error_norm` | lateral translation / normalization scale |
| `gate_vertical_error_norm` | vertical translation / scale |
| `gate_forward_distance_norm` | forward distance / scale |
| `gate_yaw_sin`, `gate_yaw_cos` | gate-plane yaw (wrapped) |
| `gate_pitch_sin`, `gate_pitch_cos` | gate-plane pitch (wrapped) |
| `gate_reprojection_error_norm` | PnP reprojection error / scale (pose quality) |
| `vision_width_fraction`, `vision_height_fraction` | apparent aperture size |
| `vision_quadrilateral_skew` | horizontal skew of the quad (oblique cue) |

## Onboard vs. evaluation-only

Everything above uses **only** the camera image and the public gate aperture size. The
simulator gate pose, global vehicle position, referee target and referee gate-passed
notifications are **never** controller inputs. Ground truth is used only offline to score
the detector (`vision_pose_capture.py`), to make synthetic labels, or in evaluation-only
tests.

See `results/rl_public/visual_pose_v2/vision_pose/` for the real-HoloOcean detector metrics.
