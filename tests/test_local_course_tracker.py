"""Robustness tests for the controller-side LocalCourseTracker.

Synthetic packet/image/DVL sequences drive the tracker through true passages,
near misses, sensor dropouts and false detections, asserting that it advances
exactly once per confirmed passage and never on partial evidence.
"""

from __future__ import annotations

import math
from typing import List, Optional

import pytest

from marine_race_arena.controllers.local_course_tracker import (
    PHASE_ADVANCE,
    PHASE_APPROACH,
    PHASE_COMMIT,
    PHASE_FINISHED,
    PHASE_SEARCH,
    PHASE_VERIFY_EXIT,
    PHASE_VISUAL_ALIGN,
    LocalCourseTracker,
    LocalCourseTrackerConfig,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

pytestmark = pytest.mark.skipif(np is None, reason="numpy required for synthetic camera frames")

DT = 0.1

GATE_COLOR = (40, 235, 150)
WATER_COLOR = (10, 32, 58)


def make_tracker(total=3, laps=1, **overrides) -> LocalCourseTracker:
    config = LocalCourseTrackerConfig(**overrides) if overrides else LocalCourseTrackerConfig()
    return LocalCourseTracker(
        initial_beacon_id="B01",
        total_beacons=total,
        laps=laps,
        config=config,
    )


def gate_frame(
    center_x_px=80,
    center_y_px=None,
    half=30,
    thickness=6,
    width=160,
    height=120,
):
    """A centered synthetic gate ring like the fallback camera would render."""
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :] = WATER_COLOR
    cy = height // 2 if center_y_px is None else center_y_px
    x0, x1 = center_x_px - half, center_x_px + half
    y0, y1 = cy - half, cy + half
    image[max(0, y0):y0 + thickness, max(0, x0):x1] = GATE_COLOR
    image[y1 - thickness:min(height, y1), max(0, x0):x1] = GATE_COLOR
    image[max(0, y0):y1, max(0, x0):x0 + thickness] = GATE_COLOR
    image[max(0, y0):y1, x1 - thickness:min(width, x1)] = GATE_COLOR
    return image


def empty_frame(width=160, height=120):
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :] = WATER_COLOR
    return image


def holoocean_like_frame(with_gate=True, width=640, height=480):
    """Yellow water/floor scene resembling the real OpenWater camera image."""
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:150, :] = (8, 3, 0)
    image[150:230, :] = (150, 108, 24)
    image[230:, :] = (202, 170, 76)
    if with_gate:
        color = (118, 82, 21)
        image[151:166, 215:425] = color
        image[326:341, 215:425] = color
        image[151:341, 215:235] = color
        image[151:341, 405:425] = color
    return image


def packet(beacon_id: str, range_m: float, t: float, bearing: float = 0.0):
    return {
        "beacon_id": beacon_id,
        "bearing_deg": bearing,
        "elevation_deg": 2.0,
        "range_m": range_m,
        "signal_strength": max(0.0, 1.0 - range_m / 90.0),
        "received_at_s": t,
    }


class Scenario:
    """Feed scripted per-step inputs into a tracker and record advancements."""

    def __init__(self, tracker: LocalCourseTracker):
        self.tracker = tracker
        self.t = 0.0
        self.advancements = 0

    def step(
        self,
        *,
        range_m: Optional[float] = None,
        bearing: float = 0.0,
        camera=None,
        dvl_vx: float = 0.0,
        beacon_id: Optional[str] = None,
        extra_packets: Optional[List[dict]] = None,
    ):
        self.t += DT
        beacons = []
        if range_m is not None:
            beacons.append(
                packet(beacon_id or self.tracker.expected_beacon_id, range_m, self.t, bearing)
            )
        if extra_packets:
            beacons.extend(extra_packets)
        result = self.tracker.update(
            local_time_s=self.t,
            beacons=beacons,
            camera_image=camera,
            dvl_velocity=[dvl_vx, 0.0, 0.0],
        )
        if result.just_advanced:
            self.advancements += 1
        return result

    def run_true_passage(self, start_range: float = 3.4, speed: float = 0.35):
        """Drive a nominal approach, commit, passage and exit confirmation."""
        range_m = start_range
        # Approach with a centered visual until COMMIT.
        for _ in range(400):
            self.step(range_m=range_m, camera=gate_frame(half=int(28 + 26 * (3.4 - range_m))), dvl_vx=speed)
            if self.tracker.phase == PHASE_COMMIT:
                break
            range_m = max(1.6, range_m - speed * DT)
        assert self.tracker.phase == PHASE_COMMIT, f"never committed (phase={self.tracker.phase})"
        # Push through the plane: range shrinks to a minimum then rises; the
        # gate leaves the camera view after the pass.
        profile = [1.4, 1.1, 0.8, 0.55, 0.42, 0.5, 0.75, 1.0, 1.3, 1.55, 1.8, 2.1, 2.4]
        for index, r in enumerate(profile):
            camera = gate_frame(half=55) if index < 3 else empty_frame()
            bearing = 175.0 if index >= 5 else 0.0
            for _ in range(3):
                result = self.step(
                    range_m=r,
                    bearing=bearing,
                    camera=camera,
                    dvl_vx=speed + 0.1,
                )
                if result.just_advanced:
                    return result
        # A few extra confirming frames if needed.
        for _ in range(20):
            result = self.step(range_m=2.6, camera=empty_frame(), dvl_vx=speed)
            if result.just_advanced:
                return result
        raise AssertionError(
            f"true passage did not advance (phase={self.tracker.phase}, diag={self.tracker.diagnostics()})"
        )


def test_synthetic_gate_frame_is_detected():
    from marine_race_arena.controllers.vision import vision_targets_from_camera

    targets = vision_targets_from_camera(gate_frame())
    assert targets, "synthetic frame must be detectable"
    best = max(targets, key=lambda t: t.confidence)
    assert abs(best.center_x) < 0.15 and best.confidence > 0.4


def test_real_camera_style_dark_gate_frame_is_detected_without_selecting_background():
    from marine_race_arena.controllers.vision import (
        select_default_visual_target,
        vision_targets_from_camera,
    )

    target = select_default_visual_target(vision_targets_from_camera(holoocean_like_frame()))

    assert target is not None
    assert abs(target.center_x) < 0.08
    assert abs(target.center_y) < 0.12
    assert 0.05 < target.area_fraction < 0.20


def test_real_camera_style_background_alone_is_not_a_gate():
    from marine_race_arena.controllers.vision import vision_targets_from_camera

    targets = vision_targets_from_camera(holoocean_like_frame(with_gate=False))

    assert not [target for target in targets if target.confidence >= 0.35]


def test_visual_association_prefers_expected_beacon_side_over_centered_next_gate():
    from marine_race_arena.controllers.vision import VisionTarget, select_visual_target_for_beacon

    expected_left = VisionTarget(
        center_x=-0.42,
        center_y=0.02,
        confidence=0.72,
        area_fraction=0.07,
        width_fraction=0.25,
        height_fraction=0.28,
    )
    centered_next_gate = VisionTarget(
        center_x=0.0,
        center_y=0.0,
        confidence=0.96,
        area_fraction=0.10,
        width_fraction=0.30,
        height_fraction=0.34,
    )

    selected = select_visual_target_for_beacon(
        [centered_next_gate, expected_left],
        bearing_deg=24.0,
        range_m=2.8,
    )

    assert selected is expected_left


def test_visual_association_rejects_front_gate_when_expected_beacon_is_behind():
    from marine_race_arena.controllers.vision import VisionTarget, select_visual_target_for_beacon

    gate_ahead = VisionTarget(
        center_x=-0.35,
        center_y=0.0,
        confidence=0.98,
        area_fraction=0.12,
        width_fraction=0.35,
        height_fraction=0.36,
    )

    assert select_visual_target_for_beacon(
        [gate_ahead], bearing_deg=178.0, range_m=0.5
    ) is None


def test_visual_association_rejects_tiny_next_gate_at_close_expected_range():
    from marine_race_arena.controllers.vision import VisionTarget, select_visual_target_for_beacon

    next_gate = VisionTarget(
        center_x=0.0,
        center_y=0.0,
        confidence=0.98,
        area_fraction=0.02,
        width_fraction=0.15,
        height_fraction=0.16,
    )

    assert select_visual_target_for_beacon(
        [next_gate], bearing_deg=1.0, range_m=1.5
    ) is None


def test_initial_visual_association_rejects_gate_far_from_expected_beacon():
    from marine_race_arena.controllers.vision import VisionTarget, select_visual_target_for_beacon

    later_gate = VisionTarget(
        center_x=-0.87,
        center_y=-0.66,
        confidence=0.99,
        area_fraction=0.08,
        width_fraction=0.28,
        height_fraction=0.30,
    )
    expected_gate = VisionTarget(
        center_x=0.0,
        center_y=0.02,
        confidence=0.62,
        area_fraction=0.05,
        width_fraction=0.22,
        height_fraction=0.24,
    )

    assert select_visual_target_for_beacon(
        [later_gate, expected_gate], bearing_deg=0.0, range_m=4.5
    ) is expected_gate


def test_default_visual_association_prefers_nearest_apparent_gate() -> None:
    from marine_race_arena.controllers.vision import (
        VisionTarget,
        select_default_visual_target,
    )

    near_gate = VisionTarget(
        center_x=0.34, center_y=0.02, confidence=0.72,
        area_fraction=0.14, width_fraction=0.38, height_fraction=0.40,
    )
    centered_far_gate = VisionTarget(
        center_x=0.0, center_y=0.0, confidence=0.96,
        area_fraction=0.025, width_fraction=0.15, height_fraction=0.16,
    )

    assert select_default_visual_target([centered_far_gate, near_gate]) is near_gate


def test_temporal_visual_tracker_confirms_smooths_and_bridges_short_flicker() -> None:
    from marine_race_arena.controllers.vision import TemporalVisionTracker, VisionTarget

    tracker = TemporalVisionTracker(max_missed_frames=2)

    def detection(x: float, area: float = 0.10) -> VisionTarget:
        return VisionTarget(
            center_x=x, center_y=0.02, confidence=0.90,
            area_fraction=area, width_fraction=0.30, height_fraction=0.32,
        )

    proposal = tracker.update([detection(0.10)], 0.0, 5.0)
    locked = tracker.update([detection(0.14, 0.11)], 0.0, 4.8)

    assert proposal is not None and proposal.confidence < 0.38
    assert locked is not None and locked.confidence > 0.60
    assert tracker.locked
    assert 0.10 < locked.center_x < 0.14

    first_gap = tracker.update([], 0.0, 4.7)
    second_gap = tracker.update([], 0.0, 4.6)
    expired = tracker.update([], 0.0, 4.5)

    assert first_gap is not None and first_gap.predicted
    assert second_gap is not None and second_gap.predicted
    assert first_gap.confidence > second_gap.confidence
    assert expired is None


def test_temporal_visual_tracker_drops_lock_in_rear_sector() -> None:
    from marine_race_arena.controllers.vision import TemporalVisionTracker, VisionTarget

    tracker = TemporalVisionTracker()
    gate_ahead = VisionTarget(0.0, 0.0, 0.95, 0.10, 0.30, 0.32)
    assert tracker.update([gate_ahead], 0.0, 2.0) is not None
    assert tracker.update([gate_ahead], 0.0, 1.8) is not None
    assert tracker.locked

    assert tracker.update([gate_ahead], 178.0, 0.5) is None
    assert not tracker.locked


def test_temporal_visual_tracker_does_not_jump_to_smaller_centered_gate() -> None:
    from marine_race_arena.controllers.vision import TemporalVisionTracker, VisionTarget

    tracker = TemporalVisionTracker()
    near = VisionTarget(0.28, 0.0, 0.82, 0.14, 0.38, 0.40)
    tracker.update([near], bearing_deg=-15.0, range_m=4.0)
    tracker.update([near], bearing_deg=-15.0, range_m=3.8)

    continued_near = VisionTarget(0.25, 0.0, 0.75, 0.16, 0.41, 0.43)
    centered_far = VisionTarget(0.0, 0.0, 0.99, 0.025, 0.15, 0.16)
    selected = tracker.update(
        [centered_far, continued_near], bearing_deg=-14.0, range_m=3.5
    )

    assert selected is not None
    assert selected.center_x > 0.20
    assert selected.area_fraction > 0.10


def test_starts_in_search_on_initial_beacon():
    tracker = make_tracker()
    assert tracker.expected_beacon_id == "B01"
    assert tracker.phase == PHASE_SEARCH
    assert tracker.local_completed == 0
    assert tracker.local_lap == 1


def test_beacon_packet_moves_search_to_approach():
    scenario = Scenario(make_tracker())
    result = scenario.step(range_m=8.0)
    assert result.phase == PHASE_APPROACH
    assert result.beacon is not None and result.beacon.beacon_id == "B01"


def test_ignores_packets_from_other_beacons():
    scenario = Scenario(make_tracker())
    result = scenario.step(range_m=None, extra_packets=[packet("B03", 5.0, 0.1)])
    assert result.phase == PHASE_SEARCH
    assert result.beacon is None


def test_visual_detection_enters_visual_align():
    scenario = Scenario(make_tracker())
    result = scenario.step(range_m=6.0, camera=gate_frame())
    assert result.phase == PHASE_VISUAL_ALIGN


def test_true_passage_advances_exactly_once():
    scenario = Scenario(make_tracker())
    result = scenario.run_true_passage()
    assert result.just_advanced
    assert scenario.tracker.expected_beacon_id == "B02"
    assert scenario.tracker.local_completed == 1
    assert scenario.tracker.phase == PHASE_ADVANCE
    # Repeated post-advance frames must not double count.
    for _ in range(60):
        scenario.step(range_m=3.0, camera=empty_frame(), dvl_vx=0.3)
    assert scenario.advancements == 1
    assert scenario.tracker.local_completed == 1


def test_progression_b01_to_b02_to_b03_and_finish():
    scenario = Scenario(make_tracker(total=2))
    scenario.run_true_passage()
    assert scenario.tracker.expected_beacon_id == "B02"
    # Wait out the exit clearance, then pass the final gate.
    for _ in range(int(3.0 / DT)):
        scenario.step(range_m=6.0, camera=empty_frame(), dvl_vx=0.3)
    scenario.run_true_passage()
    assert scenario.tracker.finished
    assert scenario.tracker.phase == PHASE_FINISHED
    assert scenario.tracker.local_completed == 2
    # Finished tracker stays finished and never advances again.
    result = scenario.step(range_m=2.0, camera=gate_frame(), dvl_vx=0.4)
    assert result.finished and scenario.advancements == 2


def test_multi_lap_wraps_back_to_b01():
    scenario = Scenario(make_tracker(total=2, laps=2))
    scenario.run_true_passage()
    for _ in range(int(3.0 / DT)):
        scenario.step(range_m=6.0, camera=empty_frame(), dvl_vx=0.3)
    scenario.run_true_passage()
    assert not scenario.tracker.finished
    assert scenario.tracker.expected_beacon_id == "B01"
    assert scenario.tracker.local_lap == 2
    for _ in range(int(3.0 / DT)):
        scenario.step(range_m=6.0, camera=empty_frame(), dvl_vx=0.3)
    scenario.run_true_passage()
    for _ in range(int(3.0 / DT)):
        scenario.step(range_m=6.0, camera=empty_frame(), dvl_vx=0.3)
    scenario.run_true_passage()
    assert scenario.tracker.finished
    assert scenario.tracker.local_completed == 4


def test_temporary_camera_loss_does_not_advance():
    scenario = Scenario(make_tracker())
    # Reach COMMIT with a stable centered view.
    for _ in range(200):
        scenario.step(range_m=3.0, camera=gate_frame(half=40), dvl_vx=0.2)
        if scenario.tracker.phase == PHASE_COMMIT:
            break
    assert scenario.tracker.phase == PHASE_COMMIT
    # One missing camera frame plus continued approach: no advancement.
    scenario.step(range_m=2.6, camera=None, dvl_vx=0.3)
    assert scenario.advancements == 0
    assert scenario.tracker.local_completed == 0


def test_single_beacon_dropout_does_not_advance():
    scenario = Scenario(make_tracker())
    for _ in range(30):
        scenario.step(range_m=5.0, camera=gate_frame(), dvl_vx=0.2)
    scenario.step(range_m=None, camera=gate_frame(), dvl_vx=0.2)  # dropout
    assert scenario.advancements == 0
    assert scenario.tracker.expected_beacon_id == "B01"


def test_false_visual_detection_alone_does_not_advance():
    scenario = Scenario(make_tracker())
    # A visual lock with no beacon in the commit envelope: never commits.
    for _ in range(120):
        scenario.step(range_m=8.0, camera=gate_frame(half=45), dvl_vx=0.3)
    assert scenario.tracker.phase in (PHASE_VISUAL_ALIGN, PHASE_APPROACH)
    assert scenario.advancements == 0


def test_centered_image_does_not_commit_with_oblique_beacon_bearing():
    scenario = Scenario(make_tracker())

    for _ in range(20):
        scenario.step(range_m=2.4, bearing=12.0, camera=gate_frame(half=40), dvl_vx=0.2)

    assert scenario.tracker.phase == PHASE_VISUAL_ALIGN
    for _ in range(6):
        scenario.step(range_m=2.4, bearing=2.0, camera=gate_frame(half=40), dvl_vx=0.2)
    assert scenario.tracker.phase == PHASE_COMMIT


def test_close_range_partial_gate_enters_commit_from_camera_and_beacon_only():
    scenario = Scenario(make_tracker())
    partial_close_gate = gate_frame(center_y_px=88, half=25)

    scenario.step(
        range_m=1.3,
        bearing=1.0,
        camera=partial_close_gate,
        dvl_vx=0.2,
    )
    assert scenario.tracker.phase == PHASE_VISUAL_ALIGN
    # The first frame above establishes VISUAL_ALIGN.  Two subsequent
    # qualifying frames are required; a single frame cannot trigger rescue.
    result = scenario.step(
        range_m=1.3,
        bearing=1.0,
        camera=partial_close_gate,
        dvl_vx=0.2,
    )
    assert result.phase == PHASE_VISUAL_ALIGN
    assert scenario.tracker.diagnostics()["close_commit_streak"] == 1
    result = scenario.step(
        range_m=1.3,
        bearing=1.0,
        camera=partial_close_gate,
        dvl_vx=0.2,
    )

    assert result.phase == PHASE_COMMIT
    assert scenario.tracker.local_completed == 0
    evidence = scenario.tracker.diagnostics()["last_commit_entry_evidence"]
    assert evidence["reason"] == "close_range_rescue"
    assert evidence["beacon_id"] == "B01"
    assert evidence["rescue_range_rise_m"] <= 0.35


def test_close_range_commit_rescue_requires_consecutive_frames():
    scenario = Scenario(make_tracker())
    partial_close_gate = gate_frame(center_y_px=88, half=25)

    scenario.step(range_m=1.3, bearing=1.0, camera=partial_close_gate)
    scenario.step(range_m=1.3, bearing=1.0, camera=partial_close_gate)
    assert scenario.tracker.diagnostics()["close_commit_streak"] == 1
    scenario.step(range_m=1.3, bearing=1.0, camera=empty_frame())
    assert scenario.tracker.diagnostics()["close_commit_streak"] == 0
    scenario.step(range_m=1.3, bearing=1.0, camera=partial_close_gate)
    assert scenario.tracker.phase == PHASE_VISUAL_ALIGN
    scenario.step(range_m=1.3, bearing=1.0, camera=partial_close_gate)

    assert scenario.tracker.phase == PHASE_COMMIT


def test_close_range_commit_rescue_rejects_post_minimum_recrossing():
    scenario = Scenario(make_tracker())
    partial_close_gate = gate_frame(center_y_px=88, half=25)

    # Observe a close first approach without a usable image, then move away.
    scenario.step(range_m=0.8, bearing=0.0, camera=empty_frame())
    for _ in range(8):
        scenario.step(
            range_m=1.3,
            bearing=1.0,
            camera=partial_close_gate,
            dvl_vx=0.2,
        )

    diagnostics = scenario.tracker.diagnostics()
    assert diagnostics["rescue_min_range_m"] == pytest.approx(0.8)
    assert diagnostics["rescue_range_rise_m"] == pytest.approx(0.5)
    assert scenario.tracker.phase == PHASE_VISUAL_ALIGN
    assert scenario.tracker.local_completed == 0


@pytest.mark.parametrize(
    ("range_m", "bearing"),
    ((2.0, 1.0), (1.3, 12.0)),
)
def test_close_range_commit_rescue_rejects_far_or_oblique_beacon(
    range_m: float,
    bearing: float,
):
    scenario = Scenario(make_tracker())
    partial_close_gate = gate_frame(center_y_px=88, half=25)

    for _ in range(8):
        scenario.step(
            range_m=range_m,
            bearing=bearing,
            camera=partial_close_gate,
            dvl_vx=0.2,
        )

    assert scenario.tracker.phase == PHASE_VISUAL_ALIGN
    assert scenario.tracker.local_completed == 0


def _local_transition_tracker(total=2):
    """Tracker thresholds used by onboard_local_transition_v1."""
    from marine_race_arena.learning.tracker_context_local_transition import (
        LOCAL_TRANSITION_TRACKER_CONFIG,
    )

    return LocalCourseTracker(
        initial_beacon_id="B01",
        total_beacons=total,
        config=LOCAL_TRANSITION_TRACKER_CONFIG,
    )


def test_local_transition_proximity_commit_accepts_cropped_offcenter_gate():
    scenario = Scenario(_local_transition_tracker())
    cropped = gate_frame(center_x_px=115, center_y_px=95, half=27)

    # The first observation enters VISUAL_ALIGN.  The following two camera +
    # close expected-beacon observations establish the local-only rescue.
    for _ in range(3):
        result = scenario.step(
            range_m=0.8, bearing=10.0, camera=cropped, dvl_vx=0.2
        )

    assert result.phase == PHASE_COMMIT
    evidence = scenario.tracker.diagnostics()["last_commit_entry_evidence"]
    assert evidence["reason"] == "proximity_passage_rescue"
    assert abs(evidence["visual_center_x"]) > 0.30


def test_local_transition_proximity_commit_requires_consecutive_camera_evidence():
    scenario = Scenario(_local_transition_tracker())
    cropped = gate_frame(center_x_px=115, center_y_px=95, half=27)

    scenario.step(range_m=0.8, bearing=10.0, camera=cropped)
    scenario.step(range_m=0.8, bearing=10.0, camera=cropped)
    assert scenario.tracker.diagnostics()["proximity_commit_streak"] == 1
    scenario.step(range_m=0.8, bearing=10.0, camera=empty_frame())
    assert scenario.tracker.diagnostics()["proximity_commit_streak"] == 0
    scenario.step(range_m=0.8, bearing=10.0, camera=cropped)

    assert scenario.tracker.phase == PHASE_VISUAL_ALIGN


def test_local_transition_cropped_true_passage_advances_from_onboard_evidence():
    scenario = Scenario(_local_transition_tracker())
    cropped = gate_frame(center_x_px=115, center_y_px=95, half=27)
    for _ in range(3):
        scenario.step(range_m=0.8, bearing=10.0, camera=cropped, dvl_vx=0.2)
    assert scenario.tracker.phase == PHASE_COMMIT

    # Establish the tight passage envelope and enough DVL displacement, then
    # require two fresh rear-sector packets plus range turnaround and visual
    # disappearance before advancing.
    scenario.step(range_m=0.55, bearing=0.0, camera=empty_frame(), dvl_vx=0.5)
    scenario.step(range_m=0.50, bearing=0.0, camera=empty_frame(), dvl_vx=0.5)
    scenario.step(range_m=0.50, bearing=0.0, camera=empty_frame(), dvl_vx=0.5)
    assert scenario.tracker.phase == PHASE_VERIFY_EXIT
    scenario.step(range_m=0.90, bearing=100.0, camera=empty_frame(), dvl_vx=0.4)
    result = scenario.step(
        range_m=0.90, bearing=100.0, camera=empty_frame(), dvl_vx=0.4
    )

    assert result.just_advanced
    assert scenario.tracker.expected_beacon_id == "B02"


def test_local_transition_proximity_near_miss_outside_tight_envelope_never_advances():
    """A transit that cannot have gone through the aperture must not advance.

    The distance is measured to the beacon, which sits 0.35 m up the gate's own
    up-axis rather than at the centre of the 1.5 m aperture. The far corner of
    a legitimate pass is therefore hypot(0.75, 0.75 + 0.35) = 1.33 m from the
    beacon, and that -- not a hand-picked number -- is what separates a pass
    from a miss.

    This test used to place the miss at 0.80 m and assert no advance. Under the
    geometry above, 0.80 m from the beacon implies at most
    sqrt(0.80^2 - 0.35^2) = 0.72 m of lateral offset, which is *inside* the
    0.75 m half-aperture: the scenario was a valid edge pass being asserted to
    fail. Rejecting it is what stalled the observation tracker on Vertical
    Serpent and Mixed Endurance, permanently, after a single rejected passage.

    A scalar range to an off-centre beacon cannot fully separate a pass from a
    miss -- a 0.9 m lateral miss sits at 0.97 m, inside the envelope. The
    envelope is the outer bound, and the exit evidence below it does the rest.
    So the miss is now placed where it is unambiguously outside.
    """
    scenario = Scenario(_local_transition_tracker())
    cropped = gate_frame(center_x_px=115, center_y_px=95, half=27)
    for _ in range(3):
        scenario.step(range_m=0.8, bearing=10.0, camera=cropped, dvl_vx=0.2)
    assert scenario.tracker.phase == PHASE_COMMIT

    # 1.6 m from the beacon: no point inside the aperture is that far away.
    for _ in range(4):
        scenario.step(range_m=1.60, bearing=0.0, camera=empty_frame(), dvl_vx=0.5)
    for _ in range(6):
        scenario.step(range_m=2.10, bearing=120.0, camera=empty_frame(), dvl_vx=0.5)

    assert scenario.advancements == 0
    assert scenario.tracker.local_completed == 0
    assert not scenario.tracker.diagnostics()["close_range_confirmed"]


def test_local_transition_envelope_admits_every_pass_through_the_aperture():
    """The envelope must not reject a transit that really went through.

    The counterpart to the test above: the corner of the aperture diagonally
    opposite the beacon is the hardest legitimate case, and it has to be
    admitted. A tracker that rejects it has no way back -- it keeps waiting for
    a gate the rover has already left behind.
    """
    import math

    from marine_race_arena.learning.tracker_context_local_transition import (
        BEACON_UP_OFFSET_M,
        GATE_APERTURE_HALF_SIZE_M,
        LOCAL_TRANSITION_TRACKER_CONFIG,
    )

    worst_legitimate_pass_m = math.hypot(
        GATE_APERTURE_HALF_SIZE_M, GATE_APERTURE_HALF_SIZE_M + BEACON_UP_OFFSET_M
    )
    assert LOCAL_TRANSITION_TRACKER_CONFIG.min_range_for_passage_m >= (
        worst_legitimate_pass_m
    )


def test_commit_without_dvl_displacement_does_not_advance():
    scenario = Scenario(make_tracker(commit_timeout_s=1.0))
    for _ in range(200):
        scenario.step(range_m=3.0, camera=gate_frame(half=40), dvl_vx=0.2)
        if scenario.tracker.phase == PHASE_COMMIT:
            break
    assert scenario.tracker.phase == PHASE_COMMIT
    # Blocked vehicle: no forward velocity, gate vanishes, range rises (drift).
    for r in (3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8):
        for _ in range(2):
            scenario.step(range_m=r, camera=empty_frame(), dvl_vx=0.0)
    assert scenario.advancements == 0
    assert scenario.tracker.local_completed == 0
    assert scenario.tracker.phase in (PHASE_SEARCH, PHASE_APPROACH, PHASE_VISUAL_ALIGN)


def test_range_increase_without_close_approach_does_not_advance():
    scenario = Scenario(make_tracker(verify_timeout_s=2.0))
    for _ in range(200):
        scenario.step(range_m=3.1, camera=gate_frame(half=40), dvl_vx=0.2)
        if scenario.tracker.phase == PHASE_COMMIT:
            break
    assert scenario.tracker.phase == PHASE_COMMIT
    # Forward displacement accumulates but the range never gets below the
    # passage envelope before rising: a pass far beside the gate.
    for r in (3.1, 3.0, 2.9, 2.8, 2.9, 3.1, 3.4, 3.7, 4.0, 4.4, 4.8):
        for _ in range(3):
            scenario.step(range_m=r, camera=empty_frame(), dvl_vx=0.5)
    assert scenario.advancements == 0
    assert scenario.tracker.local_completed == 0


def test_collision_like_near_miss_does_not_advance_without_close_envelope():
    scenario = Scenario(make_tracker())
    # Establish a valid centered COMMIT, then reproduce the evidence shape of
    # a rebound: forward DVL displacement, a 1.47 m closest approach, range
    # rise and visual disappearance.  It is close, but not an aperture pass.
    for _ in range(80):
        scenario.step(range_m=2.6, camera=gate_frame(half=45), dvl_vx=0.4)
        if scenario.tracker.phase == PHASE_COMMIT:
            break
    assert scenario.tracker.phase == PHASE_COMMIT

    for range_m in (2.2, 1.8, 1.55, 1.47, 1.55, 1.8, 2.2, 2.6, 3.0):
        for _ in range(4):
            scenario.step(range_m=range_m, camera=empty_frame(), dvl_vx=0.5)

    assert scenario.advancements == 0
    assert scenario.tracker.local_completed == 0


def test_single_low_range_outlier_cannot_finish_while_beacon_is_still_ahead():
    scenario = Scenario(make_tracker(total=1))
    for _ in range(80):
        scenario.step(range_m=2.6, camera=gate_frame(half=45), dvl_vx=0.4)
        if scenario.tracker.phase == PHASE_COMMIT:
            break
    assert scenario.tracker.phase == PHASE_COMMIT

    # Reproduces the seed-2 B12 evidence shape: one rare low range packet
    # followed by a 0.95 m rise while the same beacon remains straight ahead.
    ranges = (
        1.474,
        1.335,
        1.722,
        1.392,
        1.581,
        1.703,
        1.354,
        1.306,
        1.078,
        1.421,
        1.045,
        0.827,
        0.782,
        0.911,
        0.080,
        0.956,
        1.030,
    )
    for range_m in ranges:
        scenario.step(
            range_m=range_m,
            bearing=-1.0,
            camera=empty_frame(),
            dvl_vx=1.5,
        )

    assert scenario.advancements == 0
    assert scenario.tracker.phase == PHASE_VERIFY_EXIT
    assert scenario.tracker.diagnostics()["rear_bearing_confirmed"] is False

    scenario.step(range_m=1.1, bearing=170.0, camera=empty_frame(), dvl_vx=1.0)
    assert scenario.advancements == 0
    scenario.step(range_m=1.1, bearing=0.0, camera=empty_frame(), dvl_vx=1.0)
    assert scenario.tracker.diagnostics()["rear_bearing_streak"] == 0
    scenario.step(range_m=1.1, bearing=170.0, camera=empty_frame(), dvl_vx=1.0)
    assert scenario.advancements == 0
    scenario.step(range_m=1.1, bearing=170.0, camera=empty_frame(), dvl_vx=1.0)

    assert scenario.advancements == 1
    assert scenario.tracker.finished
    evidence = scenario.tracker.diagnostics()["last_advancement_evidence"]
    assert evidence["rear_bearing_confirmed"] is True
    assert evidence["rear_bearing_streak"] >= 2


def test_confirmed_rear_beacon_prevents_post_pass_camera_regression():
    scenario = Scenario(make_tracker(total=1))
    for _ in range(80):
        scenario.step(range_m=2.6, camera=gate_frame(half=45), dvl_vx=0.4)
        if scenario.tracker.phase == PHASE_COMMIT:
            break
    assert scenario.tracker.phase == PHASE_COMMIT

    for range_m in (1.25, 1.20, 1.15, 1.10):
        scenario.step(range_m=range_m, camera=empty_frame(), dvl_vx=2.0)
    scenario.tracker._last_commit_center_x = 0.55
    scenario.step(range_m=0.50, bearing=170.0, camera=empty_frame(), dvl_vx=2.0)
    scenario.step(range_m=0.60, bearing=170.0, camera=empty_frame(), dvl_vx=2.0)
    assert scenario.tracker.diagnostics()["rear_bearing_confirmed"] is True
    for _ in range(3):
        scenario.step(range_m=None, camera=empty_frame(), dvl_vx=2.0)

    result = scenario.step(
        range_m=2.20,
        bearing=170.0,
        camera=gate_frame(center_x_px=150, half=22),
        dvl_vx=2.0,
    )

    assert result.just_advanced
    assert scenario.tracker.finished


def _prepare_constrained_exit_scenario() -> Scenario:
    """Reach VERIFY_EXIT with all evidence except a large range turnaround."""
    scenario = Scenario(make_tracker(total=1))
    for _ in range(80):
        scenario.step(range_m=2.6, camera=gate_frame(half=45), dvl_vx=0.8)
        if scenario.tracker.phase == PHASE_COMMIT:
            break
    assert scenario.tracker.phase == PHASE_COMMIT
    for range_m in (1.2, 1.0, 0.7, 0.4):
        scenario.step(
            range_m=range_m,
            bearing=0.0,
            camera=empty_frame(),
            dvl_vx=2.0,
        )
    while scenario.tracker.phase != PHASE_VERIFY_EXIT:
        scenario.step(
            range_m=0.4,
            bearing=0.0,
            camera=empty_frame(),
            dvl_vx=2.0,
        )
    assert scenario.tracker.diagnostics()["close_range_confirmed"] is True
    return scenario


def test_persistent_rear_turnaround_confirms_constrained_true_passage():
    scenario = _prepare_constrained_exit_scenario()

    # Rear rotation alone is deliberately insufficient.  The smaller range
    # turnaround becomes eligible only after six fresh rear packets, then
    # requires three fresh consecutive samples at or above 0.35 m.
    for range_m in (0.55, 0.60, 0.65, 0.70, 0.72):
        scenario.step(
            range_m=range_m,
            bearing=125.0,
            camera=empty_frame(),
            dvl_vx=0.1,
        )
    assert scenario.advancements == 0
    for index, range_m in enumerate((0.76, 0.78, 0.80)):
        result = scenario.step(
            range_m=range_m,
            bearing=125.0,
            camera=empty_frame(),
            dvl_vx=0.1,
        )
        if index < 2:
            assert not result.just_advanced

    assert result.just_advanced
    assert scenario.tracker.finished
    evidence = scenario.tracker.diagnostics()["last_advancement_evidence"]
    assert evidence["range_rise_m"] < 0.9
    assert evidence["rear_exit_range_rise_confirmed"] is True


def test_rear_turnaround_below_recovery_margin_does_not_advance():
    scenario = _prepare_constrained_exit_scenario()

    for _ in range(30):
        scenario.step(
            range_m=0.70,
            bearing=125.0,
            camera=empty_frame(),
            dvl_vx=0.1,
        )

    assert scenario.advancements == 0
    assert scenario.tracker.diagnostics()["rear_bearing_confirmed"] is True
    assert scenario.tracker.diagnostics()["rear_exit_range_rise_confirmed"] is False


def test_replayed_packet_cannot_build_rear_or_turnaround_persistence():
    scenario = _prepare_constrained_exit_scenario()
    received_at_s = scenario.t + DT
    repeated = packet("B01", 0.80, received_at_s, bearing=125.0)

    for _ in range(20):
        scenario.t += DT
        scenario.tracker.update(
            local_time_s=scenario.t,
            beacons=[repeated],
            camera_image=empty_frame(),
            dvl_velocity=[0.1, 0.0, 0.0],
        )

    diagnostics = scenario.tracker.diagnostics()
    assert diagnostics["rear_bearing_streak"] == 1
    assert diagnostics["rear_exit_range_rise_streak"] == 0
    assert scenario.tracker.local_completed == 0


def test_rejected_range_jump_cannot_build_passage_persistence():
    scenario = _prepare_constrained_exit_scenario()

    scenario.step(
        range_m=5.0,
        bearing=125.0,
        camera=empty_frame(),
        dvl_vx=0.1,
    )

    diagnostics = scenario.tracker.diagnostics()
    assert diagnostics["rear_bearing_streak"] == 0
    assert diagnostics["rear_exit_range_rise_streak"] == 0
    assert scenario.tracker.local_completed == 0


def test_visual_disappearance_before_stable_lock_does_not_advance():
    scenario = Scenario(make_tracker())
    # Brief visual then loss before any commit: stays un-committed, no count.
    scenario.step(range_m=4.0, camera=gate_frame(), dvl_vx=0.3)
    for _ in range(80):
        scenario.step(range_m=3.0, camera=empty_frame(), dvl_vx=0.4)
    assert scenario.tracker.phase in (PHASE_SEARCH, PHASE_APPROACH)
    assert scenario.advancements == 0


def test_sudden_range_discontinuity_is_rejected():
    tracker = make_tracker()
    scenario = Scenario(tracker)
    for _ in range(5):
        scenario.step(range_m=5.0)
    assert tracker._filtered_range == pytest.approx(5.0, abs=0.01)
    # One wild packet: filtered range must hold at the previous level.
    scenario.step(range_m=25.0)
    assert tracker._filtered_range == pytest.approx(5.0, abs=0.01)
    # A second consistent packet confirms the new level.
    scenario.step(range_m=24.8)
    assert tracker._filtered_range == pytest.approx(24.8, abs=0.01)


def test_gate_sliding_out_sideways_aborts_commit():
    scenario = Scenario(make_tracker())
    for _ in range(200):
        scenario.step(range_m=3.1, camera=gate_frame(half=40), dvl_vx=0.25)
        if scenario.tracker.phase == PHASE_COMMIT:
            break
    assert scenario.tracker.phase == PHASE_COMMIT
    # The gate drifts hard toward the image edge while still far away.
    for _ in range(30):
        result = scenario.step(range_m=2.8, camera=gate_frame(center_x_px=18, half=22), dvl_vx=0.3)
        if result.phase not in (PHASE_COMMIT, PHASE_VERIFY_EXIT):
            break
    assert scenario.tracker.phase in (PHASE_VISUAL_ALIGN, PHASE_APPROACH, PHASE_SEARCH)
    assert scenario.advancements == 0


def test_local_state_resets_between_gates():
    scenario = Scenario(make_tracker())
    scenario.run_true_passage()
    tracker = scenario.tracker
    assert tracker._min_range is None
    assert tracker._commit_displacement_m == 0.0
    assert tracker._centered_streak == 0
    assert tracker._disappear_streak == 0
    assert tracker._held is None


def test_invalid_initial_beacon_id_is_rejected():
    with pytest.raises(ValueError):
        LocalCourseTracker(initial_beacon_id="G01", total_beacons=3)
    with pytest.raises(ValueError):
        LocalCourseTracker(initial_beacon_id="B09", total_beacons=3)


def test_diagnostics_snapshot_contains_local_fields_only():
    tracker = make_tracker()
    diagnostics = tracker.diagnostics()
    assert diagnostics["expected_beacon_id"] == "B01"
    assert diagnostics["rear_bearing_confirmed"] is False
    assert diagnostics["last_advancement_evidence"] is None
    forbidden = {"target_gate_id", "completed_gates", "referee", "official_time_s"}
    assert not forbidden & set(diagnostics.keys())
