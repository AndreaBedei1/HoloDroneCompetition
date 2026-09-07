"""Onboard-only temporal context for the 38-D gate-yaw contract."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from marine_race_arena.controllers.gate_pose import (
    DEFAULT_INTRINSICS,
    GatePoseTarget,
    GatePoseTracker,
    estimate_gate_pose,
)
from marine_race_arena.learning.config_local_transition_gate_yaw import (
    LocalTransitionGateYawLearningContext,
)
from marine_race_arena.learning.tracker_context_local_transition import (
    OnboardLocalTransitionContextTracker,
)


class OnboardLocalTransitionGateYawContextTracker(
    OnboardLocalTransitionContextTracker
):
    """Add a short camera-only plane-yaw filter to the proven target tracker.

    The aperture search is restricted to the visual ROI already associated
    with the *locally expected* acoustic beacon.  A target-id change resets the
    pose filter before the new image is processed, preventing the previous
    gate orientation from leaking across transitions.  No map, referee state,
    global pose or configured gate geometry is available here.
    """

    def __init__(
        self,
        total_beacons: int,
        laps: int = 1,
        initial_beacon_id: str = "B01",
        *,
        gate_width_m: float = 1.5,
        gate_height_m: float = 1.5,
    ) -> None:
        super().__init__(total_beacons, laps, initial_beacon_id)
        self.gate_width_m = float(gate_width_m)
        self.gate_height_m = float(gate_height_m)
        self._pose_tracker = GatePoseTracker(alpha=0.38, max_age_steps=4)
        self._last_gate_pose: Optional[GatePoseTarget] = None

    @property
    def last_gate_pose(self) -> Optional[GatePoseTarget]:
        """Filtered camera estimate for diagnostics; never simulator truth."""

        return self._last_gate_pose

    @property
    def gate_orientation_age_steps(self) -> int:
        """Missed camera frames behind the current filtered estimate."""

        return int(self._pose_tracker.age)

    def reset(self, first_observation: Optional[Mapping[str, Any]] = None) -> None:
        super().reset(first_observation)
        self._pose_tracker.reset()
        self._last_gate_pose = None
        base = self._last_context
        self._last_context = LocalTransitionGateYawLearningContext(**vars(base))

    def context(
        self,
        observation: Mapping[str, Any],
        *,
        dt: Optional[float],
        prev_action: Optional[Sequence[float]] = None,
    ) -> LocalTransitionGateYawLearningContext:
        expected_before = self.tracker.expected_beacon_id
        base = super().context(observation, dt=dt, prev_action=prev_action)
        if base.expected_beacon_id != expected_before:
            self._pose_tracker.reset()

        sensors = (
            observation.get("sensors")
            if isinstance(observation, Mapping)
            and isinstance(observation.get("sensors"), Mapping)
            else {}
        )
        image = sensors.get("FrontCamera")
        visual = base.visual_target
        visual_locked = bool(
            (self.tracker.diagnostics().get("visual_track") or {}).get("locked")
        )
        fresh = None
        if (
            image is not None
            and visual is not None
            and visual_locked
            and not bool(getattr(visual, "predicted", False))
        ):
            fresh = estimate_gate_pose(
                image,
                intr=DEFAULT_INTRINSICS,
                visual_target=visual,
                gate_width_m=self.gate_width_m,
                gate_height_m=self.gate_height_m,
            )
        pose = self._pose_tracker.update(fresh)
        self._last_gate_pose = pose
        valid = bool(
            pose is not None
            and pose.orientation_present
            and pose.gate_plane_yaw_deg is not None
        )
        context = LocalTransitionGateYawLearningContext(
            **vars(base),
            gate_orientation_present=valid,
            gate_yaw_deg=(float(pose.gate_plane_yaw_deg) if valid else None),
        )
        self._last_context = context
        return context
