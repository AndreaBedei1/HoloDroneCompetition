"""Onboard hybrid controller: deterministic beacon/vision navigation + learned visual servo.

``HybridGateController`` composes two existing onboard-only controllers and blends their
commands by the local-course-tracker phase, to combine the strengths of each:

* the deterministic :class:`RuleGateCenterThenCommitController` -- robust long-range beacon
  homing, gate exit and next-beacon turning, and a proven center-then-commit passage -- is the
  backbone and the sole authority in SEARCH / APPROACH / COMMIT / VERIFY_EXIT / ADVANCE;
* the learned :class:`RLGateController` (frozen BC-v1) is blended in only during VISUAL_ALIGN,
  where its learned local gate-centring can refine the deterministic command.

Both sub-controllers receive the *same* official observation each step and run their own
:class:`LocalCourseTracker`; since the tracker is deterministic in the observation stream, the
two stay phase-synchronised. The blended command is fed back to both sub-controllers so their
temporal state (BC previous-action encoding; rule command smoothing) reflects what was actually
applied. Command provenance is logged for diagnostics.

Only legal onboard information is used (received beacons, FrontCamera, depth, IMU, DVL and
controller-local state). No referee, world pose or gate pose is ever read. The controller never
does worse than the deterministic backbone in the critical passage/exit phases, where the BC
weight is exactly zero.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from marine_race_arena.controllers.local_course_tracker import (
    PHASE_ADVANCE,
    PHASE_APPROACH,
    PHASE_COMMIT,
    PHASE_SEARCH,
    PHASE_VERIFY_EXIT,
    PHASE_VISUAL_ALIGN,
)
from marine_race_arena.controllers.official_baselines import RuleGateCenterThenCommitController
from marine_race_arena.participants.controller_interface import BaseController

_AXES = ("surge", "sway", "heave", "yaw")


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class HybridGateController(BaseController):
    """Rule-backbone controller with the BC-v1 visual servo blended into VISUAL_ALIGN."""

    debug_only = False
    uses_ground_truth = False

    #: Maximum BC weight reached in VISUAL_ALIGN at full visual confidence.
    max_bc_align_weight = 0.55
    #: Minimum detection confidence before any BC blending starts.
    min_visual_confidence = 0.35

    def __init__(self, model_path: Optional[str] = None, max_bc_align_weight: Optional[float] = None) -> None:
        self._model_path = model_path
        self._rule = RuleGateCenterThenCommitController()
        self._bc = None  # lazily created in reset() (torch import only when used)
        if max_bc_align_weight is not None:
            self.max_bc_align_weight = float(max_bc_align_weight)
        self._log: List[Dict[str, Any]] = []
        self._log_limit = 4000

    # ------------------------------------------------------------------ api
    def reset(self, mission_info: Mapping[str, Any]) -> None:
        from marine_race_arena.learning.rl_controller import RLGateController  # lazy (torch)

        self._rule.reset(dict(mission_info))
        self._bc = RLGateController(model_path=self._model_path)
        self._bc.reset(mission_info)
        self._log = []

    @property
    def tracker(self):
        """The deterministic backbone's LocalCourseTracker (authoritative phase/progress)."""
        return self._rule.tracker

    @property
    def command_log(self) -> List[Dict[str, Any]]:
        return self._log

    def step(self, observation: Mapping[str, Any]) -> Dict[str, float]:
        rule_cmd = self._rule.step(observation)
        bc_cmd = self._bc.step(observation) if self._bc is not None else dict(rule_cmd)

        tracker = self._rule.tracker
        phase = tracker.phase
        visual = getattr(tracker, "_latest_visual_target", None)
        w = self._blend_weight(phase, visual)

        blended = {ax: _clamp((1.0 - w) * float(rule_cmd.get(ax, 0.0)) + w * float(bc_cmd.get(ax, 0.0)), -1.0, 1.0)
                   for ax in _AXES}
        blended = self._apply_safety_limits(blended, phase, visual)

        # Feed the applied command back so both sub-controllers' temporal state stays honest.
        self._sync_applied_command(blended)

        source = "rule" if w < 0.05 else ("bc" if w > 0.95 else "hybrid")
        if len(self._log) < self._log_limit:
            self._log.append({
                "phase": phase, "expected_beacon": tracker.expected_beacon_id,
                "blend_weight": round(w, 3), "command_source": source,
                "visual_detected": visual is not None,
                "visual_confidence": (round(float(visual.confidence), 3) if visual is not None else None),
                "visual_center_x": (round(float(visual.center_x), 3) if visual is not None else None),
                "rule_action": {k: round(float(v), 3) for k, v in rule_cmd.items()},
                "bc_action": {k: round(float(v), 3) for k, v in bc_cmd.items()},
                "blended_action": {k: round(float(v), 3) for k, v in blended.items()},
            })
        return blended

    def close(self) -> None:
        try:
            self._rule.close()
        finally:
            if self._bc is not None:
                self._bc.close()

    # -------------------------------------------------------------- internals
    def _blend_weight(self, phase: str, visual) -> float:
        """BC weight by phase. Deterministic backbone owns everything except VISUAL_ALIGN."""
        if phase != PHASE_VISUAL_ALIGN:
            # SEARCH / APPROACH / COMMIT / VERIFY_EXIT / ADVANCE / FINISHED -> pure rule backbone.
            return 0.0
        if visual is None or float(getattr(visual, "confidence", 0.0)) < self.min_visual_confidence:
            return 0.0
        conf = float(visual.confidence)
        ramp = _clamp((conf - self.min_visual_confidence) / (1.0 - self.min_visual_confidence), 0.0, 1.0)
        return self.max_bc_align_weight * ramp

    def _apply_safety_limits(self, cmd: Dict[str, float], phase: str, visual) -> Dict[str, float]:
        """Bounded deterministic guards (Section 7): brake surge when badly off-centre, and
        do not let the blend drive large simultaneous yaw and sway."""
        out = dict(cmd)
        if phase in (PHASE_VISUAL_ALIGN, PHASE_APPROACH) and visual is not None:
            if abs(float(visual.center_x)) > 0.5:
                out["surge"] = min(out["surge"], 0.12)  # gate far off-centre: do not lunge past it
        # Avoid aggressive yaw and sway at once (spins the vehicle off the gate line).
        if abs(out["yaw"]) > 0.08 and abs(out["sway"]) > 0.12:
            out["sway"] = _clamp(out["sway"], -0.12, 0.12)
        return out

    def _sync_applied_command(self, blended: Dict[str, float]) -> None:
        # Rule backbone smooths/limits from its last command -> make it the applied one.
        try:
            self._rule._last_command = dict(blended)
        except Exception:  # pragma: no cover - defensive
            pass
        # BC encodes o_(t+1) with the previous *applied* action.
        if self._bc is not None:
            try:
                import numpy as np

                self._bc._prev_action = np.asarray([blended[a] for a in _AXES], dtype=np.float32)
            except Exception:  # pragma: no cover - defensive
                pass
