"""Tests for the hybrid controller's phase-gated BC/rule blending (no simulator)."""

from marine_race_arena.controllers.hybrid_gate_controller import HybridGateController
from marine_race_arena.controllers.local_course_tracker import (
    PHASE_ADVANCE,
    PHASE_APPROACH,
    PHASE_COMMIT,
    PHASE_SEARCH,
    PHASE_VERIFY_EXIT,
    PHASE_VISUAL_ALIGN,
)


class _Vis:
    def __init__(self, confidence, center_x=0.0):
        self.confidence = confidence
        self.center_x = center_x


def _ctrl():
    return HybridGateController(model_path=None)


def test_backbone_owns_all_non_visual_align_phases():
    c = _ctrl()
    good_vis = _Vis(0.9)
    for phase in (PHASE_SEARCH, PHASE_APPROACH, PHASE_COMMIT, PHASE_VERIFY_EXIT, PHASE_ADVANCE, "FINISHED"):
        # Even with a strong detection, only VISUAL_ALIGN blends BC in.
        assert c._blend_weight(phase, good_vis) == 0.0


def test_visual_align_ramps_bc_with_confidence():
    c = _ctrl()
    assert c._blend_weight(PHASE_VISUAL_ALIGN, None) == 0.0
    assert c._blend_weight(PHASE_VISUAL_ALIGN, _Vis(0.2)) == 0.0  # below min confidence
    w_mid = c._blend_weight(PHASE_VISUAL_ALIGN, _Vis(0.6))
    w_hi = c._blend_weight(PHASE_VISUAL_ALIGN, _Vis(1.0))
    assert 0.0 < w_mid < w_hi
    assert w_hi <= c.max_bc_align_weight + 1e-9


def test_bc_weight_never_exceeds_cap():
    c = _ctrl()
    assert c._blend_weight(PHASE_VISUAL_ALIGN, _Vis(1.0)) <= c.max_bc_align_weight


def test_safety_limits_brake_surge_when_far_off_centre():
    c = _ctrl()
    cmd = {"surge": 0.4, "sway": 0.0, "heave": 0.0, "yaw": 0.0}
    out = c._apply_safety_limits(cmd, PHASE_VISUAL_ALIGN, _Vis(0.9, center_x=0.8))
    assert out["surge"] <= 0.12


def test_safety_limits_reduce_simultaneous_yaw_and_sway():
    c = _ctrl()
    cmd = {"surge": 0.2, "sway": 0.4, "heave": 0.0, "yaw": 0.2}
    out = c._apply_safety_limits(cmd, PHASE_COMMIT, None)
    assert abs(out["sway"]) <= 0.12
