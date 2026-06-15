from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

from marine_race_arena.arena.gate import Gate, canonical_gate_frame
from marine_race_arena.arena.arena_builder import ArenaBuilder
from marine_race_arena.arena.gate_factory import GateBar, GateFactory, VisualGate
from marine_race_arena.config.loader import load_track_config


TRACK_DIR = Path(__file__).resolve().parents[1] / "marine_race_arena" / "tracks"


def _visual_gate(passage_direction: tuple[float, float, float]) -> VisualGate:
    gate = Gate(
        id="GT",
        type="single",
        center=(0.0, 0.0, -4.0),
        rotation_rpy_deg=(0.0, 0.0, 123.0),
        inner_width_m=1.5,
        inner_height_m=1.5,
        bar_thickness_m=0.2,
        color="white",
        passage_direction=passage_direction,
    )
    factory = GateFactory(SimpleNamespace(track=SimpleNamespace(gate_depth_m=0.3)))
    return factory.build_visual_gate(gate)


def _bar(visual_gate: VisualGate, part: str) -> GateBar:
    return next(bar for bar in visual_gate.bars if bar.part == part)


def test_x_direction_gate_places_pillars_on_y_axis_and_top_bottom_on_z_axis() -> None:
    visual_gate = _visual_gate((1.0, 0.0, 0.0))
    left = _bar(visual_gate, "left")
    right = _bar(visual_gate, "right")
    top = _bar(visual_gate, "top")
    bottom = _bar(visual_gate, "bottom")

    assert _close_tuple(left.position, (0.0, -0.85, -4.0))
    assert _close_tuple(right.position, (0.0, 0.85, -4.0))
    assert _close_tuple(top.position, (0.0, 0.0, -3.15))
    assert _close_tuple(bottom.position, (0.0, 0.0, -4.85))
    assert left.dimensions_m == (0.3, 0.2, 1.5)
    assert right.dimensions_m == (0.3, 0.2, 1.5)
    assert top.dimensions_m == (0.3, 1.9, 0.2)
    assert bottom.dimensions_m == (0.3, 1.9, 0.2)
    assert all(_close_tuple(bar.rotation_rpy_deg, (0.0, 0.0, 0.0)) for bar in visual_gate.bars)


def test_y_direction_gate_places_pillars_on_horizontal_perpendicular_axis() -> None:
    visual_gate = _visual_gate((0.0, 1.0, 0.0))
    left = _bar(visual_gate, "left")
    right = _bar(visual_gate, "right")
    top = _bar(visual_gate, "top")

    assert _close_tuple(left.position, (0.85, 0.0, -4.0))
    assert _close_tuple(right.position, (-0.85, 0.0, -4.0))
    assert _close_tuple(top.position, (0.0, 0.0, -3.15))
    assert all(_close_tuple(bar.rotation_rpy_deg, (0.0, 0.0, 90.0)) for bar in visual_gate.bars)


def test_rotated_gate_bars_form_rectangle_around_aperture() -> None:
    root_half = math.sqrt(0.5)
    visual_gate = _visual_gate((root_half, root_half, 0.0))
    left = _bar(visual_gate, "left")
    right = _bar(visual_gate, "right")
    top = _bar(visual_gate, "top")
    bottom = _bar(visual_gate, "bottom")

    assert _close_tuple(_midpoint(left.position, right.position), (0.0, 0.0, -4.0))
    assert _close_tuple(_midpoint(top.position, bottom.position), (0.0, 0.0, -4.0))
    assert _close(_distance(left.position, right.position), 1.7)
    assert _close(_distance(top.position, bottom.position), 1.7)
    assert left.dimensions_m[2] == 1.5
    assert right.dimensions_m[2] == 1.5
    assert top.dimensions_m[2] == 0.2
    assert bottom.dimensions_m[2] == 0.2


def test_pitched_gate_uses_full_3d_frame() -> None:
    pitch_rad = math.radians(20.0)
    direction = (math.cos(pitch_rad), 0.0, math.sin(pitch_rad))
    visual_gate = _visual_gate(direction)
    normal, right_axis, up_axis = canonical_gate_frame(direction)
    left = _bar(visual_gate, "left")
    right = _bar(visual_gate, "right")
    top = _bar(visual_gate, "top")
    bottom = _bar(visual_gate, "bottom")

    center = (0.0, 0.0, -4.0)
    half_with_bar = 0.85
    assert _close_tuple(left.position, _add(center, _scale(right_axis, -half_with_bar)))
    assert _close_tuple(right.position, _add(center, _scale(right_axis, half_with_bar)))
    assert _close_tuple(top.position, _add(center, _scale(up_axis, half_with_bar)))
    assert _close_tuple(bottom.position, _add(center, _scale(up_axis, -half_with_bar)))

    actual_normal, actual_right, actual_up = _axes_from_rotation(top.rotation_rpy_deg)
    assert _close_tuple(actual_normal, normal)
    assert _close_tuple(actual_right, right_axis)
    assert _close_tuple(actual_up, up_axis)
    assert not _close(top.rotation_rpy_deg[1], 0.0)


def test_example_tracks_generate_consistent_visual_gate_bars() -> None:
    for track_name in (
        "marine_race_horseshoe_bay.json",
        "marine_race_mixed_endurance.json",
        "marine_race_vertical_serpent.json",
    ):
        arena = ArenaBuilder(load_track_config(TRACK_DIR / track_name)).build()
        for visual_gate in arena.visual_gates:
            left = _bar(visual_gate, "left")
            right = _bar(visual_gate, "right")
            top = _bar(visual_gate, "top")
            bottom = _bar(visual_gate, "bottom")

            assert left.dimensions_m[2] == 1.5
            assert right.dimensions_m[2] == 1.5
            assert top.dimensions_m[2] < left.dimensions_m[2]
            assert bottom.dimensions_m[2] < right.dimensions_m[2]
            assert _close_tuple(left.rotation_rpy_deg, right.rotation_rpy_deg)
            assert _close_tuple(top.rotation_rpy_deg, bottom.rotation_rpy_deg)
            assert _close_tuple(left.rotation_rpy_deg, top.rotation_rpy_deg)


def _midpoint(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0)


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(3)))


def _axes_from_rotation(rotation_rpy_deg: tuple[float, float, float]) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    roll, pitch, yaw = [math.radians(value) for value in rotation_rpy_deg]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    matrix = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    return (
        (matrix[0][0], matrix[1][0], matrix[2][0]),
        (matrix[0][1], matrix[1][1], matrix[2][1]),
        (matrix[0][2], matrix[1][2], matrix[2][2]),
    )


def _add(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(vector: tuple[float, float, float], scalar: float) -> tuple[float, float, float]:
    return (vector[0] * scalar, vector[1] * scalar, vector[2] * scalar)


def _close_tuple(
    actual: tuple[float, float, float],
    expected: tuple[float, float, float],
    tolerance: float = 1e-6,
) -> bool:
    return all(_close(actual[index], expected[index], tolerance) for index in range(3))


def _close(actual: float, expected: float, tolerance: float = 1e-6) -> bool:
    return abs(actual - expected) <= tolerance
