from __future__ import annotations

from marine_race_arena.arena.gate import Gate


def _gate() -> Gate:
    return Gate(
        id="G01",
        type="single",
        center=(0.0, 0.0, -4.0),
        rotation_rpy_deg=(0.0, 0.0, 0.0),
        inner_width_m=1.5,
        inner_height_m=1.5,
        bar_thickness_m=0.18,
        color="#00ff88",
        passage_direction=(1.0, 0.0, 0.0),
    )


def test_valid_gate_plane_crossing() -> None:
    gate = _gate()
    result = gate.validate_crossing((-1.0, 0.0, -4.0), (1.0, 0.0, -4.0))
    assert result.valid
    assert result.reason == "valid"
    assert result.intersection == (0.0, 0.0, -4.0)


def test_outside_aperture_is_invalid() -> None:
    gate = _gate()
    result = gate.validate_crossing((-1.0, 2.0, -4.0), (1.0, 2.0, -4.0))
    assert not result.valid
    assert result.reason == "outside_aperture"


def test_wrong_direction_is_invalid() -> None:
    gate = _gate()
    result = gate.validate_crossing((1.0, 0.0, -4.0), (-1.0, 0.0, -4.0))
    assert not result.valid
    assert result.reason == "wrong_direction"

