"""Lock the referee-visible gate geometry while onboard autonomy evolves."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

# These hashes are the canonical geometry/referee projections from HEAD before
# the onboard-only redesign.  Beacon, sensor and controller fields are
# deliberately excluded; gate apertures, poses, order and scoring are not.
EXPECTED_GEOMETRY_SHA256 = {
    "marine_race_arena/tracks/marine_race_horseshoe_bay.json": "01444cc542b9ca09cb3e3a45c2cde6e4fdf7bef6dd16ae7bae85a091f5525269",
    "marine_race_arena/tracks/marine_race_vertical_serpent.json": "6b8ad49fd356dc63561c2a0a317b4cdb8deaf95d287d9a03a35d306e4a0b13d4",
    "marine_race_arena/tracks/marine_race_mixed_endurance.json": "7d15cfc48fcc389899a6a0e74a11a1d2799924dc2a65eb0fad0db569ff4b38a9",
    "marine_race_arena/tracks/tests/four_gate_horseshoe_start.json": "356df3a4ce10447f6519be5bc34c4cf41b940a03d923f591404c1bb66deaf6af",
    "marine_race_arena/tracks/tests/single_gate_yaw_0.json": "8b9923925d359aef29292d1694c54147695ae083b878b9e751a435dad3ce370f",
    "marine_race_arena/tracks/tests/single_gate_yaw_25.json": "0d836d8bf83552cb442f5f5869070cc357d3787d03bc4d47e327da9d69297a84",
    "marine_race_arena/tracks/tests/single_gate_yaw_45.json": "1b51da30e4a006bf75191b7c19944720631a9f1e6a604aa3bfde22c0fd51e443",
    "marine_race_arena/tracks/tests/single_gate_yaw_neg25.json": "ca8afcf18d173159bfbef164c0130fa3362dec33f22ab52b17ef865d1d7e6fa8",
    "marine_race_arena/tracks/tests/single_gate_yaw_neg45.json": "1eb8fc2a84a47a1254966597a031ac5009b1c6b4d4a50c217e4fc415259b01ad",
    "marine_race_arena/tracks/tests/three_gate_s_curve.json": "932139c0eb555529cf8d20e04ea72ac4617ab286f1ee78d893f2f07140ef93f2",
    "marine_race_arena/tracks/tests/two_gate_left_curve.json": "b2d5fac2281e0400da973801f729a35a459a37bf565bd05ef8c27cfa28e5bc4f",
    "marine_race_arena/tracks/tests/two_gate_right_curve.json": "bd010478beab411818ce4e1c5e009f14d41764191c558ff43e42e4f5340054ab",
    "marine_race_arena/tracks/tests/two_gate_straight.json": "866dedb9298ca19490cfb11985c2b73ca0c13fe0048b1082552b4730006767fa",
}


@pytest.mark.parametrize("relative_path", EXPECTED_GEOMETRY_SHA256)
def test_gate_geometry_and_referee_contract_are_unchanged(relative_path: str) -> None:
    raw = json.loads((ROOT / relative_path).read_text(encoding="utf-8"))
    projection = {
        "track": {
            key: raw["track"].get(key)
            for key in (
                "declared_length_m",
                "length_tolerance_m",
                "gate_inner_size_m",
                "gate_bar_thickness_m",
                "gate_depth_m",
                "gate_sequence",
            )
        },
        "start": raw.get("start"),
        "finish": raw.get("finish"),
        "gates": [
            {
                key: gate.get(key)
                for key in (
                    "id",
                    "type",
                    "position",
                    "rotation_rpy_deg",
                    "inner_size_m",
                    "passage_direction",
                )
            }
            for gate in raw["gates"]
        ],
        "referee": raw.get("referee"),
    }
    canonical = json.dumps(projection, sort_keys=True, separators=(",", ":"))

    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == (
        EXPECTED_GEOMETRY_SHA256[relative_path]
    )
