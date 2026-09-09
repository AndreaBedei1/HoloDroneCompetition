"""Tests for the prepare-only extreme-corner demonstration tooling + offset sampler."""

import math

from marine_race_arena.learning import extreme_corner_demos as ec
from marine_race_arena.learning.curriculum import STAGE2_RANDOMIZATION
from marine_race_arena.learning.randomization import sample_offsets


def test_sample_offsets_deterministic_and_matches_frozen():
    """Deterministic and consistent with the frozen Evaluation-B offsets (seeds 1103/1139)."""
    assert sample_offsets(STAGE2_RANDOMIZATION, 1103) == sample_offsets(STAGE2_RANDOMIZATION, 1103)
    o1103 = sample_offsets(STAGE2_RANDOMIZATION, 1103)
    assert math.isclose(o1103["lateral_offset_m"], -0.9397, abs_tol=1e-3)
    assert math.isclose(o1103["yaw_offset_deg"], -13.9025, abs_tol=1e-3)
    o1139 = sample_offsets(STAGE2_RANDOMIZATION, 1139)
    assert abs(o1139["lateral_offset_m"]) >= 0.9 and abs(o1139["yaw_offset_deg"]) >= 13.0


def test_selected_seeds_are_extreme_corner_and_balanced():
    plan = ec.select_extreme_corner_seeds(n_per_quadrant=3, seed_start=20000, seed_limit=80000)
    assert len(plan["selected_seeds"]) == len(set(plan["selected_seeds"]))  # unique
    for row in plan["selected"]:
        assert ec.LAT_MIN <= abs(row["lateral_offset_m"]) <= ec.LAT_MAX
        assert ec.YAW_MIN <= abs(row["yaw_offset_deg"]) <= ec.YAW_MAX
    # all four sign quadrants represented
    assert all(c == 3 for c in plan["quadrant_counts"].values())
    assert "PREPARE-ONLY" in plan["purpose"] and "collect_demos" in plan["prepared_collect_command"]


def test_selected_seeds_disjoint_from_registry():
    from marine_race_arena.learning.seed_registry import all_used_seeds

    plan = ec.select_extreme_corner_seeds(n_per_quadrant=2, seed_start=20000, seed_limit=60000)
    assert set(plan["selected_seeds"]).isdisjoint(all_used_seeds())


def test_main_writes_plan(tmp_path):
    out = tmp_path / "plan.json"
    assert ec.main(["--out", str(out), "--n-per-quadrant", "2"]) == 0
    import json
    plan = json.loads(out.read_text(encoding="utf-8"))
    assert plan["experiment"] == "bc_extreme_corners_v2" and len(plan["selected_seeds"]) == 8
