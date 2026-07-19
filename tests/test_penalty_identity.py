"""Artifact consistency: penalized time == base time + applicable penalties.

An automated re-runnable check over the existing 78-run matrix. It reports (via
assertion) any finished run whose penalized time does not equal its official or
team-elapsed time plus its accumulated penalties. It never repairs or rewrites a
raw artifact and never launches HoloOcean.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

MATRIX = Path("results/onboard_only_validation/final_20260715")
TOL = 0.05


def _iter_summaries():
    for path in sorted(glob.glob(str(MATRIX / "**" / "*_summary.json"), recursive=True)):
        yield Path(path)


@pytest.mark.skipif(not MATRIX.exists(), reason="final matrix artifacts not present")
def test_penalized_equals_base_plus_penalties_for_every_finished_run():
    problems = []
    checked = 0
    for path in _iter_summaries():
        s = json.loads(path.read_text(encoding="utf-8"))
        team = s.get("team_summary")
        if team is not None:
            if not team.get("all_rovers_finished"):
                continue
            base = team.get("team_elapsed_time_s")
            pen = team.get("team_penalized_time_s")
            extra = team.get("total_penalties_s") or 0.0
            if base is None or pen is None:
                continue
            checked += 1
            if abs((base + extra) - pen) > TOL:
                problems.append(f"TEAM {path}: {base}+{extra} != {pen}")
            continue
        for p in s.get("participants", []):
            if p.get("status") != "FINISHED":
                continue
            base = p.get("official_time_s")
            pen = p.get("penalized_time_s")
            extra = p.get("penalties_s") or 0.0
            if base is None or pen is None:
                continue
            checked += 1
            if abs((base + extra) - pen) > TOL:
                problems.append(
                    f"{path} [{p.get('participant_id')}]: {base}+{extra} != {pen}"
                )
    assert checked > 0, "expected to check at least one finished run"
    assert not problems, "penalty identity violated:\n" + "\n".join(problems)
