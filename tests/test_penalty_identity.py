"""Artifact consistency: penalized time == base time + applicable penalties.

An automated, re-runnable check over the released benchmark rows. It reports
(via assertion) any finished run whose penalized time does not equal its
official or team-elapsed time plus its accumulated penalties. It never repairs
or rewrites an artifact and never launches HoloOcean.
"""

from __future__ import annotations

import csv
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "artifacts/paper/benchmark/runs.csv"
TOL = 0.05


def _num(value):
    return None if value in (None, "") else float(value)


def _rows():
    with RUNS.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_released_rows_are_present():
    assert RUNS.is_file(), "released benchmark rows are missing: {}".format(RUNS)
    assert len(_rows()) == 68


def test_penalized_equals_base_plus_penalties_for_every_finished_run():
    problems = []
    checked = 0
    for row in _rows():
        if row["team_elapsed_time_s"]:
            if row["all_rovers_finished"] != "True":
                continue
            base = _num(row["team_elapsed_time_s"])
            done = _num(row["team_penalized_time_s"])
        else:
            if row["status"] != "FINISHED":
                continue
            base = _num(row["official_time_s"])
            done = _num(row["penalized_time_s"])
        if base is None or done is None:
            continue
        checked += 1
        extra = _num(row["penalties_s"]) or 0.0
        if abs((base + extra) - done) > TOL:
            problems.append("{}: {}+{} != {}".format(row["run_id"], base, extra, done))
    assert checked > 0, "expected to check at least one finished run"
    assert not problems, "penalty identity violated:\n" + "\n".join(problems)


def test_every_released_run_used_the_native_backend():
    for row in _rows():
        assert row["actual_adapter"] == "holoocean", row["run_id"]
        assert row["fallback_used"] == "False", row["run_id"]
