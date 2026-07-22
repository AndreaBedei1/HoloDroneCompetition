"""Benchmark fresh vs persistent HoloOcean episode reset, and validate equivalence.

Fresh reset relaunches HoloOcean per episode; persistent reset keeps one engine alive
and teleports the vehicle. This measures both and checks that the persistent path
produces an equivalent initial observation, referee progress under a fixed action
sequence, and no residual velocity. Writes a report under
``results/rl_public/reset_benchmark/``. Real HoloOcean; run from marine_race_rl.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import List

import numpy as np

from marine_race_arena.learning.config import LearningContext
from marine_race_arena.learning.episode import PersistentRaceSession, RaceEpisode
from marine_race_arena.learning.observation_encoder import encode_observation
from marine_race_arena.learning.provenance import git_sha, now_utc, package_versions

TRACK = "marine_race_arena/tracks/training/stage1_single_gate.json"
OUT = Path("results/rl_public/reset_benchmark")
ACTIONS = [{"surge": 0.5, "sway": 0.0, "heave": 0.0, "yaw": 0.1} for _ in range(10)]


def _enc(obs):
    return encode_observation(obs, LearningContext())


def main(seeds=(2000, 2001, 2002), rollout_steps: int = 8) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds)

    # --- Fresh reset: relaunch per episode ---
    fresh_times: List[float] = []
    fresh_obs = {}
    fresh_gates = {}
    fresh_resid = {}
    for s in seeds:
        t0 = time.perf_counter()
        ep = RaceEpisode(TRACK, seed=s, dt=0.1, adapter="holoocean", allow_fallback=False, max_steps=200)
        obs = ep.reset()
        fresh_times.append(time.perf_counter() - t0)
        fresh_obs[s] = _enc(obs)
        for a in ACTIONS[:rollout_steps]:
            step = ep.step(a)
            if step.terminated or step.truncated:
                break
        fresh_gates[s] = ep.referee_progress()["valid_gate_crossings"]
        ep.close()

    # --- Persistent reset: one engine, teleport per episode ---
    persistent_times: List[float] = []
    persistent_obs = {}
    persistent_gates = {}
    persistent_resid = {}
    session = PersistentRaceSession(TRACK, seed=seeds[0], dt=0.1, adapter="holoocean", allow_fallback=False, max_steps=200)
    try:
        for s in seeds:
            t0 = time.perf_counter()
            obs = session.reset_episode(seed=s, start_randomization=None)
            persistent_times.append(time.perf_counter() - t0)
            persistent_obs[s] = _enc(obs)
            persistent_resid[s] = session.dvl_speed()
            for a in ACTIONS[:rollout_steps]:
                step = session.step(a)
                if step.terminated or step.truncated:
                    break
            persistent_gates[s] = session.referee_progress()["valid_gate_crossings"]
    finally:
        session.close()

    # --- Equivalence ---
    obs_max_diff = max(float(np.max(np.abs(fresh_obs[s] - persistent_obs[s]))) for s in seeds)
    gate_match = all(fresh_gates[s] == persistent_gates[s] for s in seeds)
    max_residual = max(persistent_resid.values()) if persistent_resid else 0.0

    def stats(xs):
        return {"mean_s": round(statistics.mean(xs), 2), "std_s": round(statistics.pstdev(xs), 2) if len(xs) > 1 else 0.0,
                "n": len(xs), "values_s": [round(x, 2) for x in xs]}

    equivalent = obs_max_diff < 1e-3 and gate_match and max_residual < 0.05
    report = {
        "generated_utc": now_utc(),
        "git_sha": git_sha(),
        "track": TRACK,
        "seeds": seeds,
        "rollout_steps": rollout_steps,
        "fresh_reset": stats(fresh_times),
        "persistent_reset": stats(persistent_times),
        "speedup_x": round(statistics.mean(fresh_times) / max(1e-9, statistics.mean(persistent_times)), 1),
        "equivalence": {
            "initial_observation_max_abs_diff": obs_max_diff,
            "referee_gate_progress_matches": gate_match,
            "max_residual_dvl_speed_m_s": round(max_residual, 4),
            "equivalent": equivalent,
        },
        "recommended_training_mode": ("persistent_reset" if equivalent else "fresh_reset (persistent not validated equivalent here)"),
        "known_limitations": [
            "Validated only on the noise-free Stage-1 track with yaw-0 base start.",
            "Persistent reset re-uses the arena beacon manager; on tracks with beacon noise the per-episode noise stream differs from fresh reset until separately validated.",
            "Frozen correctness evaluations use fresh reset; persistent reset is for training throughput only.",
        ],
        "packages": package_versions(),
    }
    (OUT / "reset_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
