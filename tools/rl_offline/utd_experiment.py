"""Offline update-to-data experiment for the SAC critic.

STATUS: the first run of this harness was INVALID and its comparison must not
be used.  It drew batches with `rng.integers(0, visible, size=bs)` -- uniform
sampling -- while the live learner uses StratifiedReplayBuffer.sample(), which
forces a fixed composition of 35% general / 30% transition / 20% success /
15% safety.  Safety transitions are only 1.18% of the buffer naturally and
carry one-time returns of -38 to -99, so uniform batches average -0.186 while
stratified batches average -6.537: a 35x difference in the target signal.  The
offline baseline therefore drove mean Q to +6.5 with loss ~30 and never
destabilised, whereas the live v7 run sat at mean Q -16 with loss ~190 and
rebuilt its critics.  A candidate compared against that baseline proves nothing.

This file now uses the real stratified sampler over a growing buffer.  Rerun
before drawing any UTD conclusion.

UTD semantics (this is the subtle part). Running the same 30k updates under
different labels would prove nothing: UTD is *optimizer updates per newly
collected environment transition*. So the harness replays the real TRAIN buffer
IN COLLECTION ORDER and simulates the data stream:

  - each simulated environment step reveals n_envs transitions into a growing
    visible buffer;
  - update credit accrues at `updates_per_transition`;
  - batches are drawn only from what has been revealed so far.

Both arms therefore consume the SAME environment data and differ only in how
much optimization is applied to it -- which is exactly the hypothesis. Lower UTD
performs fewer total updates and reuses each sample less often.

The actor is frozen throughout; no actor or entropy updates occur.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"C:\Users\andrea.bedei3\Desktop\HoloDroneCompetition-sac")
sys.path.insert(0, str(ROOT))

from marine_race_arena.learning.sac_health_gates import (  # noqa: E402
    PHASE_HEALTHY, CriticHealthMonitor,
)
from marine_race_arena.learning.sac_replay_buffer import StratifiedReplayBuffer  # noqa: E402
from marine_race_arena.learning.sac_transition_policy import (  # noqa: E402
    actor_parameter_sha256, rebuild_critics_from_actor,
)
from marine_race_arena.learning.train_sac_transition import (  # noqa: E402
    _load_config, _new_critic_health_monitor,
)

V7 = ROOT / "results/rl/universal_transition/sac/universal_transition_sac_v7_stable_critic_seed23001"
CONFIG = ROOT / "configs/rl/sac_universal_transition_v7_stable_critic.json"
ACTOR = V7 / "checkpoints/sac_100096_steps.pt"
OUT = ROOT / "results/rl_public/sac_utd_experiment"

LEARNING_STARTS = 20_000
N_ENVS = 2
HELDOUT = 20_000          # last slice reserved for generalisation diagnostics only


def load_stream():
    f = sorted(V7.glob("checkpoints/*.replay.npz"), key=lambda p: p.stat().st_mtime)[-1]
    replay = StratifiedReplayBuffer.load(f)
    n = replay.size
    data = {
        "observations": np.asarray(replay.observations[:n]),
        "actions": np.asarray(replay.actions[:n]),
        "rewards": np.asarray(replay.rewards[:n]),
        "next_observations": np.asarray(replay.next_observations[:n]),
        "discounts": np.asarray(replay.discounts[:n]),
    }
    return f.name, n, data


def batch_at(data, idx):
    return {k: v[idx] for k, v in data.items()}


def run_arm(name, utd, data, usable, config, updates_cap):
    """Simulate the data stream at the given update-to-data ratio."""
    sac = config["sac"]
    torch.manual_seed(23001)                      # identical critic init per arm
    agent, _ = rebuild_critics_from_actor(
        ACTOR,
        anchor_coefficient=float(sac["anchor"]["initial_coefficient"]),
        critic_learning_rate=float(sac["critic_learning_rate"]),
        actor_learning_rate=float(sac["actor_learning_rate"]),
        entropy_learning_rate=float(sac.get("entropy_learning_rate", 1e-6)),
        tau=float(sac["tau"]),
    )
    sha0 = actor_parameter_sha256(agent)
    monitor: CriticHealthMonitor = _new_critic_health_monitor(config)
    rng = np.random.default_rng(4242)             # identical sampling stream
    bs = int(sac["batch_size"])
    clip = float(sac["gradient_clip_critic"])

    # Grow a REAL StratifiedReplayBuffer so batches carry the live composition
    # (35/30/20/15).  Uniform sampling under-represents safety transitions by
    # more than an order of magnitude and does not reproduce the phenomenon.
    growing = StratifiedReplayBuffer(
        capacity=int(sac["replay_capacity"]), seed=4242,
        composition=dict(sac.get("replay_composition") or {}),
    )
    reuse = np.zeros(usable, dtype=np.int64)
    credit, updates, visible = 0.0, 0, 0
    track, warnings, rebuild_at = [], 0, None

    for step in range(0, usable, N_ENVS):
        visible = min(step + N_ENVS, usable)
        if visible < LEARNING_STARTS or visible < bs:
            continue
        credit += N_ENVS * utd
        while credit >= 1.0 and updates < updates_cap:
            L = agent.update(growing.sample(bs), update_actor=False, update_entropy=False)
            credit -= 1.0
            updates += 1
            rec = monitor.observe(L, updates=updates)
            if rec["cumulative_state"]:
                warnings += 1
            if rec["should_pause"] and rebuild_at is None:
                rebuild_at = {"updates": updates, "transitions": visible,
                              "reasons": rec["reasons"], "phase": rec["phase"]}
            if updates % 500 == 0:
                track.append({
                    "u": updates, "tx": visible,
                    "critic_loss": round(float(L["critic_loss"]), 2),
                    "mean_q": round(float(L["mean_q"]), 3),
                    "q_p05": round(float(L["q_p05"]), 3),
                    "q_p95": round(float(L["q_p95"]), 3),
                    "target_q": round(float(L["mean_target_q"]), 3),
                    "td_p50": round(float(L["td_error_p50"]), 3),
                    "td_p95": round(float(L["td_error_p95"]), 2),
                    "td_max": round(float(L["td_error_max"]), 2),
                    "grad": round(float(L["critic_gradient_norm"]), 2),
                    "phase": rec["phase"],
                })
    assert actor_parameter_sha256(agent) == sha0, "actor must stay frozen"

    # Held-out TRAIN generalisation (diagnostics only, never fitted).
    hold = np.arange(usable, usable + HELDOUT)
    with torch.no_grad():
        hb = batch_at(data, hold)
        o = torch.as_tensor(hb["observations"], dtype=torch.float32)
        a = torch.as_tensor(hb["actions"], dtype=torch.float32)
        q1, q2 = agent.critic(o, a)
        qh = torch.minimum(q1, q2)
    first, last = track[0], track[-1]
    seen = reuse[reuse > 0]
    return {
        "name": name, "updates_per_transition": utd,
        "updates": updates, "transitions_consumed": visible,
        "measured_utd": round(updates / max(1, visible - LEARNING_STARTS), 6),
        "track": track,
        "q_displacement": round(abs(last["mean_q"] - first["mean_q"]), 3),
        "q_p05_displacement": round(abs(last["q_p05"] - first["q_p05"]), 3),
        "q_span_first": round(abs(first["q_p95"] - first["q_p05"]), 3),
        "q_span_last": round(abs(last["q_p95"] - last["q_p05"]), 3),
        "target_q_displacement": round(abs(last["target_q"] - first["target_q"]), 3),
        "final_mean_q": last["mean_q"], "final_q_p05": last["q_p05"],
        "final_td_p95": last["td_p95"], "final_td_max": last["td_max"],
        "mean_td_p95": round(float(np.mean([t["td_p95"] for t in track])), 2),
        "final_critic_loss": last["critic_loss"],
        "mean_grad": round(float(np.mean([t["grad"] for t in track])), 2),
        "max_grad": round(float(np.max([t["grad"] for t in track])), 2),
        "clip_fraction_final": round(monitor.clip_hit_fraction(), 4),
        "cumulative_warnings": warnings,
        "rebuild_triggered": rebuild_at,
        "final_phase": monitor.phase,
        "mean_sample_reuse": round(float(seen.mean()), 3) if seen.size else 0.0,
        "max_sample_reuse": int(seen.max()) if seen.size else 0,
        "heldout_q_mean": round(float(qh.mean()), 3),
        "heldout_q_std": round(float(qh.std()), 3),
    }


def main():
    config = _load_config(CONFIG)
    name, n, data = load_stream()
    usable = n - HELDOUT
    base_utd = float(config["sac"]["updates_per_transition"])
    cap = int((usable - LEARNING_STARTS) * base_utd) + 5
    print(f"replay {name}  n={n}  usable={usable}  heldout={HELDOUT}")
    print(f"baseline UTD={base_utd}  update cap={cap}\n", flush=True)

    arms = [("baseline", base_utd), ("candidate_A", base_utd / 2.0)]
    results = []
    for label, utd in arms:
        print(f"--- {label}: UTD={utd} ---", flush=True)
        r = run_arm(label, utd, data, usable, config, cap)
        results.append(r)
        for t in r["track"][::max(1, len(r["track"]) // 8)] + [r["track"][-1]]:
            print(f"   u={t['u']:<6} tx={t['tx']:<7} loss={t['critic_loss']:<8} "
                  f"mean_q={t['mean_q']:<9} q_p05={t['q_p05']:<9} td_p95={t['td_p95']:<7} "
                  f"grad={t['grad']:<7} {t['phase']}")
        print(f"   updates={r['updates']} measured_utd={r['measured_utd']} "
              f"|dQ|={r['q_displacement']} |dq_p05|={r['q_p05_displacement']} "
              f"reuse={r['mean_sample_reuse']} rebuild={r['rebuild_triggered']}\n", flush=True)

    a, b = results
    ratios = {
        "q_displacement": round(b["q_displacement"] / max(1e-9, a["q_displacement"]), 3),
        "q_p05_displacement": round(b["q_p05_displacement"] / max(1e-9, a["q_p05_displacement"]), 3),
        "q_span_growth": round((b["q_span_last"] / max(1e-9, b["q_span_first"])) /
                               max(1e-9, a["q_span_last"] / max(1e-9, a["q_span_first"])), 3),
        "target_q_displacement": round(b["target_q_displacement"] / max(1e-9, a["target_q_displacement"]), 3),
        "mean_td_p95": round(b["mean_td_p95"] / max(1e-9, a["mean_td_p95"]), 3),
        "final_td_max": round(b["final_td_max"] / max(1e-9, a["final_td_max"]), 3),
        "mean_grad": round(b["mean_grad"] / max(1e-9, a["mean_grad"]), 3),
        "final_critic_loss": round(b["final_critic_loss"] / max(1e-9, a["final_critic_loss"]), 3),
        "sample_reuse": round(b["mean_sample_reuse"] / max(1e-9, a["mean_sample_reuse"]), 3),
    }
    better = sum(1 for k in ("q_displacement", "q_p05_displacement", "q_span_growth",
                             "target_q_displacement", "mean_td_p95", "mean_grad")
                 if ratios[k] < 0.85)
    verdict = {
        "ratios_candidate_over_baseline": ratios,
        "corroborating_metrics_improved": better,
        "baseline_reproduced_instability": bool(
            a["rebuild_triggered"] or a["cumulative_warnings"] > 0
            or a["q_displacement"] > 5.0),
        "candidate_supported": better >= 3,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "utd_comparison.json").write_text(
        json.dumps({"replay": name, "arms": results, "verdict": verdict}, indent=2),
        encoding="utf-8")
    print("=== VERDICT ===")
    for k, v in ratios.items():
        print(f"  {k:<24} {v}")
    for k in ("baseline_reproduced_instability", "corroborating_metrics_improved",
              "candidate_supported"):
        print(f"  {k} = {verdict[k]}")
    print(f"written -> {OUT/'utd_comparison.json'}")


if __name__ == "__main__":
    raise SystemExit(main())
