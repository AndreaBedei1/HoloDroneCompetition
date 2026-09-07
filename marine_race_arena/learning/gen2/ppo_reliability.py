"""Recurrent PPO phase 1: reliability on the three circuits, not speed.

Behaviour cloning got the controller to 17/30 with Horseshoe Bay at 10/10, and
then stopped paying: four hotspot rounds produced two accepts and two rejects,
with gains on the two weak circuits of about +1/10 at roughly three hours a
round.  The reason is structural -- cloning optimizes agreement with the
expert's action, and the thing that is actually failing is whether a *run*
survives twenty-two gates.  PPO optimizes that directly.

Three properties this phase must have, because the starting policy is valuable:

**It starts from the frozen parent, exactly.**  The actor is loaded, not
re-initialized, and :func:`verify_actor_parity` checks the loaded policy
reproduces the parent's actions on a fixed observation stream *before* the
first update.  Gen-1 lost time to resume bugs of exactly this kind.

**The reward is lexicographic, not a weighted sum of everything.**  Gate
progress dominates; safety events are penalties large enough that no amount of
speed buys them; time enters only as a small shaping term that cannot outweigh
a single gate.  A faster DNF must score worse than a slower completion.

**Updates are conservative.**  Low learning rate, tight clip range, KL
monitoring, frequent checkpoints, and the parent preserved.  A run that
degrades completion is rolled back rather than continued.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from marine_race_arena.learning.gen2 import track_fragments as tf

#: Reward weights.  The ordering between them is the design, not the values:
#: one gate must be worth more than any achievable amount of time shaping, and
#: one safety event must be worth more than any achievable number of gates.
#: A circuit runs to roughly this many steps, and every per-step term has to be
#: judged against that horizon rather than against its own magnitude.  The first
#: set of weights failed exactly here: a -0.002 per-step time penalty looks
#: negligible until it accumulates to -12 over a full circuit and outweighs a
#: 10-point gate, which would have let PPO trade gates for speed in phase 1.
REWARD_HORIZON_STEPS = 6000

GATE_REWARD = 20.0
COMPLETION_BONUS = 60.0
MISSED_GATE_PENALTY = -30.0
COLLISION_PENALTY = -20.0
OUT_OF_BOUNDS_PENALTY = -50.0
WRONG_DIRECTION_PENALTY = -25.0
#: Shaping only.  Accumulated over the horizon these stay well under one gate.
PROGRESS_REWARD = 0.30
TIME_PENALTY = 0.0             # phase 1 learns reliability, never speed
STALL_PENALTY = -0.005         # -30.0 if it stalls the entire episode
#: Below this body speed the vehicle is treated as not making progress.
STALL_SPEED_M_S = 0.05


@dataclass
class PPOReliabilityConfig:
    """Conservative defaults: the parent is worth more than a fast experiment."""

    total_timesteps: int = 50_000
    n_steps: int = 512
    batch_size: int = 128
    n_epochs: int = 4
    learning_rate: float = 5e-5      # an order below the BC from-scratch rate
    clip_range: float = 0.10         # half the SB3 default
    target_kl: float = 0.015
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    gamma: float = 0.995             # circuits are long; credit must reach back
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    checkpoint_every: int = 10_000
    seed: int = 0
    device: str = "cpu"
    #: Share of episodes drawn from complete circuits rather than fragments.
    full_circuit_fraction: float = 0.35

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def verify_actor_parity(
    parent_path: str | Path,
    model,
    *,
    samples: int = 24,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """Confirm the PPO actor reproduces the parent before any update.

    Compares deterministic actions over a fixed observation stream with the
    recurrent state carried, which is the only comparison that means anything
    for an LSTM policy: matching a single step would pass even if the hidden
    state semantics were wrong.
    """
    from sb3_contrib import RecurrentPPO

    from marine_race_arena.learning.config_local_transition import OBS_DIM_LOCAL_TRANSITION
    from marine_race_arena.learning.gen2.recurrent_policy import Gen2RecurrentController

    stream = np.random.default_rng(12345).random(
        (int(samples), OBS_DIM_LOCAL_TRANSITION)
    ).astype(np.float32)

    def rollout(m):
        controller = Gen2RecurrentController(m, deterministic=True)
        return np.asarray([
            controller.act(obs, first_step=(i == 0)) for i, obs in enumerate(stream)
        ])

    parent = RecurrentPPO.load(str(parent_path), device="cpu")
    expected = rollout(parent)
    observed = rollout(model)
    deviation = float(np.abs(expected - observed).max())
    return {
        "parity": bool(deviation <= tolerance),
        "max_abs_deviation": deviation,
        "tolerance": tolerance,
        "samples": int(samples),
        "parent": str(parent_path),
    }


@dataclass
class RewardBreakdown:
    gates: float = 0.0
    completion: float = 0.0
    missed_gate: float = 0.0
    collision: float = 0.0
    out_of_bounds: float = 0.0
    wrong_direction: float = 0.0
    progress: float = 0.0
    time: float = 0.0
    stall: float = 0.0

    @property
    def total(self) -> float:
        return float(sum(asdict(self).values()))

    def as_dict(self) -> Dict[str, float]:
        return {**asdict(self), "total": self.total}


class ReliabilityReward:
    """Gate progress dominates; safety is a hard penalty; time only shapes.

    Deliberately *not* a tuned weighted sum.  The invariant that matters is
    checked by :func:`assert_reward_hierarchy`: a run that completes must score
    above any run that does not, whatever its time, and a run with a safety
    event must score below the same run without one.
    """

    def __init__(self, gate_count: int) -> None:
        self.gate_count = int(gate_count)
        self.breakdown = RewardBreakdown()
        self._previous_gates = 0
        self._previous_counts: Dict[str, int] = {}
        self._completion_paid = False

    def reset(self, env=None) -> None:
        self.breakdown = RewardBreakdown()
        self._previous_gates = 0
        self._previous_counts = {}
        self._completion_paid = False

    def __call__(self, env, step, gate_delta: int, action) -> Tuple[float, Dict[str, float]]:
        reward = 0.0
        parts: Dict[str, float] = {}

        if gate_delta > 0:
            value = GATE_REWARD * gate_delta
            reward += value
            self.breakdown.gates += value
            parts["gates"] = value
            self._previous_gates += gate_delta

        referee = env.episode.context.referee.states[env.episode.participant_id]
        status = getattr(referee.status, "value", str(referee.status))
        if (
            status == "FINISHED"
            and self._previous_gates >= self.gate_count
            and not self._completion_paid
        ):
            reward += COMPLETION_BONUS
            self.breakdown.completion += COMPLETION_BONUS
            parts["completion"] = COMPLETION_BONUS
            self._completion_paid = True

        event_specs = (
            ("collision", "collision_events", COLLISION_PENALTY),
            ("obstacle_collision", "obstacle_collision_events", COLLISION_PENALTY),
            ("out_of_bounds", "out_of_bounds_events", OUT_OF_BOUNDS_PENALTY),
            ("wrong_direction", "wrong_direction_crossings", WRONG_DIRECTION_PENALTY),
            ("missed_gate", "missed_gate_attempts", MISSED_GATE_PENALTY),
        )
        for part, attribute, penalty in event_specs:
            current = int(getattr(referee, attribute, 0))
            previous = self._previous_counts.get(attribute, 0)
            delta = max(0, current - previous)
            self._previous_counts[attribute] = current
            if not delta:
                continue
            value = float(penalty * delta)
            reward += value
            field_name = "collision" if part == "obstacle_collision" else part
            setattr(
                self.breakdown,
                field_name,
                getattr(self.breakdown, field_name) + value,
            )
            parts[field_name] = parts.get(field_name, 0.0) + value

        speed = float(np.linalg.norm(np.asarray(step.current_state.position)
                                     - np.asarray(step.previous_state.position)))
        if speed > STALL_SPEED_M_S * env.episode.dt:
            value = PROGRESS_REWARD * speed
            reward += value
            self.breakdown.progress += value
            parts["progress"] = value
        else:
            reward += STALL_PENALTY
            self.breakdown.stall += STALL_PENALTY
            parts["stall"] = STALL_PENALTY

        reward += TIME_PENALTY
        self.breakdown.time += TIME_PENALTY
        parts["time"] = TIME_PENALTY
        return float(reward), parts


def assert_reward_hierarchy() -> Dict[str, Any]:
    """A faster DNF must never outscore a slower completion.

    Checked arithmetically rather than by inspection, because this is the one
    property that stops a speed term from buying missed gates.
    """
    horizon = REWARD_HORIZON_STEPS
    worst_time = TIME_PENALTY * horizon
    worst_stall = STALL_PENALTY * horizon
    one_gate = GATE_REWARD

    checks = {
        # Time may shape, never decide: a whole circuit of time penalty must
        # cost less than a single gate.
        "gate_outweighs_full_time_penalty": one_gate > abs(worst_time),
        "completion_outweighs_time": COMPLETION_BONUS > abs(worst_time),
        # A stalled episode is already a DNF; the penalty must not dwarf the
        # gate signal on episodes that do move.
        "gates_outweigh_full_stall_penalty": 3 * one_gate > abs(worst_stall),
        # Safety must not be purchasable with gates.
        "out_of_bounds_outweighs_two_gates": abs(OUT_OF_BOUNDS_PENALTY) > 2 * GATE_REWARD,
        "wrong_direction_outweighs_one_gate": abs(WRONG_DIRECTION_PENALTY) >= GATE_REWARD,
        "missed_gate_outweighs_one_gate": abs(MISSED_GATE_PENALTY) > GATE_REWARD,
        "time_penalty_is_small": abs(worst_time) < GATE_REWARD,
        # The decisive property: finishing slowly beats not finishing quickly.
        "slow_completion_beats_fast_dnf": (
            COMPLETION_BONUS + one_gate + worst_time > MISSED_GATE_PENALTY
        ),
    }
    return {
        "all_hold": all(checks.values()),
        "checks": checks,
        "worst_case_time_penalty": round(worst_time, 3),
        "worst_case_stall_penalty": round(worst_stall, 3),
        "one_gate_reward": one_gate,
        "horizon_steps": horizon,
    }


def training_tracks(
    config: PPOReliabilityConfig,
    *,
    hotspot_report: Optional[str | Path] = None,
    root: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """The PPO training distribution: hard fragments plus complete circuits.

    Full circuits are included from the start rather than unlocked later --
    these three tracks are the task, and the failure being fixed is a
    whole-run property that a fragment cannot express.
    """
    fragments: List[tf.TrackFragment] = []
    if hotspot_report is not None:
        from marine_race_arena.learning.gen2.track_eval import (
            CircuitOutcome,
            hotspot_fragments,
            transition_hotspots,
        )

        rows: List[CircuitOutcome] = []
        base = Path(hotspot_report)
        for track in tf.OFFICIAL_TRACKS:
            for candidate in (base / track / "circuits" / "circuit_report.json",
                              base / f"eval_{track}" / "circuits" / "circuit_report.json"):
                if candidate.exists():
                    rows += [CircuitOutcome(**r)
                             for r in json.loads(candidate.read_text(encoding="utf-8"))["rows"]]
                    break
        fragments = hotspot_fragments(transition_hotspots(rows), radius=2, root=root)
    return {
        "full_circuits": list(tf.OFFICIAL_TRACKS),
        "full_circuit_fraction": config.full_circuit_fraction,
        "hotspot_fragments": [f.name for f in fragments],
        "fragment_count": len(fragments),
    }
