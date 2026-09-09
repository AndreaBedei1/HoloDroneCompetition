"""Generation-2 learned controller: expert bootstrapping -> recurrent BC -> DAgger -> recurrent PPO.

Generation 1 (``feature/rl-universal-gate-transition``) was a pure feed-forward
PPO campaign.  It is CLOSED and its artifacts are preserved verbatim; nothing in
this package may overwrite them.  See ``docs/rl_generations.md``.

The Generation-2 inference contract is deliberately narrow::

    action = learned_policy(observation, recurrent_state)

No rules fallback, no action blending, no hybrid controller, no privileged
referee information, no global pose, no circuit identity, no future gate
geometry, no final-track identifier.  The rule controller appears ONLY as a
teacher during dataset generation and DAgger labelling.
"""

GEN2_EXPERIMENT_ID = "rl_gen2_recurrent_dagger"
GEN2_OBS_CONTRACT = "onboard_local_transition_v1"
GEN2_ACTION_CONTRACT = "surge_sway_heave_yaw_pm1_v1"
GEN2_EXPERT_ID = "rule_gate_center_then_commit"

__all__ = [
    "GEN2_EXPERIMENT_ID",
    "GEN2_OBS_CONTRACT",
    "GEN2_ACTION_CONTRACT",
    "GEN2_EXPERT_ID",
]
