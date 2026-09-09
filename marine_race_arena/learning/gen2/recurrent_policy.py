"""The Generation-2 recurrent actor-critic.

Architecture (the brief's shape, realized inside sb3-contrib so nothing has to
be copied between frameworks)::

        27-D onboard observation
      -> frozen normalization (mean/std latched from the expert corpus)
      -> MLP encoder  [35 -> 256 -> 128]      (features_extractor)
      -> LSTM         [128 -> 128]            (lstm_actor / lstm_critic)
      -> policy head  [128 -> 128 -> 4]       (mlp_extractor.policy_net + action_net)
      -> value head   [128 -> 128 -> 1]       (mlp_extractor.value_net + value_net)

Why the encoder is a ``BaseFeaturesExtractor``: in
:class:`~sb3_contrib.common.recurrent.policies.RecurrentActorCriticPolicy` the
features extractor is the only hook that runs *before* the LSTM.  Putting the
encoder there gives exactly the requested ordering.

**Why BC trains the SB3 policy.**  The legacy Generation 1 path lost time to a class of
BC-to-PPO transfer bugs: BC trained a standalone ``nn.Module`` and the weights
were then folded into a PPO policy by hand, so any layout mismatch showed up as
a silent competence loss.  Gen-2 removes the copy entirely -- behaviour cloning
optimizes the parameters of an actual :class:`RecurrentPPO` policy, and the
"transfer" into PPO is loading the same object.  Parity is therefore an
identity, and ``tests/learning/gen2/test_gen2_transfer.py`` asserts it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces
from torch import nn
from torch.nn import functional as F

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from marine_race_arena.learning.config import ACTION_DIM
from marine_race_arena.learning.config_local_transition import (
    FEATURE_NAMES_LOCAL_TRANSITION,
    OBS_DIM_LOCAL_TRANSITION,
    OBS_ENCODING_VERSION_LOCAL_TRANSITION,
)
from marine_race_arena.learning.config_local_transition_gate_yaw import (
    FEATURE_BOUNDS_LOCAL_TRANSITION_GATE_YAW,
    FEATURE_NAMES_LOCAL_TRANSITION_GATE_YAW,
    OBS_DIM_LOCAL_TRANSITION_GATE_YAW,
    OBS_ENCODING_VERSION_LOCAL_TRANSITION_GATE_YAW,
)
from marine_race_arena.learning.config_local_transition_27d import (
    FEATURE_BOUNDS_LOCAL_TRANSITION_27D,
    FEATURE_NAMES_LOCAL_TRANSITION_27D,
    OBS_DIM_LOCAL_TRANSITION_27D,
    OBS_ENCODING_VERSION_LOCAL_TRANSITION_27D,
)
from marine_race_arena.learning.gen2 import GEN2_ACTION_CONTRACT

GEN2_ARCHITECTURE_ID = "gen2_recurrent_lstm_v1"


@dataclass(frozen=True)
class Gen2Architecture:
    """The frozen Gen-2 network shape.  Changing it changes the experiment."""

    architecture_id: str = GEN2_ARCHITECTURE_ID
    obs_dim: int = OBS_DIM_LOCAL_TRANSITION
    action_dim: int = ACTION_DIM
    encoder_hidden: Tuple[int, ...] = (256, 128)
    lstm_hidden_size: int = 128
    n_lstm_layers: int = 1
    head_hidden: Tuple[int, ...] = (128,)
    shared_lstm: bool = False
    enable_critic_lstm: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


GEN2_ARCH = Gen2Architecture()
GEN2_GATE_YAW_ARCH = Gen2Architecture(
    architecture_id="gen2_recurrent_lstm_gate_yaw_v2",
    obs_dim=OBS_DIM_LOCAL_TRANSITION_GATE_YAW,
)
GEN2_27D_ARCH = Gen2Architecture(
    architecture_id="gen2_recurrent_lstm_27d_v1",
    obs_dim=OBS_DIM_LOCAL_TRANSITION_27D,
)


def _observation_contract(architecture: Gen2Architecture):
    if int(architecture.obs_dim) == OBS_DIM_LOCAL_TRANSITION:
        from marine_race_arena.learning.config_local_transition import (
            FEATURE_BOUNDS_LOCAL_TRANSITION,
        )

        return (
            OBS_ENCODING_VERSION_LOCAL_TRANSITION,
            FEATURE_NAMES_LOCAL_TRANSITION,
            FEATURE_BOUNDS_LOCAL_TRANSITION,
        )
    if int(architecture.obs_dim) == OBS_DIM_LOCAL_TRANSITION_GATE_YAW:
        return (
            OBS_ENCODING_VERSION_LOCAL_TRANSITION_GATE_YAW,
            FEATURE_NAMES_LOCAL_TRANSITION_GATE_YAW,
            FEATURE_BOUNDS_LOCAL_TRANSITION_GATE_YAW,
        )
    if int(architecture.obs_dim) == OBS_DIM_LOCAL_TRANSITION_27D:
        return (
            OBS_ENCODING_VERSION_LOCAL_TRANSITION_27D,
            FEATURE_NAMES_LOCAL_TRANSITION_27D,
            FEATURE_BOUNDS_LOCAL_TRANSITION_27D,
        )
    raise ValueError(f"unsupported Gen-2 observation dimension {architecture.obs_dim}")


class Gen2ObsEncoder(BaseFeaturesExtractor):
    """Frozen input normalization followed by a small tanh MLP encoder.

    The normalization statistics are **buffers**, so they travel inside the
    checkpoint ``state_dict`` and can never drift between BC, DAgger, PPO and
    inference.  They are latched once from the expert corpus and then frozen;
    a running normalizer would make a DAgger round silently rescale everything
    the previous round learned.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        *,
        hidden: Sequence[int] = GEN2_ARCH.encoder_hidden,
        obs_mean: Optional[Sequence[float]] = None,
        obs_std: Optional[Sequence[float]] = None,
        expected_obs_dim: int = OBS_DIM_LOCAL_TRANSITION,
        expected_contract: str = OBS_ENCODING_VERSION_LOCAL_TRANSITION,
        legacy_prefix_dim: int = OBS_DIM_LOCAL_TRANSITION,
    ) -> None:
        features_dim = int(hidden[-1])
        super().__init__(observation_space, features_dim=features_dim)
        obs_dim = int(observation_space.shape[0])
        if obs_dim != int(expected_obs_dim):
            raise ValueError(
                f"Gen-2 requires the {int(expected_obs_dim)}-feature contract "
                f"{expected_contract!r}, got obs_dim={obs_dim}"
            )
        mean = np.zeros(obs_dim, np.float32) if obs_mean is None else np.asarray(obs_mean, np.float32)
        std = np.ones(obs_dim, np.float32) if obs_std is None else np.asarray(obs_std, np.float32)
        if mean.shape != (obs_dim,) or std.shape != (obs_dim,):
            raise ValueError("normalization statistics must be one value per feature")
        self.register_buffer("obs_mean", th.as_tensor(mean, dtype=th.float32))
        self.register_buffer("obs_std", th.as_tensor(np.maximum(std, 1e-3), dtype=th.float32))
        self.legacy_prefix_dim = int(legacy_prefix_dim)

        layers: list = []
        previous = obs_dim
        for size in hidden:
            layers.append(nn.Linear(previous, int(size)))
            layers.append(nn.Tanh())
            previous = int(size)
        self.encoder = nn.Sequential(*layers)

    def set_normalization(self, mean: Sequence[float], std: Sequence[float]) -> None:
        """Latch the corpus statistics.  Call once, before behaviour cloning."""
        mean_array = np.asarray(mean, dtype=np.float32).reshape(-1)
        std_array = np.asarray(std, dtype=np.float32).reshape(-1)
        if mean_array.shape != self.obs_mean.shape or std_array.shape != self.obs_std.shape:
            raise ValueError("normalization statistics have the wrong shape")
        with th.no_grad():
            self.obs_mean.copy_(th.as_tensor(mean_array))
            self.obs_std.copy_(th.as_tensor(np.maximum(std_array, 1e-3)))

    def forward(self, observations: th.Tensor) -> th.Tensor:
        normalized = (observations - self.obs_mean) / self.obs_std
        # Preserve the parent's exact 35-column GEMM when a contract appends
        # neutral, zero-weight features.  Evaluating one 38-column GEMM can
        # change floating-point reduction order even when its last columns are
        # zero.  Splitting the legacy and appended contributions makes neutral
        # recurrent parity exact while still allowing PPO to learn the new
        # columns normally after the first update.
        if normalized.shape[-1] > self.legacy_prefix_dim:
            first = self.encoder[0]
            legacy = F.linear(
                normalized[..., : self.legacy_prefix_dim],
                # The slice has a 38-column stride; make it physically 35-wide
                # so PyTorch selects the same GEMM/reduction as the parent.
                first.weight[..., : self.legacy_prefix_dim].contiguous(),
                first.bias,
            )
            appended = F.linear(
                normalized[..., self.legacy_prefix_dim :],
                first.weight[..., self.legacy_prefix_dim :],
                None,
            )
            encoded = th.tanh(legacy + appended)
            return self.encoder[2:](encoded)
        return self.encoder(normalized)


def build_gen2_recurrent_ppo(
    env,
    *,
    architecture: Gen2Architecture = GEN2_ARCH,
    learning_rate: float = 3e-4,
    n_steps: int = 256,
    batch_size: int = 256,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    target_kl: Optional[float] = 0.02,
    seed: Optional[int] = None,
    device: str = "cpu",
    obs_mean: Optional[Sequence[float]] = None,
    obs_std: Optional[Sequence[float]] = None,
    tensorboard_log: Optional[str] = None,
    verbose: int = 0,
):
    """Construct the Gen-2 ``RecurrentPPO``.  BC optimizes this object directly."""
    from sb3_contrib import RecurrentPPO

    observation_contract, feature_names, _bounds = _observation_contract(architecture)

    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=list(architecture.head_hidden), vf=list(architecture.head_hidden)),
        features_extractor_class=Gen2ObsEncoder,
        features_extractor_kwargs=dict(
            hidden=tuple(architecture.encoder_hidden),
            obs_mean=obs_mean,
            obs_std=obs_std,
            expected_obs_dim=int(architecture.obs_dim),
            expected_contract=observation_contract,
            legacy_prefix_dim=(
                int(architecture.obs_dim)
                if int(architecture.obs_dim) <= OBS_DIM_LOCAL_TRANSITION
                else OBS_DIM_LOCAL_TRANSITION
            ),
        ),
        share_features_extractor=True,
        lstm_hidden_size=int(architecture.lstm_hidden_size),
        n_lstm_layers=int(architecture.n_lstm_layers),
        shared_lstm=bool(architecture.shared_lstm),
        enable_critic_lstm=bool(architecture.enable_critic_lstm),
    )
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=int(n_steps),
        batch_size=int(batch_size),
        n_epochs=int(n_epochs),
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        target_kl=target_kl,
        policy_kwargs=policy_kwargs,
        seed=seed,
        device=device,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
    )
    # Provenance travels with the checkpoint so a stale policy can never be
    # loaded into a different contract by accident.
    model.gen2_architecture = architecture.as_dict()
    model.gen2_obs_contract = observation_contract
    model.gen2_action_contract = GEN2_ACTION_CONTRACT
    model.gen2_feature_names = list(feature_names)
    return model


def gen2_observation_space(
    architecture: Gen2Architecture = GEN2_ARCH,
) -> spaces.Box:
    _contract, _names, bounds = _observation_contract(architecture)
    low = np.asarray([b[0] for b in bounds], dtype=np.float32)
    high = np.asarray([b[1] for b in bounds], dtype=np.float32)
    return spaces.Box(
        low=low,
        high=high,
        shape=(int(architecture.obs_dim),),
        dtype=np.float32,
    )


def gen2_action_space() -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)


def build_gen2_policy_for_training(
    *,
    architecture: Gen2Architecture = GEN2_ARCH,
    obs_mean: Optional[Sequence[float]] = None,
    obs_std: Optional[Sequence[float]] = None,
    seed: Optional[int] = None,
    device: str = "cpu",
    **ppo_kwargs: Any,
):
    """Build the Gen-2 ``RecurrentPPO`` without needing a live simulator."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([lambda: _GymContractEnv(architecture)])
    return build_gen2_recurrent_ppo(
        env,
        architecture=architecture,
        obs_mean=obs_mean,
        obs_std=obs_std,
        seed=seed,
        device=device,
        **ppo_kwargs,
    )


class _GymContractEnv(gym.Env):
    """A gymnasium stub carrying only the Gen-2 contract spaces.

    ``RecurrentPPO`` needs an env to read spaces from at construction time.
    Behaviour cloning never rolls this out -- it exists so building a network
    does not have to pay a HoloOcean launch.
    """

    metadata: Dict[str, Any] = {"render_modes": []}
    render_mode = None

    def __init__(self, architecture: Gen2Architecture = GEN2_ARCH) -> None:
        super().__init__()
        self.architecture = architecture
        self.observation_space = gen2_observation_space(architecture)
        self.action_space = gen2_action_space()

    def reset(self, *, seed: Optional[int] = None, options=None):
        return np.zeros(int(self.architecture.obs_dim), np.float32), {}

    def step(self, _action):
        return (
            np.zeros(int(self.architecture.obs_dim), np.float32),
            0.0,
            True,
            False,
            {},
        )

    def close(self) -> None:
        return None

    def render(self):
        return None


class Gen2RecurrentController:
    """Inference wrapper: ``action = learned_policy(observation, recurrent_state)``.

    This is the ONLY thing that drives a vehicle at Gen-2 inference time.  It
    has no expert, no rules, no fallback, no blending and no course knowledge.
    It owns exactly one piece of state -- the LSTM hidden/cell pair -- and
    resets it on, and only on, an episode boundary.
    """

    def __init__(self, model, *, deterministic: bool = True) -> None:
        self.model = model
        self.deterministic = bool(deterministic)
        self._states: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._episode_started = True
        self.steps = 0

    def reset(self) -> None:
        """Drop the recurrent state.  Call exactly once per episode."""
        self._states = None
        self._episode_started = True
        self.steps = 0

    @property
    def recurrent_state(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return self._states

    def act(self, observation: np.ndarray, first_step: bool = False) -> np.ndarray:
        """Return the 4-D action for one 35-feature observation."""
        if first_step:
            self.reset()
        vector = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        expected_dim = int(self.model.observation_space.shape[0])
        if vector.shape[1] != expected_dim:
            raise ValueError(
                f"Gen-2 controller expects {expected_dim} features, "
                f"got {vector.shape[1]}"
            )
        starts = np.asarray([self._episode_started], dtype=bool)
        action, self._states = self.model.predict(
            vector,
            state=self._states,
            episode_start=starts,
            deterministic=self.deterministic,
        )
        self._episode_started = False
        self.steps += 1
        return np.clip(
            np.asarray(action, dtype=np.float32).reshape(ACTION_DIM), -1.0, 1.0
        )

    def __call__(self, observation: np.ndarray, first_step: bool = False) -> np.ndarray:
        return self.act(observation, first_step)


def policy_parameter_count(model) -> Dict[str, int]:
    """Parameter counts by block, for the report and for transfer assertions."""
    policy = model.policy
    blocks = {
        "features_extractor": getattr(policy, "features_extractor", None),
        "lstm_actor": getattr(policy, "lstm_actor", None),
        "lstm_critic": getattr(policy, "lstm_critic", None),
        "mlp_extractor": getattr(policy, "mlp_extractor", None),
        "action_net": getattr(policy, "action_net", None),
        "value_net": getattr(policy, "value_net", None),
    }
    counts = {
        name: int(sum(p.numel() for p in module.parameters()))
        for name, module in blocks.items()
        if module is not None
    }
    counts["log_std"] = int(policy.log_std.numel()) if hasattr(policy, "log_std") else 0
    counts["total"] = int(sum(p.numel() for p in policy.parameters()))
    counts["trainable"] = int(
        sum(p.numel() for p in policy.parameters() if p.requires_grad)
    )
    return counts


def actor_state_dict(model) -> Dict[str, th.Tensor]:
    """The parameters behaviour cloning is allowed to move."""
    policy = model.policy
    out: Dict[str, th.Tensor] = {}
    for prefix in ("features_extractor", "lstm_actor", "action_net"):
        module = getattr(policy, prefix, None)
        if module is None:
            continue
        for key, value in module.state_dict().items():
            out[f"{prefix}.{key}"] = value.detach().clone()
    extractor = getattr(policy, "mlp_extractor", None)
    if extractor is not None and hasattr(extractor, "policy_net"):
        for key, value in extractor.policy_net.state_dict().items():
            out[f"mlp_extractor.policy_net.{key}"] = value.detach().clone()
    return out


def write_policy_manifest(model, path: str | Path, **extra: Any) -> Path:
    """Record architecture, contracts and parameter counts next to a checkpoint."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    architecture_data = getattr(model, "gen2_architecture", GEN2_ARCH.as_dict())
    obs_dim = int(model.observation_space.shape[0])
    if obs_dim == OBS_DIM_LOCAL_TRANSITION_GATE_YAW:
        contract = OBS_ENCODING_VERSION_LOCAL_TRANSITION_GATE_YAW
        feature_names = FEATURE_NAMES_LOCAL_TRANSITION_GATE_YAW
    else:
        contract = OBS_ENCODING_VERSION_LOCAL_TRANSITION
        feature_names = FEATURE_NAMES_LOCAL_TRANSITION
    payload = {
        "schema_version": "gen2_policy_manifest_v1",
        "architecture": architecture_data,
        "observation_contract": contract,
        "action_contract": GEN2_ACTION_CONTRACT,
        "observation_dim": obs_dim,
        "action_dim": ACTION_DIM,
        "feature_names": list(feature_names),
        "parameters": policy_parameter_count(model),
        **extra,
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target
