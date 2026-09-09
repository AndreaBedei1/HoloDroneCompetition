"""Isolated learning package for Marine Race Arena (feature/rl-controller).

This package adds imitation- and reinforcement-learning support *on top of* the
existing benchmark without changing the official observation contract, the
independent referee, gate validation, official scoring or the official tracks.

Nothing here is imported by the benchmark runtime; the normal runner, referee
and rule-based controllers do not depend on this package. Heavy optional
dependencies (Gymnasium, PyTorch, Stable-Baselines3) are imported lazily inside
the modules that need them, so the numpy-only pieces (observation encoding,
reward, dataset) remain importable in the plain benchmark environment.

Layers, kept strictly separate:
  * policy observation  -> onboard-only (``observation_encoder``);
  * training reward      -> may use privileged simulator/referee state (``reward``);
  * official evaluation  -> the unchanged referee and benchmark metrics.
"""

from marine_race_arena.learning.config import (
    ACTION_AXES,
    ACTION_DIM,
    FEATURE_NAMES,
    OBS_DIM,
    LearningContext,
)

__all__ = [
    "ACTION_AXES",
    "ACTION_DIM",
    "FEATURE_NAMES",
    "OBS_DIM",
    "LearningContext",
]
