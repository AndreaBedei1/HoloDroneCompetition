"""Built-in marine race controllers."""

from marine_race_arena.controllers.acoustic_gate_follower import AcousticGateFollowerController
from marine_race_arena.controllers.keyboard_manual import KeyboardManualController
from marine_race_arena.controllers.official_baselines import (
    AcousticBaselineController,
    AcousticVisionBaselineController,
    VisionGateBaselineController,
)
from marine_race_arena.controllers.oracle_gate_follower import OracleGateFollowerController
from marine_race_arena.controllers.pygame_manual import PygameManualController
from marine_race_arena.controllers.student_template import StudentController

__all__ = [
    "AcousticGateFollowerController",
    "AcousticBaselineController",
    "AcousticVisionBaselineController",
    "VisionGateBaselineController",
    "KeyboardManualController",
    "OracleGateFollowerController",
    "PygameManualController",
    "StudentController",
]
