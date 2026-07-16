"""Built-in marine race controllers."""

from marine_race_arena.controllers.keyboard_manual import KeyboardManualController
from marine_race_arena.controllers.leader_follower import LeaderFollowerController
from marine_race_arena.controllers.local_course_tracker import (
    LocalCourseTracker,
    LocalCourseTrackerConfig,
)
from marine_race_arena.controllers.official_baselines import (
    RuleGateBaselineController,
    RuleGateCenterThenCommitController,
)
from marine_race_arena.controllers.oracle_gate_follower import OracleGateFollowerController
from marine_race_arena.controllers.pygame_manual import PygameManualController
from marine_race_arena.controllers.student_template import StudentController

__all__ = [
    "RuleGateBaselineController",
    "RuleGateCenterThenCommitController",
    "KeyboardManualController",
    "LeaderFollowerController",
    "LocalCourseTracker",
    "LocalCourseTrackerConfig",
    "OracleGateFollowerController",
    "PygameManualController",
    "StudentController",
]
