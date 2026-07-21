"""Load built-in or external participant controllers."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Optional

from marine_race_arena.participants.controller_interface import validate_controller_instance


class ControllerError(RuntimeError):
    """Raised when a participant controller cannot be loaded or executed."""


class ControllerLoader:
    BUILT_INS = {
        "oracle": (
            "marine_race_arena.controllers.oracle_gate_follower",
            "OracleGateFollowerController",
        ),
        "rule_gate_baseline": (
            "marine_race_arena.controllers.official_baselines",
            "RuleGateBaselineController",
        ),
        "rule_gate_center_then_commit": (
            "marine_race_arena.controllers.official_baselines",
            "RuleGateCenterThenCommitController",
        ),
        "leader_follower": (
            "marine_race_arena.controllers.leader_follower",
            "LeaderFollowerController",
        ),
        # Learned controller (feature/rl-controller). Imported lazily only when
        # selected; requires the RL dependencies (requirements-rl.txt) and a
        # trained model via model_path / $MARINE_RACE_RL_MODEL.
        "rl_gate_controller": (
            "marine_race_arena.learning.rl_controller",
            "RLGateController",
        ),
        "keyboard": (
            "marine_race_arena.controllers.keyboard_manual",
            "KeyboardManualController",
        ),
        "manual": (
            "marine_race_arena.controllers.keyboard_manual",
            "KeyboardManualController",
        ),
        "manual_keyboard": (
            "marine_race_arena.controllers.keyboard_manual",
            "KeyboardManualController",
        ),
        "pygame": (
            "marine_race_arena.controllers.pygame_manual",
            "PygameManualController",
        ),
        "pygame_keyboard": (
            "marine_race_arena.controllers.pygame_manual",
            "PygameManualController",
        ),
        "student_template": (
            "marine_race_arena.controllers.student_template",
            "StudentController",
        ),
    }

    def load(
        self,
        controller_reference: str,
        controller_class: Optional[str] = None,
        constructor_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> object:
        """Load a controller. ``constructor_kwargs`` are passed to the controller
        constructor, but only the ones its ``__init__`` actually accepts, so
        controllers that do not declare an option (e.g. the rule baselines) receive
        no unsupported argument."""
        if controller_reference in self.BUILT_INS:
            module_name, class_name = self.BUILT_INS[controller_reference]
            module = importlib.import_module(module_name)
            return self._instantiate(module, class_name, constructor_kwargs)

        if controller_reference.endswith(".py"):
            module = self._load_from_file(Path(controller_reference))
            if not controller_class:
                raise ControllerError(
                    "controller_class is required when loading a controller from a file path."
                )
            return self._instantiate(module, controller_class, constructor_kwargs)

        module_name, class_name = self._split_module_and_class(controller_reference, controller_class)
        module = importlib.import_module(module_name)
        return self._instantiate(module, class_name, constructor_kwargs)

    def _instantiate(
        self,
        module: ModuleType,
        class_name: str,
        constructor_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> object:
        controller_type = getattr(module, class_name, None)
        if controller_type is None:
            raise ControllerError(f"Controller class '{class_name}' was not found in {module.__name__}.")
        kwargs = self._accepted_kwargs(controller_type, constructor_kwargs)
        controller = controller_type(**kwargs)
        validate_controller_instance(controller)
        return controller

    @staticmethod
    def _accepted_kwargs(
        controller_type: type, constructor_kwargs: Optional[Mapping[str, Any]]
    ) -> dict:
        """Keep only the constructor keywords this controller class accepts.

        Drops ``None`` values (so an unset CLI option is a no-op and, for example,
        an RL controller falls back to its environment variable). Controllers with a
        ``**kwargs`` constructor receive everything non-None."""
        if not constructor_kwargs:
            return {}
        provided = {key: value for key, value in constructor_kwargs.items() if value is not None}
        if not provided:
            return {}
        init = getattr(controller_type, "__init__", None)
        # A class that does not define its own __init__ (inherits object.__init__,
        # whose reported signature is (*args, **kwargs)) accepts no keyword arguments.
        if init is None or init is object.__init__:
            return {}
        try:
            parameters = inspect.signature(init).parameters
        except (TypeError, ValueError):  # pragma: no cover - builtin/edge constructors
            return {}
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            return dict(provided)
        return {key: value for key, value in provided.items() if key in parameters}

    def _load_from_file(self, path: Path) -> ModuleType:
        if not path.exists():
            raise ControllerError(f"Controller file does not exist: {path}")
        module_name = f"marine_race_external_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ControllerError(f"Could not create an import spec for controller file: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _split_module_and_class(
        self, controller_reference: str, controller_class: Optional[str]
    ) -> tuple[str, str]:
        if ":" in controller_reference:
            module_name, class_name = controller_reference.split(":", 1)
            return module_name, controller_class or class_name
        if controller_class:
            return controller_reference, controller_class
        parts = controller_reference.split(".")
        if len(parts) < 2:
            raise ControllerError(
                "Controller reference must be a built-in name, file path, module:Class, "
                "or fully qualified module.Class."
            )
        return ".".join(parts[:-1]), parts[-1]
