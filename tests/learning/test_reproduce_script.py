"""The generated PPO reproduction script is complete and executable (no ellipsis)."""

import pytest

from marine_race_arena.learning.reward import RewardConfig
from marine_race_arena.learning.train_workflow import build_reproduce_script


def _info(**over):
    info = {
        "track": "marine_race_arena/tracks/training/stage1_single_gate.json",
        "stage": "stage1",
        "algorithm": "ppo",
        "total_timesteps": 50000,
        "train_seed": 0,
        "eval_seeds": [900, 901],
        "hidden_sizes": [256, 256],
        "reward_config": RewardConfig(),
        "adapter_requested": "holoocean",
        "allow_fallback": False,
        "bc_initialized": True,
        "checkpoint_freq": 5000,
        "eval_freq": 5000,
        "ppo_kwargs": {"n_steps": 2048, "batch_size": 64, "n_epochs": 10, "learning_rate": 0.0003},
        "env_kwargs": {"adapter": "holoocean", "allow_fallback": False, "max_steps": 400},
        "bc_model_path": "results/rl/stage1/bc_rand_combined/best_model.pt",
        "bc_model_sha256": "deadbeef",
        "output_root": "results/rl",
        "run_dir": "results/rl/stage1/ppo/20260101_000000",
    }
    info.update(over)
    return info


def _python_body(script: str) -> str:
    start = script.index("python - <<'PY'") + len("python - <<'PY'")
    end = script.index("\nPY", start)
    return script[start:end]


def test_reproduce_script_is_complete_and_has_no_ellipsis():
    script = build_reproduce_script(_info(), commit="abc123")
    assert "..." not in script  # no unusable placeholders
    for token in (
        "abc123", "stage1_single_gate.json", "total_timesteps=50000", "train_seed=0",
        "eval_seeds=[900, 901]", "hidden_sizes=[256, 256]", "n_steps", "RewardConfig(",
        "adapter", "OUTPUT_ROOT", "resume=True", "BC_MODEL_PATH", "deadbeef",
    ):
        assert token in script, f"missing {token!r}"


def test_reproduce_python_body_compiles():
    script = build_reproduce_script(_info(), commit="abc123")
    body = _python_body(script)
    compile(body, "<reproduce>", "exec")  # syntactically valid Python


def test_reproduce_with_randomization_reconstructs_spec():
    from marine_race_arena.learning.randomization import StartRandomization

    env_kwargs = {"adapter": "holoocean", "allow_fallback": False,
                  "start_randomization": StartRandomization(lateral_offset_m=1.0)}
    script = build_reproduce_script(_info(env_kwargs=env_kwargs), commit="abc123")
    assert "StartRandomization(**" in script
    compile(_python_body(script), "<reproduce>", "exec")


def test_reproduce_without_bc_omits_bc_lines():
    script = build_reproduce_script(_info(bc_initialized=False, bc_model_path=None, bc_model_sha256=None), commit="abc123")
    assert "BC_MODEL_PATH" not in script
    assert "bc_model_path=" not in script
    compile(_python_body(script), "<reproduce>", "exec")
