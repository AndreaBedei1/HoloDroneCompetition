"""Autonomous Gen-2 phase pipeline: collect -> clone -> DAgger.

Runs the phases in order, gates each one on the previous phase's measured
result, and writes ``pipeline_state.json`` after every step so status is a file
read rather than a poll of a process.

    python -m marine_race_arena.learning.gen2.pipeline \\
        --base results/rl/gen2 --episodes 400 --workers 5

Every phase is resumable.  Collection skips shards that already exist, BC
retrains from the corpus on disk, and a DAgger round skips episodes it has
already recorded, so re-issuing the same command after an interruption
continues rather than restarts.

The pipeline stops rather than escalating when a phase does not earn the next
one: if behaviour cloning cannot drive 2 and 3 gates, running DAgger on 5-8
gate courses would only aggregate labels for states the learner reaches by
accident.  A stop with a recorded reason is the useful outcome there.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

STATE_SCHEMA = "gen2_pipeline_state_v1"


class PipelineAlreadyRunning(RuntimeError):
    """Raised when another pipeline already owns this base directory."""


def _process_alive(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(int(pid))
    except Exception:
        return True  # assume alive: refusing is safer than racing


class Pipeline:
    """Owns one base directory, exclusively.

    The exclusivity is not cosmetic.  Two pipelines on the same base write the
    same BC checkpoint, the same DAgger round directory and the same state
    file, and each one's closed-loop evaluation competes with the other's for
    simulator engines.  The results are then neither run's -- which is exactly
    what happened on 2026-08-18, when a stopped shell left its detached child
    alive and a second pipeline started beside it.  The two runs graded stage B
    at 0.90 and 0.7667 on identical seeds.
    """

    def __init__(self, base: str | Path, *, force: bool = False) -> None:
        self.base = Path(base)
        self.base.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.base / "pipeline.lock"
        self._acquire_lock(force=force)
        self.state_path = self.base / "pipeline_state.json"
        self.state: Dict[str, Any] = {
            "schema_version": STATE_SCHEMA,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "phases": [],
            "current": None,
            "finished": False,
        }
        if self.state_path.exists():
            try:
                self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        self._flush()

    def _acquire_lock(self, *, force: bool = False) -> None:
        import os

        if self.lock_path.exists() and not force:
            try:
                holder = json.loads(self.lock_path.read_text(encoding="utf-8"))
            except Exception:
                holder = {}
            pid = int(holder.get("pid", -1))
            if pid > 0 and _process_alive(pid):
                raise PipelineAlreadyRunning(
                    f"pipeline {pid} already owns {self.base} (started "
                    f"{holder.get('started_utc')}). Stop it first, or pass "
                    f"force=True if you are certain it is dead. Note that "
                    f"stopping a shell does not stop its detached child."
                )
        self.lock_path.write_text(json.dumps({
            "pid": os.getpid(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "base": str(self.base),
        }, indent=2), encoding="utf-8")

    def release(self) -> None:
        try:
            self.lock_path.unlink()
        except OSError:
            pass

    def _flush(self) -> None:
        tmp = self.state_path.with_suffix(".json.partial")
        tmp.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def begin(self, name: str, detail: Mapping[str, Any]) -> None:
        self.state["current"] = {
            "phase": name,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "detail": dict(detail),
        }
        self._flush()

    def finish(self, name: str, result: Mapping[str, Any], *, ok: bool = True) -> None:
        entry = dict(self.state.get("current") or {"phase": name})
        entry.update({
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ok": bool(ok),
            "result": dict(result),
        })
        self.state["phases"].append(entry)
        self.state["current"] = None
        self._flush()

    def stop(self, reason: str) -> None:
        self.state["finished"] = True
        self.state["stop_reason"] = reason
        self._flush()


def phase_collect(
    pipeline: Pipeline, *, episodes: int, workers: int, adapter: str, weights: str
) -> Dict[str, Any]:
    from marine_race_arena.learning.gen2.collect_expert import GATE_WEIGHT_PRESETS, collect

    out = pipeline.base / "expert_corpus"
    pipeline.begin("collect", {"episodes": episodes, "workers": workers, "out": str(out)})
    summary = collect(
        out, episodes=episodes, workers=workers, adapter=adapter,
        gate_weights=GATE_WEIGHT_PRESETS[weights],
    )
    pipeline.finish("collect", {
        "corpus_sha256": summary["corpus_sha256"],
        "statistics": summary["statistics"],
        "wall_time_s": summary["wall_time_s"],
    })
    return summary


def phase_bc(
    pipeline: Pipeline, *, epochs: int, stages: Sequence[str],
    episodes_per_stage: int, adapter: str, seed: int,
) -> Dict[str, Any]:
    from marine_race_arena.learning.gen2.bc_recurrent import BCConfig
    from marine_race_arena.learning.gen2.train_bc import train

    out = pipeline.base / "bc_v1"
    pipeline.begin("bc", {"out": str(out), "stages": list(stages), "epochs": epochs})
    summary = train(
        [pipeline.base / "expert_corpus"], out,
        config=BCConfig(epochs=epochs, seed=seed),
        stages=stages, episodes_per_stage=episodes_per_stage, adapter=adapter,
    )
    pipeline.finish("bc", {
        "checkpoint": summary["checkpoint"],
        "validation_mse": summary["bc"]["validation"]["mse"],
        "per_axis_correlation": summary["bc"]["validation"]["per_axis_correlation"],
        "stages": summary["stages"],
        "recommendation": summary["recommendation"],
        "parameters": summary["parameters"]["total"],
    })
    return summary


def phase_dagger(
    pipeline: Pipeline, *, bc_checkpoint: str, rounds: Sequence[int],
    episodes_per_round: int, workers: int, adapter: str, epochs: int, seed: int,
) -> Dict[str, Any]:
    from marine_race_arena.learning.gen2.bc_recurrent import BCConfig
    from marine_race_arena.learning.gen2.run_dagger import run_campaign

    pipeline.begin("dagger", {"rounds": list(rounds), "episodes_per_round": episodes_per_round})
    summary = run_campaign(
        pipeline.base, bc_checkpoint,
        rounds=rounds, episodes_per_round=episodes_per_round,
        workers=workers, adapter=adapter, config=BCConfig(epochs=epochs, seed=seed),
    )
    pipeline.finish("dagger", {
        "best_dagger": summary["best_dagger"],
        "rounds": [
            {
                "round": r["round_index"],
                "lengths": r["lengths"],
                "validation_completion": r["validation"].get("overall_completion_rate"),
                "gate1_to_gate2": r["validation"].get("gate1_to_gate2_transition_rate"),
                "pure_learner_rollout": r["blending_audit"]["pure_learner_rollout"],
                "advance": r["advance"],
                "reason": r["advance_reason"],
            }
            for r in summary["rounds"]
        ],
    })
    return summary


def run(
    base: str | Path,
    *,
    episodes: int = 400,
    workers: int = 5,
    adapter: str = "holoocean",
    weights: str = "stage1",
    bc_epochs: int = 40,
    stages: Sequence[str] = ("A", "B", "C"),
    episodes_per_stage: int = 30,
    dagger_rounds: Sequence[int] = (1, 2, 3),
    episodes_per_round: int = 120,
    seed: int = 0,
    skip_collect: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    pipeline = Pipeline(base, force=force)
    try:
        if not skip_collect:
            phase_collect(
                pipeline, episodes=episodes, workers=workers,
                adapter=adapter, weights=weights,
            )

        bc = phase_bc(
            pipeline, epochs=bc_epochs, stages=stages,
            episodes_per_stage=episodes_per_stage, adapter=adapter, seed=seed,
        )
        recommendation = bc["recommendation"]
        if recommendation["phase"] == "more_bc":
            pipeline.stop(
                "behaviour cloning has not earned DAgger yet: " + recommendation["reason"]
            )
            return pipeline.state

        phase_dagger(
            pipeline, bc_checkpoint=bc["checkpoint"], rounds=dagger_rounds,
            episodes_per_round=episodes_per_round, workers=workers,
            adapter=adapter, epochs=bc_epochs, seed=seed,
        )
        pipeline.stop("DAgger campaign complete; next gate is the pre-PPO readiness check")
    except Exception as exc:
        pipeline.state["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc()[-4000:],
        }
        pipeline.stop(f"pipeline failed in phase {pipeline.state.get('current')}: {exc}")
        raise
    finally:
        pipeline.release()
    return pipeline.state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Gen-2 phase pipeline")
    parser.add_argument("--base", required=True)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--adapter", default="holoocean", choices=("holoocean", "fallback"))
    parser.add_argument("--weights", default="stage1")
    parser.add_argument("--bc-epochs", type=int, default=40)
    parser.add_argument("--stages", default="A,B,C")
    parser.add_argument("--episodes-per-stage", type=int, default=30)
    parser.add_argument("--dagger-rounds", default="1,2,3")
    parser.add_argument("--episodes-per-round", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument(
        "--force", action="store_true",
        help="take the base-directory lock even if another pipeline holds it",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    state = run(
        args.base,
        episodes=args.episodes, workers=args.workers, adapter=args.adapter,
        weights=args.weights, bc_epochs=args.bc_epochs,
        stages=tuple(s.strip().upper() for s in args.stages.split(",") if s.strip()),
        episodes_per_stage=args.episodes_per_stage,
        dagger_rounds=[int(v) for v in args.dagger_rounds.split(",") if v.strip()],
        episodes_per_round=args.episodes_per_round, seed=args.seed,
        skip_collect=args.skip_collect, force=args.force,
    )
    print(json.dumps({
        "phases": [
            {"phase": p["phase"], "ok": p.get("ok"), "result": p.get("result")}
            for p in state.get("phases", [])
        ],
        "stop_reason": state.get("stop_reason"),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
