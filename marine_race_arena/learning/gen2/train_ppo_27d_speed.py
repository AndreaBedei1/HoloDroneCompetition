"""Full-circuit 27-D PPO campaign with a bounded time-efficiency objective."""
from __future__ import annotations

import json, shutil, time, subprocess
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import RecurrentPPO

from marine_race_arena.config.benchmark_tasks import BENCHMARK_TASK_CLEAN_GATE
from marine_race_arena.learning.config_local_transition_27d import OBS_ENCODING_VERSION_LOCAL_TRANSITION_27D
from marine_race_arena.learning.gen2 import track_fragments as tf
from marine_race_arena.learning.gen2.fog_contract import verify_fog_sources
from marine_race_arena.learning.gen2.ppo_reliability import ReliabilityReward
from marine_race_arena.learning.gen2.train_ppo_27d import TrainingSource27d
from marine_race_arena.learning.gym_env import MarineRaceGymEnv

SPEED_REWARD = {
    "gate_crossing_bonus": 20.0,
    "completion_bonus": 50.0,
    "time_penalty_per_step": -0.01,
    "terminal_failure_penalty": -55.0,
    "collision_penalty": -35.0,
    "energy_penalty": 0.0,
    "jerk_penalty": 0.0,
}
TRACKS = tuple(tf.OFFICIAL_TRACKS)
MILESTONES = (50_000, 100_000, 150_000, 200_000, 300_000, 400_000, 500_000)
ENV_RECYCLE_INTERVAL = 10_000


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def audit_processes(path: Path, label: str) -> None:
    command = "Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^(python|pythonw|Holodeck)(\\.exe)?$'} | Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Depth 4 -Compress"
    raw = subprocess.check_output(["powershell", "-NoProfile", "-Command", command], text=True, encoding="utf-8", errors="replace")
    try:
        rows = json.loads(raw) if raw.strip() else []
    except json.JSONDecodeError:
        rows = {"parse_error": raw}
    write_json(path, {"label": label, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "processes": rows})


def sources() -> list[TrainingSource27d]:
    out: list[TrainingSource27d] = []
    for track in TRACKS:
        path = tf.track_path(track)
        gate_count = len(tf.load_track(track)["track"]["gate_sequence"])
        for copy in range(2):
            out.append(TrainingSource27d(
                name=f"speed_full_{track}_{copy:02d}", track=track,
                kind="full_circuit", path=str(path), gate_count=gate_count,
                max_steps=max(2000, gate_count * 900),
            ))
    return out


def make_env(source: TrainingSource27d, seed: int):
    return MarineRaceGymEnv(
        source.path, seed=int(seed), adapter="holoocean", allow_fallback=False,
        max_steps=int(source.max_steps), official=True, current_profile="none",
        benchmark_task=BENCHMARK_TASK_CLEAN_GATE,
        reward_fn=ReliabilityReward(
            source.gate_count,
            collision_penalty=SPEED_REWARD["collision_penalty"],
            completion_bonus=SPEED_REWARD["completion_bonus"],
            time_penalty=SPEED_REWARD["time_penalty_per_step"],
            terminal_failure_penalty=SPEED_REWARD["terminal_failure_penalty"],
        ),
        observation_encoding_version=OBS_ENCODING_VERSION_LOCAL_TRANSITION_27D,
    )


def make_vec(run_dir: Path, seed: int, copy_id: int = 0):
    from stable_baselines3.common.vec_env import SubprocVecEnv
    # Keep one worker per real circuit to avoid simulator restart storms on
    # this host.  Copy 0/1 are alternated at environment recycle boundaries,
    # preserving two balanced source passes per circuit over the campaign.
    src = [s for s in sources() if s.name.endswith(f"_{copy_id:02d}")]
    factories = [partial(make_env, s, int(seed) + i * 10007) for i, s in enumerate(src)]
    return SubprocVecEnv(factories, start_method="spawn"), src


def callback_class(run_dir: Path):
    from stable_baselines3.common.callbacks import BaseCallback
    class Callback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=1)
            self.records=[]; self.episodes=[]; self.rewards=[]; self.action_sum=np.zeros(4); self.action_sq=np.zeros(4); self.action_n=0; self.saturated=0
        def _on_step(self):
            r=self.locals.get("rewards")
            if r is not None: self.rewards.extend(float(x) for x in np.asarray(r).reshape(-1))
            a=self.locals.get("actions")
            if a is not None:
                arr=np.asarray(a); self.action_sum += arr.sum(axis=0); self.action_sq += (arr*arr).sum(axis=0); self.action_n += arr.shape[0]; self.saturated += int(np.sum(np.abs(arr)>=.995))
            dones = self.locals.get("dones")
            infos = self.locals.get("infos")
            for done,info in zip(dones if dones is not None else [], infos if infos is not None else []):
                if done:
                    self.episodes.append({"timestep":int(self.num_timesteps),"completed":bool(info.get("completed",False)),"gates":int(info.get("gates_completed",0)),"status":str(info.get("status","")),"collision_events":int(info.get("collision_events",0))+int(info.get("obstacle_collision_events",0)),"missed_gate_attempts":int(info.get("missed_gate_attempts",0)),"out_of_bounds_events":int(info.get("out_of_bounds_events",0))})
            if not self.records or int(self.num_timesteps)-self.records[-1]["timestep"] >= 2048:
                self._record()
            return True
        def _record(self):
            n=max(1,len(self.episodes)); stats={"timestep":int(self.num_timesteps),"episodes":len(self.episodes),"completion_rate":sum(x["completed"] for x in self.episodes)/n,"mean_gates":sum(x["gates"] for x in self.episodes)/n,"safety_events":sum(x["collision_events"]+x["missed_gate_attempts"]+x["out_of_bounds_events"] for x in self.episodes),"mean_reward_last500":float(np.mean(self.rewards[-500:])) if self.rewards else 0.0,"action_mean":(self.action_sum/max(1,self.action_n)).tolist(),"action_std":np.sqrt(np.maximum(0,self.action_sq/max(1,self.action_n)-(self.action_sum/max(1,self.action_n))**2)).tolist(),"action_saturation":self.saturated/max(1,4*self.action_n)}
            logs=getattr(self.logger,"name_to_value",{}) if self.logger else {}; stats.update({"approx_kl":float(logs.get("train/approx_kl",0.0)),"policy_loss":float(logs.get("train/policy_gradient_loss",0.0)),"value_loss":float(logs.get("train/value_loss",0.0)),"entropy_loss":float(logs.get("train/entropy_loss",0.0)),"clip_fraction":float(logs.get("train/clip_fraction",0.0))}); self.records.append(stats); write_json(run_dir/"training_metrics.json",{"records":self.records,"episodes":self.episodes}); print(json.dumps(stats),flush=True)
        def _on_rollout_end(self): self._record()
        def _on_training_end(self): self._record()
    return Callback


def quick_eval(checkpoint: Path, out: Path, seed: int) -> dict:
    from marine_race_arena.learning.gen2.evaluation import run_policy_episode
    from marine_race_arena.learning.gen2.recurrent_policy import Gen2RecurrentController
    rows=[]
    for i,track in enumerate(TRACKS):
        model=RecurrentPPO.load(str(checkpoint),device="cpu"); controller=Gen2RecurrentController(model,deterministic=True)
        row=run_policy_episode(controller,tf.track_path(track),seed=int(seed+i),adapter="holoocean",allow_fallback=False,max_steps=max(2000,len(tf.load_track(track)["track"]["gate_sequence"])*900)).as_row(); row.update({"track":track,"checkpoint":str(checkpoint)}); rows.append(row); controller=None; model=None
    report={"rows":rows,"actual_adapter":"holoocean","fallback_used":False,"fog":verify_fog_sources([tf.track_path(t) for t in TRACKS])}; write_json(out,report); return report


def run(parent: str, out_dir: str, max_timesteps: int = 500_000, seed: int = 20260908):
    from stable_baselines3.common.utils import get_schedule_fn
    from marine_race_arena.learning.train_ppo_transition import close_vec_env_safely
    run_dir=Path(out_dir); run_dir.mkdir(parents=True,exist_ok=True); parent_path=Path(parent); immutable=parent_path.read_bytes()
    sources_list=sources(); fog=verify_fog_sources([Path(s.path) for s in sources_list]); write_json(run_dir/"campaign_config.json",{"start_checkpoint":str(parent_path),"observation_contract":OBS_ENCODING_VERSION_LOCAL_TRANSITION_27D,"observation_dim":27,"sources":[asdict(s) for s in sources_list],"source_balance":"two alternating full-circuit copies per track; no fragments/synthetic","worker_count":3,"copy_schedule":"copy 0/1 alternated at each environment recycle","fog":fog,"reward":SPEED_REWARD,"ppo":{"learning_rate":1e-5,"clip_range":0.08,"target_kl":0.01,"n_epochs":2,"batch_size":128,"n_steps":256,"max_grad_norm":0.5,"ent_coef":0.0},"milestones":MILESTONES,"environment_recycle_interval":ENV_RECYCLE_INTERVAL})
    env=None; stages=[]
    try:
        copy_id=0
        env,_=make_vec(run_dir,seed,copy_id)
        # Loading with env is required when the source checkpoint was trained
        # with a different worker count (the candidate used nine sources).
        model=RecurrentPPO.load(str(parent_path), env=env, device="cpu")
        model.learning_rate=1e-5; model.lr_schedule=get_schedule_fn(1e-5); model.clip_range=get_schedule_fn(0.08); model.target_kl=0.01; model.n_epochs=2; model.max_grad_norm=0.5; model.ent_coef=0.0
        start=int(model.num_timesteps); current=start
        write_json(run_dir/"start_audit.json",{"parent_bytes_unchanged":parent_path.read_bytes()==immutable,"initial_num_timesteps":start,"all_parameters_trainable":all(p.requires_grad for p in model.policy.parameters()),"policy_log_std":model.policy.log_std.detach().cpu().numpy().tolist()})
        Callback=callback_class(run_dir)
        for target in [x for x in MILESTONES if x<=max_timesteps]:
            if target<=current: continue
            callback=Callback()
            # HoloOcean can occasionally lose a simulator after a long
            # multi-worker rollout.  Recycle the six owned environments at
            # bounded intervals while preserving the recurrent policy state.
            while current < target:
                chunk=min(ENV_RECYCLE_INTERVAL, target-current)
                model.learn(total_timesteps=chunk,callback=callback,reset_num_timesteps=False,progress_bar=False); current=int(model.num_timesteps)
                if current < target:
                    close_vec_env_safely(env); env=None
                    audit_processes(run_dir/f"process_audit_recycle_{current:06d}.json", f"recycle_{current}")
                    copy_id=1-copy_id
                    env,_=make_vec(run_dir,seed+target+current,copy_id); model.set_env(env)
            ck=run_dir/f"checkpoint_{target:06d}.zip"; model.save(str(ck)); shutil.copy2(ck,run_dir/"latest.zip"); write_json(run_dir/f"checkpoint_{target:06d}_sha.json",{"sha256":__import__('hashlib').sha256(ck.read_bytes()).hexdigest(),"timesteps":current})
            close_vec_env_safely(env); env=None
            audit_processes(run_dir/f"process_audit_{target:06d}_pre_eval.json", f"after_training_{target}")
            ev=quick_eval(ck,run_dir/f"evaluation_{target:06d}.json",seed+target); stages.append({"target":target,"actual_timesteps":current,"checkpoint":str(ck),"evaluation":ev}); write_json(run_dir/"training_decision_log.json",{"stages":stages,"selection":"closed_loop_completion_then_safety_then_time"})
            audit_processes(run_dir/f"process_audit_{target:06d}_post_eval.json", f"after_evaluation_{target}")
            copy_id=1-copy_id
            env,_=make_vec(run_dir,seed+target,copy_id); model.set_env(env)
        write_json(run_dir/"ppo_training_summary.json",{"actual_timesteps":current,"stages":stages,"start_checkpoint":str(parent_path)})
        return {"actual_timesteps":current,"stages":stages}
    finally:
        if env is not None:
            try: close_vec_env_safely(env)
            except Exception:
                try: env.close()
                except Exception: pass
        write_json(run_dir/"environment_close.json",{"closed":True})
        audit_processes(run_dir/"process_audit_end.json", "after_speed_campaign")

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--parent",required=True); ap.add_argument("--out",required=True); ap.add_argument("--max-timesteps",type=int,default=500000); ap.add_argument("--seed",type=int,default=20260908); a=ap.parse_args(); print(json.dumps(run(a.parent,a.out,a.max_timesteps,a.seed),indent=2))
