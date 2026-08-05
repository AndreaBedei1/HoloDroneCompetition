"""Exact transition-track snapshots, geometry rendering, and video helpers."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from marine_race_arena.learning.holoocean_capacity import reserve_holoocean_engines
from marine_race_arena.learning.longrun_checkpoint import atomic_write_json
from marine_race_arena.learning.transition_curriculum import (
    DIFFICULTY_LEVELS,
    FULL_SEQUENCE_LENGTHS,
    TransitionGeometry,
    TransitionGeometrySampler,
    generate_transition_track,
)
from marine_race_arena.learning.transition_evaluation import (
    _benchmark_cases,
    aggregate_transition_benchmark,
    evaluate_local_transition_episode,
)


def snapshot_json(path: str | Path, *, attempts: int = 10) -> Dict[str, Any]:
    """Read an atomically replaced active track without touching the source."""

    source = Path(path)
    for _ in range(int(attempts)):
        before = source.stat()
        payload = source.read_bytes()
        after = source.stat()
        if (
            before.st_mtime_ns == after.st_mtime_ns
            and before.st_size == after.st_size == len(payload)
        ):
            return json.loads(payload.decode("utf-8"))
        time.sleep(0.01)
    raise RuntimeError(f"active track changed repeatedly while snapshotting: {source}")


def geometry_from_track(track: Mapping[str, Any]) -> TransitionGeometry:
    value = dict(track.get("universal_transition") or {})
    if not value:
        raise ValueError("track has no universal_transition geometry")
    for key in (
        "spacings_m", "turn_deltas_deg", "vertical_deltas_m",
        "initial_body_velocity_m_s",
    ):
        value[key] = tuple(value[key])
    return TransitionGeometry(**value)


def track_summary(track: Mapping[str, Any]) -> Dict[str, Any]:
    geometry = geometry_from_track(track)
    gates = list(track.get("gates") or [])
    positions = np.asarray([gate["position"] for gate in gates], dtype=float)
    return {
        "difficulty": geometry.difficulty,
        "episode_type": geometry.episode_type,
        "focused_transition": geometry.episode_type == "transition_focus",
        "full_sequence": geometry.episode_type == "full_sequence",
        "pattern": geometry.pattern,
        "seed": geometry.seed,
        "sequence_length": geometry.gate_count,
        "gate_order": [gate["id"] for gate in gates],
        "gate_centers": positions.tolist(),
        "turn_angles_deg": list(geometry.turn_deltas_deg),
        "vertical_changes_m": list(geometry.vertical_deltas_m),
        "inter_gate_distances_m": list(geometry.spacings_m),
        "initial_position": list(track["start"]["position"]),
        "initial_orientation_rpy_deg": list(track["start"]["rotation_rpy_deg"]),
        "expected_beacon_sequence": list(track["track"]["gate_sequence"]),
        "currents_disabled": not bool(track.get("currents")),
    }


def sample_geometries(
    *,
    seed_start: int,
    num_tracks: int,
    difficulties: Sequence[str],
    episode_types: Sequence[str],
    lengths: Sequence[int],
) -> list[TransitionGeometry]:
    if not difficulties or not episode_types:
        raise ValueError("at least one difficulty and episode type is required")
    samplers = {
        difficulty: TransitionGeometrySampler(
            seed=int(seed_start) + index * 1_000_003,
            difficulty=difficulty,
            transition_focus_fraction=0.5,
        )
        for index, difficulty in enumerate(difficulties)
    }
    rows = []
    attempts = 0
    while len(rows) < int(num_tracks):
        difficulty = difficulties[len(rows) % len(difficulties)]
        episode_type = episode_types[(len(rows) // len(difficulties)) % len(episode_types)]
        geometry = samplers[difficulty].sample(force_episode_type=episode_type)
        attempts += 1
        if episode_type == "full_sequence" and lengths and geometry.gate_count not in lengths:
            if attempts > int(num_tracks) * 20_000:
                raise RuntimeError("could not sample requested long-sequence lengths")
            continue
        rows.append(geometry)
    return rows


def write_sampled_tracks(
    geometries: Iterable[TransitionGeometry],
    output_dir: str | Path,
    *,
    frames_per_sec: bool | int = False,
) -> list[Path]:
    output = Path(output_dir)
    paths = []
    for index, geometry in enumerate(geometries):
        paths.append(generate_transition_track(
            geometry,
            output / f"track_{index:05d}_{geometry.difficulty}_{geometry.episode_type}.json",
            frames_per_sec=frames_per_sec,
        ))
    return paths


def _gate_frame(gate: Mapping[str, Any], half_width: float, half_height: float):
    center = np.asarray(gate["position"], dtype=float)
    forward = np.asarray(gate.get("passage_direction", (1.0, 0.0, 0.0)), dtype=float)
    forward /= max(np.linalg.norm(forward), 1e-9)
    lateral = np.asarray((-forward[1], forward[0], 0.0), dtype=float)
    vertical = np.asarray((0.0, 0.0, 1.0), dtype=float)
    corners = np.asarray([
        center - half_width * lateral - half_height * vertical,
        center + half_width * lateral - half_height * vertical,
        center + half_width * lateral + half_height * vertical,
        center - half_width * lateral + half_height * vertical,
        center - half_width * lateral - half_height * vertical,
    ])
    return center, forward, lateral, vertical, corners


def render_geometry_track(
    track: Mapping[str, Any],
    *,
    speed: float = 1.0,
    loop: bool = False,
    pause_between_tracks: float = 0.0,
    show_gate_frames: bool = False,
    show_camera_frustum: bool = False,
    record: Optional[str | Path] = None,
    show: bool = True,
) -> Dict[str, Any]:
    """Render only JSON geometry; importing/calling this launches no simulator."""

    import matplotlib.pyplot as plt
    from matplotlib import animation

    summary = track_summary(track)
    gates = list(track["gates"])
    positions = np.asarray([
        track["start"]["position"], *(gate["position"] for gate in gates)
    ], dtype=float)
    inner = track.get("track", {}).get("gate_inner_size_m", (1.5, 1.5))
    half_width, half_height = float(inner[0]) / 2.0, float(inner[1]) / 2.0
    figure = plt.figure(figsize=(11, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(positions[:, 0], positions[:, 1], positions[:, 2], "--", color="0.45")
    for index, gate in enumerate(gates):
        center, forward, lateral, vertical, corners = _gate_frame(
            gate, half_width, half_height
        )
        axis.plot(corners[:, 0], corners[:, 1], corners[:, 2], color="tab:green")
        axis.text(*center, f" {index + 1}:{gate['id']}")
        if show_gate_frames:
            axis.quiver(*center, *forward, length=0.9, color="tab:red")
            axis.quiver(*center, *lateral, length=0.7, color="tab:blue")
            axis.quiver(*center, *vertical, length=0.7, color="tab:purple")
    start = positions[0]
    axis.scatter(*start, marker="^", s=80, color="tab:orange", label="vehicle start")
    if show_camera_frustum:
        yaw = math.radians(float(track["start"]["rotation_rpy_deg"][2]))
        for delta in (-45.0, 45.0):
            angle = yaw + math.radians(delta)
            end = start + np.asarray((3.0 * math.cos(angle), 3.0 * math.sin(angle), 0.0))
            axis.plot((start[0], end[0]), (start[1], end[1]), (start[2], end[2]), color="tab:orange")
    vehicle, = axis.plot([start[0]], [start[1]], [start[2]], "o", color="black")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.set_title(
        f"{summary['difficulty']} {summary['episode_type']} | "
        f"seed={summary['seed']} gates={summary['sequence_length']}"
    )
    axis.legend(loc="upper left")
    axis.set_box_aspect(np.maximum(np.ptp(positions, axis=0), 1.0))
    segment_frames = max(2, int(round(20.0 / max(float(speed), 0.05))))
    trajectory = []
    for previous, current in zip(positions, positions[1:]):
        for fraction in np.linspace(0.0, 1.0, segment_frames, endpoint=False):
            trajectory.append(previous + fraction * (current - previous))
    trajectory.append(positions[-1])

    def update(index):
        point = trajectory[int(index)]
        vehicle.set_data_3d([point[0]], [point[1]], [point[2]])
        return (vehicle,)

    movie = animation.FuncAnimation(
        figure, update, frames=len(trajectory), interval=50,
        blit=False, repeat=bool(loop), repeat_delay=int(1000 * pause_between_tracks),
    )
    if record:
        target = Path(record)
        target.parent.mkdir(parents=True, exist_ok=True)
        writer = "pillow" if target.suffix.lower() == ".gif" else "ffmpeg"
        movie.save(str(target), writer=writer, fps=20)
    if show:
        plt.show()
    else:
        # Force one draw so headless tests validate all artists.
        figure.canvas.draw()
    plt.close(figure)
    return summary


def _camera_frame(env: Any, sensor_name: str = "FrontCamera") -> Optional[np.ndarray]:
    context = env.episode.context
    if sensor_name == "FrontCamera":
        data = context.adapter.get_allowed_sensor_data(
            env.episode.participant_id, context.participant.config.sensors
        )
    else:
        state = context.adapter.get_participant_state(env.episode.participant_id)
        data = dict(state.raw_sensors)
    value = data.get(sensor_name)
    if value is None:
        return None
    frame = np.asarray(value)
    if frame.ndim != 3 or frame.shape[2] < 3:
        return None
    # HoloOcean 2.3 emits BGR/BGRA; imageio expects RGB.
    return np.ascontiguousarray(frame[..., :3][..., ::-1])


class FrameRecorder:
    def __init__(
        self,
        *,
        live: bool = False,
        output_video: Optional[str | Path] = None,
        fps: int = 30,
        window_name: str = "Universal transition evaluation",
        sensor_name: str = "FrontCamera",
    ) -> None:
        self.live = bool(live)
        self.output_video = None if output_video is None else Path(output_video)
        self.fps = int(fps)
        self.window_name = str(window_name)
        self.sensor_name = str(sensor_name)
        self._writer = None
        self.frames = 0

    def __call__(self, env: Any, step: int) -> None:
        frame = _camera_frame(env, self.sensor_name)
        if frame is None:
            return
        if self.output_video is not None:
            if self._writer is None:
                import imageio.v2 as imageio

                self.output_video.parent.mkdir(parents=True, exist_ok=True)
                self._writer = imageio.get_writer(
                    self.output_video, fps=self.fps, codec="libx264"
                )
            self._writer.append_data(frame)
        if self.live:
            import cv2

            cv2.imshow(self.window_name, frame[..., ::-1])
            cv2.waitKey(1)
        self.frames += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self.live:
            try:
                import cv2

                cv2.destroyWindow(self.window_name)
            except Exception:
                pass


def holoocean_flythrough(
    track_path: str | Path,
    *,
    speed: float = 1.0,
    live: bool = False,
    output_video: Optional[str | Path] = None,
    camera: str = "spectator",
) -> Dict[str, Any]:
    """Teleport a preview-only camera vehicle along exact gate centers."""

    from marine_race_arena.learning.gym_env import MarineRaceGymEnv
    from marine_race_arena.learning.reward_local_transition import LocalTransitionTrainingReward

    track = snapshot_json(track_path)
    points = np.asarray([
        track["start"]["position"], *(gate["position"] for gate in track["gates"])
    ], dtype=float)
    recorder = FrameRecorder(live=live, output_video=output_video)
    env = None
    with reserve_holoocean_engines(1, owner=f"track preview {track_path}"):
        try:
            env = MarineRaceGymEnv(
                str(track_path), seed=int(track["universal_transition"]["seed"]),
                adapter="holoocean", allow_fallback=False, current_profile="none",
                max_steps=3600, observation_encoding_version="onboard_local_transition_v1",
                reward_fn=LocalTransitionTrainingReward(),
            )
            env.reset(seed=int(track["universal_transition"]["seed"]))
            context = env.episode.context
            frames_per_segment = max(2, int(round(30.0 / max(float(speed), 0.05))))
            frame_index = 0
            for previous, current in zip(points, points[1:]):
                delta = current - previous
                yaw = math.degrees(math.atan2(delta[1], delta[0]))
                for fraction in np.linspace(0.0, 1.0, frames_per_segment, endpoint=False):
                    point = previous + fraction * delta
                    forward = delta / max(float(np.linalg.norm(delta)), 1e-9)
                    pitch = 0.0
                    if camera == "chase":
                        point = point - 3.0 * forward + np.asarray([0.0, 0.0, 1.2])
                        pitch = -12.0
                    elif camera == "spectator":
                        point = point - 6.0 * forward + np.asarray([0.0, 0.0, 4.0])
                        pitch = -28.0
                    elif camera != "first_person":
                        raise ValueError(f"unknown preview camera {camera!r}")
                    context.adapter.teleport_participant(
                        env.episode.participant_id, point.tolist(), (0.0, pitch, yaw)
                    )
                    context.adapter.step(env.episode.dt)
                    recorder(env, frame_index)
                    frame_index += 1
            return {
                **track_summary(track),
                "backend": "holoocean",
                "camera": camera,
                "preview_controller": "privileged_teleport_flythrough_not_used_by_training",
                "frames": recorder.frames,
                "output_video": None if output_video is None else str(output_video),
            }
        finally:
            recorder.close()
            if env is not None:
                env.close()


def resolve_checkpoint(run: str | Path, checkpoint: str, algorithm: str) -> Path:
    candidate = Path(checkpoint)
    if candidate.exists():
        return candidate
    status = json.loads((Path(run) / "status.json").read_text(encoding="utf-8"))
    aliases = dict(status.get("checkpoint_aliases") or {})
    if checkpoint not in aliases or not aliases[checkpoint]:
        raise ValueError(f"checkpoint alias {checkpoint!r} is not assigned")
    path = Path(aliases[checkpoint])
    if path.exists():
        return path
    # A run in the sibling PPO worktree stores repository-relative aliases.
    relative = Path(run).resolve().parents[4] / path
    if relative.exists():
        return relative
    raise FileNotFoundError(path)


def rendered_long_sequence_evaluation(
    *,
    algorithm: str,
    run: str | Path,
    checkpoint: str,
    lengths: Sequence[int] = FULL_SEQUENCE_LENGTHS,
    episodes_per_length: int = 2,
    output_dir: str | Path,
    difficulty: str = "G6",
    seed: int = 88001,
    live: bool = False,
    record: bool = False,
    camera: str = "chase",
) -> Dict[str, Any]:
    if algorithm == "ppo":
        from stable_baselines3 import PPO

        model_path = resolve_checkpoint(run, checkpoint, algorithm)
        model = PPO.load(str(model_path), device="cpu")
    elif algorithm == "sac":
        from marine_race_arena.learning.sac_transition_policy import load_sac_transition_policy

        model_path = resolve_checkpoint(run, checkpoint, algorithm)
        model = load_sac_transition_policy(model_path)
    else:
        raise ValueError(algorithm)
    output = Path(output_dir)
    all_cases = _benchmark_cases(
        output=output, seed=int(seed), difficulty=difficulty,
        transition_cases=0, full_cases_per_length=int(episodes_per_length),
    )
    rows = []
    for ordinal, geometry, case_output, _ in all_cases:
        if geometry.gate_count not in set(map(int, lengths)):
            continue
        video = case_output / f"{algorithm}_{camera}.mp4" if record else None
        recorder = FrameRecorder(
            live=live, output_video=video,
            window_name=f"{algorithm.upper()} length {geometry.gate_count}",
            sensor_name="FrontCamera" if camera == "first_person" else "RenderCamera",
        )
        with reserve_holoocean_engines(1, owner=f"rendered {algorithm} evaluation"):
            try:
                row = evaluate_local_transition_episode(
                    model, geometry=geometry, output_dir=case_output,
                    adapter="holoocean", frames_per_sec=False,
                    max_steps=3600, record_trajectory=True,
                    frame_callback=recorder,
                    action_source=f"{algorithm}_policy",
                    render_camera=camera,
                )
            finally:
                recorder.close()
        atomic_write_json(case_output / "episode.json", row)
        rows.append(row)
    report = {
        "schema_version": "rendered_universal_transition_evaluation_v1",
        "algorithm": algorithm,
        "checkpoint": str(model_path),
        "seed": int(seed),
        "difficulty": difficulty,
        "lengths": list(map(int, lengths)),
        "episodes_per_length": int(episodes_per_length),
        "camera": camera,
        "video_rendering_changes_policy_actions": False,
        "all_actions_policy_generated": True,
        "metrics": aggregate_transition_benchmark(rows),
        "episodes": rows,
    }
    atomic_write_json(output / "evaluation.json", report)
    return report
