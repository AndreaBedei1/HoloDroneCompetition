from __future__ import annotations

import json
from contextlib import contextmanager

import numpy as np
import pytest

from marine_race_arena.learning.transition_curriculum import (
    TransitionGeometrySampler,
    generate_transition_track,
)
from marine_race_arena.learning.transition_evaluation import (
    _configure_render_camera,
    aggregate_transition_benchmark,
    evaluate_local_transition_episode,
)
from marine_race_arena.learning.transition_track_visualization import (
    geometry_from_track,
    holoocean_flythrough,
    render_geometry_track,
    sample_geometries,
    snapshot_json,
    track_summary,
)


class _ZeroPolicy:
    def predict(self, observation, deterministic=True):
        array = np.asarray(observation)
        shape = (4,) if array.ndim == 1 else (array.shape[0], 4)
        return np.zeros(shape, np.float32), None


def test_sampler_preview_reproduces_exact_geometries_for_same_seed():
    kwargs = dict(
        seed_start=43001, num_tracks=24,
        difficulties=["G1", "G3", "G6"],
        episode_types=["transition_focus", "full_sequence"],
        lengths=[3, 5, 8, 12, 17, 22],
    )
    assert sample_geometries(**kwargs) == sample_geometries(**kwargs)


def test_track_snapshot_preserves_gate_order_orientation_and_diagnostics(tmp_path):
    geometry = TransitionGeometrySampler(
        seed=44, difficulty="G4", transition_focus_fraction=1.0
    ).sample(force_episode_type="transition_focus")
    path = generate_transition_track(geometry, tmp_path / "track.json")
    track = snapshot_json(path)
    restored = geometry_from_track(track)
    summary = track_summary(track)
    assert restored == geometry
    assert summary["gate_order"] == track["track"]["gate_sequence"]
    assert summary["gate_centers"] == [gate["position"] for gate in track["gates"]]
    assert summary["currents_disabled"]


def test_geometry_backend_requires_no_holoocean(monkeypatch, tmp_path):
    geometry = TransitionGeometrySampler(
        seed=45, difficulty="G1", transition_focus_fraction=1.0
    ).sample(force_episode_type="transition_focus")
    path = generate_transition_track(geometry, tmp_path / "track.json")
    summary = render_geometry_track(snapshot_json(path), speed=20, show=False)
    assert summary["seed"] == geometry.seed
    assert summary["sequence_length"] == 2


def test_holoocean_backend_obeys_engine_cap(monkeypatch, tmp_path):
    geometry = TransitionGeometrySampler(
        seed=46, difficulty="G1", transition_focus_fraction=1.0
    ).sample(force_episode_type="transition_focus")
    path = generate_transition_track(geometry, tmp_path / "track.json")

    @contextmanager
    def denied(*args, **kwargs):
        raise RuntimeError("HoloOcean engine cap would be exceeded")
        yield

    monkeypatch.setattr(
        "marine_race_arena.learning.transition_track_visualization.reserve_holoocean_engines",
        denied,
    )
    with pytest.raises(RuntimeError, match="cap"):
        holoocean_flythrough(path)


def test_render_callback_does_not_change_headless_evaluation_metrics(tmp_path):
    geometry = TransitionGeometrySampler(
        seed=47, difficulty="G1", transition_focus_fraction=1.0
    ).sample(force_episode_type="transition_focus")
    policy = _ZeroPolicy()
    headless = evaluate_local_transition_episode(
        policy, geometry=geometry, output_dir=tmp_path / "headless",
        adapter="fallback", max_steps=20,
    )
    frames = []
    rendered = evaluate_local_transition_episode(
        policy, geometry=geometry, output_dir=tmp_path / "rendered",
        adapter="fallback", max_steps=20,
        frame_callback=lambda env, step: frames.append(step),
    )
    assert rendered == headless
    assert frames


def test_rendered_sac_evaluation_records_sac_policy_provenance(tmp_path):
    geometry = TransitionGeometrySampler(
        seed=48, difficulty="G1", transition_focus_fraction=1.0
    ).sample(force_episode_type="transition_focus")
    row = evaluate_local_transition_episode(
        _ZeroPolicy(), geometry=geometry, output_dir=tmp_path / "sac",
        adapter="fallback", max_steps=5, action_source="sac_policy",
    )
    assert row["action_source"] == "sac_policy"
    assert aggregate_transition_benchmark([row])["all_actions_policy_generated"]


def test_render_camera_is_not_exposed_to_policy_sensor_profile(tmp_path):
    geometry = TransitionGeometrySampler(
        seed=49, difficulty="G1", transition_focus_fraction=1.0
    ).sample(force_episode_type="transition_focus")
    path = generate_transition_track(geometry, tmp_path / "track.json")
    assert _configure_render_camera(path, "chase") == "RenderCamera"
    track = json.loads(path.read_text(encoding="utf-8"))
    sensors = track["participants"][0]["sensors"]
    assert "RenderCamera" not in sensors["allowed_sensors"]
    render = [
        value for value in sensors["holoocean_sensors"]
        if value.get("sensor_name") == "RenderCamera"
    ]
    assert len(render) == 1
    assert render[0]["location"] == [-3.0, 0.0, 1.2]
