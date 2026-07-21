# Learning-Based Controller — Progress (feature/rl-controller)

Status of the imitation- and reinforcement-learning extension. This work lives
**only** on `feature/rl-controller`. It does not modify the official observation
contract, the independent referee, gate validation, official scoring, the official
track geometry, the rule-based baselines or the frozen 78-run results, and it adds
no dependencies to the benchmark `requirements.txt`.

**Honest headline:** the full learning *pipeline* is implemented and tested end to
end — record → dataset → BC train → deploy through the unchanged runner + referee,
plus a Gymnasium env, a training-only reward and an exact BC→PPO warm-start. Open-loop
behavioral cloning fits the expert well on engine-free demonstrations. **No closed-loop
learned-policy success is claimed**: collecting real demonstrations and training to
reliable gate completion requires HoloOcean training runs that have not been executed
here (the engine-free fallback cannot complete gates because the reference controllers
are camera-gated). See *Training status* and *Next steps*.

## Package layout (`marine_race_arena/learning/`)

| Module | Role |
| --- | --- |
| `config.py` | Fixed 36-feature observation layout, action axes, normalization scales, `LearningContext`. |
| `observation_encoder.py` | Legal onboard-only encoder → fixed `float32[36]` with explicit missing-data masks. |
| `episode.py` | `RaceEpisode`: step-wise single-vehicle engine reusing the runner's construction + referee. |
| `gym_env.py` | `MarineRaceGymEnv`: Gymnasium wrapper (encoded obs, `[-1,1]^4` action, reward hook). |
| `tracker_context.py` | `OnboardContextTracker`: shared tracker-driven context (identical at train/inference). |
| `reward.py` | Training-only reward (`score_step` + `TrainingReward`); privileged, never in the observation. |
| `trajectory_recorder.py` | Full-rate expert demonstration recorder. |
| `dataset.py` | `BCDataset`: integrity checks, episode-level train/val split, npz IO. |
| `bc_train.py` | `BCPolicy` (Tanh-MLP + linear head + obs norm) and `train_bc`. |
| `rl_train.py` | PPO builder, **exact** BC→PPO weight transfer, `train_ppo`. |
| `rl_controller.py` | `RLGateController`: deployable `BaseController`, alias `rl_gate_controller`. |
| `evaluate_policy.py` | Held-out evaluation of any controller under the unchanged referee. |
| `curriculum.py` | Staged tasks + success criteria. |

RL dependencies (separate `requirements-rl.txt`, verified together on Python 3.9.25 /
numpy 2.0.2): `torch 2.8.0`, `stable-baselines3 2.7.1`, `gymnasium 1.0.0`.

## Observation encoding (legal, onboard-only)

Fixed `float32` vector of length **36**, every feature normalized, finite and clipped
to a declared range, derived only from the official observation plus controller-local
state:

- **Beacon (7)** for the locally expected beacon: present flag, sin/cos bearing,
  normalized elevation, normalized range, signal strength, packet age.
- **Vision (5)** from FrontCamera via the official vision pipeline: present flag,
  centre x/y, area fraction, confidence.
- **Depth/motion (10)**: depth + mask, depth error vs a local reference + mask,
  DVL surge/sway/heave + mask, IMU yaw-rate + mask.
- **Controller-local (14)**: 7-way LocalCourseTracker phase one-hot, normalized local
  beacon index and lap, visual-lock flag, previous action (4).

Missing beacon/camera/sensor data is signalled by explicit `*_present` masks, never by
ground-truth substitution. Referee state, simulator pose, world velocity, true gate
geometry and current vectors are **never** encoded (test-enforced).

## Action space

`Box(-1, 1, shape=(4,))` → normalized body-frame `surge, sway, heave, yaw`. Mapped
straight to the command; the adapter clamps to the vehicle's control limits.

## Training-only reward

Documented, tunable components (privileged geometry/referee counters used only here):
ratcheted gate-approach **progress** and **alignment** (only a new closest distance /
lowest lateral offset earns reward — oscillation cannot farm it), a per-gate crossing
bonus driven by the referee's own count (once), a completion bonus, a per-step time
cost, and penalties from per-step deltas of the referee's authoritative collision /
obstacle / out-of-bounds / missed-gate / stuck counters, plus terminal DNF/timeout/stuck
and an action-change penalty. Every component is logged. Tests cover sign, gate-bonus-
once, no progress farming, terminal-once, and no leakage into the observation.

## Algorithms

1. **Behavioral cloning** — small Tanh-MLP (default `[256, 256]`), linear action head,
   internal obs normalization from the *training* split only, MSE with optional per-axis
   weighting, early stopping, checkpointing, CSV + per-axis logging.
2. **PPO from scratch** — Stable-Baselines3 PPO whose policy net mirrors the BC net.
3. **BC-initialized PPO** — `transfer_bc_to_ppo` copies the BC extractor + action head
   into the PPO policy. A test verifies PPO's deterministic action equals the BC output
   (max error `0.0` over sampled observations), so the warm-start is exact.

## Curriculum (`curriculum.py`, training tracks under `tracks/training/`)

Training tracks preserve the official 1.5×1.5 m aperture, beacon model, observation,
action mapping and referee.

| Stage | Task | Track | Eval eps | Criterion |
| --- | --- | --- | ---: | ---: |
| 0 | API/actuation sanity (no training) | stage1 | 0 | — |
| 1 | Single gate, easy start | `tracks/training/stage1_single_gate.json` | 20 | ≥ 90% |
| 2 | Single gate, randomized start + beacon noise | stage1 (+randomization) | 20 | ≥ 90% |
| 3 | Three ordered gates (S-curve) | `tracks/training/stage3_three_gates.json` | 20 | ≥ 80% |
| 5 | Full Horseshoe Bay (official, unchanged) | official | 20 | ≥ 80% |
| 6 | Vertical Serpent / Mixed generalization | official | 20 | ≥ 80% |
| 7 | Disturbance robustness (currents) | official | 20 | ≥ 80% |

Do not advance a stage until its held-out completion criterion is met. Training seeds,
demonstration seeds and evaluation seeds are kept separate; official tracks are used
unchanged and any artifacts go to fresh directories.

## Training status

**Verified (tested, engine-free fallback):**
- `RaceEpisode` is step-for-step equivalent to the real `_run_race_loop` (observations
  and referee gate progress match tick-for-tick).
- `MarineRaceGymEnv` passes the Gymnasium `env_checker`; reset determinism, action
  clipping, truncation, seeded determinism, reward-component reporting.
- Trajectory recording → `BCDataset` integrity + episode-level split.
- BC training reduces validation MSE by ~2 orders of magnitude on fallback demonstrations
  (e.g. `0.0205 → 0.0002`), i.e. it imitates the expert open-loop.
- `RLGateController` loads a BC or PPO model, runs **closed-loop through the unchanged
  runner + referee**, needs no Gym/referee/Gymnasium at inference, and is loadable via
  the `rl_gate_controller` alias.
- Exact BC→PPO warm-start; PPO and BC-initialized-PPO training plumbing run on the
  fallback backend.

**Not done (requires HoloOcean compute, out of scope for this session):**
- Collecting real HoloOcean expert demonstrations. The fallback adapter cannot complete
  gates (the reference controllers are camera-gated), so fallback data trains the pipeline
  but yields no gate-completing policy.
- Actual BC / PPO / BC+PPO training runs to the Stage-1 completion criterion and beyond.
- Any closed-loop performance number on training or official tracks.

Test coverage: **82 learning tests** (`tests/learning/`). The RL-only tests
skip automatically when torch/SB3 are absent, so the benchmark environment is unaffected
(benchmark suite: 432 passed, 4 skipped in `ocean`).

## Failures / learnings

- HoloOcean 2.3.0 is not on PyPI; the RL env extends the source-installed benchmark stack
  (documented in `requirements-rl.txt`).
- The engine-free **fallback** adapter is for fast API/plumbing tests only — camera-gated
  policies (rule baselines and their BC clones) stall there, so it must not be read as a
  performance result. Test tracks carry zero beacon noise (fully reproducible).
- To make BC→PPO warm-start exact, the BC policy uses a linear action head (+ clip)
  matching SB3's `MlpPolicy` rather than a tanh output head; both respect `[-1, 1]`.

## Next steps (to resume)

1. Set up the RL environment (once):
   ```bash
   conda create -n marine_race_rl python=3.9 -y && conda activate marine_race_rl
   pip install <HoloOcean-2.3.0 source>/client && pip install -r requirements.txt
   pip install -r requirements-rl.txt
   ```
2. Collect real Stage-1 demonstrations with HoloOcean (fresh output dir):
   record episodes via `trajectory_recorder.collect_dataset(stage1_track,
   controller="rule_gate_center_then_commit", seeds=range(0,30), adapter="holoocean",
   allow_fallback=False)` and save the `BCDataset`.
3. Train BC (`bc_train.train_bc`), then evaluate closed-loop with
   `evaluate_policy.evaluate_controller(stage1_track, lambda: RLGateController(model_path),
   seeds=<held-out>, adapter="holoocean", allow_fallback=False)`. Require ≥ 90%
   completion over 20 held-out episodes before Stage 2.
4. Then PPO from scratch and BC-initialized PPO on Stage 1 (short smoke → longer runs),
   advancing the curriculum only when each stage's criterion is met. Write runs to
   `results/rl/<stage>/<algorithm>/<timestamp>/` with config/seed/commit/versions.
5. Only after clean Horseshoe Bay succeeds on held-out seeds with BC, PPO and BC+PPO
   compared should any manuscript change be proposed. Until then the paper keeps the
   current conservative future-work wording.

## Commits (this branch)

```
5e2ab6e Add legal fixed-size learning observation encoder
6f92397 Add Gymnasium-compatible step-wise race environment
87016ad Add training-only reward and wire it into the Gym env
1d41141 Add trajectory recording and BC dataset pipeline
6c50ede Add behavioral-cloning trainer, PPO scaffold and deployable RL controller
```
