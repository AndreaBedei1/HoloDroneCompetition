# Learning-Based Controller — Progress (feature/rl-controller)

Status of the imitation- and reinforcement-learning extension. This work lives
**only** on `feature/rl-controller`. It does not modify the official observation
contract, the independent referee, gate validation, official scoring, the official
track geometry, the rule-based baselines or the frozen 78-run results, and it adds
no dependencies to the benchmark `requirements.txt`.

**Honest headline:** the full learning pipeline is implemented, tested, and
**demonstrated on the real HoloOcean engine**. Two frozen 50-seed held-out
evaluations of the 34-demo BC controller (unchanged referee, onboard-only
observations, corrected metric/randomization code; see `results/rl_public/stage1/`):

- **Evaluation A — fixed start (Stage 1):** **50/50 = 100%**, Wilson 95% CI
  [0.929, 1.000], 0 collisions/OOB/wrong-direction → **Stage 1 PASS**.
- **Evaluation B — randomized start (Stage 2):** **48/50 = 96%**, Wilson 95% CI
  [0.865, 0.989] → **Stage 2 PASS on the point estimate**, but the 95% CI lower bound
  dips below 0.90 (reported honestly, not rounded up); both failures are at
  near-maximum randomization offsets.

Reaching this required diagnosing and fixing a real covariate-shift failure
(fixed-start demonstrations gave 0% closed-loop despite a tiny open-loop error;
Stage-2 start randomization fixed it: 0% → 69% → 100%). PPO is **not** trained to
convergence — only its workflow was validated (fallback tests + a 300-step real-HoloOcean
plumbing smoke); PPO-to-convergence and the BC/PPO/BC+PPO comparison are the documented
next step. No gate geometry or referee margin was ever weakened. See *Training status*.

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
| `bc_ppo_init.py` | Safe stochastic warm-start: per-axis exploration `log_std` from BC residuals. |
| `train_workflow.py` | Resumable PPO workflow: run metadata, timestep-zero eval, best-model, reproduce. |
| `launch_stage1_ppo.py` | User-facing PPO launcher (safe defaults, hash checks, `--dry-run`, resume). |
| `collect_demos.py` | Resumable demonstration collection with a full provenance manifest. |
| `closed_loop_eval.py` | Held-out closed-loop eval with an `evaluation_manifest.json` resume gate. |
| `rl_controller.py` | `RLGateController`: deployable `BaseController`, alias `rl_gate_controller`. |
| `evaluate_policy.py` | Held-out evaluation; records referee status + `evaluation_end_reason`. |
| `curriculum.py` | Staged tasks + success criteria. |
| `build_public_package.py` / `build_ppo_smoke_package.py` | Derive the compact public audit packages. |
| `reset_benchmark.py` | Fresh-vs-persistent reset benchmark (persistent stays experimental). |

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

## Running the learned controller (CLI)

The trained controller runs through the normal runner. The model path comes from
`--controller-model-path` (highest precedence), then `$MARINE_RACE_RL_MODEL`, then a
Python constructor argument. Controllers that do not accept a model path ignore the
option, so rule baselines are unaffected.

```bash
python -m marine_race_arena.scripts.run_marine_race \
  --track marine_race_arena/tracks/training/stage1_single_gate.json \
  --controller rl_gate_controller \
  --controller-model-path results/rl/stage1/bc/best_model.pt \
  --adapter holoocean \
  --official
```

Config-driven `run.py` accepts `controller.model_path` in the JSON, which maps to the
same flag.

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

### Stage 1 — real HoloOcean BC result (ACHIEVED)

Demonstrations, training and evaluation were run on the **real HoloOcean adapter**
(`--adapter holoocean`, no fallback) in the dedicated `marine_race_rl` environment,
on the Stage-1 single-gate training track, scored by the unchanged referee. Training
and evaluation seeds are disjoint.

| Demonstrations (expert `rule_gate_center_then_commit`) | Held-out closed-loop BC completion |
| --- | --- |
| 21 demos, **fixed start / zero beacon noise** | **0/16 (0%)** — catastrophic covariate shift |
| 18 demos, **Stage-2 seeded start/beacon randomization** | 11/16 (69%) |
| 34 demos, randomized | **20/20 (100%)**, 0 collisions, 0 out-of-bounds |

- Expert demonstrations complete the gate 100% (both fixed and randomized start).
- Open-loop BC validation MSE is tiny (`~2.3e-5` fixed, `~6.5e-5` combined), but that
  alone does **not** imply closed-loop success — the fixed-start set fails completely.
- **Diagnosis and remedy:** the fixed-start track produces near-identical trajectories,
  so BC sees almost no state diversity and small closed-loop errors compound into unseen
  states (out-of-bounds/collisions). Collecting demonstrations under the Stage-2 seeded
  start/beacon randomization restores diversity; **34 randomized demonstrations reach
  100% closed-loop completion over 20 held-out seeds**, exceeding the Stage-1 ≥90%
  criterion. No gate or referee margin was weakened.
- The final Stage-1 BC controller runs through the unchanged runner + referee via
  `--controller rl_gate_controller --controller-model-path <best_model.pt>`, using only
  legal onboard observations.
- The 20/20 above is a **randomized-start (Stage-2)** development evaluation. The
  **authoritative** verdicts come from two frozen 50-seed evaluations on unused seeds
  with the corrected metric/randomization code — **Evaluation A (fixed, Stage 1): 100%**,
  **Evaluation B (randomized, Stage 2): 96%** — published for external audit under
  `results/rl_public/stage1/` (`frozen_evaluations.md`, `result_manifest.json`,
  per-seed `evaluation_fixed_50/` and `evaluation_randomized_50/`, and the committed
  0.3 MB model with its SHA-256).

Artifacts: the compact, externally inspectable package is `results/rl_public/stage1/`;
the heavy raw datasets/checkpoints stay under the git-ignored `results/rl/`.

**PPO status (workflow + 1k smokes validated on real HoloOcean, NOT converged):**
- **Safe BC→PPO stochastic warm-start** (`bc_ppo_init.py`): the exact normalization-aware
  transfer reproduces the BC mean, and the PPO exploration `log_std` is set to a small
  per-axis value derived from the BC validation residuals (`sqrt(MSE)`, clamped to
  `[0.05, 0.15]`; documented fallback `log_std=-2.5`). For the Stage-1 model every axis
  floors at std `0.05` (residuals are tiny). Without this, SB3's default std ≈ 1.0 would
  saturate actions and destroy the warm-start on the first update.
- **Timestep-zero held-out evaluation** runs (deterministically) before `model.learn()`
  and writes `evaluation/initial_eval.json` + a `timesteps=0` row; the timestep-0 policy
  is saved as the initial best. This verifies the warm-start starts near the BC baseline.
- **One-command launcher** `launch_stage1_ppo` (+ `scripts/*.bat`, `docs/rl_quickstart.md`):
  safe defaults (holoocean, no fallback, **fresh reset**, committed public BC model with
  hash check, dev seeds 1200–1204, conservative config lr 5e-5 / target_kl 0.01 /
  clip 0.1), `--dry-run`, refuses a non-empty output dir, verifies branch + hashes.
- **1,000-step smokes run on the real HoloOcean engine (both arms, no fallback)** —
  plumbing only, **not convergence**. Compact results: `results/rl_public/stage1/ppo_smoke/`.
  BC-init timestep-zero completion matched the BC baseline (warm-start intact); resume to
  1,500 steps was verified (eval history appended, timestep-0 not duplicated, best model
  preserved). No superiority claim from 1,000 steps; the staged 5k/10k plan and exact
  commands are in `docs/ppo_plan.md`, and the final evaluation must use new unseen seeds.
- Convergence and the scientific BC-vs-scratch comparison remain the documented next step.

### Test coverage (measured, not copied)

Counts from actually running the suites in each environment:

| Suite / environment | Result |
| --- | --- |
| Benchmark environment (`ocean`, no RL deps), full `pytest` | **562 passed, 7 skipped** |
| — of which non-learning benchmark tests | 387 |
| — of which learning tests that run without RL deps (numpy-only) | 175 |
| — skipped (the 7 RL-dependency test files) | 7 |
| RL environment (torch + SB3 + Gymnasium), `tests/learning` | **224 passed** |
| RL environment, full `pytest` | **611 passed** |
| Learning tests that REQUIRE RL deps (torch/SB3/Gymnasium) | 49 (= 224 − 175) |
| Tests that launch the HoloOcean engine | **0** (all use the fallback adapter) |

Machine-readable verification (commands, counts, versions, provenance) is published at
`results/rl_public/test_verification.json`.

Test categories:
- **Benchmark tests** (387): the original suite; unaffected by the learning package.
- **RL unit/integration tests** (224 total; 175 numpy-only + 49 needing torch/SB3/Gymnasium):
  observation encoder, episode equivalence, reward, dataset, curriculum, randomization,
  controller model-path, temporal alignment, Gym env, BC, RL controller, BC→PPO transfer +
  safe stochastic warm-start, PPO workflow + timestep-zero eval, evaluation end-reasons +
  resume manifests, demonstration provenance, launcher dry-run/safety, public-package +
  PPO-smoke validation.
- **Skipped in the benchmark env** (7 files): they `pytest.importorskip` torch/SB3/Gymnasium
  and run only in the RL environment, so the benchmark environment stays RL-free.
- **HoloOcean-engine tests**: none. Every automated test uses the engine-free **fallback**
  adapter (fast, deterministic). Real-HoloOcean validation is a manual smoke command
  (`--adapter holoocean`, no `--allow-fallback`) or the `launch_stage1_ppo` launcher, never
  a `pytest` test — so CI never launches Unreal.

### Continuous integration — decision

No automated CI workflow is added at this time, deliberately. HoloOcean 2.3.0 is not
installable in a hosted runner (it is not on PyPI, needs its multi-GB Unreal world and
a GPU), and part of the benchmark suite constructs the HoloOcean adapter (with mocks).
A CI that omitted HoloOcean could not be validated green from here without a Linux
runner, so adding one now risks exactly the misleading/fragile failures to avoid. The
observation/reward/dataset/curriculum logic is instead covered by the fast, engine-free
tests below, run locally in both environments.

Reproduce the reported counts locally:

```bash
# Benchmark environment (no RL deps): 509 passed, 6 skipped
conda run -n ocean python -m pytest -q

# RL environment (torch + SB3 + Gymnasium): 548 passed (161 in tests/learning)
conda run -n marine_race_rl python -m pytest -q
```

A hosted CI job for the numpy/torch/SB3 tests (fallback only, no HoloOcean) can be
added later once the HoloOcean-free subset is confirmed green on a Linux runner.

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

## Foundation corrections (post-scaffold)

Before demonstration collection / long training, these correctness fixes landed:

1. **Temporal alignment** — the Gym env encoded the next observation before updating
   `prev_action`, so `o_(t+1)` carried `a_(t-1)`; fixed to carry `a_t`, matching the
   recorder and controller (train/inference parity is test-enforced).
2. **Normalization-aware BC→PPO transfer** — the BC observation normalization is folded
   into PPO's first layer (`W/std`, `b − W/std·mean`), so the warm-start is exact for
   *any* normalization (tested with non-identity mean/std).
3. **Model-path CLI** — `--controller-model-path` (precedence over `$MARINE_RACE_RL_MODEL`);
   `ControllerLoader` passes only constructor kwargs a controller accepts, so rule
   baselines are unaffected.
4. **Official referee margin** — training tracks restored to `vehicle_clearance_margin_m
   = 0.10` (was 0.05); a test pins each training track to the official geometry/referee.
5. **Directional reward** — progress uses the signed distance to the gate plane, gated to
   the legal entry side; wrong-side approach earns nothing and wrong-direction crossings
   are penalized.
6. **Executable curriculum** — Stage 4 (six gates), real seeded Stage-2 randomization,
   Stage 6 over both official tracks, Stage 7 as `none→medium→strong`, and a runner that
   refuses to advance until the held-out criterion is met (disjoint train/val/eval seeds).
7. **Resumable PPO workflow** — timestamped non-overwriting run dirs, checkpoints, resume,
   completion-based best-model, full provenance (config/seeds/reward/track hash/git SHA/
   versions/adapter/reproduce).

## Commits (this branch)

See `git log main..feature/rl-controller`. Scaffold: `5e2ab6e` encoder, `6f92397` gym
env, `87016ad` reward, `1d41141` recorder/dataset, `6c50ede` BC/PPO/controller,
`9ef2a4a` curriculum/eval/docs. Foundation fixes: `40e5672` temporal alignment,
`7bbcec9` normalization-aware transfer, `7364d52` model-path CLI, `51678bd` referee
margin, `37693ab` directional reward, `5d27209` curriculum, `d42182f` PPO workflow.
