# Clean-Environment Installation Verification — 2026-07-21

This report records a from-scratch installation and smoke test of Marine Race
Arena in a dedicated, throwaway conda environment. It follows the corrected
`README` / `requirements.txt` installation procedure. It does **not** touch the
frozen 78-run benchmark matrix under
`results/onboard_only_validation/final_20260715/`.

## Environment

| Item | Value |
| --- | --- |
| Conda env | `marine_race_verify` (created fresh for this check) |
| Python | 3.9.25 |
| OS | Windows 11 Enterprise (Windows-10-10.0.26200-SP0) |
| HoloOcean | 2.3.0, installed from the official source `client` package |
| HoloOcean worlds | `['Ocean']` (shared per-version user install, reused — no re-download) |
| Run artifacts | `results/install_verification/20260721_214541/` (git-ignored) |
| Full `pip freeze` | `results/install_verification/20260721_214541/pip_freeze.txt` |

Pinned dependency versions observed in the clean env (match `requirements.txt`):
`numpy==2.0.2`, `opencv-python==4.12.0.88`, `pygame==2.6.1`, `Pillow==11.3.0`,
`pytest==8.4.2`. HoloOcean pulls in `scipy==1.13.1` and `matplotlib==3.9.4`
(unpinned upstream).

## Installation defect found and fixed

`requirements.txt` previously pinned `holoocean==2.3.0`, and the README told users
to run `pip install -r requirements.txt`. **HoloOcean 2.3.0 is not published on
PyPI** (PyPI ships only `holoocean 0.5.8`), so that command fails in a clean
environment:

```
ERROR: Could not find a version that satisfies the requirement holoocean==2.3.0
       (from versions: 0.5.8)
```

The `ocean` env that produced the benchmark had HoloOcean 2.3.0 installed from a
local source tree (`.../HoloOcean-2.3.0/client`), so the broken pin was masked
there. The official HoloOcean docs confirm the client is installed from source
(`cd holoocean/client && pip install .`), never from PyPI.

Fix applied (main branch, benchmark dependencies only — no RL packages added):

- `requirements.txt`: removed the non-installable `holoocean==2.3.0` PyPI pin and
  documented that the HoloOcean 2.3.0 client is installed separately from source
  (it pulls `numpy/scipy/matplotlib`; the `numpy==2.0.2` pin is retained).
- `README.md` §2: added the ordered install — (1) install the HoloOcean 2.3.0
  client from source, (2) `pip install -r requirements.txt`, (3)
  `holoocean.install('Ocean')` — with a note that PyPI does not carry 2.3.0.

## Exact commands used

```bash
conda create -n marine_race_verify python=3.9 -y
# 1. HoloOcean 2.3.0 client from source (not on PyPI):
python -m pip install <HoloOcean-2.3.0 source>/client
# 2. Pinned Python dependencies:
python -m pip install -r requirements.txt
# 3. World package already present (shared per-version user install):
python -c "import holoocean; print(holoocean.installed_packages())"   # -> ['Ocean']
```

## Checks and results

| # | Check | Command (in `marine_race_verify`) | Result |
| --- | --- | --- | --- |
| 1 | HoloOcean import + version | `python -c "import holoocean; holoocean.__version__"` | `2.3.0` ✓ |
| 2 | Ocean world visible | `holoocean.installed_packages()` | `['Ocean']` ✓ |
| 3 | Byte-compile | `python -m compileall -q marine_race_arena tests run.py validate_final_matrix.py summarize_final_results.py` | exit 0 ✓ |
| 4 | Unit + integration tests | `python -m pytest -q` | **387 passed** in 13.3 s ✓ |
| 5 | Config validation | `python -m marine_race_arena.scripts.validate_track_config --track .../marine_race_vertical_serpent.json --benchmark-task current_gate --current-profile medium` | `Validation passed.` (5 currents, 17 gates) ✓ |
| 6 | Load both official controllers | `ControllerLoader().load("rule_gate_baseline" / "rule_gate_center_then_commit")` | both instantiated ✓ |
| 7 | Load custom controller | file-path + `--controller-class MyController`, `reset/step/close` | returns `{surge,sway,heave,yaw}` ✓ |
| 8 | Fallback smoke | `run_marine_race --track tests/single_gate_yaw_0.json --controller rule_gate_baseline --adapter fallback --allow-fallback --official --duration 3 --dt 0.1 --headless` | exit 0, adapter `fallback`, summary+jsonl written ✓ |
| 9 | **Real-HoloOcean smoke** | `run_marine_race --track tests/single_gate_yaw_0.json --controller rule_gate_baseline --adapter holoocean --official --duration 20 --dt 0.033 --seed 0 --headless` (no `--allow-fallback`) | exit 0, **adapter `holoocean`**, status **FINISHED**, `physical_current_coupling_active=true` ✓ |
| 10 | Paper build | `cd article && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` | 15 pages, 0 overfull boxes, 0 undefined refs/citations ✓ |

Notes:
- Check 9 used `--adapter holoocean` **without** `--allow-fallback`, so a HoloOcean
  launch failure would have aborted rather than silently switched to fallback. The
  summary confirms the real adapter was used and the referee scored the single-gate
  test track as FINISHED.
- The fallback smoke (check 8) intentionally does not complete a gate: the
  camera-gated baseline cannot perceive the gate through the engine-free fallback
  camera, so it stalls by design. This is a fast plumbing check only.
- Paper build uses the system MiKTeX toolchain, independent of the conda env.

## Conclusion

Following the corrected README, a clean Python 3.9 environment reaches a fully
working install: HoloOcean 2.3.0 imports, the Ocean world is available, all 387
tests pass, the runner executes under both the fallback and the real HoloOcean
adapters, custom controllers load through the documented interface, and the paper
compiles. The only installation defect (the non-installable `holoocean==2.3.0`
PyPI pin) has been corrected in `requirements.txt` and `README.md`.
