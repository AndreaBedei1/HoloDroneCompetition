# Revision manifest — Marine Race Arena manuscript revision

This manifest records the *manuscript-and-schema revision* and keeps it clearly
separate from the frozen experiment provenance. The 78-run HoloOcean matrix was
**not** rerun; its metadata, source fingerprint and commit association are
unchanged. Do not read the revision commit as the commit that generated the
experiments.

## Frozen experiment provenance (unchanged)

Recorded in the per-run artifact metadata under
`results/onboard_only_validation/final_20260715/` and its
`complete_experiment_manifest.{json,csv,md}`:

| Field | Value |
| --- | --- |
| Experiment-generation commit | `df098ef470ccf4598acf5644cb5a2e72f423d213` |
| Source-tree fingerprint (`marine_race_arena/**` .py+.json) | `e7d3107784ea53056febcc3966b267ef59ee6d0f24d523c5c9b9446efca044b8` |
| HoloOcean version | 2.3.0 |
| Python | 3.9.25 (MSC v.1929 64-bit) |
| OS | Windows-10-10.0.26200 |
| Fallback | disabled (`fallback_allowed = false`) |
| Physical current coupling | recorded per run (`set_ocean_currents`) |

Every result table traces to these raw files, an aggregation command
(`python article/regenerate_tables.py`), and the fingerprint above.

## This revision (documentation + behavior-preserving schema refactor)

- Base commit of the working tree: `2943bf9fbb32c574246dc849320589538d38d692`
  (the revision itself is committed separately from the experiment commit).
- Post-refactor source-tree fingerprint: `41697c8b3c3de0f3d66a6c90a9bf039a68eb79b6a423c52ff9c92fb300462ebc`.
  This **differs** from the experiment fingerprint because the beacon schema was
  refactored and the track JSONs migrated; the change is proven numerically
  inert (below), so the difference does not affect any reported value.

### Beacon schema change — validated numerically equivalent

`beacon.noise_std` (one scalar) was split into `angular_noise_std_deg` (deg,
bearing/elevation) and `range_noise_std_m` (m, range), migrated value-equal on
every official and test track. The Gaussian draw order (bearing, elevation,
range) and the single range-noise application are preserved.

- Legacy-equivalence test: `tests/test_beacon_noise_migration.py`
  (`test_migrated_packets_match_legacy_bitwise`,
  `test_manager_stream_matches_legacy_over_a_trajectory`) — **PASS**, byte-level
  equality of bearing, elevation, range, signal strength and dropout decisions
  across the three official noise levels and several seeds. The refactor is a
  units/schema correction, not a re-calibration of sensor statistics.

### Coordination default

`LeaderFollowerController.MIN_GATE_GAP` and the CLI default changed from 2 to 1
(LF(1) recommended, LF(2) conservative comparison). The existing coordination
artifacts recorded their `min_gate_gap` explicitly, so the default change is
inert for reproduction; both LF(1) and LF(2) artifacts remain unchanged.

## Manuscript build environment

- Engine: MiKTeX `pdflatex` via `latexmk -pdf`.
- Class: `IEEEtran` (conference).
- Build: `cd article && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`.

## Regeneration commands (post-processing of existing artifacts only)

```
# result tables (sample std, from the frozen matrix; also runs the penalty-identity check)
python article/regenerate_tables.py

# figures (track layouts + result plot from artifacts; task schematic from geometry)
python article/figures/generate_figures.py

# optional photo-real HoloOcean render (requires HoloOcean; visualization only)
python -m marine_race_arena.scripts.capture_environment_screenshot
```

## Unchanged artifact directories (not modified by this revision)

- `results/onboard_only_validation/final_20260715/` — the full 78-run matrix,
  per-run summaries, event logs, aggregated results and audits.
- `results/capability_checks/` — the setup checks.
