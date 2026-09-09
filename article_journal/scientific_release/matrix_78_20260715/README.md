# Compact 78-run scientific release package

This directory contains a machine-readable, compact projection of the frozen
78-run HoloOcean benchmark matrix used by the journal manuscript. It was
derived from `results/onboard_only_validation/final_20260715/` in the
authoritative local results tree; the raw per-run logs remain outside Git and
were not modified.

Contents:

- `manifest.json`: source fingerprint, counts and audit status;
- `runs.json` and `runs.csv`: one normalized row per run, including outcomes,
  event counts, controller/condition metadata and reproduction paths;
- `SHA256SUMS`: hashes for the package data files.

Verify the package without HoloOcean:

```text
python article_journal/scripts/package_78_matrix.py verify
```

Rebuild it from an available raw matrix (post-processing only):

```text
python article_journal/scripts/package_78_matrix.py build \
  --source path/to/results/onboard_only_validation/final_20260715
```

The package does not include neural-network weights, videos or raw simulator
logs. It is sufficient to reproduce the manuscript's aggregate table inputs;
the source manifest hash identifies the original raw tree.
