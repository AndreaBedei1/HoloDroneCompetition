# Marine Race Arena IEEE Paper

This folder contains the v0.1 release-candidate paper source for Marine Race Arena.

Main file: `main.tex`

Recommended local build:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Equivalent Overleaf build sequence:

1. pdfLaTeX
2. BibTeX
3. pdfLaTeX
4. pdfLaTeX

The paper describes one official onboard-only contract. Controllers receive participant-local time, allow-listed onboard sensors, physically received beacon packets and optional controller-authored team messages; no referee progress or target selection reaches autonomy. Both camera-assisted rule controllers maintain course progress with their own `LocalCourseTracker`. The 78-run HoloOcean matrix covers both controllers on all three clean circuits, both controllers under the Horseshoe Bay medium/strong profiles, homogeneous two-rover fleet sweeps, matched three-rover coordination and a yield-margin ablation. The article tables are derived only from the fresh matrix artifacts under `results/onboard_only_validation/final_20260715`: exact coverage and the artifact contract pass, while the overall audit explicitly reports controller-local/referee progress mismatches.

All reported experiments use the reference BlueROV2-class vehicle, real HoloOcean, fallback disabled and unchanged gate geometry. Inter-vehicle proximity detection remains a referee-side feature with off, diagnostic and penalize modes; diagnostic mode is used for the reported fleet experiments.
