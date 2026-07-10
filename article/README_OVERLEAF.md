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

The paper reports the official clean-gate benchmark infrastructure, rule-based baseline, HoloOcean BlueROV2 integration, staggered low-contact two-rover smoke result and baseline degradation under the medium and strong current profiles. On the deterministic kinematic substrate, a second conservative gate controller differs measurably from the first in completion time and motion smoothness. A leader-follower controller uses the optional acoustic channel to reduce a staggered team's inter-vehicle proximity events to the single-rover level while every rover finishes, at the cost of a longer team time. HoloOcean diagnostic validation covers only clean Horseshoe Bay with three rovers, no currents or obstacles, two start gaps and three seeds. All six coordinated runs finish with no inter-vehicle proximity events or gate/world collisions. Uncoordinated teams incur many gate/world collisions and sometimes fail to finish. Inter-vehicle proximity detection is a referee-side feature with off, diagnostic and penalize modes; diagnostic mode is the default.

Current-rejecting control remains future work.
