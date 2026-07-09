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

The paper reports the official clean-gate benchmark infrastructure, the rule-based baseline, HoloOcean BlueROV2 integration, the staggered low-contact two-rover fleet and team smoke result, and the honest degradation of the baseline under medium and strong current profiles. It also adds a second, conservative gate controller and a leader-follower team-coordination controller over the optional acoustic channel, compared on a deterministic kinematic substrate: the two controllers differ measurably in completion time and motion smoothness, and coordination removes a staggered team's inter-vehicle proximity events (to the single-rover level) while every rover still finishes, at the cost of a longer team time. Inter-rover proximity is a plain referee-side feature with off, diagnostic and penalize modes (diagnostic by default). Current-rejecting control is identified as future work.
