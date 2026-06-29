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

The paper reports the official clean-gate benchmark infrastructure, the rule-based baseline, HoloOcean BlueROV2 integration, the staggered low-contact two-rover fleet and team smoke result, and the honest degradation of the baseline under medium and strong current profiles. Inter-rover proximity is a plain referee-side feature with off, diagnostic and penalize modes (diagnostic by default). Current-rejecting control is identified as future work.
