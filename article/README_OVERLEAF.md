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

The paper reports the official clean-gate benchmark infrastructure, the rule-based baseline, HoloOcean BlueROV2 integration, and the stable staggered fleet/team smoke result. Current compensation, DVL observers, close-proximity fleet racing, and fully calibrated inter-vehicle collision penalties are intentionally described as experimental or future work rather than solved v0.1 claims.
