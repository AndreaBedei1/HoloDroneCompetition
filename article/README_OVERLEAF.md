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

The paper reports the official clean-gate benchmark infrastructure, staggered low-contact fleet validation and baseline degradation under the medium and strong current profiles. It also compares continuous visual servoing with center-then-commit gate passage as single-rover strategies in HoloOcean. Both finish all five clean Horseshoe Bay seeds. Center-then-commit records no gate/world collisions and reduces mean official time from 231.2 to 184.9 s. HoloOcean diagnostic validation covers only clean Horseshoe Bay with three rovers, no currents or obstacles, two start gaps and three seeds. All six coordinated runs finish with no inter-vehicle proximity events or gate/world collisions. Uncoordinated teams incur many gate/world collisions and sometimes fail to finish. Inter-vehicle proximity detection is a referee-side feature with off, diagnostic and penalize modes; diagnostic mode is the default.

Current-rejecting control remains future work.
