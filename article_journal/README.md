# Marine Race Arena — journal manuscript

Target venue: **Robotics and Autonomous Systems** (Elsevier).

`main.tex` is the only manuscript source and `main.pdf` is the compiled
manuscript. Every change is made to the files in this directory.

## Build

```bash
cd article_journal
latexmk -C
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Requires MiKTeX or TeX Live with `elsarticle`, `booktabs`, `multirow`,
`tikz`, `listings` and `threeparttable`.

The layout is one class option in `main.tex`: `preprint,3p,times` (default,
submission format), `review,3p,times` (double-spaced), `final,5p,times`
(two-column published look).

## Layout

```text
main.tex            root document and front matter
refs_journal.bib    bibliography
sections/           01 introduction … 14 conclusion
tables/             one table body per file; seven are regenerated (below)
figures/            figure wrappers and TikZ sources
  captures/         static HoloOcean render
  generated/        built by the scripts below, each with a provenance JSON
scripts/            the four post-processing scripts
highlights.txt      Elsevier highlights
competing_interests.txt
```

## Regenerating tables and figures

```bash
python article_journal/scripts/regenerate_tables.py    # the seven data-driven tables
python article_journal/scripts/generate_figures.py     # track layouts, controller plot
python article_journal/scripts/make_perception_figure.py
```

All three read only `artifacts/paper/`, launch no simulator and modify no
artifact. `regenerate_tables.py` rewrites the tables byte-identically, so a
clean checkout plus a regeneration leaves an empty `git diff`, and it re-checks
the penalty identity on every finished run. The figure scripts write a
provenance JSON next to each output recording their inputs.

## Verifying the claims

```bash
python article_journal/scripts/verify_claims.py
```

Recomputes every quantity stated in the text and the tables from
`artifacts/paper/`, verifies the artifact manifest hashes, and exits non-zero
on any mismatch.

## Updating the learning results

Section 9 (methodology) is checkpoint-independent. To refresh Table
`tab:learning` with a new frozen checkpoint:

1. place the checkpoint under `artifacts/paper/ppo/model/` and record its
   SHA-256, training step and observation/action contract in
   `artifacts/paper/ppo/provenance.json`;
2. run three validation episodes per official circuit under the same
   participant-level interface and write them to
   `artifacts/paper/ppo/validation/`;
3. rebuild `artifacts/paper/manifest.json`, then run `regenerate_tables.py`
   and `verify_claims.py` and update the expected values the latter asserts.

## Before submitting

The following cannot be derived from this repository and must be supplied by
the authors: CRediT author roles, corresponding-author name, email, ORCID and
postal address, the funding statement if any, prior-publication disclosure,
and confirmation of the provenance of the conceptual figures
`figures/p1.png`–`figures/p5.png`.
