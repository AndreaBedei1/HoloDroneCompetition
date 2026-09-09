# Producing a flat LaTeX submission for Elsevier Editorial Manager

During development the manuscript uses subdirectories (`sections/`, `tables/`,
`figures/`) because that keeps the sources reviewable. Elsevier's Editorial
Manager can require every source file at a single level with no subdirectories.
This document describes how to produce that flat form.

The development tree stays the canonical source. The flat directory is a
**build product** and is deliberately not committed: it duplicates every figure
and would double the repository weight for no benefit. It is listed in
`.gitignore`.

## One command

```bash
python article_journal/scripts/flatten_submission.py --build
```

This writes `article_journal/submission_flat/` and test-compiles it. Omit
`--build` to skip the compile. Use `--out <dir>` to write elsewhere.

Build the manuscript at least once first, so `main.bbl` exists:

```bash
cd article_journal && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## What the script does

1. Parses `main.tex` and follows every `\input{...}` recursively.
2. Rewrites each `\input{sections/05_perception}` to `\input{05_perception}` and
   each `\includegraphics{figures/generated/x.pdf}` to `\includegraphics{x.pdf}`.
3. Copies every referenced `.tex`, `.pdf` and `.png` into the flat directory,
   prefixing a basename with its source subdirectory only if a collision would
   otherwise occur.
4. Copies `main.bbl` so the submission compiles without a BibTeX pass, and
   `refs_journal.bib` alongside it for the production editor.
5. Copies `elsarticle.cls` and `elsarticle-num.bst` when `kpsewhich` can locate
   them, so the submission is self-contained even if the compilation host lacks
   the Elsevier package.
6. Reports any `\input` or `\includegraphics` target it could not resolve, on
   stderr, without silently dropping it.

The script never writes outside the output directory and never modifies the
development tree. It rebuilds the output directory from scratch on every run, so
a stale file cannot survive into a submission.

## Verifying the result

The flattened build must be identical to the development build. Check all four:

```bash
cd article_journal/submission_flat
pdfinfo main.pdf | grep Pages          # must match the development build
grep -c "LaTeX Warning: Reference" main.log   # must be 0
grep -c "LaTeX Warning: Citation"  main.log   # must be 0
grep -c "^!"                        main.log   # must be 0
```

The compressed development build produces 38 pages with zero undefined
references, zero undefined citations, zero errors and zero overfull/underfull
box warnings.

## What to upload

| Item | File | Notes |
| --- | --- | --- |
| Manuscript source | every `.tex` in `submission_flat/` | `main.tex` is the root |
| Bibliography | `main.bbl` (and `refs_journal.bib`) | the `.bbl` is what compiles |
| Figures | every `.pdf` and `.png` in `submission_flat/` | vector where possible |
| Class and style | `elsarticle.cls`, `elsarticle-num.bst` | only if the system asks |
| Highlights | `highlights.txt` | separate upload item, 3–5 points, ≤ 85 characters each |
| Graphical abstract | produced from `graphical_abstract_prompt.txt` | separate upload item |
| Cover letter | from `cover_letter_draft.md` | resolve every `TODO` first |
| Declaration of interest | `competing_interests.txt` | also present in `main.tex` |
| Data availability | `data_availability_statement.txt` | also present in `main.tex` |
| CRediT statement | from `author_contributions_CRediT.md` | **unresolved TODOs** — must be completed by the authors |

## Switching the layout for submission

`main.tex` currently uses `\documentclass[preprint,3p,times]{elsarticle}`, the
single-column preprint format. Two alternatives, changed on that one line:

* `[review,3p,times]` — double-spaced review format, if the editor requests it;
* `[final,5p,times]` — the two-column published look, useful for checking how
  wide tables will behave in print.

Re-run the flattening script after changing the class option, and re-verify the
four checks above: the wide tables (`system_comparison`, `perception_audit`,
`fleet`, `learning`) are the elements most likely to overflow in `5p`.
