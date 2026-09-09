# Marine Race Arena — journal manuscript

Journal version of the Marine Race Arena paper, targeting **Robotics and
Autonomous Systems** (Elsevier).

The conference manuscript in [`../article/`](../article/) is **untouched** and
remains the record of that submission. This directory is a separate, substantially
extended manuscript; see [`JOURNAL_CHANGELOG.md`](JOURNAL_CHANGELOG.md) for the
section-by-section relationship between the two.

## Current state

| | |
| --- | --- |
| Target journal | Robotics and Autonomous Systems (Elsevier) |
| Document class | `elsarticle`, `preprint,3p,times` (single column) |
| Length | 39 pages after the final figure pass, 14 sections, 16 main figures, 11 main tables, 36 references |
| Build | clean — 0 errors, 0 undefined references, 0 undefined citations, 0 missing files, 0 overfull/underfull boxes |
| Claim audit | 78 checks, 78 verified, 0 mismatched |
| Blocking work remaining | CRediT roles, corresponding-author details, graphical abstract, figure provenance confirmation |

## Build

```bash
cd article_journal
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Requires MiKTeX or TeX Live with `elsarticle`, `booktabs`, `tikz`, `pgfplots`,
`listings` and `threeparttable`. Output: `main.pdf`.

To switch layout, change the single class option in `main.tex`:
`preprint,3p,times` (default, submission format) · `review,3p,times`
(double-spaced) · `final,5p,times` (two-column published look).

## Structure

```
article_journal/
  main.tex                       root document, front matter, class options
  refs_journal.bib               21 entries, all cited
  sections/                      01 intro … 14 conclusion
  tables/                        16 table bodies, one file each
  figures/
    *.tex                        figure wrappers and TikZ sources
    existing/                    assets reused from article/ and results/
    generated/                   built by the scripts below, with provenance JSON
  scripts/
    make_perception_figure.py    composes the gate-perception panel figure
    verify_claims.py             executable claim audit
    package_78_matrix.py          immutable 78-run release package/checker
    flatten_submission.py        flat directory for Editorial Manager
  SCIENTIFIC_EVIDENCE_MAP.md     what evidence exists and how it is classified
  CLAIM_AUDIT.md                 every number, its source, its status
  FIGURE_SOURCE_AUDIT.md         what was reused, what was built, what was deferred
  JOURNAL_CHANGELOG.md           conference -> journal disposition
  ai_image_disclosure_todo.md    provenance/disclosure TODO for supplied schematics
  flatten_submission.md          how to produce the flat submission
  submission_checklist.md        pre-submission checklist
  highlights.txt                 5 highlights, each <= 85 characters
  graphical_abstract_prompt.txt  detailed brief for the graphical abstract
  cover_letter_draft.md          cover letter (contains TODOs)
  author_contributions_CRediT.md CRediT template (roles UNASSIGNED)
  competing_interests.txt        declaration
  data_availability_statement.txt
```

## Where the figures come from

`FIGURE_SOURCE_AUDIT.md` has the full table. In summary:

* **Reused unchanged** — the HoloOcean course render, the track-layout and
  controller-comparison PDFs, five conference TikZ sources, two trajectory plots.
* **Reused with post-processing** — three committed native-HoloOcean simulator
  captures, cropped and relabelled in English by
  `scripts/make_perception_figure.py`.
* **Computed from frozen artifacts** — the survival curve and the per-group
  completion chart is retained only as historical supporting material; the
  current main manuscript uses the final Gen-2 validation table.
* **New HoloOcean captures — none.** A PPO training campaign held every engine
  instance while this manuscript was written; no figure script launches the
  simulator, and each writes a provenance JSON asserting so.

```bash
python article_journal/scripts/make_perception_figure.py
python article_journal/scripts/package_78_matrix.py verify
```

## Schematic figures

Five conceptual figures are now inserted in Sections 3, 5, 6, 8 and 9. Their
provenance/disclosure status is tracked in `ai_image_disclosure_todo.md`.

Five conceptual figures previously needed graphic design rather than an auto-generated
diagram. `figure_todo_nanobanana.md` contains a complete generation prompt for
each — layout, arrows, forbidden connections, colour groups, verbatim text,
aspect ratio, caption and exact LaTeX insertion point:

1. Overall architecture and the information boundary (the paper's central figure)
2. Onboard gate-perception pipeline
3. LocalCourseTracker and the Center-then-Commit passage (refinement)
4. Learning pipeline and curriculum
5. Fleet architecture and team scoring

TikZ placeholders compile today, so the manuscript is not blocked on these.

## Regenerating the result tables

Tables 10–12 come from the frozen 78-run matrix, which is git-ignored. With the
raw artifacts present:

```bash
python article/regenerate_tables.py --check
```

This prints the aggregation and verifies the penalty identity without writing
anything. The values reproduce the committed conference tables byte-for-byte;
the journal tables split them by condition and add the control rate.

Tables 13–15 (learning) come from `results/rl_public/final_benchmark/`, which
**is** tracked, so they can be re-derived in any checkout.

## Updating the PPO results

The learning section is written so its tables can be refreshed without touching
the prose. To add or replace a checkpoint:

1. Record its SHA-256 in the benchmark package manifest.
2. Run the same matched suite on the same registered seeds.
3. Regenerate `tables/learning.tex` and the supporting paired/readiness tables
   from the new `aggregate_by_group.csv` and paired-comparison output.
4. Re-run `python article_journal/scripts/make_learning_figures.py`.
5. Re-run `python article_journal/scripts/verify_claims.py` and update the
   expected values it asserts.

Section 9 (methodology) is checkpoint-independent and does not change.
Section 10.6 states its numbers in prose as well as in tables, so update both;
`CLAIM_AUDIT.md` lists exactly which sentences carry numbers.

**Do not add any Generation-2 result until it is frozen, hashed and has passed
a readiness verdict.** Nothing from that line appears in this manuscript, by
design (see `SCIENTIFIC_EVIDENCE_MAP.md` §D).

## Verifying the manuscript

```bash
python article_journal/scripts/verify_claims.py
```

Recomputes 95 headline numbers from the frozen artifacts and exits non-zero on
any mismatch. It launches nothing and modifies nothing. The 78-run matrix rows
read from the main checkout via `MRA_MATRIX_ROOT`; if that tree is absent those
rows report `SKIP` rather than failing.

## Preparing the submission

```bash
python article_journal/scripts/flatten_submission.py --build
```

Writes `submission_flat/` with every file at one level and test-compiles it. The
flat directory is a build product and is git-ignored. See
`flatten_submission.md` for what to upload and how to verify the result.

## Before submitting

Work through `submission_checklist.md`. The items that cannot be resolved from
this repository, and that must be supplied by the authors, are:

* **CRediT author roles** — unassigned in `author_contributions_CRediT.md`.
  Individual contributions are not documented anywhere in the repository, and
  inferring them from commit metadata would be fabrication.
* **Corresponding author** name, email, ORCID and postal address.
* **Funding statement**, if any funding applies.
* **Prior-publication disclosure** — whether any part appeared in a conference
  proceedings, and if so how this version extends it.
