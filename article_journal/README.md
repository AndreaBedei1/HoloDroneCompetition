# Marine Race Arena: A Configurable HoloOcean Benchmark for Underwater Gate Racing and Team-Level Fleet Evaluation

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
| Length | 33 pages in the current build, 14 sections |
| Build | latest build succeeds; 0 errors, 0 undefined references and 0 undefined citations |
| Claim audit | current manuscript headline values verified; historical release artifacts retained separately |
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

* **Reused unchanged** — the HoloOcean course render, five conference TikZ
  sources and two trajectory plots.
* **Computed from committed result artifacts** — the controller-comparison
  figure reports the three clean circuits and the Horseshoe Bay medium-current
  condition used in the current manuscript.
* **Reused with post-processing** — three committed native-HoloOcean simulator
  captures, cropped and relabelled in English by
  `scripts/make_perception_figure.py`.
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

The current manuscript contains committed table projections for current-free,
clean-track, medium-current, fleet and recurrent-PPO results. A historical
78-run release package is retained under
`scientific_release/matrix_78_20260715/` for artifact provenance and is not
presented as the complete evaluation matrix of the current manuscript.

With the raw artifacts present, the post-processing check can be run with:

```bash
python article/regenerate_tables.py --check
```

This prints the aggregation and verifies the penalty identity without writing
anything.

The recurrent-PPO table is derived from the committed learning result artifacts.

## Updating the PPO results

The learning section is written so its tables can be refreshed without touching
the prose. To add or replace a checkpoint:

1. Record its SHA-256 in the benchmark package manifest.
2. Run three validation episodes per official circuit under the reported
   participant-level interface.
3. Regenerate `tables/learning.tex` from the per-episode result files.
4. Re-run `python article_journal/scripts/make_learning_figures.py`.
5. Re-run `python article_journal/scripts/verify_claims.py` and update the
   expected values it asserts.

Section 9 (methodology) is checkpoint-independent and does not change.
Section 10.6 states its numbers in prose as well as in tables, so update both;
`CLAIM_AUDIT.md` lists exactly which sentences carry numbers.

The current learning result is reported as a recurrent-PPO validation
demonstration; superseded learning artifacts remain historical repository
material and are not used as current manuscript evidence.

## Verifying the manuscript

```bash
python article_journal/scripts/verify_claims.py
```

Recomputes the retained headline values from available artifacts and exits
non-zero on any mismatch. It launches nothing and modifies nothing. The
historical release package can be verified separately with
`python article_journal/scripts/package_78_matrix.py verify`.

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
