# Submission checklist — Robotics and Autonomous Systems

Status legend: `[x]` done and verified · `[ ]` outstanding · `[!]` blocked on
information that cannot be derived from the repository.

## Manuscript

- [x] Target journal set in `main.tex` (`\journal{Robotics and Autonomous Systems}`)
- [x] `elsarticle` document class (not `IEEEtran`)
- [x] Title, authors, affiliation in Elsevier front matter
- [x] Abstract present, single paragraph, no citations, no undefined acronyms
- [x] Keywords present (8 terms, `\begin{keyword}`)
- [x] Declaration of competing interest in the manuscript
- [x] Data availability statement in the manuscript
- [x] Bibliography style `elsarticle-num`
- [ ] CRediT authorship contribution statement inserted into `main.tex`
- [!] Corresponding author, email, ORCID, postal address — see the `TODO` block in `main.tex`
- [!] Funding statement — no funding record exists in the repository

## Build

- [x] Compiles with `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- [x] 0 errors
- [x] 0 undefined references
- [x] 0 undefined citations
- [x] 0 missing files
- [x] 0 overfull hboxes, 0 underfull hboxes, 0 overfull vboxes
- [x] Every bibliography entry is cited (21 of 21)
- [x] PDF reviewed page by page; no figure cropped, no table beyond the margin
- [x] Flat submission directory builds identically (44 pages, same warning counts)

## Scientific integrity

- [x] Every quantitative claim traced to an artifact (`CLAIM_AUDIT.md`)
- [x] Claim audit is executable: `python article_journal/scripts/verify_claims.py` — 95 checks, 95 verified
- [x] Evidence classified VALIDATED / PRELIMINARY / DIAGNOSTIC / DEVELOPMENT ONLY (`SCIENTIFIC_EVIDENCE_MAP.md`)
- [x] No result reported from a run that was still in progress
- [x] Negative results reported: pose front end under obliquity, failed readiness gate, 0/30 holdout
- [x] Overclaim sweep completed: `first`, `only`, `novel`, `state-of-the-art`, `robust`, `reliable`, `general`, `universal`, `proves`, `demonstrates` all reviewed in context
- [x] Novelty claim hedged and immediately bounded by an explicit list of what is not claimed
- [x] Comparison table uses verifiable criteria and states what it omits
- [x] Control rates (30 Hz and 10 Hz) never pooled; stated in every affected table
- [x] Official circuits labelled a retrospective diagnostic, not a held-out set
- [x] Statistical tests named (Wilson intervals, two-sided sign tests) with sample sizes

## Figures and tables

- [x] All figures referenced from the text
- [x] All tables referenced from the text
- [x] Captions above tables, below figures
- [x] `booktabs` rules, no vertical rules, no `\hline` stacks
- [x] Metric direction marked where relevant (`Gates ↑`, `Time ↓`)
- [x] Consistent numeric precision within each column
- [x] Figure provenance recorded (`figures/generated/*.provenance.json`)
- [x] No HoloOcean run launched to produce any figure
- [ ] Schematic figures regenerated from `figure_todo_nanobanana.md` (Schemas 1, 2, 4, 5; Schema 3 refinement) — TikZ placeholders compile in the meantime
- [ ] Graphical abstract produced from `graphical_abstract_prompt.txt`

## Separate upload items

- [x] `highlights.txt` — 5 highlights, all ≤ 85 characters (verified: 70, 76, 73, 82, 79)
- [ ] Graphical abstract image (≥ 1328 × 531 px, or vector)
- [ ] Cover letter — draft in `cover_letter_draft.md`, contains unresolved `TODO`s
- [x] `data_availability_statement.txt`
- [x] `competing_interests.txt`
- [!] `author_contributions_CRediT.md` — roles are **unassigned**; each author must confirm their own
- [ ] Suggested reviewers, if requested by the journal

## Before pressing submit

- [ ] Every `TODO` resolved in `main.tex`, `cover_letter_draft.md`, `author_contributions_CRediT.md`, `competing_interests.txt`, `data_availability_statement.txt`
- [ ] Confirm whether any part of this work appeared in a conference proceedings; if so, name the venue and state the extension explicitly in the cover letter
- [ ] Repository made public and the URL in the manuscript verified to resolve
- [ ] Consider an archival snapshot (e.g. Zenodo) for a citable DOI, and add it to the data availability statement
- [ ] Re-run `python article_journal/scripts/verify_claims.py` and confirm 0 mismatches
- [ ] Re-run `python article_journal/scripts/flatten_submission.py --build` and re-check the four build counters
- [ ] Read the final PDF once more end to end
