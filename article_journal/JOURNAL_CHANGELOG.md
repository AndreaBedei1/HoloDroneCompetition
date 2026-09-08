# Journal changelog — what changed from the conference manuscript

This records the relationship between `article/` (the conference manuscript,
**left completely untouched**) and `article_journal/` (this manuscript). Nothing
in `article/` was edited, moved or deleted.

## Repositioning

| | Conference | Journal |
| --- | --- | --- |
| Class | `IEEEtran` (conference) | `elsarticle` (`preprint,3p,times`) |
| Title | *A Configurable HoloOcean Benchmark for Underwater Gate Racing and Team-Level Fleet Evaluation* | *A Reproducible Benchmark for Onboard-Only Autonomous Underwater Gate Racing, Multi-Vehicle Evaluation and Learned Control* |
| Framing | "we built a configurable HoloOcean environment" | "we define an evaluation contract above a simulator backend" |
| HoloOcean's role | close to the contribution | a replaceable backend reached through an adapter |
| Structure | 10 sections, chronological evaluation | 14 sections, evaluation organized by five research questions |
| Contributions | 5 | 7 |
| Pages | 8 | 44 |

The title change is the substantive one. The conference title names the backend;
the journal title names the property that makes the work reusable
(*reproducible*, *onboard-only*) and the two dimensions the conference title
omitted (*multi-vehicle*, *learned control*). Both candidate titles proposed for
this revision were considered; the chosen title merges them, because after the
manuscript was written neither "benchmarking perception, control and
multi-robot autonomy" nor "a reproducible benchmark for onboard-only gate
racing" alone covered the learning extension, which is now a full section with
its own experimental programme.

## Section-by-section disposition

| Conference section | Disposition | Journal destination |
| --- | --- | --- |
| 1 Introduction | **EXPAND** — same opening argument; new paragraphs on why the boundary is measurable, on the three recurring findings, and 7 contributions instead of 5 | §1 |
| 2 Related Work (3 subsections) | **EXPAND** — restructured into 5 topical subsections plus a positioning subsection; MarineGym promoted from a sentence to a dedicated complementarity argument; new subsection on underwater perception | §2 |
| 3 Benchmark Model | **KEEP** — Definitions 1 and 2, the gate frame and the requirement table preserved verbatim; added a paragraph on why the action space fixes comparability | §3 |
| 4 Vehicle, Simulator and Sensing | **KEEP + UPDATE** — all models preserved; **new**: explicit two-control-rate discipline (0.033 s vs 0.1 s, never pooled); explicit statement of the three withheld quantities | §4 |
| — | **NEW** | §5 Onboard Gate Perception and Target Association |
| 5 Controller Interface and Rule Baseline | **KEEP + EXPAND** — both controllers preserved; restructured into motivation / design / advantage; **new**: quantified cost of the tracker's three-modality criterion | §6 |
| 6 Referee | **KEEP + EXPAND** — every equation preserved; **new**: formal two-sided boundary $\Phi_i \perp \sigma(\cdot)$, rationale for the arbitration order, and the boundary measurement subsection | §7 |
| 7 Fleet | **KEEP + EXPAND** — all equations preserved; **new**: explanation of why the team-finish conjunction and the stuck penalty on a yielding vehicle are the right design | §8 |
| — | **NEW** | §9 Learning-Based Extension and Training Pipeline |
| 8 Evaluation | **MOVE + EXPAND** — reorganized from chronological to RQ1–RQ5; all conference results retained and re-presented as systematic benchmark evaluation | §10 |
| 9 Discussion, Limitations, Future Work | **SPLIT** — discussion expanded into 7 analytical subsections; limitations promoted to their own section and expanded from 5 sentences to 12 numbered items; future work moved into the conclusion | §11, §12, §14 |
| 9 Reproducibility (subsection of 8) | **MOVE + EXPAND** — promoted to a full section with three provenance levels and regeneration commands | §13 |
| 10 Conclusion | **EXPAND** — four conclusions with numbers, the central finding named, four future directions | §14 |

## New scientific content

1. **§5 Onboard Gate Perception and Target Association.** The detector and the
   acoustic--visual association were implemented but never described. The section
   documents the full pipeline with its thresholds, formalizes the normalized
   image error and the confidence, and explains why the framework must *not*
   resolve the association on the participant's behalf.
2. **§5.3–5.4 Perception evaluation, including a negative result.** The
   exploratory pose front end is reported with its measured failure under strong
   obliquity (11.7 % pose availability, 48.2° median yaw error, 0.00 orientation
   and sign accuracy) and separately with its accuracy in the near-frontal
   regime the controller actually uses.
3. **§7.7 Measuring the information boundary.** New table quantifying
   controller-local versus referee progression over 1474 advancements. The
   conference paper asserted the separation; the journal version measures it.
4. **§9 Learning extension.** Observation encoding, reward design, curriculum,
   PPO configuration, retention regularizer, pre-registered readiness gate and
   evaluation protocol. Entirely new.
5. **§10.2 RQ1 current-free completion.** New nine-run demonstration with a
   verified zero-current manifest, kept explicitly separate from the 78-run
   matrix.
6. **§10.6 RQ5 matched learned-controller benchmark.** New 474-episode,
   six-controller, paired comparison, plus the pre-registered readiness gate and
   the 0/30 holdout.

## Results retained from the conference paper

All of them, reorganized and none weakened:

* 78 real-HoloOcean runs under a frozen source fingerprint → §10.1
* Clean-track completion for both controllers on three circuits → Table 10
* Medium and strong current degradation → Table 11
* Homogeneous two-vehicle fleet → Table 12 (upper block)
* Three-vehicle leader--follower coordination, LF(1) / LF(2) / uncoordinated → Table 12 (lower block)
* Controller comparison figure → Fig. 11

The numbers were re-derived from the raw artifacts with
`article/regenerate_tables.py --check` and reproduce the committed conference
tables byte-for-byte. No experiment was rerun.

## Bibliography

`refs_journal.bib` starts from `article/refs.bib` (19 entries, all retained and
all still cited) and adds two:

* `schulman2017ppo` — PPO, the algorithm actually used;
* `raffin2021sb3` — Stable-Baselines3, pinned at 2.7.1 in `requirements-rl.txt`
  and recorded in the artifact provenance.

No entry was removed and no metadata was invented. MarineGym
(`chu2025marinegym`) was already present with the published IROS 2025 record and
is now the anchor of §2.5.

## Figures

| Figure | Origin |
| --- | --- |
| 1 Teaser | reused: `article/figures/world.png` |
| 2 Architecture | reused TikZ; replacement specified as Schema 1 |
| 3 Track layouts | reused: `article/figures/tracks_layout.pdf` |
| 4 Perception pipeline | new TikZ; replacement specified as Schema 2 |
| 5 Perception panels | **new**, composed from three committed HoloOcean captures |
| 6 Tracker state machine | reused TikZ, redrawn |
| 7 Center-then-commit states | reused TikZ; refinement specified as Schema 3 |
| 8 Gate geometry | reused TikZ |
| 9 Participant lifecycle | reused TikZ |
| 10 Fleet scoring | reused TikZ; Schema 5 adds a fleet architecture figure |
| 11 Controller comparison | reused: `article/figures/controller_comparison.pdf` |
| 12 Learned completion per group | **new**, computed from `aggregate_by_group.csv` |
| 13 Multi-gate survival | **new**, computed from `readiness_verdict.json` |

**Zero HoloOcean runs were launched for this manuscript.** A PPO training
campaign held every available engine instance throughout; see
`FIGURE_SOURCE_AUDIT.md`.

## Tables

New: system comparison (rebuilt with additional criteria), perception audit,
controller-vs-referee agreement, current-free completion, matched learned
benchmark, paired learned comparison, readiness gate and holdout.
Retained: sensors, official tracks, observation contract, requirements,
commands, penalties. Split for readability: the conference validation table
became three (clean tracks, currents, fleet).

## Not carried over

* The conference `\input{tables/validation_results}` and
  `\input{tables/holoocean_coordination}` composite tables — split into
  condition-specific tables so each carries one message.
* The conference "Future Work" subsection — merged into the conclusion so the
  limitations section is not diluted by directions.

## Deliberate exclusions

See `SCIENTIFIC_EVIDENCE_MAP.md` §D. In short: no Generation-2 result (training
was running while this was written), no SAC result, no G1-only matched selection
as a headline, no obstacle-avoidance claim, no acoustic-channel characterization.
