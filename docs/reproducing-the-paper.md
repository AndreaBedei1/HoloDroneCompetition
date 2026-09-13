# Reproducing the paper

Every quantity reported in the manuscript is recomputed from a small evidence
package that ships with the repository. **You do not need HoloOcean, a GPU, or
any training to verify the paper** — the four commands below read files and
nothing else.

If you want to *re-run* experiments rather than verify results, see
[experiments.md](experiments.md) instead.

---

## What you need

```bash
pip install -r requirements-dev.txt
```

That is matplotlib, Pillow and pytest. Verifying the claims needs only the
Python standard library; the figure scripts need the other two.

---

## 1. Verify every reported number

```bash
python article_journal/scripts/verify_claims.py
```

It recomputes each value stated in the text and in the tables from
`artifacts/paper/`, checks the artifact manifest hashes, prints one line per
check, and exits non-zero on any mismatch.

```text
YES  current-free completions                got='9/9' expected='9/9'
YES  clean mean time [horseshoe/ctc]         got=197.7 expected=197.7
YES  medium official mean [servo]            got=337.9 expected=337.9
YES  convoy team time [gap8/LF1]             got=266.0 expected=266.0
YES  PPO gates completed                     got=153 expected=153
...
257 checks: 257 verified, 0 mismatched
```

The checks cover the clean-circuit comparison, the medium-current condition, the
homogeneous fleet and the three-vehicle convoy, the current-free completion
demonstration, the recurrent-PPO validation, the perception audit, the
controller-local versus referee progression audit, the penalty identity on every
finished run, and the SHA-256 of all 40 released artifacts.

---

## 2. Regenerate the result tables

```bash
python article_journal/scripts/regenerate_tables.py
```

This rewrites the seven data-driven tables in `article_journal/tables/` from
`artifacts/paper/` and re-checks that
`penalized_time = official_time + penalties` on every finished run.

The tables are **byte-identical** by construction, so on a clean checkout:

```bash
git diff --stat article_journal/tables/    # prints nothing
```

An empty diff means the committed tables really are the artifacts. If a table
changes, an artifact changed.

Use `--check` to print the tables and the comparison without writing anything.

---

## 3. Regenerate the figures

```bash
python article_journal/scripts/generate_figures.py       # track layouts, controller comparison
python article_journal/scripts/make_perception_figure.py # onboard perception panels
```

Both write into `article_journal/figures/generated/` together with a provenance
JSON recording their inputs and asserting that no simulator was launched. Their
PDF output suppresses the embedded creation timestamp, so these are
byte-reproducible too — a full regeneration on a clean checkout leaves
`git status` clean.

`generate_figures.py` additionally asserts the completion counts it plots
against the values reported in the manuscript, and fails loudly rather than
drawing a figure that disagrees with the text.

---

## 4. Build the manuscript

```bash
cd article_journal
latexmk -C
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Needs a TeX distribution with `elsarticle`, `booktabs`, `multirow`, `tikz`,
`listings` and `threeparttable`. The result is a 33-page `main.pdf` with zero
undefined references and zero undefined citations.

---

## Everything in one pass

```bash
python article_journal/scripts/verify_claims.py \
  && python article_journal/scripts/regenerate_tables.py \
  && python article_journal/scripts/generate_figures.py \
  && python article_journal/scripts/make_perception_figure.py \
  && git status --short
```

A clean `git status` at the end is the strongest statement this repository can
make: the committed tables, figures and manuscript are exactly what the released
artifacts produce.

---

## What the evidence package contains

`artifacts/paper/` is 40 files and 5.3 MB. Each one is listed in
[`manifest.json`](../artifacts/paper/manifest.json) with its purpose, size,
SHA-256 and the manuscript section, table or figure it supports.

| Directory | Contents |
|---|---|
| `benchmark/` | One row per benchmark run actually reported: clean circuits, the medium-current condition, the homogeneous fleet and the three-vehicle convoy. Plus the controller-local versus referee progression audit. |
| `current_free/` | The nine current-free onboard-only runs, per circuit, with their evaluation manifests and the seed registry. |
| `ppo/` | The frozen recurrent-PPO checkpoint, its training configuration, its provenance record, and the validation episodes. |
| `perception/` | The perception audit metrics, the gate-pose estimator metrics, and the three HoloOcean captures the perception figure uses. |

Two deliberate scoping decisions are recorded in the artifacts themselves rather
than left implicit:

* `benchmark/runs.csv` holds the **68** runs the manuscript reports, not all 78
  that were executed. The ten strong-current runs are excluded because the
  current manuscript does not report them; `runs.json` states this.
* `benchmark/local_vs_referee.json` is the one aggregate still computed over
  **all 78** executed runs. It is an audit of the information boundary across
  every run that was executed, not a per-condition result, and its `scope` field
  says so.

---

## Identifying the learned controller

The released checkpoint is
`artifacts/paper/ppo/model/policy_recurrent_ppo_27d.zip`
(SHA-256 `8d6a2958…`, 4,100,072 bytes, 50,688 training steps, 27-D observation,
4-D action, 10 Hz).

Its identity was not taken from a directory name. The validation run recorded no
checkpoint path, and the evaluator's hardcoded default pointed at a *different*
training campaign, so the checkpoint was identified from behavioural signatures
— per-circuit path length, completion time and surge/sway statistics — which
match one candidate and exclude the others by margins far outside seed-to-seed
variation. The full reasoning is in
[`artifacts/paper/ppo/provenance.json`](../artifacts/paper/ppo/provenance.json).

You can confirm the file is the one the manifest describes:

```bash
python -c "import hashlib,pathlib; print(hashlib.sha256(pathlib.Path('artifacts/paper/ppo/model/policy_recurrent_ppo_27d.zip').read_bytes()).hexdigest())"
```

---

## A note on re-running experiments

Verifying the paper and re-running its experiments are different things.

The reported benchmark matrix is **frozen**. Re-running a condition on HoloOcean
will reproduce the referee's qualitative outcome — which gates were completed,
whether the run finished, how many events were charged — but not the exact
official time to the decimal: engine stepping is sensitive to machine state, and
a repeat of a recorded seed can land tens of seconds away from the recorded
value. That is a property of the simulator, not of the benchmark logic, and it
is why the manuscript reports the frozen artifacts rather than asking readers to
re-run 78 races.

Every recorded run carries its exact reproduction command:

```bash
python -c "import csv; rows=list(csv.DictReader(open('artifacts/paper/benchmark/runs.csv',encoding='utf-8'))); print(rows[0]['reproduction_command'])"
```
