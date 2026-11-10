# Decisions: what was chosen, and why

Each entry says what the code does, the reason, and what it costs. Add new entries at the bottom
with a date. If a decision is reversed, keep the old entry and add a new one that says so.

Dates are the day the decision was written down, not necessarily the day the code was written.

## D1. Two normalization modes: joint and frozen (2026-11)

**What.** `dataset.normalization.mode` is `joint` (default) or `frozen`.
- `joint`: cf-normalization and TMM constants are computed from all samples in the file at once
  (`normalization.normalize_counts`). This is the paper's method.
- `frozen`: constants are fitted once on `reference_cohorts` (`normalization.fit_normalizer`) and
  applied to every sample on its own (`normalization.apply_normalizer`).

**Why.** Joint normalization lets every sample influence every other sample's values, including
the validation cohort's influence on the training cohort. A real new patient arrives alone, so
only frozen mode matches how the model would be used, and only frozen mode can make a bundle
(`training.train_bundle` refuses without constants).

**How to use them.** Joint reproduces the paper. Frozen with `reference_cohorts: [CEDAR]` is the
honest validation (BCC is normalized with constants that never saw it). Frozen with
`[CEDAR, BCC]` is for the final model.

**Cost.** Joint and frozen numbers differ (29 vs 36 genes with CEDAR as reference), so every
result must say which mode it used.

## D2. SMOTE lives inside the pipeline (2026-11)

**What.** `models.build_paper_pipeline` is `log` (log1p), `smote`, `rf`, an imblearn Pipeline.
SMOTE adds `n_synthetic_pdac` synthetic PDAC samples (default 100; `null` balances exactly) and
only while fitting.

**Why.** Inside the pipeline SMOTE sees only the training rows it is given and is skipped when
scoring. Resampling beforehand would put synthetic neighbours of validation samples into
training.

**Cost.** The default of 100 synthetic samples is a fixed number, not a balanced class; `null` is one
`--set` away if you want to compare ([../configuration.md](../configuration.md)).

## D3. Genes are chosen on training rows only (2026-11)

**What.** `selection.select_de_genes` (variance above 1, Wilcoxon test per gene, Benjamini-Hochberg
below 0.05) is run on the training cohort, and the experiment setting `gene_selection_task` fixes which labels
it uses (`training.select_experiment_genes`).

**Why.** Selecting genes with the validation labels would leak them into the model.

**Cost.** The list depends on the training set. CEDAR has only 20 PDAC samples, so lists are
unstable: 29 genes (joint), 36 (frozen, CEDAR), and 872 for a model trained on CEDAR plus BCC. The
872-gene case is unresolved; its threshold would need its own choice, and the gene-selection rule
for the final model still needs deciding.

## D4. Scores are averaged over seeds, AUC reported with spread and a bootstrap interval (2026-11)

**What.** `n_seeds` (default 10) models with seeds `seed, seed+1, ...`. The run reports the mean
AUC with its spread (`ddof=1`), the AUC of the seed-averaged score, and a bootstrap 95% interval
(`evaluation.bootstrap_auc_ci`).

**Why.** SMOTE and the forest are random; a single seed can mislead. With 21 PDAC samples in BCC
the interval is wide (about plus or minus 0.06), and it should be quoted next to every AUC.

## D5. The threshold is chosen on the BCC scores, and an honest estimate is reported (2026-11)

**What.** `scripts/select_threshold.py` scores BCC with the CEDAR-trained model, picks the
threshold with `thresholds.choose_threshold` (default rule F-beta with beta 2, which values missing
a cancer twice as much as a false alarm), and reports two things:
- the apparent performance at that threshold (optimistic, because the threshold was tuned on the
  same samples), and
- `thresholds.cv_threshold_performance`: repeatedly choose on four fifths of BCC, test on the
  remaining fifth. This is the number to quote.

**Why not choose on CEDAR first (the textbook route).** It is implemented
(`--choose-on train`, using `thresholds.oof_scores`) and it does not work at this sample size.
With in-fold gene selection and 20 PDAC samples, the gene list changes a lot between folds
(some folds select no gene), so the out-of-fold scores do not describe the final model. With
genes fixed beforehand, the held-out labels leak into the selection. Neither gives a threshold
that transfers.

**Cost.** BCC is used both to pick the threshold and to test it. The cross-validated estimate
removes most of the optimism but BCC is no longer a pure hold-out. A real hold-out needs a third
cohort.

## D6. One config set, changed per run with `--set` and `--patch` (2026-11)

**What.** Defaults live in `configs/dataset.yaml`, `configs/experiment.yaml` and
`configs/model.yaml`. A run changes them with `--set section.key=value` (value parsed as YAML) or
`--patch file.yaml`, merged in `run_config.load_run_config`. If anything was changed and no name
was given, the run name is the base name plus 8 hex characters of a hash of the settings
(`run_config._auto_name`).

**Why.** One file per experiment drifts apart. With one default set and recorded changes, any run
can be reproduced from its `resolved_config.yaml`, and two runs with the same changes get the same
name.

**Cost.** Auto names are not readable; pass `--name` for runs you want to find later.

## D7. Every run is a folder with its own record (2026-11)

**What.** `runs.tracked_run` takes a file lock, creates `runs/<timestamp>_<name>/`, saves
`resolved_config.yaml`, saves `code_changes.diff` if the git tree is dirty, and keeps
`run_metadata.json` with the status (`running`, `success`, `failed`), package versions and
results.

**Why.** Results should be traceable to the exact settings and code. The lock stops two runs from
writing at once.

**Cost.** Failed runs leave a folder behind (cleaning them is on the robustness list).

## D8. Strict configs (2026-11)

**What.** All config classes are pydantic with `extra="forbid"` and `frozen=True`.

**Why.** A misspelt setting should stop the run, not be ignored. A frozen config cannot be
changed halfway through a run.

## D9. A bundle is a pickle plus a readable sidecar (2026-11)

**What.** `bundle.save_bundle` writes `model.joblib` and `model.json` (the metadata only).
`bundle.load_bundle` rejects a bundle saved with a different major.minor of scikit-learn or
imbalanced-learn.

**Why.** Pickles break silently across library versions. The version check turns that into a clear
message. The JSON lets a person read what a model is without loading it.

**Cost.** Pickles can run code on load. Only load your own bundles.

## D10. The sex-gene removal was deliberately not implemented (2026-11)

**What.** There is no step that removes sex-linked genes, no `exclude_genes` function or
setting, and no related tests or docs. Early code for it was removed on request.

**Why.** It was dropped on request and is not part of this project. Do not describe the pipeline as removing
sex genes, and do not compare gene lists as if it did (`scripts/compare_genes.py` compares the
selected genes with the paper's list exactly as selected).

## D11. Scoring external data needs unfiltered raw counts (2026-11)

**What.** `scripts/evaluate_external.py` and `scripts/predict.py` take raw counts, genes x samples.
The external file `expr_ml.csv` (50 samples) is log-scale and filtered, so it cannot be scored by
the model: normalization needs the original counts, and the filtered file lacks genes the model
needs (`predict.predict_scores` raises on missing genes).

**Why.** Normalizing an already normalized, filtered table gives numbers on the wrong scale, and
scoring them would look like a result. The scripts refuse rather than guess.

**Next step.** Ask for the raw count file of those 50 samples. Labels come from the sample names:
a PDAC suffix is class 1, `Control`/`Controles` class 0, anything else raises
(`scripts/evaluate_external.py`).

## D12. The docs are checked by tests (2026-11)

**What.** `docs/code/api_reference.md` is generated by `scripts/make_api_reference.py`; the other
pages are hand-written. `tests/test_docs.py` fails if the reference is stale or a hand-written page
names a `module.name` or file that does not exist.

**Why.** Documentation of code goes wrong when functions are renamed. A failing test is the only
reminder that does not rely on anyone remembering.

**Cost.** 44 functions still have no docstring and show as `_no docstring_` in the reference.

## Open items

- Decide the gene-selection rule and threshold for the final CEDAR plus BCC model (872 genes in
  the sandbox).
- Obtain raw counts for the 50 external samples to get a real hold-out.
- Robustness: load each dataset once per run, record input file hashes, validate counts in
  `scripts/predict.py`, clean failed run folders, split the tests into fast and slow.
- Add the missing docstrings.
