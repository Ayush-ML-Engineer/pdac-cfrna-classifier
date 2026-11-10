# Data contracts: what every table and file looks like

Every function in this repo passes tables around. Most bugs here come from a wrong orientation
(genes in rows vs samples in rows) or a wrong index. This page lists the shape of everything that
crosses a boundary: inputs, in-memory objects, and files written to a run folder.

For where each object is created see [execution_flows.md](execution_flows.md); for every function
signature see [api_reference.md](api_reference.md) (generated).

## 1. The one rule about orientation

| Where | Orientation | Rows | Columns |
| --- | --- | --- | --- |
| Raw counts CSV (`pdac_genecount.csv`) | genes x samples | genes | samples |
| Anything inside `normalization` (`normalization.cf_norm`, `normalization.normalize_counts`, `normalization.fit_normalizer`, `normalization.apply_normalizer`) | genes x samples | genes | samples |
| Input of `predict.predict_scores` and `scripts/predict.py --counts` | genes x samples | genes | samples |
| `data.Dataset.X` and everything after it (selection, models, thresholds) | samples x genes | samples | genes |

The switch happens once, in `data.load_dataset` (step 5, `X = X.T`). A matrix file stored the
other way round is declared with `matrix.orientation: samples_by_genes` in `configs/dataset.yaml`
and flipped at step 2, so the normalization code always sees genes x samples.

`predict.predict_scores` raises if gene names are found in the columns but not in the rows (a
transposed file), instead of silently scoring nonsense.

## 2. Input files

All paths in `configs/dataset.yaml` use `${PDAC_DATA_ROOT}`, read from `.env` (see
[../configuration.md](../configuration.md)).

| File | Key setting | Layout |
| --- | --- | --- |
| `pdac_genecount.csv` | `matrix.path` | genes x samples, first column = gene name, raw integer counts |
| `updated_intrinsic.csv` | the `factors` setting in the normalization block | genes as index, numeric columns `Intrinsic` and `Extrinsic` (`normalization.FACTOR_COLUMNS`) |
| `pdac_meta.csv` | `metadata.path` | one row per sample; `SeqID` = sample id, `Pancreatitis_type` = diagnosis, `Cohort` = CEDAR or BCC |

Rules the loader enforces (all in `data.load_dataset`):

- All three files are checked to exist before any is read, with a message naming the config path
  and the `.env` variable.
- Sample ids are turned into strings on both sides (matrix columns and metadata `SeqID`).
- Every matrix sample must appear in the metadata, otherwise `ValueError`.
- Labels are stripped of spaces, then looked up in `class_map`. A label not in the map is an error
  when `unmapped: error` and a dropped sample when `unmapped: drop`. The task chosen in the
  experiment replaces `class_map` and `unmapped` (`data.apply_task`).
- Normalization sees every sample in the file; dropping by `class_map` happens afterwards.

The factor table is also what decides which genes exist downstream: `normalization.fit_normalizer`
keeps only genes present in both the counts and the factor table.

## 3. In-memory objects

### `data.Dataset`

| Field | Type | Meaning |
| --- | --- | --- |
| `X` | DataFrame, samples x genes | normalized values (or the matrix as read, if `normalization` is `null`); index named `sample_id`, strings |
| `y` | int Series, same index as `X` | class id from `class_map`, named `y`; 1 = PDAC |
| `meta` | DataFrame, same index as `X` | the metadata rows of the kept samples |
| `config` | `config.DatasetConfig` | the settings that built it |
| `normalizer` | `normalization.NormalizationConstants` or None | set only in frozen mode |

`Dataset.train_mask` and `Dataset.validation_mask` are boolean Series indexed like `X`, built
from `metadata.cohort_column` and `split.train_cohorts` / `split.validation_cohorts`. An empty
`validation_cohorts` list gives an all-False validation mask (the final model on everything).
Use them as `ds.X.loc[ds.train_mask.to_numpy(), genes]` and `ds.y[ds.train_mask]`; the rows of
`X` and `y` always line up.

### `normalization.NormalizationConstants`

Everything a single new sample needs to be normalized like the reference batch:

| Field | Type | Meaning |
| --- | --- | --- |
| `factors` | DataFrame | `Intrinsic` / `Extrinsic` of the genes the reference batch covers |
| `median_intrinsic` | float | median over the reference batch of the intrinsic totals |
| `median_batch` | float | median over the reference batch of the batch-part totals |
| `tmm_reference` | Series by gene | cf-normalized counts of the TMM reference sample |
| `tmm_scale` | float | geometric mean of the reference batch's raw TMM factors |

Created by `normalization.fit_normalizer` from at least 2 reference samples; used by
`normalization.apply_normalizer`. In joint mode none exists, which is why joint mode cannot score
a single new sample and cannot make a bundle.

### `bundle.ModelBundle`

| Field | Type | Meaning |
| --- | --- | --- |
| `pipeline` | fitted imblearn Pipeline | steps `log` (log1p), `smote` (fit only), `rf` |
| `genes` | list of str | training genes, in the column order the pipeline expects |
| `normalizer` | `NormalizationConstants` | frozen constants |
| `metadata` | dict | see below |

Saved as `model.joblib` plus a readable `model.json` (same name) by `bundle.save_bundle`. The
JSON is only the metadata, not the model. `bundle.load_bundle` refuses a bundle saved with a
different major.minor of scikit-learn or imbalanced-learn, and joblib files are pickles: load only
your own.

`ModelBundle.metadata` keys: `format_version`, `created_at`, `experiment`, `dataset`, `task`,
`genes`, `n_train`, `n_train_pdac`, `seed`, `n_synthetic_pdac`, `model`, `overrides`, `config`,
`library_versions`, `git`. `scripts/train_model.py` adds `validation_auc` and `n_validation` when
the split has a validation cohort.

### The scoring contract

`predict.predict_scores(counts, bundle)` takes raw counts, genes x samples, any number of samples
(one is enough), and returns a Series named `pdac_score`, indexed by sample, values in 0 to 1 =
probability of class 1. Scores of one sample never depend on the other samples in the table.
It raises `ValueError` if the table looks transposed, if model genes are missing, or if the
counts are invalid (negative, missing, repeated names, a sample with no counts).

## 4. Settings (`config.*`, `experiment.*`, `model_config.*`)

`run_config.RunConfig` has four parts: `dataset`, `experiment`, `model` and `overrides`.
All are pydantic models with `extra="forbid"` and `frozen=True`, so a misspelt key is an error
and a config cannot change after loading. `overrides` is provenance only (the `--set` / `--patch`
arguments applied).

| Section | File | Fields |
| --- | --- | --- |
| dataset | `configs/dataset.yaml` | `name`, `matrix` (`path`, `orientation`), `normalization` (`factors`, `mode`, `reference_cohorts`), `metadata` (`path`, `sample_id_column`, `label_column`, `cohort_column`), `class_map`, `positive_class`, `unmapped`, `split` (`train_cohorts`, `validation_cohorts`) |
| experiment | `configs/experiment.yaml` | `name`, `task`, `gene_selection_task`, `variance_cutoff`, `alpha`, `log_variance`, `n_synthetic_pdac`, `seed`, `n_seeds` |
| model | `configs/model.yaml` | `name`, `n_estimators`, `max_features`, `min_samples_leaf`, `bootstrap` |

Constraints worth knowing: `reference_cohorts` is only allowed with `mode: frozen`;
the experiment `name` and the model `name` may only contain letters, digits, `_` and `-`;
`task` and `gene_selection_task` must be keys of `experiment.TASKS`; `n_synthetic_pdac: null`
means SMOTE balances the classes exactly.

## 5. Files in a run folder

`runs/<timestamp>_<name>/` is created by `runs.tracked_run`. These three are written by every
run:

| File | Content |
| --- | --- |
| `resolved_config.yaml` | the final merged settings (can be reused with `--patch`) |
| `run_metadata.json` | see below |
| `code_changes.diff` | only if the git tree had uncommitted changes: the diff |

`run_metadata.json` keys: `experiment_name`, `overrides`, `task`, `seed`, `config`, `git`,
`started_at`, `python`, `packages`, `status` (`running`, then `success` or `failed`),
`finished_at`, `duration_seconds`, and `results`. A failed run also has `error`.
`results` is whatever the script put in `run.extra`:

| Script | Keys in `results` |
| --- | --- |
| `scripts/run_paper_replication.py` | `n_genes`, `n_train`, `n_train_pdac`, `n_test`, `n_test_pdac`, `seeds`, `auc_per_seed`, `mean_auc`, `std_auc`, `seed_averaged_auc`, `seed_averaged_ci_low`, `seed_averaged_ci_high` |
| `scripts/train_model.py` | `n_genes`, `n_train`, `n_train_pdac`, `validation_auc` |
| `scripts/select_threshold.py` | the contents of `threshold.json` (below) |

`reporting.runs_table` only uses runs with `status: success` and a `mean_auc` in `results`, so
model-training and threshold runs never appear in the summary table.

Other files, by script:

| File | Written by | Layout |
| --- | --- | --- |
| `de_genes.csv` | `scripts/run_paper_replication.py`, `scripts/replicate_de_selection.py` | `gene,p_value,q_value,auc,direction,significant`; one row per gene that passed the variance filter |
| `test_scores.csv` | `scripts/run_paper_replication.py` | index `sample_id`; `y_true`; one column per seed; `mean_score` |
| `model.joblib`, `model.json` | `scripts/train_model.py` | the bundle (section 3) |
| `gene_comparison.md` | `scripts/compare_genes.py` | table of overlap with `reference/paper_genes.yaml` |
| `validation_scores.csv` | `scripts/select_threshold.py` | `sample_id,label,pdac_score` |
| `oof_scores.csv` | `scripts/select_threshold.py --choose-on train` | out-of-fold training scores |
| `threshold_table.csv` | `scripts/select_threshold.py` | one row per threshold from 0.01 to 0.99: `threshold,tp,fn,tn,fp,sensitivity,specificity,precision,f1,f2,youden` |
| `threshold.json` | `scripts/select_threshold.py` | see below |
| `pr_curves.png`, `confusion_matrices.png` | `scripts/select_threshold.py` | the two plots |

`threshold.json` keys: `choose_on` (`validation` or `train`), `rule`, `n_final_genes`,
`validation_apparent` (`counts` = tp/fn/tn/fp, `rates` = `[estimate, low, high]` Wilson interval
for sensitivity, specificity, ppv, npv), `validation_cv` (`threshold_median`, `threshold_min`,
`threshold_max`, `sensitivity`, `specificity`, `ppv`: the honest estimate from choosing the
threshold on part of the samples and testing on the rest), and `threshold`, the chosen value.
With `--choose-on train` the file has `gene_selection`, `genes_per_fold`, `training_oof` and
`validation` (the threshold was locked beforehand) instead of the apparent and cv blocks.
`oof_scores.csv` is `sample_id,label,oof_score`.
Read `validation_apparent` as optimistic and `validation_cv` as the number to report.

## 6. Quick checks when something looks wrong

| Symptom | Check |
| --- | --- |
| "table looks transposed" | the counts file must have genes in rows; see section 1 |
| A model gene missing at prediction | the file is filtered; scoring needs unfiltered raw counts |
| `ValueError: a model bundle needs frozen normalization` | add `--set dataset.normalization.mode=frozen` and a `reference_cohorts` |
| NaN scores | a gene-table function was used where a gene list was needed (`training.select_experiment_table` returns a table, `training.select_experiment_genes` returns names) |
| Run missing from the summary | it failed, or `results` has no `mean_auc` |
