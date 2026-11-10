# Configuration

Three files hold the defaults. They describe the **paper replication** (raw counts, joint
normalization, PDAC vs all, trained on CEDAR, tested on BCC):

| File | Holds | Python class |
|---|---|---|
| `configs/dataset.yaml` | data files, normalization, label column, class map, cohort split | `DatasetConfig` |
| `configs/experiment.yaml` | task, gene-selection cutoffs, SMOTE, seeds, run name | `ExperimentConfig` |
| `configs/model.yaml` | random-forest settings | `ModelConfig` |

A run changes only what it needs. Everything is validated together into one `RunConfig`; unknown
keys, wrong types and impossible combinations stop the run before any run folder is created.

## Changing settings

Bash and PowerShell use `\` or a backtick for line continuation; cmd.exe uses `^`. Use double
quotes around any value that contains `[` `]`.

```bash
# one change at a time; the value is read as YAML (null, true, 3, 0.01, [A, B] all work)
uv run python scripts/run_paper_replication.py --set experiment.task=pdac_vs_benign

# several changes, and a name for the run
uv run python scripts/run_paper_replication.py --name frozen_pdac_vs_all \
    --set dataset.normalization.mode=frozen \
    --set "dataset.normalization.reference_cohorts=[CEDAR]"

# many changes you want to keep: a small YAML that holds only the differences
uv run python scripts/run_paper_replication.py --patch my_changes.yaml
```

Rules:

- Keys start with the section: `dataset.`, `experiment.` or `model.`.
- `null` switches an optional block off (`dataset.normalization=null`). Settings that are not needed
  stay at their default or `null`; nothing has to be deleted from the file.
- Patch files use the same structure as the defaults but list only what changes:

  ```yaml
  dataset:
    normalization:
      mode: frozen
      reference_cohorts: [CEDAR]
  experiment:
    task: pdac_vs_benign
  ```

- Order of application: defaults, then every `--patch` in order, then every `--set` in order.
- The run name is `--name`, or `--set experiment.name=...`, or (when something changed) the default
  name plus 8 characters derived from the settings, so the same settings always get the same name.
- `${PDAC_DATA_ROOT}` in a path is replaced by the environment variable when the file is read.
  Put it in `.env` (copy `.env.example`).

## Every run saves its settings

`runs/<timestamp>_<name>/resolved_config.yaml` is the complete configuration of that run. Repeat
the run exactly with `--patch runs/<folder>/resolved_config.yaml`.

## Settings reference

### `dataset.yaml`

| Key | Meaning |
|---|---|
| `matrix.path`, `matrix.orientation` | the counts file; `genes_by_samples` or `samples_by_genes` |
| `normalization` | block that normalizes raw counts on load; `null` if the matrix is already normalized |
| `normalization.factors` | per-gene intrinsic / extrinsic factor CSV |
| `normalization.mode` | `joint` (paper) or `frozen` (constants fitted on `reference_cohorts`, applied per sample) |
| `normalization.reference_cohorts` | only with `frozen`; `null` means the split's training cohorts |
| `metadata.*` | metadata file and the columns for sample id, diagnosis and cohort |
| `class_map`, `unmapped` | diagnosis to 0/1; what to do with unlisted diagnoses (`error` or `drop`). The experiment's task replaces the map |
| `split.train_cohorts`, `split.validation_cohorts` | which cohort values train and validate; an empty validation list means no held-out cohort |

### `experiment.yaml`

| Key | Meaning |
|---|---|
| `task` | `pdac_vs_all`, `pdac_vs_non_cancer` or `pdac_vs_benign` |
| `gene_selection_task` | the task whose labels pick the genes (default `pdac_vs_all`, as in the paper) |
| `variance_cutoff`, `alpha`, `log_variance` | selection settings: variance threshold, FDR level, whether to take the log before the variance |
| `n_synthetic_pdac` | synthetic PDAC samples SMOTE adds; `null` balances the classes exactly |
| `seed`, `n_seeds` | first seed, and how many seeds to run (`seed`, `seed+1`, ...) |

### `model.yaml`

`n_estimators`, `max_features`, `min_samples_leaf`, `bootstrap` for the random forest.

## Recipes

These are checked by `tests/test_run_config.py`, so they cannot silently go stale.

| Goal | Changes |
|---|---|
| The authors' already-normalized matrix | `dataset.matrix.path=${PDAC_DATA_ROOT}/normalized_pdac_genecount.csv` and `dataset.normalization=null` |
| Frozen normalization fitted on CEDAR | `dataset.normalization.mode=frozen` and `dataset.normalization.reference_cohorts=[CEDAR]` |
| Our own NMF factors | `dataset.normalization.factors=${PDAC_DATA_ROOT}/updated_factors.csv` |
| SMOTE balances the classes exactly | `experiment.n_synthetic_pdac=null` |
| Another task | `experiment.task=pdac_vs_benign` (or `pdac_vs_non_cancer`) |
| Normalizer fitted on both cohorts, genes and model from CEDAR only | frozen mode with `reference_cohorts=[CEDAR, BCC]` (the split stays CEDAR/BCC) |
| Final model on every sample (`train_model.py` only) | frozen mode with `reference_cohorts=[CEDAR, BCC]`, `dataset.split.train_cohorts=[CEDAR, BCC]`, `dataset.split.validation_cohorts=[]` |

The last recipe has no held-out data, so it reports no AUC. It also selects many more genes (872 in
our run, against 29 or 36), because 248 samples give far more statistical power and the two
cohorts differ from each other. Treat it as unvalidated until the gene selection is reviewed.
