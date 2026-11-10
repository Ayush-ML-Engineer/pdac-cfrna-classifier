# Execution flows: what happens when you run each script

How to read this page: every box is a real function (`module.function`). The first four sections
are steps that many scripts share; each script section then lists its own steps and says which
shared steps it calls. Signatures and one-line descriptions of every function are in
[api_reference.md](api_reference.md). Where the code does something that looks odd or wasteful it
is marked **Note**.

## Shared step A: from the command line to one validated `RunConfig`

Used by every script that calls `run_config.add_config_arguments`: `run_paper_replication.py`,
`train_model.py`, `select_threshold.py`, `replicate_de_selection.py`, `check_task_splits.py`.

```mermaid
flowchart TB
    A["script: parser.parse_args()"] --> B["run_config.config_from_args"]
    B --> C["run_config.load_run_config"]
    C --> D["config.load_env<br/>reads .env, sets PDAC_DATA_ROOT"]
    D --> E["run_config.read_yaml_mapping<br/>configs/dataset.yaml, experiment.yaml, model.yaml"]
    E --> F["--patch files: run_config.deep_merge<br/>(applied in the order given)"]
    F --> G["--set arguments: run_config.parse_set<br/>then run_config._set_path"]
    G --> H{"--name given?"}
    H -->|no, and something was changed| I["run_config._auto_name<br/>base name + 8 hex digits of the settings"]
    H -->|yes| J["use it"]
    I --> K["RunConfig(**tree) validates all three sections<br/>(config.DatasetConfig, experiment.ExperimentConfig, model_config.ModelConfig)"]
    J --> K
```

What each step does, in order:

1. `config.load_env` loads `.env` from the repo root. It never overrides a real environment
   variable. This is where `PDAC_DATA_ROOT` comes from.
2. `run_config.read_yaml_mapping` reads the three default YAMLs into one dict, one key per section
   (`dataset`, `experiment`, `model`).
3. Each `--patch FILE` is merged on top with `run_config.deep_merge` (mappings merge key by key,
   everything else replaces). A `resolved_config.yaml` from an old run works as a patch.
4. Each `--set a.b.c=value` goes through `run_config.parse_set` (the value is read as YAML, so
   `null`, `true`, `3`, `[CEDAR]` work; one pair of wrapping quotes is stripped for cmd.exe) and
   `run_config._set_path` writes it into the dict. An unknown path fails here.
5. If anything was changed and no name was given, `run_config._auto_name` makes
   `experiment.name = <base>-<hash of the settings>`: the same settings always get the same name.
6. `RunConfig(**tree)` runs pydantic validation. The models forbid unknown keys, so a typo fails
   before any data is loaded.
7. The `RunConfig.overrides` list keeps the text of every patch and set, which later shows up in
   `run_metadata.json` and in the results table.

## Shared step B: the run folder, `runs.tracked_run`

Every script that produces results wraps its work in `with tracked_run(cfg) as run:`.

1. `runs.run_lock` takes a file lock (`runs/.run.lock`), so two runs cannot start at the same
   moment.
2. `runs.create_run_dir` makes `runs/<timestamp>_<name>/`.
3. `run_config.dump_resolved` writes `resolved_config.yaml` (usable as a `--patch` to repeat the run).
4. `runs.git_info` records commit, branch and dirty state. If the working tree is dirty the diff
   is written to `code_changes.diff`.
5. `run_metadata.json` is written with `"status": "running"`, the overrides, the full config and
   package versions.
6. The script body runs. It puts results into `run.extra` (a dict).
7. On exit, the metadata is rewritten: status `success` or `failed` (with the error text), the
   finish time, the duration, and `results = run.extra`.

Because `run_metadata.json` is rewritten at the end even after an error, a crashed run is still
visible (status `failed`). `reporting.runs_table` and `gene_comparison.read_run_genes` both ignore
runs that are not `success`.

## Shared step C: loading a dataset, `training.load_for_task`

```mermaid
flowchart TB
    A["training.load_for_task(dataset_cfg, task_name)"] --> B["experiment.TASKS[task_name]<br/>class map + unmapped policy"]
    B --> C["data.apply_task<br/>copy of DatasetConfig with the task's class map, re-validated"]
    C --> D["data.load_dataset"]
    D --> D1["1. check all files exist"]
    D1 --> D2["2. read counts, genes x samples"]
    D2 --> D3["3. read metadata, index by sample id"]
    D3 --> D4{"4. normalization block?"}
    D4 -->|none| D5["use the matrix as it is<br/>(already normalized file)"]
    D4 -->|joint| D6["normalization.normalize_counts<br/>all samples together"]
    D4 -->|frozen| D7["pick reference_cohorts samples<br/>normalization.fit_normalizer on them<br/>normalization.apply_normalizer on every sample"]
    D5 --> D8["5. transpose: samples x genes"]
    D6 --> D8
    D7 --> D8
    D8 --> D9["6. every sample must have metadata"]
    D9 --> D10["7. map diagnoses to 0 / 1 with the class map,<br/>drop unmapped samples or raise"]
    D10 --> E["data.Dataset: X, y, meta, config, normalizer"]
```

Points to remember:

* The normalization (step 4) sees **every** sample in the file, and happens **before** samples
  are dropped by the class map. So a task that drops samples still normalizes with them.
* In joint mode the constants come from all samples at once and are thrown away (no
  `normalizer` on the `Dataset`). In frozen mode the constants are fitted on the reference cohorts
  only and kept in `Dataset.normalizer`; this is what a model bundle needs.
* `Dataset.train_mask` and `Dataset.validation_mask` are boolean Series built from
  `dataset.split` and the cohort column of the metadata. Rows are never split into separate
  tables: the code always selects with `X[ds.train_mask]`.
* **Note:** every call to `training.load_for_task` re-reads the files and re-fits the normalization.
  `run_paper_replication.py`, `train_model.py` and `select_threshold.py` each call it twice (once
  inside the gene selection, once for the experiment task), so the normalization runs twice per
  run. It is correct but slow; caching the loaded dataset within a run is on the robustness list.

## Shared step D: gene selection and the classifier

**Gene selection** (`training.select_experiment_table`, then `training.select_experiment_genes`):

1. `training.load_for_task(cfg.dataset, exp.gene_selection_task)` (default `pdac_vs_all`).
2. Take the training rows only: `ds.X[ds.train_mask]`, `ds.y[ds.train_mask]`.
3. `selection.select_de_genes`: `selection.filter_variance` keeps genes with sample variance above
   the cutoff; `selection.wilcoxon_per_gene` gives a p-value, an AUC and a direction per gene;
   `selection.adjust_pvalues` applies Benjamini-Hochberg; a gene is `significant` when
   `q_value < alpha`. The table is sorted by q-value.
4. `training.select_experiment_genes` returns the names of the significant genes.

**Classifier** (`models.build_paper_pipeline`, used by `models.score_task` and
`training.train_bundle`): an imblearn `Pipeline` with three steps, `log` (`np.log1p`), `smote`
(adds `n_synthetic_pdac` synthetic PDAC samples, or balances the classes when it is `null`; only
active while fitting) and `rf` (random forest from `configs/model.yaml`). `models.score_task`
fits it and returns `predict_proba` for class 1 as a Series named `pdac_score`.

---

## `scripts/run_paper_replication.py`: replicate the paper

```mermaid
flowchart TB
    A["Shared step A: RunConfig"] --> B{"validation cohort configured?"}
    B -->|no| X["parser.error"]
    B -->|yes| C["Shared step B: tracked_run"]
    C --> D["training.select_experiment_table<br/>(Shared step D, training rows only)"]
    D --> E["write de_genes.csv"]
    E --> F["training.load_for_task(exp.task)<br/>(Shared step C, second load)"]
    F --> G["X[genes], split by train_mask and validation_mask"]
    G --> H["for each seed in seed .. seed + n_seeds - 1:<br/>models.score_task, evaluation.auc"]
    H --> I["test_scores.csv: one column per seed + mean_score"]
    I --> J["mean AUC, std (ddof=1), seed-averaged AUC,<br/>evaluation.bootstrap_auc_ci"]
    J --> K["run.extra, console summary"]
```

* Genes are chosen once, on the training cohort, with the gene-selection task's labels, and reused
  for every seed.
* The seeds change SMOTE and the forest only; the genes and data do not change.
* Two uncertainties are reported separately: spread over seeds (`std_auc`) and the small test set
  (bootstrap CI of the seed-averaged score).
* Needs: `configs/*.yaml`, counts, metadata and the factor file. Produces: `de_genes.csv`,
  `test_scores.csv`, `run_metadata.json` (results block).

## `scripts/train_model.py`: train a deployable bundle

```mermaid
flowchart TB
    A["Shared step A: RunConfig"] --> B{"normalization mode is frozen?"}
    B -->|no| X["parser.error"]
    B -->|yes| C["Shared step B: tracked_run"]
    C --> D["training.select_experiment_genes"]
    D --> E["training.load_for_task(exp.task)"]
    E --> F["training.train_bundle"]
    F --> F1["checks: ds.normalizer exists, genes are in ds.X"]
    F1 --> F2["fit models.build_paper_pipeline on the training rows"]
    F2 --> F3["metadata: config, genes, sizes, library versions, git"]
    F3 --> G{"validation cohort?"}
    G -->|yes| H["predict_proba on it, sklearn roc_auc_score,<br/>stored in bundle.metadata"]
    G -->|no| I["no reference AUC"]
    H --> J["bundle.save_bundle: model.joblib + model.json"]
    I --> J
```

* A bundle (`bundle.ModelBundle`) holds the fitted pipeline, the gene list, the frozen
  `normalization.NormalizationConstants` and a metadata dict. That is everything `scripts/predict.py`
  needs, which is why joint mode cannot produce one.
* `bundle.save_bundle` writes the joblib file and a readable JSON file next to it.

## `scripts/predict.py` and `scripts/evaluate_external.py`: score new raw counts

```mermaid
flowchart TB
    A["bundle.load_bundle<br/>checks scikit-learn / imbalanced-learn major.minor"] --> B["read raw counts CSV, genes x samples"]
    B --> C["predict.predict_scores"]
    C --> C1["table transposed? (gene names in columns) -> error"]
    C1 --> C2["normalization.apply_normalizer<br/>each sample on its own with the saved constants"]
    C2 --> C3["all model genes present? else error"]
    C3 --> C4["X = normalized.loc[genes].T, predict_proba class 1"]
    C4 --> D{"which script?"}
    D -->|predict.py| E["bundle.scores_to_frame, write CSV"]
    D -->|evaluate_external.py| F["labels_from_names (suffix PDAC = 1, Controles / Control = 0)<br/>evaluation.auc, bootstrap CI, confusion table at --threshold"]
```

* Every sample is normalized on its own, so one sample is enough and the batch makes no
  difference.
* The counts must be **raw counts** with the genes the factor table needs. Already normalized or
  log-scale tables give meaningless scores.

## `scripts/select_threshold.py`: decision threshold and confusion matrix

```mermaid
flowchart TB
    A["Shared step A, frozen check, Shared step B"] --> B["training.load_for_task, masks"]
    B --> C["training.select_experiment_genes (36 genes in the validation setup)"]
    C --> D["validation scores: mean over seeds of<br/>models.score_task(train rows -> validation rows)"]
    D --> E{"--choose-on"}
    E -->|"validation (default)"| F["thresholds.threshold_table on validation scores<br/>thresholds.choose_threshold (rule: fbeta, sensitivity, youden)"]
    F --> G["report counts at that threshold (optimistic)<br/>thresholds.cv_threshold_performance (honest estimate)"]
    E -->|train| H["thresholds.oof_scores on the training cohort<br/>(per fold: select_de_genes + pipeline, or fixed genes)"]
    H --> I["threshold_table + choose_threshold on out-of-fold scores"]
    I --> J["lock the threshold, report on the validation cohort"]
    G --> K["threshold_plots.plot_pr_curves, plot_confusion_matrices"]
    J --> K
    K --> L["threshold.json, threshold_table.csv, validation_scores.csv, PNGs"]
```

* `thresholds.threshold_table` evaluates every threshold of the grid 0.01 to 0.99;
  `thresholds.choose_threshold` picks one by the rule (for ties it takes the middle of the plateau).
* In the default mode the threshold is tuned on the same validation samples it is reported on.
  `thresholds.cv_threshold_performance` repeats "choose on 4/5, test on 1/5" to show the honest
  performance. It is the default because `--choose-on train` fails with 20 PDAC samples in CEDAR:
  with genes re-selected in every fold the out-of-fold AUC is 0.60 (threshold 0.05, unusable);
  with the 36 genes fixed, the held-out labels leak into the gene choice, the out-of-fold AUC is
  0.97 (against 0.895 on BCC) and the threshold 0.35 does not transfer (BCC sensitivity 0.67).
* `thresholds.oof_scores` skips a fold that selects no gene (it records 0 for that fold).

## `scripts/compare_genes.py`: selected genes against the paper

1. `gene_comparison.load_paper_genes` reads `reference/paper_genes.yaml` (25 named genes,
   `reported_total` 29).
2. `gene_comparison.collect_runs` visits every folder in `runs/`; `gene_comparison.read_run_genes`
   returns a run's genes from `de_genes.csv` (the `significant` rows) or from `model.json`, and
   `None` for runs that failed or saved no genes (threshold runs are skipped this way).
3. For each run `gene_comparison.compare` splits the genes into matched, selected-but-not-named and
   named-but-not-selected; `gene_comparison.render_run_report` writes `gene_comparison.md` into the
   run folder.
4. `gene_comparison.latest_per_name` keeps the newest run of each experiment name and
   `gene_comparison.render_summary` writes `reports/gene_overlap.md` (genes stable across runs).

## `scripts/summarize_runs.py`

`reporting.runs_table` reads `run_metadata.json` of every run, keeps the newest successful run per
experiment that reports an AUC, and `reporting.to_markdown` prints the table.

## `scripts/replicate_de_selection.py` and `scripts/check_task_splits.py`

* `replicate_de_selection.py`: Shared steps A and B, then `training.load_for_task`, training rows,
  `selection.select_de_genes`, `de_genes.csv`, and a comparison with the paper's genes written
  inside the script (the `PAPER_UP`, `PAPER_DOWN`, `PAPER_OTHER` sets).
* `check_task_splits.py`: Shared step A only (no run folder). Selects the genes once, asserts there
  are 29 (so it only works with the default joint settings), then loads each task from
  `experiment.TASKS` and asserts the train / test sizes against the lab sheet (the `EXPECTED` dict).

## `scripts/normalize_counts.py` and `scripts/fit_reference_factors.py`

Standalone tools, not part of a run.

* `normalize_counts.py`: `normalization.load_factors` then `normalization.normalize_counts`
  (`normalization.cf_norm` followed by `normalization.tmm`), writes the normalized CSV.
* `fit_reference_factors.py`: `nmf.fit_reference_factors` learns intrinsic and extrinsic fractions
  per gene from healthy reference samples using the constrained `nmf.nmf`; with `--counts` it also
  calls `nmf.adjust_factors` to extend the factors to the genes of a patient dataset.
