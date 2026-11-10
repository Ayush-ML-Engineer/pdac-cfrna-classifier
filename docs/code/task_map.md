# Task map: what to run for what

Read this first. Each row says which command does a job, which code it goes through, what it needs
and what it produces. For the step-by-step trace of a command see
[execution_flows.md](execution_flows.md); for every function see
[api_reference.md](api_reference.md) (generated).

All commands run from the repo root with `uv run python ...`. Settings come from `configs/*.yaml`
and are changed per run with `--set section.key=value` or `--patch file.yaml`
([../configuration.md](../configuration.md)).

## Tasks

| I want to... | Run | Main code it uses | Output |
| --- | --- | --- | --- |
| Replicate the paper (train CEDAR, test BCC, AUC over 10 seeds) | `scripts/run_paper_replication.py` | `training.select_experiment_table`, `models.score_task`, `evaluation.auc` | run folder: `de_genes.csv`, `test_scores.csv`, metrics in `run_metadata.json` |
| Run it with frozen normalization (the honest validation) | `scripts/run_paper_replication.py --set dataset.normalization.mode=frozen --set 'dataset.normalization.reference_cohorts=[CEDAR]'` | as above plus `normalization.fit_normalizer`, `normalization.apply_normalizer` | as above |
| Try another task or setting | `scripts/run_paper_replication.py --set experiment.task=pdac_vs_benign` (any `--set`) | as above | as above, run named from the changes |
| See only the gene selection (29 / 36 genes) | `scripts/replicate_de_selection.py` | `selection.select_de_genes` | `de_genes.csv` |
| Check that the train/test splits match the lab sheet | `scripts/check_task_splits.py` | `training.select_experiment_genes`, `data.Dataset` masks | printout, no run folder |
| Train a deployable model | `scripts/train_model.py` (needs frozen mode) | `training.train_bundle`, `bundle.save_bundle` | run folder: `model.joblib`, `model.json` |
| Train the final model on CEDAR + BCC | `scripts/train_model.py` with the `model_final` settings in its docstring | same | same, no validation AUC |
| Score new raw counts (no labels) | `scripts/predict.py --model M.joblib --counts RAW.csv --out SCORES.csv` | `predict.predict_scores`, `normalization.apply_normalizer` | scores CSV |
| Score a labelled external raw counts file and check it | `scripts/evaluate_external.py --model M.joblib --counts RAW.csv --out S.csv` | `predict.predict_scores`, `evaluation.bootstrap_auc_ci` | scores CSV, AUC and confusion table in the console |
| Choose the decision threshold, get the confusion matrix and PR curve | `scripts/select_threshold.py` (frozen settings, see its docstring) | `thresholds.choose_threshold`, `thresholds.cv_threshold_performance`, `threshold_plots` | run folder: `threshold.json`, `threshold_table.csv`, `validation_scores.csv`, `pr_curves.png`, `confusion_matrices.png` |
| Compare selected genes with the paper's genes | `scripts/compare_genes.py` | `gene_comparison.read_run_genes`, `gene_comparison.compare` | `gene_comparison.md` in each run folder, `reports/gene_overlap.md` |
| Put all runs in one results table | `scripts/summarize_runs.py` | `reporting.runs_table` | markdown table (console, optional file) |
| Normalize a raw count table once (joint, as in the paper) | `scripts/normalize_counts.py` | `normalization.normalize_counts` | normalized CSV |
| Learn the gene factors with NMF | `scripts/fit_reference_factors.py` | `nmf.fit_reference_factors`, `nmf.adjust_factors` | factor CSV(s) |
| Regenerate the API reference | `scripts/make_api_reference.py` | `ast` only | `docs/code/api_reference.md` |

## Which input files a run needs

The data files live outside the repo, in the folder named by `PDAC_DATA_ROOT` (in `.env`).
`configs/dataset.yaml` names them.

| File | Needed by | What it is |
| --- | --- | --- |
| `pdac_genecount.csv` | every run on raw counts | raw gene counts, genes in rows, samples in columns |
| `pdac_meta.csv` | every run | per-sample diagnosis and cohort (CEDAR / BCC) |
| `updated_intrinsic.csv` | joint and frozen normalization (default) | the paper's intrinsic / extrinsic gene factors |
| `updated_factors.csv` | runs with our own factors | factors we fitted ourselves with NMF |
| `normalized_pdac_genecount.csv` | runs on the paper's already normalized matrix | the shipped normalized matrix (`dataset.normalization=null`) |
| `reference_data_counts.csv` | `scripts/fit_reference_factors.py` | healthy reference counts used to fit the factors |
| `reference/paper_genes.yaml` (in the repo) | `scripts/compare_genes.py` | the genes the paper names |

## Which settings file controls what

| File | Controls | Read by |
| --- | --- | --- |
| `configs/dataset.yaml` | data files, normalization block (mode, factors, reference cohorts), class map, train / validation cohorts | `config.DatasetConfig` through `run_config.load_run_config` |
| `configs/experiment.yaml` | experiment name, task, variance cutoff, alpha, seed, number of seeds, SMOTE synthetic samples | `experiment.ExperimentConfig` |
| `configs/model.yaml` | random forest settings | `model_config.ModelConfig` |
| `.env` | `PDAC_DATA_ROOT` (data folder) | `config.load_env`, `config.resolve_path` |
