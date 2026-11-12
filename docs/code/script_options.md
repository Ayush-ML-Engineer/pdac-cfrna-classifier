# Script options: every command-line option of every script

Two kinds of settings exist, and they are given differently:

| Kind | Given as | Defined in | Covered in |
| --- | --- | --- | --- |
| Config settings (data, task, model) | `--set section.key=value`, `--patch file.yaml` | `configs/dataset.yaml`, `configs/experiment.yaml`, `configs/model.yaml` | [../configuration.md](../configuration.md), [data_contracts.md](data_contracts.md) section 4 |
| Script options (what this one script does) | `--option value` | the script itself | this page |

All commands run from the repo root as `uv run python scripts/<name>.py ...`. In cmd.exe put the
whole command on one line and do not use single quotes (see the end of this page). To print the
options of any script yourself: `uv run python scripts/<name>.py --help`.

## 1. Scripts that take the config options

These accept `--set`, `--patch`, `--name` and `--config-dir`, added by
`run_config.add_config_arguments`:

| Option | Meaning | Default |
| --- | --- | --- |
| `--set section.key=value` | change one config setting; repeatable; the value is read as YAML | none |
| `--patch FILE` | a YAML file holding only the settings to change; repeatable, applied in order | none |
| `--name NAME` | name of the run folder | the experiment name, plus 8 hex characters if anything was changed |
| `--config-dir DIR` | folder holding the three default YAML files | `configs/` |

Scripts with these: `scripts/run_paper_replication.py`, `scripts/replicate_de_selection.py`,
`scripts/check_task_splits.py`, `scripts/train_model.py`, `scripts/select_threshold.py`.
`scripts/run_paper_replication.py`, `scripts/replicate_de_selection.py`,
`scripts/train_model.py` and `scripts/select_threshold.py` write a run folder;
`scripts/check_task_splits.py` only prints.

These five have no further options except `scripts/select_threshold.py`, below.

## 2. `scripts/select_threshold.py`

Chooses the decision threshold and reports the confusion matrix. Needs frozen normalization:
`--set dataset.normalization.mode=frozen --set dataset.normalization.reference_cohorts=[CEDAR]`.

| Option | Values | Default | Meaning |
| --- | --- | --- | --- |
| `--choose-on` | `validation`, `train` | `validation` | `validation`: pick the threshold on BCC scores of the model trained on CEDAR, and also report a cross-validated estimate. `train`: pick it on out-of-fold CEDAR scores, lock it, apply it once to BCC. |
| `--rule` | `fbeta`, `sensitivity`, `youden` | `fbeta` | how the best threshold is chosen |
| `--beta` | number | 2.0 | `fbeta` only; 2 means recall counts twice as much as precision |
| `--target-sensitivity` | number | 0.9 | `sensitivity` rule only: the highest threshold that still catches this share of PDAC |
| `--folds` | integer | 5 | CV folds for the estimate |
| `--repeats` | integer | 5 | `train` mode only: repeats of the CEDAR out-of-fold CV |
| `--gene-selection` | `in_fold`, `fixed` | `in_fold` | `train` mode only. `in_fold`: genes are selected again inside every fold (no leak, but unstable with about 20 PDAC; a fold may select no gene and is then skipped). `fixed`: use the genes chosen on all of CEDAR (stable, but the held-out labels influenced the genes, so the scores look too good). |

Which options matter when:

- Normal use: only the config `--set` lines; leave everything else at the defaults.
- To change what counts as "best": `--rule`, plus `--beta` or `--target-sensitivity`.
- Only with `--choose-on train`: `--repeats` and `--gene-selection`.
- `--choose-on train` also needs the experiment settings `task` and `gene_selection_task` to be
  equal; `--choose-on validation` accepts different ones, so the fixed gene list of
  `pdac_vs_all` can be used for the other tasks.

Outputs: see [data_contracts.md](data_contracts.md) section 5.

## 3. Scripts for scoring new data (no config options)

| Script | Options | Meaning |
| --- | --- | --- |
| `scripts/predict.py` | `--model` (required), `--counts` (required), `--out` (required) | score a raw counts file (genes x samples) with a saved bundle; writes a scores CSV |
| `scripts/evaluate_external.py` | `--model`, `--counts`, `--out` (all required), `--threshold` (default 0.5) | the same, plus AUC, a bootstrap interval and a confusion table at `--threshold`; labels come from the sample names |

The model file is the `model.joblib` written by `scripts/train_model.py`.

## 4. Scripts for normalization and factors

| Script | Options | Meaning |
| --- | --- | --- |
| `scripts/normalize_counts.py` | `--counts`, `--factors`, `--out` (all required) | cf-normalization then TMM of a raw counts file (all samples together, the joint mode) |
| `scripts/fit_reference_factors.py` | `--reference` (required), `--out` (required), `--counts`, `--out-updated`, `--seed` (default 0) | learn the intrinsic and extrinsic gene factors with NMF from a reference counts file; with `--counts` and `--out-updated` also extend the table to a patient dataset |

## 5. Scripts for reports and docs

| Script | Options | Default | Meaning |
| --- | --- | --- | --- |
| `scripts/compare_genes.py` | `--runs` | the `runs/` folder | folder holding the run folders |
| | `--run` | all runs | only this run folder; repeatable |
| | `--paper-genes` | `reference/paper_genes.yaml` | the paper's gene list |
| | `--summary` | `reports/gene_overlap.md` | where to write the page comparing all runs |
| | `--no-summary` | off | skip that page |
| `scripts/summarize_runs.py` | `--runs` | the `runs/` folder | folder holding the run folders |
| | `--out` | none | also write the table to this file |
| `scripts/make_api_reference.py` | `--check` | off | do not write; fail if `docs/code/api_reference.md` is out of date |

## 6. Typing options in cmd.exe and PowerShell

- cmd.exe: one line per command. Quote only with double quotes, and only when the value contains
  spaces. `[CEDAR]` needs no quotes.
- PowerShell: a backtick at the end of a line continues the command; single quotes work.
- Single quotes in cmd.exe are not removed, so the setting name arrives with a quote in it and the
  run stops with "must start with one of ['dataset', 'experiment', 'model']".
