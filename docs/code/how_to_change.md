# How to change things

Recipes for the changes you are most likely to make. Each lists the exact files to touch, the
test to add and the check to run. After any change, finish with section 8.

Conventions that apply to every recipe: type hints on all functions, docstrings on public
functions, lines up to 100 characters, `ruff` and `mypy` clean. Settings live in YAML and are
validated by pydantic, never read with bare `dict` lookups.

## 1. Add a setting to a config

Example: a new experiment field `min_auc`.

1. Add the field to the right pydantic class with a default and a validation bound:
   `experiment.ExperimentConfig` (experiment), `config.DatasetConfig` or one of its parts
   (`config.MatrixSpec`, `config.MetadataSpec`, `config.NormalizationSpec`, `config.SplitSpec`), or
   `model_config.ModelConfig` (model).
2. Add the same key with its default to `configs/experiment.yaml` (or `dataset.yaml` / `model.yaml`)
   so it appears in `resolved_config.yaml`.
3. Use it where it matters (`cfg.experiment.min_auc`). Nothing else reads YAML directly.
4. Test: a case in `tests/test_experiment_fields.py` (experiment), `tests/test_model_config.py`
   (model) or `tests/test_run_config.py` (merging and `--set`). Check the default, a good value
   and a bad value that must raise.
5. Document the key in `docs/configuration.md` and in the table of
   [data_contracts.md](data_contracts.md) section 4.

Because every config class forbids unknown keys, a key present in the YAML but missing from the
class fails at load time. That is the intended safety net.

## 2. Add a task (a new set of classes)

Example: PDAC vs pancreatitis only.

1. Add an entry to `experiment.TASKS`: a `class_map` from the metadata label text to 0 or 1 and
   `unmapped` (`drop` to ignore other labels, `error` to demand that all labels are covered).
2. The task is then valid for the experiment settings `task` and `gene_selection_task` with no other
   change, because `experiment.ExperimentConfig` checks the name against `experiment.TASKS`.
3. Run it: `uv run python scripts/run_paper_replication.py --set experiment.task=<new_name>`.
4. Test: `tests/test_experiment.py` already loops over every key of `experiment.TASKS`, so the new
   task is checked automatically; add a case there only for task-specific behaviour (for example,
   which labels are dropped).
5. Add a row to the tasks table in [task_map.md](task_map.md).

Remember that the genes are chosen with `gene_selection_task`, which defaults to `pdac_vs_all`.
Changing only `task` keeps the paper's gene list; change both to select genes for the new task.

## 3. Change the classifier

Small changes (more trees, different `max_features`) are only settings:
`--set model.n_estimators=500`.

Changing the structure (another algorithm, another step in the pipeline):

1. Edit `models.build_paper_pipeline`. Keep the step names `log`, `smote`, `rf` unless you also
   update everything that looks them up by name.
2. If the new model needs settings, add them to `model_config.ModelConfig` and `configs/model.yaml`
   (recipe 1).
3. `predict.predict_scores` uses `pipeline.classes_` and `predict_proba`, so the new model must
   provide both.
4. Old bundles still load but were trained with the old pipeline. Retrain with
   `scripts/train_model.py` and bump `bundle.BUNDLE_FORMAT_VERSION` if the saved structure changed
   in a way old code cannot read.
5. Test: `tests/test_models.py` (and `tests/test_models_balance.py` for the SMOTE settings).

SMOTE must stay inside the pipeline. It only acts while fitting, so it never touches validation or
new samples. Do not resample before calling `models.score_task`.

## 4. Add a threshold rule

Example: "highest specificity subject to precision at least 0.5".

1. In `thresholds.py`, add the name to the `Rule` literal type and a branch in
   `thresholds.choose_threshold`. The table passed in has the columns listed in
   [data_contracts.md](data_contracts.md) section 5, so use those.
2. If the rule needs a number, add a parameter with a default to `thresholds.choose_threshold`
   and to `thresholds.cv_threshold_performance`, and pass it through in `scripts/select_threshold.py`
   (a new `--` option, and a mention in the script docstring).
3. Add the name to `--rule` choices in `scripts/select_threshold.py`.
4. Test: `tests/test_thresholds.py`. Use a small hand-made table whose best threshold you can
   compute by hand, plus a case where no threshold qualifies (it should raise `ValueError`).

Tie handling matters: `thresholds._middle_of_best` returns the middle of a plateau of equal
values. Reuse it rather than taking the first or last.

## 5. Change normalization

- A new normalization mode: extend the `Literal` in `config.NormalizationSpec`, handle it in
  `data.load_dataset` step 4, and add the validation rules to `config.NormalizationSpec`.
  Tests: `tests/test_data_normalization.py`.
- A change to the maths in `normalization.py`: the golden tests (`tests/test_normalization_golden.py`
  and `tests/test_raw_experiments_golden.py`) compare against stored numbers from the paper's
  method. If they fail after your change, the change altered the results. Only update the stored
  numbers if you can explain why, and write that in [decisions.md](decisions.md).
- Anything that is fitted on data (new constants) belongs in
  `normalization.NormalizationConstants`, fitted by `normalization.fit_normalizer`, and applied by
  `normalization.apply_normalizer`. If it is not in that dataclass it is not in the bundle, and new
  samples will be scored without it. Test: `tests/test_frozen_normalizer.py`.

## 6. Add a script

1. Copy the shape of `scripts/select_threshold.py`: parse arguments with
   `run_config.add_config_arguments`, build the settings with `run_config.config_from_args`, load
   data with `training.load_for_task`, and do the work inside `runs.tracked_run`.
2. Put the logic in a module in `src/pdac_cfrna/` and keep the script thin. Logic in a module can
   be tested; logic in a script usually is not.
3. Store numbers worth keeping in `run.extra` (they land in `run_metadata.json` under `results`)
   and write files into `run.dir`.
4. Add a row in [task_map.md](task_map.md) and a flow in [execution_flows.md](execution_flows.md).
5. Regenerate the reference: `uv run python scripts/make_api_reference.py`.

## 7. Add a module

New file `src/pdac_cfrna/<name>.py` with a module docstring. Regenerate the API reference
(section 8). If you name it in a hand-written page, write it as `module.name` in backticks, for
example `thresholds.choose_threshold`; the docs test checks that it exists.

## 8. Checks to run before committing

Run from the repo root, in cmd.exe or PowerShell:

```
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
uv run python scripts/make_api_reference.py
uv run python scripts/make_api_reference.py --check
```

What each one catches:

| Command | Catches |
| --- | --- |
| `ruff check` / `ruff format --check` | style, unused imports, lines over 100 characters |
| `mypy src` | wrong types, wrong return values |
| `pytest` | behaviour changes, including the golden numbers |
| `make_api_reference.py` | rewrites `docs/code/api_reference.md` from the code |
| `make_api_reference.py --check` | fails if the reference is stale (the same check as `tests/test_docs.py`) |

`tests/test_docs.py` also fails when a hand-written page in `docs/code/` names a `module.name`
or a `.py` / `.yaml` path that does not exist. If you rename or delete a function, the failing
test tells you which page still mentions it.

## 9. Things to keep true

- `Dataset.X` is samples x genes; `normalization` works on genes x samples (see
  [data_contracts.md](data_contracts.md)).
- Gene selection sees training rows only.
- Anything fitted (SMOTE, constants, thresholds) is fitted on the training side only. When a
  threshold is chosen on the validation cohort, report the cross-validated estimate, not the
  tuned number.
- Never present anything that is not implemented as implemented. If a feature is dropped, remove
  its code, its config key and its doc lines together.
