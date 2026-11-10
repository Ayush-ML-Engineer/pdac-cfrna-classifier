"""Gene selection and model training for one resolved run configuration."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from pdac_cfrna.bundle import BUNDLE_FORMAT_VERSION, ModelBundle, library_versions
from pdac_cfrna.config import REPO_ROOT, DatasetConfig
from pdac_cfrna.data import Dataset, apply_task, load_dataset
from pdac_cfrna.experiment import TASKS
from pdac_cfrna.models import build_paper_pipeline
from pdac_cfrna.run_config import RunConfig
from pdac_cfrna.runs import git_info
from pdac_cfrna.selection import select_de_genes


def load_for_task(dataset_cfg: DatasetConfig, task_name: str) -> Dataset:
    """Load a dataset with one task's labels (and sample selection) applied."""
    task = TASKS[task_name]
    return load_dataset(apply_task(dataset_cfg, task.class_map, task.unmapped))


def select_experiment_table(cfg: RunConfig) -> pd.DataFrame:
    """The gene-selection table (variance filter, Wilcoxon, BH) on the training rows only."""
    exp = cfg.experiment
    ds = load_for_task(cfg.dataset, exp.gene_selection_task)
    return select_de_genes(
        ds.X[ds.train_mask],
        ds.y[ds.train_mask],
        exp.variance_cutoff,
        exp.alpha,
        exp.log_variance,
    )


def select_experiment_genes(cfg: RunConfig) -> list[str]:
    """The genes of a run: the significant genes of the gene-selection table."""
    table = select_experiment_table(cfg)
    return list(table.index[table["significant"]])


def train_bundle(
    ds: Dataset,
    cfg: RunConfig,
    genes: list[str],
    extra_metadata: dict[str, Any] | None = None,
) -> ModelBundle:
    """Fit the paper's pipeline on the training rows of `ds` and package it with its normalizer.

    Raises:
        ValueError: the dataset does not use frozen normalization (there would be no constants to
            save), or some of `genes` are not columns of the dataset.
    """
    if ds.normalizer is None:
        raise ValueError(
            "a model bundle needs frozen normalization: set normalization.mode to 'frozen' "
            "in the dataset config"
        )
    unknown = [g for g in genes if g not in ds.X.columns]
    if unknown:
        raise ValueError(f"{len(unknown)} gene(s) are not in the dataset, e.g. {unknown[:5]}")

    X_train = ds.X.loc[ds.train_mask.to_numpy(), genes]
    y_train = ds.y[ds.train_mask]
    exp, model = cfg.experiment, cfg.model
    pipeline = build_paper_pipeline(int(y_train.sum()), exp.seed, model, exp.n_synthetic_pdac)
    pipeline.fit(X_train, y_train)

    git = git_info(REPO_ROOT)
    git.pop("diff", None)
    metadata: dict[str, Any] = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment": exp.name,
        "dataset": cfg.dataset.name,
        "task": exp.task,
        "genes": list(genes),
        "n_train": len(y_train),
        "n_train_pdac": int(y_train.sum()),
        "seed": exp.seed,
        "n_synthetic_pdac": exp.n_synthetic_pdac,
        "model": model.model_dump(),
        "overrides": list(cfg.overrides),
        "config": cfg.model_dump(mode="json"),
        "library_versions": library_versions(),
        "git": git,
        **(extra_metadata or {}),
    }
    return ModelBundle(
        pipeline=pipeline, genes=list(genes), normalizer=ds.normalizer, metadata=metadata
    )
