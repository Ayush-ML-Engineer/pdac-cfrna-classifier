"""Training a model bundle from an experiment file."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pdac_cfrna.bundle import BUNDLE_FORMAT_VERSION, ModelBundle, library_versions
from pdac_cfrna.config import REPO_ROOT, load_dataset_config
from pdac_cfrna.data import Dataset, apply_task, load_dataset
from pdac_cfrna.experiment import TASKS, ExperimentConfig
from pdac_cfrna.model_config import load_model_config
from pdac_cfrna.models import build_paper_pipeline
from pdac_cfrna.runs import git_info
from pdac_cfrna.selection import select_de_genes


def load_for_task(dataset_name: str, task_name: str) -> Dataset:
    """Load a dataset config with one task's labels (and sample selection) applied."""
    task = TASKS[task_name]
    cfg = apply_task(load_dataset_config(dataset_name), task.class_map, task.unmapped)
    return load_dataset(cfg)


def select_experiment_genes(exp: ExperimentConfig) -> list[str]:
    """The genes of an experiment: variance filter, Wilcoxon, BH on the training rows only."""
    ds = load_for_task(exp.dataset, exp.gene_selection_task)
    table = select_de_genes(
        ds.X[ds.train_mask],
        ds.y[ds.train_mask],
        exp.variance_cutoff,
        exp.alpha,
        exp.log_variance,
    )
    return list(table.index[table["significant"]])


def train_bundle(
    ds: Dataset,
    exp: ExperimentConfig,
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
    model = load_model_config(exp.model)
    pipeline = build_paper_pipeline(int(y_train.sum()), exp.seed, model, exp.n_synthetic_pdac)
    pipeline.fit(X_train, y_train)

    git = git_info(REPO_ROOT)
    git.pop("diff", None)
    metadata: dict[str, Any] = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment": exp.name,
        "dataset": exp.dataset,
        "task": exp.task,
        "genes": list(genes),
        "n_train": len(y_train),
        "n_train_pdac": int(y_train.sum()),
        "seed": exp.seed,
        "n_synthetic_pdac": exp.n_synthetic_pdac,
        "model": model.model_dump(),
        "library_versions": library_versions(),
        "git": git,
        **(extra_metadata or {}),
    }
    return ModelBundle(
        pipeline=pipeline, genes=list(genes), normalizer=ds.normalizer, metadata=metadata
    )
