import os
from pathlib import Path

import numpy as np
import pytest

from pdac_cfrna.config import DATASET_CONFIG_DIR, load_dataset_config, load_env
from pdac_cfrna.data import apply_task, load_dataset
from pdac_cfrna.evaluation import auc
from pdac_cfrna.experiment import (
    EXPERIMENT_CONFIG_DIR,
    TASKS,
    load_experiment_config,
    resolve_experiment_path,
)
from pdac_cfrna.model_config import load_model_config
from pdac_cfrna.models import score_task
from pdac_cfrna.selection import select_de_genes

load_env()
PROCESSED_ROOT = Path(os.environ.get("PROCESSED_DATA_PATH", "."))
needs_data = pytest.mark.skipif(
    not os.environ.get("PROCESSED_DATA_PATH"), reason="needs the data folder"
)


def _run_experiment(name: str, n_seeds: int = 3) -> tuple[list[str], float]:
    """Gene selection + classifier for one experiment file (a shortened replication run)."""
    exp = load_experiment_config(resolve_experiment_path(name))

    def dataset(task_name: str):
        task = TASKS[task_name]
        return load_dataset(
            apply_task(load_dataset_config(exp.dataset), task.class_map, task.unmapped)
        )

    ds_sel = dataset(exp.gene_selection_task)
    table = select_de_genes(
        ds_sel.X[ds_sel.train_mask],
        ds_sel.y[ds_sel.train_mask],
        exp.variance_cutoff,
        exp.alpha,
        exp.log_variance,
    )
    genes = list(table.index[table["significant"]])

    ds = dataset(exp.task)
    X = ds.X[genes]
    X_train, y_train = X[ds.train_mask], ds.y[ds.train_mask]
    X_test, y_test = X[ds.validation_mask], ds.y[ds.validation_mask]
    model = load_model_config(exp.model)
    aucs = [
        auc(y_test, score_task(X_train, y_train, X_test, seed, model, exp.n_synthetic_pdac))
        for seed in range(exp.seed, exp.seed + n_seeds)
    ]
    return genes, float(np.mean(aucs))


REFERENCE_AUC = {"pdac_vs_benign": 0.927, "pdac_vs_non_cancer": 0.932, "pdac_vs_all": 0.891}


def test_every_shipped_experiment_names_an_existing_dataset_config():
    for path in sorted(EXPERIMENT_CONFIG_DIR.glob("*.yaml")):
        exp = load_experiment_config(path)
        assert (
            DATASET_CONFIG_DIR / f"{exp.dataset}.yaml"
        ).is_file(), f"{path.name}: unknown dataset {exp.dataset!r}"


@pytest.mark.slow
@needs_data
@pytest.mark.parametrize("task", sorted(REFERENCE_AUC))
def test_raw_counts_with_the_authors_factors_reproduce_the_paper_replication(task):
    genes, mean_auc = _run_experiment(f"paper_replication_raw_{task}")
    assert len(genes) == 29
    assert mean_auc == pytest.approx(REFERENCE_AUC[task], abs=0.02)


@pytest.mark.slow
@needs_data
@pytest.mark.skipif(
    not (PROCESSED_ROOT / "our_updated_factors.csv").exists(),
    reason="run scripts/fit_reference_factors.py with --out-updated first",
)
@pytest.mark.parametrize("task", sorted(REFERENCE_AUC))
def test_raw_counts_with_our_own_factors_reproduce_it_too(task):
    genes, mean_auc = _run_experiment(f"paper_replication_ownfactors_{task}")
    assert len(genes) == 29
    assert mean_auc == pytest.approx(REFERENCE_AUC[task], abs=0.02)
