import os
from pathlib import Path

import numpy as np
import pytest

from pdac_cfrna.config import load_env
from pdac_cfrna.evaluation import auc
from pdac_cfrna.models import score_task
from pdac_cfrna.run_config import load_run_config
from pdac_cfrna.training import load_for_task, select_experiment_table

load_env()

ROOT = Path(os.environ.get("PDAC_DATA_ROOT", "."))
needs_data = pytest.mark.skipif(
    not os.environ.get("PDAC_DATA_ROOT"), reason="needs the data folder"
)

PROCESSED_ROOT = Path(os.environ.get("PROCESSED_DATA_PATH", "."))
needs_data = pytest.mark.skipif(
    not os.environ.get("PROCESSED_DATA_PATH"), reason="needs the data folder"
)

OWN_FACTORS = [
    f"dataset.normalization.factors={(PROCESSED_ROOT / 'updated_factors.csv').as_posix()}"
]
BALANCED = ["experiment.n_synthetic_pdac=null"]
NO_SEX_GENES = ["experiment.exclude_genes=[USP9Y, UTY, USP9X]"]


def _run_experiment(set_args: list[str], task: str, n_seeds: int = 3) -> tuple[list[str], float]:
    """Gene selection + classifier for one configuration (a shortened replication run)."""
    cfg = load_run_config([f"experiment.task={task}", *set_args])
    exp = cfg.experiment
    table = select_experiment_table(cfg)
    genes = list(table.index[table["significant"]])

    ds = load_for_task(cfg.dataset, exp.task)
    X = ds.X[genes]
    X_train, y_train = X[ds.train_mask], ds.y[ds.train_mask]
    X_test, y_test = X[ds.validation_mask], ds.y[ds.validation_mask]
    aucs = [
        auc(y_test, score_task(X_train, y_train, X_test, seed, cfg.model, exp.n_synthetic_pdac))
        for seed in range(exp.seed, exp.seed + n_seeds)
    ]
    return genes, float(np.mean(aucs))


REFERENCE_AUC = {"pdac_vs_benign": 0.927, "pdac_vs_non_cancer": 0.932, "pdac_vs_all": 0.891}
BALANCED_AUC = {"pdac_vs_benign": 0.922, "pdac_vs_non_cancer": 0.928, "pdac_vs_all": 0.893}
NO_SEX_GENES_AUC = {"pdac_vs_benign": 0.887, "pdac_vs_non_cancer": 0.906, "pdac_vs_all": 0.855}


@pytest.mark.slow
@needs_data
@pytest.mark.parametrize("task", sorted(REFERENCE_AUC))
def test_the_default_config_reproduces_the_paper_replication(task):
    genes, mean_auc = _run_experiment([], task)
    assert len(genes) == 29
    assert mean_auc == pytest.approx(REFERENCE_AUC[task], abs=0.02)


@pytest.mark.slow
@needs_data
@pytest.mark.skipif(
    not (PROCESSED_ROOT / "updated_factors.csv").exists(),
    reason="run scripts/fit_reference_factors.py with --out-updated first",
)
@pytest.mark.parametrize("task", sorted(REFERENCE_AUC))
def test_our_own_factors_reproduce_it_too(task):
    genes, mean_auc = _run_experiment(OWN_FACTORS, task)
    assert len(genes) == 29
    assert mean_auc == pytest.approx(REFERENCE_AUC[task], abs=0.02)


@pytest.mark.slow
@needs_data
@pytest.mark.parametrize("task", sorted(BALANCED_AUC))
def test_balancing_smote_exactly_changes_the_result_very_little(task):
    genes, mean_auc = _run_experiment(BALANCED, task)
    assert len(genes) == 29
    assert mean_auc == pytest.approx(BALANCED_AUC[task], abs=0.02)
