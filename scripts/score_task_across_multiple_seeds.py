"""Train the paper pipeline on CEDAR and score BCC for several seeds (one task, many seeds).

Run from the repo root:
    uv run python scripts/score_task_across_multiple_seeds.py --task pdac_vs_benign
Temporary exploration script: Section 7 replaces it with the full experiment script.
"""

from __future__ import annotations

import argparse

import numpy as np

from pdac_cfrna.config import load_dataset_config
from pdac_cfrna.data import Dataset, apply_task, load_dataset
from pdac_cfrna.evaluation import auc
from pdac_cfrna.experiment import TASKS, load_experiment_config, resolve_experiment_path
from pdac_cfrna.model_config import load_model_config
from pdac_cfrna.models import score_task
from pdac_cfrna.selection import select_de_genes

SELECTION_EXPERIMENT = "de_selection_pdac_vs_all"  # settings for the gene selection
MODEL = "random_forest_paper"
N_SYNTHETIC = 100
N_SEEDS = 10


def load_for_task(dataset_name: str, task_name: str) -> Dataset:
    task = TASKS[task_name]
    cfg = apply_task(load_dataset_config(dataset_name), task.class_map, task.unmapped)
    return load_dataset(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="pdac_vs_all", choices=sorted(TASKS))
    args = parser.parse_args()

    exp = load_experiment_config(resolve_experiment_path(SELECTION_EXPERIMENT))

    # 1. Genes: chosen once, with PDAC vs all, on CEDAR rows only.
    ds_sel = load_for_task(exp.dataset, exp.gene_selection_task)
    table = select_de_genes(
        ds_sel.X[ds_sel.train_mask],
        ds_sel.y[ds_sel.train_mask],
        exp.variance_cutoff,
        exp.alpha,
        exp.log_variance,
    )
    genes = list(table.index[table["significant"]])

    # 2. Train on CEDAR, test on BCC for the chosen task, using the same genes.
    ds = load_for_task(exp.dataset, args.task)
    X = ds.X[genes]
    X_train, y_train = X[ds.train_mask], ds.y[ds.train_mask]
    X_test, y_test = X[ds.validation_mask], ds.y[ds.validation_mask]

    # 3. One AUC per seed. The model config is loaded once, outside the loop.
    model = load_model_config(MODEL)
    aucs = []
    for seed in range(N_SEEDS):
        scores = score_task(X_train, y_train, X_test, seed, model, N_SYNTHETIC)
        aucs.append(auc(y_test, scores))

    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs, ddof=1))
    print(f"{args.task}: AUC over {N_SEEDS} seeds = {mean_auc:.3f} +/- {std_auc:.3f}")
    print(f"  min {min(aucs):.3f}, max {max(aucs):.3f}")
    print("  per seed:", ", ".join(f"{a:.3f}" for a in aucs))
    assert std_auc > 0, "SD of exactly 0 means the seed is being ignored somewhere"


if __name__ == "__main__":
    main()
