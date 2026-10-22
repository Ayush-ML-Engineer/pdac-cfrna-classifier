"""Train the paper pipeline on CEDAR and score BCC once (one task, one seed).

Run from the repo root:
    uv run python scripts/score_task_once.py --task pdac_vs_benign --seed 0
Temporary exploration script: Section 7 replaces it with the full experiment script.
"""

from __future__ import annotations

import argparse

from pdac_cfrna.config import load_dataset_config
from pdac_cfrna.data import Dataset, apply_task, load_dataset
from pdac_cfrna.evaluation import auc, bootstrap_auc_ci
from pdac_cfrna.experiment import TASKS, load_experiment_config, resolve_experiment_path
from pdac_cfrna.model_config import load_model_config
from pdac_cfrna.models import score_task
from pdac_cfrna.selection import select_de_genes

SELECTION_EXPERIMENT = "de_selection_pdac_vs_all"  # the 29 genes always come from PDAC vs all
MODEL = "random_forest_paper"
N_SYNTHETIC = 100


def load_for_task(dataset_name: str, task_name: str) -> Dataset:
    task = TASKS[task_name]
    cfg = apply_task(load_dataset_config(dataset_name), task.class_map, task.unmapped)
    return load_dataset(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="pdac_vs_all", choices=sorted(TASKS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    exp = load_experiment_config(resolve_experiment_path(SELECTION_EXPERIMENT))

    # 1. Genes: PDAC vs all, CEDAR rows only.
    ds_sel = load_for_task(exp.dataset, exp.task)
    table = select_de_genes(
        ds_sel.X[ds_sel.train_mask],
        ds_sel.y[ds_sel.train_mask],
        exp.variance_cutoff,
        exp.alpha,
        exp.log_variance,
    )
    genes = list(table.index[table["significant"]])

    # 2. Split the chosen task: train on CEDAR, test on BCC, same 29 genes.
    ds = load_for_task(exp.dataset, args.task)
    X = ds.X[genes]
    X_train, y_train = X[ds.train_mask], ds.y[ds.train_mask]
    X_test, y_test = X[ds.validation_mask], ds.y[ds.validation_mask]

    # 3. Score and evaluate.
    scores = score_task(
        X_train,
        y_train,
        X_test,
        seed=args.seed,
        model=load_model_config(MODEL),
        n_synthetic=N_SYNTHETIC,
    )

    low, high = bootstrap_auc_ci(y_test, scores)
    print(f"BCC Cohort, {args.task}, seed {args.seed}: AUC {auc(y_test, scores):.3f}", end="  ")
    print(f"(95% CI = {low:.3f} to {high:.3f})")
    print(f"test samples {len(y_test)} (PDAC {int(y_test.sum())})")


if __name__ == "__main__":
    main()
