"""Step 3, Section 1: get the 29 genes and check the train/test split of every task.

Run from the repo root:
    uv run python scripts/check_task_splits.py
"""

from __future__ import annotations

from pdac_cfrna.config import load_dataset_config
from pdac_cfrna.data import Dataset, apply_task, load_dataset
from pdac_cfrna.experiment import TASKS, load_experiment_config, resolve_experiment_path
from pdac_cfrna.selection import select_de_genes

# The experiment whose settings (and task, PDAC vs all) define the gene list.
SELECTION_EXPERIMENT = "de_selection_pdac_vs_all"

# From the paper: task -> (train samples, train PDAC, test samples, test PDAC).
EXPECTED = {
    "pdac_vs_benign": (68, 20, 44, 21),
    "pdac_vs_non_cancer": (143, 20, 78, 21),
    "pdac_vs_all": (153, 20, 95, 21),
}


def load_for_task(dataset_name: str, task_name: str) -> Dataset:
    """Load the dataset with one task's labels (and sample selection) applied."""
    task = TASKS[task_name]
    cfg = apply_task(load_dataset_config(dataset_name), task.class_map, task.unmapped)
    return load_dataset(cfg)


def main() -> None:
    exp = load_experiment_config(resolve_experiment_path(SELECTION_EXPERIMENT))

    # 1. Pick the genes ONCE: PDAC vs all, CEDAR (training) rows only.
    ds_sel = load_for_task(exp.dataset, exp.task)
    table = select_de_genes(
        ds_sel.X[ds_sel.train_mask],
        ds_sel.y[ds_sel.train_mask],
        exp.variance_cutoff,
        exp.alpha,
        exp.log_variance,
    )
    genes = list(table.index[table["significant"]])
    print(f"Selected genes: {len(genes)}")
    assert len(genes) == 29, f"expected 29 genes, got {len(genes)}"

    # 2. The same 29 genes are used for every task; only the labels and samples change.
    for task_name in TASKS:
        ds = load_for_task(exp.dataset, task_name)
        X = ds.X[genes]
        X_train, y_train = X[ds.train_mask], ds.y[ds.train_mask]
        X_test, y_test = X[ds.validation_mask], ds.y[ds.validation_mask]

        got = (len(y_train), int(y_train.sum()), len(y_test), int(y_test.sum()))
        print(f"{task_name:20s} train {X_train.shape}, PDAC {got[1]}", end=" | ")
        print(f"test {X_test.shape}, PDAC {got[3]}")
        assert got == EXPECTED[task_name], f"{task_name}: expected {EXPECTED[task_name]}, got {got}"

    print("All split sizes match the paper.")


if __name__ == "__main__":
    main()
