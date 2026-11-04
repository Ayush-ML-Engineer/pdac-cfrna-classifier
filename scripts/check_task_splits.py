"""Step 3, Section 1: get the 29 genes and check the train/test split of every task.

Run from the repo root:
    uv run python scripts/check_task_splits.py
"""

import argparse

from pdac_cfrna.experiment import TASKS
from pdac_cfrna.run_config import add_config_arguments, config_from_args
from pdac_cfrna.training import load_for_task, select_experiment_genes

EXPECTED = {
    "pdac_vs_benign": (68, 20, 44, 21),
    "pdac_vs_non_cancer": (143, 20, 78, 21),
    "pdac_vs_all": (153, 20, 95, 21),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    cfg = config_from_args(parser.parse_args())

    # 1. Pick the genes ONCE: PDAC vs all, training-cohort rows only.
    genes = select_experiment_genes(cfg)
    print(f"Selected genes: {len(genes)}")
    assert len(genes) == 29, f"expected 29 genes, got {len(genes)}"

    # 2. From the paper: task -> (train samples, train PDAC, test samples, test PDAC).
    for task_name in TASKS:
        ds = load_for_task(cfg.dataset, task_name)
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
