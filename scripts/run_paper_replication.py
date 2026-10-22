"""Run the paper's classifier as described by an experiment YAML: train on CEDAR, test on BCC.

Run from the repo root:
    uv run python scripts/run_paper_replication.py paper_replication_pdac_vs_all
(the argument is a file name in configs/experiments/, without .yaml, or a path)
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from pdac_cfrna.config import load_dataset_config
from pdac_cfrna.data import Dataset, apply_task, load_dataset
from pdac_cfrna.evaluation import auc, bootstrap_auc_ci
from pdac_cfrna.experiment import TASKS, load_experiment_config, resolve_experiment_path
from pdac_cfrna.model_config import load_model_config
from pdac_cfrna.models import score_task
from pdac_cfrna.runs import tracked_run
from pdac_cfrna.selection import select_de_genes


def load_for_task(dataset_name: str, task_name: str) -> Dataset:
    task = TASKS[task_name]
    cfg = apply_task(load_dataset_config(dataset_name), task.class_map, task.unmapped)
    return load_dataset(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="experiment name or path to its YAML file")
    args = parser.parse_args()

    # Validate before taking the lock or creating a run folder.
    exp_path = resolve_experiment_path(args.experiment)
    exp = load_experiment_config(exp_path)

    with tracked_run(exp, exp_path) as run:
        # 1. Genes: chosen once, with the gene-selection task's labels, on CEDAR rows only.
        ds_sel = load_for_task(exp.dataset, exp.gene_selection_task)
        table = select_de_genes(
            ds_sel.X[ds_sel.train_mask],
            ds_sel.y[ds_sel.train_mask],
            exp.variance_cutoff,
            exp.alpha,
            exp.log_variance,
        )
        genes = list(table.index[table["significant"]])
        table.to_csv(run.dir / "de_genes.csv", index_label="gene")

        # 2. The experiment's own task: train on CEDAR, test on BCC, same genes.
        ds = load_for_task(exp.dataset, exp.task)
        X = ds.X[genes]
        X_train, y_train = X[ds.train_mask], ds.y[ds.train_mask]
        X_test, y_test = X[ds.validation_mask], ds.y[ds.validation_mask]

        # 3. One score per test sample per seed. The seeds are seed, seed+1, ...
        model = load_model_config(exp.model)
        seeds = [exp.seed + i for i in range(exp.n_seeds)]
        scores_by_seed = {}
        aucs = []
        for seed in seeds:
            scores = score_task(X_train, y_train, X_test, seed, model, exp.n_synthetic_pdac)
            scores_by_seed[f"seed_{seed}"] = scores
            aucs.append(auc(y_test, scores))

        # 4. Two different kinds of uncertainty, reported separately:
        #    - spread across seeds (randomness of SMOTE and the forest): mean and SD of the AUCs
        #    - spread across test samples (only 21 PDAC): bootstrap CI of the seed-averaged score
        score_table = pd.DataFrame(scores_by_seed)
        score_table.insert(0, "y_true", y_test)
        score_table["mean_score"] = score_table[list(scores_by_seed)].mean(axis=1)
        score_table.to_csv(run.dir / "test_scores.csv", index_label="sample_id")

        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else None
        avg_auc = auc(y_test, score_table["mean_score"])
        ci_low, ci_high = bootstrap_auc_ci(y_test, score_table["mean_score"])

        run.extra.update(
            n_genes=len(genes),
            n_train=len(y_train),
            n_train_pdac=int(y_train.sum()),
            n_test=len(y_test),
            n_test_pdac=int(y_test.sum()),
            seeds=seeds,
            auc_per_seed=aucs,
            mean_auc=mean_auc,
            std_auc=std_auc,
            seed_averaged_auc=avg_auc,
            seed_averaged_ci_low=ci_low,
            seed_averaged_ci_high=ci_high,
        )

        print(f"{exp.name}: {len(genes)} genes, train {len(y_train)}, test {len(y_test)}")
        spread = f" +/- {std_auc:.3f}" if std_auc is not None else ""
        print(f"  AUC over {len(seeds)} seeds: {mean_auc:.3f}{spread}")
        print(f"  Seed-averaged score: AUC {avg_auc:.3f} (95% CI {ci_low:.3f} to {ci_high:.3f})")
        print(f"  Run folder: {run.dir}")


if __name__ == "__main__":
    main()
