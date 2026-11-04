"""Run the paper's classifier: train on the training cohort, test on the validation cohort.

Run from the repo root (defaults = configs/*.yaml; change settings with --set / --patch):
    uv run python scripts/run_paper_replication.py
    uv run python scripts/run_paper_replication.py --set experiment.task=pdac_vs_benign
    uv run python scripts/run_paper_replication.py --set dataset.normalization.mode=frozen `
        --set 'dataset.normalization.reference_cohorts=[CEDAR]' --name frozen_vs_all
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from pdac_cfrna.evaluation import auc, bootstrap_auc_ci
from pdac_cfrna.models import score_task
from pdac_cfrna.run_config import add_config_arguments, config_from_args
from pdac_cfrna.runs import tracked_run
from pdac_cfrna.training import load_for_task, select_experiment_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    cfg = config_from_args(parser.parse_args())  # validated BEFORE the lock and the run folder
    exp = cfg.experiment
    split = cfg.dataset.split
    if split is None or not split.validation_cohorts:
        parser.error(
            "this run has no validation cohort (dataset.split.validation_cohorts is empty); "
            "use scripts/train_model.py to train a model on everything"
        )

    with tracked_run(cfg) as run:
        # 1. Genes: chosen once, with the gene-selection task's labels, on training rows only.
        table = select_experiment_table(cfg)
        selected = list(table.index[table["significant"]])
        genes = selected
        table.to_csv(run.dir / "de_genes.csv", index_label="gene")

        # 2. The experiment's own task: train on the training cohort, test on the validation one.
        ds = load_for_task(cfg.dataset, exp.task)
        X = ds.X[genes]
        X_train, y_train = X[ds.train_mask], ds.y[ds.train_mask]
        X_test, y_test = X[ds.validation_mask], ds.y[ds.validation_mask]

        # 3. One score per test sample per seed. The seeds are seed, seed+1, ...
        seeds = [exp.seed + i for i in range(exp.n_seeds)]
        scores_by_seed = {}
        aucs = []
        for seed in seeds:
            scores = score_task(X_train, y_train, X_test, seed, cfg.model, exp.n_synthetic_pdac)
            scores_by_seed[f"seed_{seed}"] = scores
            aucs.append(auc(y_test, scores))

        # 4. Two kinds of uncertainty, reported separately: seeds, and the small test set.
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
            excluded_genes=list(exp.exclude_genes),
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

        print(f"{cfg.name}: {len(genes)} genes, train {len(y_train)}, test {len(y_test)}")
        spread = f" +/- {std_auc:.3f}" if std_auc is not None else ""
        print(f"  AUC over {len(seeds)} seeds: {mean_auc:.3f}{spread}")
        print(f"  Seed-averaged score: AUC {avg_auc:.3f} (95% CI {ci_low:.3f} to {ci_high:.3f})")
        print(f"  Run folder: {run.dir}")


if __name__ == "__main__":
    main()
