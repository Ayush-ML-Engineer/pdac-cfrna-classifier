"""Run the paper's DE gene selection as described by an experiment YAML.

Run from the repo root:
    uv run python scripts/replicate_de_selection.py de_selection_pdac_vs_all
(the argument is a file name in configs/experiments/, without .yaml, or a path)
"""

from __future__ import annotations

import argparse

from pdac_cfrna.config import load_dataset_config
from pdac_cfrna.data import apply_task, load_dataset
from pdac_cfrna.experiment import load_experiment_config, resolve_experiment_path
from pdac_cfrna.runs import tracked_run
from pdac_cfrna.selection import select_de_genes

# Genes named in the paper's Fig. 3a / Results (24 of the 29; direction as reported).
PAPER_UP = {"SERPINA1", "ACOT12", "CRP", "AFMID"}
PAPER_DOWN = {"RPS4X", "RPS13", "RPS14", "CD37", "MS4A1", "TNFRSF13C", "PAX5", "FAM117A"}
PAPER_OTHER = {
    "THAP12",
    "MYC",
    "FOXO1",
    "RIOX1",
    "SMAD3",
    "USP9X",
    "USP9Y",
    "UTY",
    "GCLC",
    "LARP1B",
    "OR4K13",
    "RAB13",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="experiment name or path to its YAML file")
    args = parser.parse_args()

    # Validate the experiment BEFORE taking the lock or creating a run folder.
    yaml_path = resolve_experiment_path(args.experiment)
    exp = load_experiment_config(yaml_path)

    with tracked_run(exp, yaml_path) as run:
        # 1. Dataset config + the experiment's task override.
        task = exp.resolved_task
        dataset_cfg = apply_task(load_dataset_config(exp.dataset), task.class_map, task.unmapped)
        ds = load_dataset(dataset_cfg)

        # 2. Training cohort only (CEDAR). The validation cohort is never touched here.
        X_train = ds.X[ds.train_mask]
        y_train = ds.y[ds.train_mask]
        print(f"Training samples: {len(y_train)} (PDAC: {int(y_train.sum())})")

        # 3. Variance filter -> Wilcoxon -> BH -> flag.
        table = select_de_genes(
            X_train,
            y_train,
            min_variance=exp.variance_cutoff,
            alpha=exp.alpha,
            log_before_variance=exp.log_variance,
        )
        selected = table[table["significant"]]
        table.to_csv(run.dir / "de_genes.csv", index_label="gene")

        # 4. Compare with the paper's named genes (most meaningful for pdac_vs_all).
        found = set(selected.index)
        named = PAPER_UP | PAPER_DOWN | PAPER_OTHER
        up_ok = [g for g in PAPER_UP & found if selected.loc[g, "direction"] == "up"]
        down_ok = [g for g in PAPER_DOWN & found if selected.loc[g, "direction"] == "down"]
        run.extra.update(
            n_train_samples=len(y_train),
            n_train_pdac=int(y_train.sum()),
            n_genes_tested=len(table),
            n_significant=len(selected),
            paper_genes_recovered=len(found & named),
            paper_genes_named=len(named),
            paper_direction_correct=len(up_ok) + len(down_ok),
        )

        print(f"Genes tested after variance filter: {len(table)}")
        print(f"Genes with q <= {exp.alpha}: {len(selected)}")
        print(f"Paper genes recovered: {len(found & named)} of {len(named)}")
        print(f"Run folder: {run.dir}")


if __name__ == "__main__":
    main()
