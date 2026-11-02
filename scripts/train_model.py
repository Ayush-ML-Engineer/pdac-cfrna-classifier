"""Train a model bundle from an experiment YAML (frozen normalization, training cohort only).

Run from the repo root:
    uv run python scripts/train_model.py model_frozen_pdac_vs_all
(the argument is a file name in configs/experiments/, without .yaml, or a path)

The bundle is written into the run folder as model.joblib, with model.json next to it.
"""

from __future__ import annotations

import argparse

from sklearn.metrics import roc_auc_score

from pdac_cfrna.bundle import save_bundle
from pdac_cfrna.experiment import load_experiment_config, resolve_experiment_path
from pdac_cfrna.runs import tracked_run
from pdac_cfrna.training import load_for_task, select_experiment_genes, train_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="experiment name or path to its YAML file")
    args = parser.parse_args()

    # Validate before taking the lock or creating a run folder.
    exp_path = resolve_experiment_path(args.experiment)
    exp = load_experiment_config(exp_path)

    with tracked_run(exp, exp_path) as run:
        genes = select_experiment_genes(exp)
        ds = load_for_task(exp.dataset, exp.task)
        bundle = train_bundle(ds, exp, genes)

        # Reference performance on the validation cohort (never used for training).
        validation_auc = None
        if ds.validation_mask.any():
            X_val = ds.X.loc[ds.validation_mask.to_numpy(), genes]
            y_val = ds.y[ds.validation_mask]
            column = list(bundle.pipeline.classes_).index(1)
            scores = bundle.pipeline.predict_proba(X_val)[:, column]
            validation_auc = float(roc_auc_score(y_val, scores))
            bundle.metadata["validation_auc"] = validation_auc
            bundle.metadata["n_validation"] = len(y_val)

        path = save_bundle(bundle, run.dir / "model.joblib")
        run.extra.update(
            n_genes=len(genes),
            n_train=bundle.metadata["n_train"],
            n_train_pdac=bundle.metadata["n_train_pdac"],
            validation_auc=validation_auc,
        )
        print(f"{exp.name}: {len(genes)} genes, trained on {bundle.metadata['n_train']} samples")
        if validation_auc is not None:
            print(f"  Validation cohort AUC (reference): {validation_auc:.3f}")
        print(f"  Bundle: {path}")


if __name__ == "__main__":
    main()
