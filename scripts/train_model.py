"""Train a model bundle (needs frozen normalization) on the training cohorts.

Run from the repo root:
    uv run python scripts/train_model.py --name model_frozen_vs_all `
        --set dataset.normalization.mode=frozen `
        --set 'dataset.normalization.reference_cohorts=[CEDAR]'

Final model on every sample (no validation cohort, so no reference AUC is printed):
    uv run python scripts/train_model.py --name model_final `
        --set dataset.normalization.mode=frozen `
        --set 'dataset.normalization.reference_cohorts=[CEDAR,BCC]' `
        --set 'dataset.split.train_cohorts=[CEDAR,BCC]' `
        --set 'dataset.split.validation_cohorts=[]'

The bundle is written into the run folder as model.joblib, with model.json next to it.
"""

from __future__ import annotations

import argparse

from sklearn.metrics import roc_auc_score

from pdac_cfrna.bundle import save_bundle
from pdac_cfrna.run_config import add_config_arguments, config_from_args
from pdac_cfrna.runs import tracked_run
from pdac_cfrna.training import load_for_task, select_experiment_genes, train_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    cfg = config_from_args(parser.parse_args())  # validated BEFORE the lock and the run folder
    norm = cfg.dataset.normalization
    if norm is None or norm.mode != "frozen":
        parser.error(
            "a model bundle needs frozen normalization: add "
            "--set dataset.normalization.mode=frozen "
            "--set 'dataset.normalization.reference_cohorts=[CEDAR]'"
        )

    with tracked_run(cfg) as run:
        genes = select_experiment_genes(cfg)
        ds = load_for_task(cfg.dataset, cfg.experiment.task)
        bundle = train_bundle(ds, cfg, genes)
        # ... the validation-AUC block, save_bundle and prints stay as they were, except that
        # the last print uses {cfg.name} instead of {exp.name}.

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
        print(f"{cfg.name}: {len(genes)} genes, trained on {bundle.metadata['n_train']} samples")
        if validation_auc is not None:
            print(f"  Validation cohort AUC (reference): {validation_auc:.3f}")
        print(f"  Bundle: {path}")


if __name__ == "__main__":
    main()
