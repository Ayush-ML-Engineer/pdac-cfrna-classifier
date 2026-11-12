"""Choose a decision threshold for the PDAC score and report the confusion matrix.

Run from the repo root (validation setup: normalization constants and training on CEDAR only):
    uv run python scripts/select_threshold.py --name threshold_f2 `
        --set dataset.normalization.mode=frozen `
        --set 'dataset.normalization.reference_cohorts=[CEDAR]'

Where the threshold is chosen (--choose-on):
  validation (default)  on the validation-cohort scores of the model trained on the training
        cohort. Those scores come from a model that never saw the cohort, so they are honest
        scores, but the threshold is tuned on the same samples it is then reported on. To show how
        much that matters, the script also estimates the performance by cross-validation on the
        validation cohort: choose on part of it, test on the rest, repeat.
  train  on out-of-fold scores of the training cohort (repeated CV; --gene-selection says whether
        genes are selected again inside every fold). The threshold is then locked and applied once
        to the validation cohort. Needs enough PDAC samples for gene selection to be stable; with
        about 20 it is not, see docs.

--rule: fbeta (default, beta 2: recall counts twice as much as precision), sensitivity (the
highest threshold that still catches --target-sensitivity of the PDAC samples) or youden.

Writes into the run folder: threshold.json, threshold_table.csv, validation_scores.csv,
oof_scores.csv (train mode), pr_curves.png, confusion_matrices.png.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pdac_cfrna.models import score_task
from pdac_cfrna.run_config import add_config_arguments, config_from_args
from pdac_cfrna.runs import tracked_run
from pdac_cfrna.threshold_plots import plot_confusion_matrices, plot_pr_curves
from pdac_cfrna.thresholds import (
    choose_threshold,
    confusion_counts,
    cv_threshold_performance,
    oof_scores,
    summarize,
    threshold_table,
    wilson_interval,
)
from pdac_cfrna.training import load_for_task, select_experiment_genes


def _report(name: str, y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, object]:
    counts = confusion_counts(y, score, threshold)
    rates = summarize(counts)
    print(f"\n{name}  ({int(y.sum())} PDAC, {int((1 - y).sum())} controls)")
    print(f"  AUC {roc_auc_score(y, score):.3f}")
    print(
        f"  confusion: TP {counts['tp']}  FN {counts['fn']}  TN {counts['tn']}  FP {counts['fp']}"
    )
    for key, (value, low, high) in rates.items():
        print(f"  {key:<12}{value:.3f}  (95% CI {low:.3f} to {high:.3f})")
    return {"counts": counts, "rates": {k: list(v) for k, v in rates.items()}}


def _report_cv(estimate: dict[str, object], n_pos: int, n_neg: int) -> dict[str, object]:
    c: dict[str, float] = estimate["counts"]  # type: ignore[assignment]
    t = np.array(estimate["thresholds"], dtype=float)
    print(
        "\nHonest estimate: threshold chosen on part of the validation cohort, tested on the rest"
    )
    print(f"  thresholds chosen: median {np.median(t):.2f}, range {t.min():.2f} to {t.max():.2f}")
    for key, k, n in (
        ("sensitivity", c["tp"], n_pos),
        ("specificity", c["tn"], n_neg),
    ):
        low, high = wilson_interval(round(k), n)
        print(f"  {key:<12}{estimate[key]:.3f}  (approx. 95% CI {low:.3f} to {high:.3f})")
    print(f"  ppv         {estimate['ppv']:.3f}")
    return {
        "threshold_median": float(np.median(t)),
        "threshold_min": float(t.min()),
        "threshold_max": float(t.max()),
        "sensitivity": estimate["sensitivity"],
        "specificity": estimate["specificity"],
        "ppv": estimate["ppv"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--choose-on", choices=["validation", "train"], default="validation")
    parser.add_argument("--rule", choices=["fbeta", "sensitivity", "youden"], default="fbeta")
    parser.add_argument("--beta", type=float, default=2.0, help="beta of the fbeta rule")
    parser.add_argument("--target-sensitivity", type=float, default=0.90)
    parser.add_argument("--folds", type=int, default=5, help="CV folds")
    parser.add_argument("--repeats", type=int, default=5, help="repeats of the training-cohort CV")
    parser.add_argument(
        "--gene-selection",
        choices=["in_fold", "fixed"],
        default="in_fold",
        help="train mode only. in_fold: select genes again in every fold (honest, but unstable "
        "with few PDAC). fixed: use the genes chosen on all training rows (leaks the held-out "
        "labels, so the scores look too good; for comparison only)",
    )
    args = parser.parse_args()
    cfg = config_from_args(args)
    norm = cfg.dataset.normalization

    if norm is None or norm.mode != "frozen":
        parser.error("add --set dataset.normalization.mode=frozen (see the docstring)")

    exp = cfg.experiment

    if args.choose_on == "train" and exp.gene_selection_task != exp.task:
        parser.error(
            "--choose-on train needs experiment.gene_selection_task to equal experiment.task"
        )

    with tracked_run(cfg) as run:
        ds = load_for_task(cfg.dataset, exp.task)
        train, val = ds.train_mask.to_numpy(), ds.validation_mask.to_numpy()
        if not val.any():
            parser.error("no validation cohort configured: there is nothing to report on")
        X_tr, y_tr = ds.X.loc[train], ds.y[train]
        X_val, y_val = ds.X.loc[val], ds.y[val]
        genes = select_experiment_genes(cfg)

        # Validation scores: the mean over the seeds of models trained on the training rows.
        seeds = range(exp.seed, exp.seed + exp.n_seeds)
        s_val = (
            pd.concat(
                [
                    score_task(X_tr[genes], y_tr, X_val[genes], s, cfg.model, exp.n_synthetic_pdac)
                    for s in seeds
                ],
                axis=1,
            )
            .mean(axis=1)
            .reindex(X_val.index)
        )
        y_v, s_v = y_val.to_numpy(), s_val.to_numpy()
        pd.DataFrame({"sample_id": s_val.index, "label": y_v, "pdac_score": s_v}).to_csv(
            run.dir / "validation_scores.csv", index=False
        )
        print(f"{len(genes)} genes; model trained on {len(y_tr)} samples; {len(y_v)} validation")

        results: dict[str, object] = {
            "choose_on": args.choose_on,
            "rule": args.rule,
            "n_final_genes": len(genes),
        }
        sets: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        if args.choose_on == "train":
            oof, n_genes = oof_scores(
                X_tr,
                y_tr,
                exp,
                cfg.model,
                args.folds,
                args.repeats,
                fixed_genes=genes if args.gene_selection == "fixed" else None,
            )
            y_oof, s_oof = y_tr.to_numpy(), oof.to_numpy()
            table = threshold_table(y_oof, s_oof)
            threshold = choose_threshold(table, args.rule, args.beta, args.target_sensitivity)
            skipped = sum(n == 0 for n in n_genes)
            print(
                f"Training cohort out-of-fold ({args.gene_selection} genes): genes per fold "
                f"min {min(n_genes)}, median {int(np.median(n_genes))}, max {max(n_genes)}; "
                f"{skipped} of {len(n_genes)} folds selected no gene"
            )
            print(f"Rule {args.rule}: threshold {threshold:.2f}")
            pd.DataFrame({"sample_id": oof.index, "label": y_oof, "oof_score": s_oof}).to_csv(
                run.dir / "oof_scores.csv", index=False
            )
            results.update(gene_selection=args.gene_selection, genes_per_fold=n_genes)
            results["training_oof"] = _report(
                "Training cohort, out-of-fold", y_oof, s_oof, threshold
            )
            results["validation"] = _report(
                f"Validation cohort, threshold {threshold:.2f} locked beforehand",
                y_v,
                s_v,
                threshold,
            )
            sets["Training (out-of-fold)"] = (y_oof, s_oof)
        else:
            table = threshold_table(y_v, s_v)
            threshold = choose_threshold(table, args.rule, args.beta, args.target_sensitivity)
            print(f"Rule {args.rule}: threshold {threshold:.2f} (chosen on the validation scores)")
            results["validation_apparent"] = _report(
                f"Validation cohort, threshold {threshold:.2f} tuned on these samples (optimistic)",
                y_v,
                s_v,
                threshold,
            )
            estimate = cv_threshold_performance(
                y_v, s_v, args.rule, args.beta, args.target_sensitivity, args.folds, 20, exp.seed
            )
            results["validation_cv"] = _report_cv(estimate, int(y_v.sum()), int((1 - y_v).sum()))
        sets["Validation"] = (y_v, s_v)

        results["threshold"] = threshold
        table.to_csv(run.dir / "threshold_table.csv", index=False)
        plot_pr_curves(sets, threshold, run.dir / "pr_curves.png")
        plot_confusion_matrices(sets, threshold, run.dir / "confusion_matrices.png")
        (run.dir / "threshold.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        run.extra.update(results)
        print(f"\nWritten to {run.dir}")


if __name__ == "__main__":
    main()
