"""Score an external raw counts file with a saved model and check the scores against labels.

Run from the repo root:
    uv run python scripts/evaluate_external.py --model runs/<run>/model.joblib `
        --counts RAW.csv --out reports/external_scores.csv

RAW.csv: genes in rows, samples in columns, RAW COUNTS (not log values, not CPM).
Labels come from the end of each sample name: names ending in "PDAC" are 1 (cancer), names
ending in "Controles" or "Control" are 0. Any other name stops the script, so no sample is
labelled by guess. The labels are used only after scoring; the model never sees them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from pdac_cfrna.bundle import load_bundle
from pdac_cfrna.evaluation import auc, bootstrap_auc_ci
from pdac_cfrna.predict import predict_scores

POSITIVE_SUFFIXES = ("PDAC",)
NEGATIVE_SUFFIXES = ("Controles", "Control")


def labels_from_names(names: list[str]) -> pd.Series:
    """1 for PDAC samples, 0 for controls, read from the end of the sample name."""
    labels: dict[str, int] = {}
    unknown: list[str] = []
    for name in names:
        if name.endswith(POSITIVE_SUFFIXES):
            labels[name] = 1
        elif name.endswith(NEGATIVE_SUFFIXES):
            labels[name] = 0
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(f"cannot read a label from {len(unknown)} sample name(s): {unknown[:5]}")
    return pd.Series(labels, name="label").reindex(names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="model bundle written by train_model.py")
    parser.add_argument("--counts", required=True, help="raw counts CSV (genes x samples)")
    parser.add_argument("--out", required=True, help="where to write the scores and labels CSV")
    parser.add_argument("--threshold", type=float, default=0.5, help="cutoff for the table below")
    args = parser.parse_args()

    bundle = load_bundle(args.model)
    counts = pd.read_csv(args.counts, index_col=0)
    labels = labels_from_names(list(counts.columns))
    scores = predict_scores(counts, bundle).reindex(labels.index)

    result = pd.DataFrame({"sample_id": labels.index, "label": labels.to_numpy()})
    result["pdac_score"] = scores.to_numpy()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    y, s = result["label"].to_numpy(), result["pdac_score"].to_numpy()
    low, high = bootstrap_auc_ci(y, s)
    pred = s >= args.threshold
    tp, fn = int(((y == 1) & pred).sum()), int(((y == 1) & ~pred).sum())
    tn, fp = int(((y == 0) & ~pred).sum()), int(((y == 0) & pred).sum())
    fpr, tpr, thr = roc_curve(y, s)

    print(f"Samples: {len(y)} ({int(y.sum())} PDAC, {int((1 - y).sum())} controls)")
    print(f"AUC: {auc(y, s):.3f}  (95% bootstrap CI {low:.3f} to {high:.3f})")
    print(f"At threshold {args.threshold}: sensitivity {tp}/{tp + fn}, specificity {tn}/{tn + fp}")
    print(f"  confusion: TP {tp}  FN {fn}  TN {tn}  FP {fp}")
    best = int(np.argmax(tpr - fpr))
    print(f"Best Youden cutoff on THESE samples: {thr[best]:.3f} (only for looking; not a choice)")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
