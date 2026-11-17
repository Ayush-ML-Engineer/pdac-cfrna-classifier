"""Score a raw count table with a saved model bundle.

Run from the repo root:
    uv run python scripts/predict.py --model MODEL.joblib --counts RAW.csv --out SCORES.csv

RAW.csv: genes in rows, samples in columns, raw counts. Every sample is scored on its own.
SCORES.csv: one row per sample with its PDAC score between 0 and 1.

To also get a label (1 = PDAC, 0 = not PDAC; a sample is called PDAC when its score is at least the
threshold), give the threshold as a number or as the threshold.json of a select_threshold.py run:
    ... --threshold 0.28
    ... --threshold-file runs/<run folder>/threshold.json
Without either, only scores are written. A threshold belongs to one model: use the one chosen for
the model (same genes, same normalization settings) you are scoring with.
Only load model files you made yourself or trust (they are pickles).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pdac_cfrna.bundle import load_bundle, scores_to_frame
from pdac_cfrna.predict import label_scores, predict_scores, read_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="model bundle written by train_model.py")
    parser.add_argument("--counts", required=True, help="raw counts CSV (genes x samples)")
    parser.add_argument("--out", required=True, help="where to write the scores CSV")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--threshold", type=float, help="also write labels: score >= this is PDAC")
    group.add_argument(
        "--threshold-file", help="read the threshold from a threshold.json (select_threshold.py)"
    )
    args = parser.parse_args()
    threshold = read_threshold(args.threshold_file) if args.threshold_file else args.threshold

    bundle = load_bundle(args.model)
    counts = pd.read_csv(args.counts, index_col=0)
    scores = predict_scores(counts, bundle)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = scores_to_frame(scores)
    if threshold is not None:
        labels = label_scores(scores, threshold)
        table["pdac_label"] = labels.to_numpy()
    table.to_csv(out_path, index=False)

    print(f"Samples scored: {len(scores)}")
    print(f"Scores: min {scores.min():.3f}, median {scores.median():.3f}, max {scores.max():.3f}")

    if threshold is not None:
        n_pdac = int(labels.sum())
        print(f"Threshold {threshold:.3f}: {n_pdac} called PDAC, {len(labels) - n_pdac} not")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
