"""Score a raw count table with a saved model bundle.

Run from the repo root:
    uv run python scripts/predict.py --model MODEL.joblib --counts RAW.csv --out SCORES.csv

RAW.csv: genes in rows, samples in columns, raw counts. Every sample is scored on its own.
SCORES.csv: one row per sample with its PDAC score between 0 and 1.
Only load model files you made yourself or trust (they are pickles).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pdac_cfrna.bundle import load_bundle, scores_to_frame
from pdac_cfrna.predict import predict_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="model bundle written by train_model.py")
    parser.add_argument("--counts", required=True, help="raw counts CSV (genes x samples)")
    parser.add_argument("--out", required=True, help="where to write the scores CSV")
    args = parser.parse_args()

    bundle = load_bundle(args.model)
    counts = pd.read_csv(args.counts, index_col=0)
    scores = predict_scores(counts, bundle)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores_to_frame(scores).to_csv(out_path, index=False)

    print(f"Samples scored: {len(scores)}")
    print(f"Scores: min {scores.min():.3f}, median {scores.median():.3f}, max {scores.max():.3f}")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
