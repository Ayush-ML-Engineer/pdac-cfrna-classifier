"""Normalize a raw count table (genes x samples) with cf-normalization followed by TMM.

Run from the repo root:
    uv run python scripts/normalize_counts.py --counts RAW.csv --factors FACTORS.csv --out OUT.csv

Note: results depend on which samples are in the file (see normalize_counts).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pdac_cfrna.normalization import load_factors, normalize_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counts", required=True, help="raw counts CSV: genes in rows, samples in columns"
    )
    parser.add_argument(
        "--factors", required=True, help="factor CSV with Intrinsic and Extrinsic columns"
    )
    parser.add_argument("--out", required=True, help="where to write the normalized CSV")
    args = parser.parse_args()

    counts = pd.read_csv(args.counts, index_col=0)
    factors = load_factors(args.factors)
    result = normalize_counts(counts, factors)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path)

    dropped = counts.shape[0] - result.shape[0]
    print(f"Samples: {result.shape[1]}")
    print(f"Genes: {counts.shape[0]} in, {result.shape[0]} out ({dropped} not in the factor table)")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
