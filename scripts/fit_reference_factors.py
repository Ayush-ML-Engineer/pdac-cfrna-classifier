"""Learn the intrinsic/extrinsic gene factors (NMF), and optionally extend them to a dataset.

Reference factors only (run from the repo root):
    uv run python scripts/fit_reference_factors.py --reference REF.csv --out REF_FACTORS.csv

Also extend them to the genes the healthy reference never saw (adjust_factors, fitted on
every sample in --counts):
    uv run python scripts/fit_reference_factors.py --reference REF.csv --out REF_FACTORS.csv \\
        --counts RAW_COUNTS.csv --out-updated UPDATED_FACTORS.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pdac_cfrna.nmf import adjust_factors, fit_reference_factors


def _write(table: pd.DataFrame, path: str) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="reference counts CSV (genes x samples)")
    parser.add_argument("--out", required=True, help="where to write the reference factor table")
    parser.add_argument("--counts", help="raw patient counts CSV (genes x samples)")
    parser.add_argument("--out-updated", help="where to write the extended factor table")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if bool(args.counts) != bool(args.out_updated):
        parser.error("--counts and --out-updated must be given together")

    reference = pd.read_csv(args.reference, index_col=0)
    factors = fit_reference_factors(reference, seed=args.seed)
    path = _write(factors, args.out)
    print(f"Reference samples: {reference.shape[1]}, genes: {reference.shape[0]}")
    print(f"Genes with factors: {int(factors['Intrinsic'].notna().sum())}")
    print(f"Genes without factors (zero everywhere): {int(factors['Intrinsic'].isna().sum())}")
    print(f"Written: {path}")

    if args.counts:
        counts = pd.read_csv(args.counts, index_col=0)
        updated = adjust_factors(counts, factors, seed=args.seed)
        path = _write(updated, args.out_updated)
        print(f"Patient samples: {counts.shape[1]}, genes: {counts.shape[0]}")
        print(f"Genes with factors after adjusting: {int(updated['Intrinsic'].notna().sum())}")
        print(f"Written: {path}")


if __name__ == "__main__":
    main()
