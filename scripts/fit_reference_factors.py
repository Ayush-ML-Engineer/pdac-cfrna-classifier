"""Learn the intrinsic/extrinsic gene factors from the healthy reference samples (NMF).

Run from the repo root:
    uv run python scripts/fit_reference_factors.py --reference REFERENCE.csv --out OUT.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pdac_cfrna.nmf import fit_reference_factors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="reference counts CSV (genes x samples)")
    parser.add_argument("--out", required=True, help="where to write the factor table")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    counts = pd.read_csv(args.reference, index_col=0)
    factors = fit_reference_factors(counts, seed=args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    factors.to_csv(out_path)

    print(f"Reference samples: {counts.shape[1]}, genes: {counts.shape[0]}")
    print(f"Genes with factors: {int(factors['Intrinsic'].notna().sum())}")
    print(f"Genes without factors (zero everywhere): {int(factors['Intrinsic'].isna().sum())}")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
