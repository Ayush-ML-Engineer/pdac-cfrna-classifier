"""Replicate the paper's DE gene selection on the CEDAR (training) cohort.

Run from the repo root:
    uv run python scripts/replicate_de_selection.py
    uv run python scripts/replicate_de_selection.py --log-variance
"""

from __future__ import annotations

import argparse

from pdac_cfrna.config import REPO_ROOT, load_dataset_config
from pdac_cfrna.data import load_dataset
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
    parser.add_argument("--dataset", default="moore2025_cfnorm")
    parser.add_argument("--min-variance", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--log-variance",
        action="store_true",
        help="compute the variance filter on log1p(counts)",
    )
    parser.add_argument("--out", default="data/interim/de_genes_cedar.csv")
    args = parser.parse_args()

    # 1. Load the dataset and keep ONLY the training cohort (CEDAR). BCC is never touched.
    ds = load_dataset(load_dataset_config(args.dataset))
    X_train = ds.X[ds.train_mask]
    y_train = ds.y[ds.train_mask]
    print(f"Training samples: {len(y_train)} (PDAC: {int(y_train.sum())})")

    # 2. Variance filter -> Wilcoxon -> BH -> flag.
    table = select_de_genes(
        X_train,
        y_train,
        min_variance=args.min_variance,
        alpha=args.alpha,
        log_before_variance=args.log_variance,
    )
    selected = table[table["significant"]]
    print(f"Genes tested after variance filter: {len(table)}")
    print(f"Genes with q <= {args.alpha}: {len(selected)}")

    # 3. Save the full table.
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index_label="gene")
    print(f"Saved: {out_path}")

    # 4. Compare with the paper's named genes.
    found = set(selected.index)
    named = PAPER_UP | PAPER_DOWN | PAPER_OTHER
    print(f"\nPaper genes recovered: {len(found & named)} of {len(named)}")
    print("  missing:", sorted(named - found))
    for label, expected_dir, genes in (("up", "up", PAPER_UP), ("down", "down", PAPER_DOWN)):
        hits = [g for g in genes if g in found]
        correct = [g for g in hits if selected.loc[g, "direction"] == expected_dir]
        print(f"  expected {label}: found {len(hits)}/{len(genes)}, right direction {len(correct)}")
    print("\nTop 10 genes:")
    print(selected.head(10).to_string())


if __name__ == "__main__":
    main()
