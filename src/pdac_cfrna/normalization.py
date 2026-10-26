"""cf-normalization (Moore et al. 2025): rescale the intrinsic and batch parts of each gene.

Tables in this module are genes x samples, as in the paper's R code and CSV files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FACTOR_COLUMNS = ["Intrinsic", "Extrinsic"]


def cf_norm(counts: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Apply cf-normalization to a genes x samples count table.

    Every gene's counts are split into an intrinsic share (factors["Intrinsic"]) and a batch
    share (factors["Extrinsic"]). Each share is rescaled per sample by how large it is compared
    with the median sample:

        result[g, s] = counts[g, s] * (I_g / fact_i[s] + E_g / fact_b[s])

    LIMITATION: the medians come from the samples passed in, so a sample's result depends on
    which other samples are in the table. A batch of one sample is returned unchanged.

    Args:
        counts: raw counts, genes in rows, samples in columns.
        factors: table indexed by gene with columns "Intrinsic" and "Extrinsic".

    Returns:
        Normalized table: the genes found in both inputs, in the factor table's order.

    Raises:
        ValueError: no gene is shared, or a sample has no counts in the intrinsic or the
            batch part (it cannot be rescaled).
    """
    genes = factors.index.intersection(counts.index)
    if len(genes) == 0:
        raise ValueError("counts and factors share no genes")
    counts, f = counts.loc[genes], factors.loc[genes]

    intrinsic = counts.mul(f["Intrinsic"], axis=0).sum()  # per sample: intrinsic part total
    batch = counts.mul(f["Extrinsic"], axis=0).sum()  # per sample: batch part total
    empty = intrinsic.index[(intrinsic == 0) | (batch == 0)]
    if len(empty) > 0:
        raise ValueError(
            f"cannot normalize samples with no intrinsic or batch counts: {list(empty)}"
        )

    fact_i = intrinsic / intrinsic.median()
    fact_b = batch / batch.median()
    scale = (
        f["Intrinsic"].to_numpy()[:, None] / fact_i.to_numpy()[None, :]
        + f["Extrinsic"].to_numpy()[:, None] / fact_b.to_numpy()[None, :]
    )
    return pd.DataFrame(counts.to_numpy() * scale, index=counts.index, columns=counts.columns)


def load_factors(path: str | Path, tolerance: float = 1e-6) -> pd.DataFrame:
    """Read a factor table (e.g. updated_intrinsic.csv) and check that it is usable.

    The file has the gene name in its first column and the columns "Intrinsic" and "Extrinsic".
    Genes without factors (empty rows) are dropped, as in the paper's code. Extra columns are
    ignored.

    Raises:
        FileNotFoundError: the file does not exist.
        ValueError: a required column is missing or not numeric, gene names repeat, a gene has
            only one of the two factors, a factor lies outside [0, 1], a gene's two factors do
            not add up to 1, or no gene has factors at all.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Factor file not found: {path}")
    table = pd.read_csv(path, index_col=0)

    missing = [c for c in FACTOR_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"{path.name} lacks the column(s) {missing}; found {list(table.columns)}")
    table = table[FACTOR_COLUMNS]
    if not all(pd.api.types.is_numeric_dtype(table[c]) for c in FACTOR_COLUMNS):
        raise ValueError(f"{path.name}: the factor columns must be numeric")

    table = table[table["Intrinsic"].notna()]  # genes with no factors are dropped
    if table.empty:
        raise ValueError(f"{path.name}: no gene has factors")
    if table["Extrinsic"].isna().any():
        raise ValueError(f"{path.name}: some genes have an Intrinsic but no Extrinsic factor")
    if table.index.has_duplicates:
        raise ValueError(
            f"{path.name}: duplicate gene names: {list(table.index[table.index.duplicated()][:5])}"
        )
    if ((table < -tolerance) | (table > 1 + tolerance)).any().any():
        raise ValueError(f"{path.name}: factors must lie between 0 and 1")
    if ((table["Intrinsic"] + table["Extrinsic"] - 1).abs() > tolerance).any():
        raise ValueError(f"{path.name}: for every gene Intrinsic + Extrinsic must sum to 1")
    return table.astype(float)
