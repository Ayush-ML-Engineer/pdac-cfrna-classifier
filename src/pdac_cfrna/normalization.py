"""cf-normalization (Moore et al. 2025): rescale the intrinsic and batch parts of each gene.

Tables in this module are genes x samples, as in the paper's R code and CSV files.
"""

from __future__ import annotations

import pandas as pd


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
