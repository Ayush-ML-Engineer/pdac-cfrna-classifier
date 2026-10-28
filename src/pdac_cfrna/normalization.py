"""cf-normalization (Moore et al. 2025): rescale the intrinsic and batch parts of each gene.

Tables in this module are genes x samples, as in the paper's R code and CSV files.
"""

from __future__ import annotations

from collections.abc import Hashable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

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


def tmm_reference_sample(counts: pd.DataFrame) -> Hashable:
    """Label of the reference sample used by edgeR's TMM.

    Each sample's 75th percentile of (count / library size) is computed; the reference is the
    sample whose value is closest to the average of those values.
    """
    x = counts.loc[(counts > 0).any(axis=1)]
    f75 = (x / x.sum()).quantile(0.75)
    return (f75 - f75.mean()).abs().idxmin()


def _tmm_factor_vs_reference(
    obs: np.ndarray, ref: np.ndarray, logratio_trim: float, sum_trim: float
) -> float:
    """TMM scale factor of one sample against the reference (both are count vectors)."""
    n_o, n_r = obs.sum(), ref.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        p_o, p_r = obs / n_o, ref / n_r
        m = np.log2(p_o / p_r)  # log-ratio of the two proportions
        a = (np.log2(p_o) + np.log2(p_r)) / 2  # average log expression
        v = (n_o - obs) / n_o / obs + (n_r - ref) / n_r / ref  # approximate variance of m

    usable = np.isfinite(m) & np.isfinite(a)  # genes counted in both samples
    m, a, v = m[usable], a[usable], v[usable]
    n = len(m)
    if n == 0:
        raise ValueError("a sample shares no expressed genes with the reference sample")

    lo_m = int(np.floor(n * logratio_trim)) + 1
    hi_m = n + 1 - lo_m
    lo_a = int(np.floor(n * sum_trim)) + 1
    hi_a = n + 1 - lo_a
    rank_m, rank_a = rankdata(m), rankdata(a)
    keep = (rank_m >= lo_m) & (rank_m <= hi_m) & (rank_a >= lo_a) & (rank_a <= hi_a)
    if not keep.any():
        raise ValueError("no genes left after trimming")

    weights = 1 / v[keep]
    return float(2 ** (np.sum(m[keep] * weights) / np.sum(weights)))


def tmm_scale_factors(
    counts: pd.DataFrame, logratio_trim: float = 0.3, sum_trim: float = 0.05
) -> pd.Series:
    """TMM scale factors (Robinson & Oshlack 2010) as computed by edgeR::calcNormFactors.

    Args:
        counts: genes x samples, non-negative.

    Returns:
        One factor per sample (indexed by sample), with a geometric mean of 1. The reference
        sample's factor is 1 before that final rescaling.

    Raises:
        ValueError: negative or missing counts, a sample without any counts, or no gene with
            counts at all.
    """
    if counts.isna().any().any() or (counts < 0).any().any():
        raise ValueError("counts must be non-negative numbers without missing values")
    x = counts.loc[(counts > 0).any(axis=1)]  # drop genes that are zero in every sample
    if x.empty:
        raise ValueError("no gene has counts in any sample")
    empty = x.columns[x.sum() == 0]
    if len(empty) > 0:
        raise ValueError(f"samples without any counts: {list(empty)}")

    ref = tmm_reference_sample(x)
    ref_counts = x[ref].to_numpy(dtype=float)
    factors = {
        sample: 1.0
        if sample == ref
        else _tmm_factor_vs_reference(
            x[sample].to_numpy(dtype=float), ref_counts, logratio_trim, sum_trim
        )
        for sample in x.columns
    }
    f = pd.Series(factors, index=x.columns, dtype=float)
    geometric_mean = float(np.exp(np.log(f).mean()))
    return f / geometric_mean  # now the geometric mean is 1


def tmm(counts: pd.DataFrame) -> pd.DataFrame:
    """Divide every sample (column) by its TMM scale factor."""
    factors = tmm_scale_factors(counts)
    return pd.DataFrame(
        counts.to_numpy() / factors.to_numpy()[None, :],
        index=counts.index,
        columns=counts.columns,
    )


def normalize_counts(counts: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """The paper's normalization: cf_norm followed by TMM (main_script.R, line 43).

    Args:
        counts: raw counts, genes in rows and samples in columns, at least two samples.
        factors: cleaned factor table, normally from load_factors().

    Returns:
        Normalized table with the genes shared with the factor table, in its order.

    LIMITATION: both steps use the other samples in `counts` (a median for cf_norm, a reference
    sample for TMM), so a sample's result depends on which samples are normalized together.

    Raises:
        TypeError: counts is not a DataFrame.
        ValueError: fewer than two samples, repeated gene or sample names, non-numeric,
            missing or negative counts, or a factor table without Intrinsic/Extrinsic columns.
    """
    if not isinstance(counts, pd.DataFrame):
        raise TypeError("counts must be a pandas DataFrame (genes x samples)")
    if counts.shape[1] < 2:
        raise ValueError("at least 2 samples are needed to normalize a batch")
    if not counts.index.is_unique:
        raise ValueError("gene names must be unique")
    if not counts.columns.is_unique:
        raise ValueError("sample names must be unique")
    if not all(pd.api.types.is_numeric_dtype(counts[c]) for c in counts.columns):
        raise ValueError("counts must contain numbers only")
    if counts.isna().any().any():
        raise ValueError("counts must not contain missing values")
    if (counts < 0).any().any():
        raise ValueError("counts must not be negative")
    missing = [c for c in FACTOR_COLUMNS if c not in factors.columns]
    if missing:
        raise ValueError(f"factors lack the column(s) {missing}")

    return tmm(cf_norm(counts, factors))


def cpm(counts: pd.DataFrame) -> pd.DataFrame:
    """Counts per million: every sample (column) divided by its total, times 1,000,000.

    Raises:
        ValueError: missing or negative counts, or a sample whose total is zero.
    """
    if counts.isna().any().any():
        raise ValueError("counts must not contain missing values")
    if (counts < 0).any().any():
        raise ValueError("counts must not be negative")

    totals = counts.sum()
    empty = totals.index[totals == 0]

    if len(empty) > 0:
        raise ValueError(f"samples with a total of zero: {list(empty)}")

    return counts / totals * 1e6
