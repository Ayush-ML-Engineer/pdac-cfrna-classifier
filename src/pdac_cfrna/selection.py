"""Differential-expression gene selection (Moore et al. 2025, Methods: "DE gene discovery").

Every function here is "pure": it takes tables in and returns tables out. None of them
reads files or knows about cohorts, so the same code can later run inside each CV fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def filter_variance(X: pd.DataFrame, min_variance: float = 1.0) -> pd.DataFrame:
    """Keep genes (columns) whose SAMPLE variance (n-1 divisor) is > min_variance.

    Column order is preserved. If no gene passes, an empty-column table is returned.
    """
    variances = X.var(axis=0, ddof=1)
    return X.loc[:, variances > min_variance]


def wilcoxon_per_gene(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Two-sided Wilcoxon rank-sum (Mann-Whitney U) test per gene: class 1 vs class 0.

    Returns a table indexed by gene with columns:
      p_value   raw p-value
      auc       P(random class-1 value > random class-0 value); > 0.5 means higher in class 1
      direction "up", "down" or "none" (auc exactly 0.5)
    """
    _check_aligned(X, y)
    pos = X[(y == 1).to_numpy()].to_numpy(dtype=float)
    neg = X[(y == 0).to_numpy()].to_numpy(dtype=float)

    res = stats.mannwhitneyu(pos, neg, axis=0, alternative="two-sided", use_continuity=True)
    p_values = np.nan_to_num(res.pvalue, nan=1.0)  # all-tied gene: no evidence, p = 1
    auc = res.statistic / (len(pos) * len(neg))

    direction = np.where(auc > 0.5, "up", np.where(auc < 0.5, "down", "none"))
    return pd.DataFrame({"p_value": p_values, "auc": auc, "direction": direction}, index=X.columns)


def adjust_pvalues(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg false-discovery-rate adjustment. Keeps the input order and index."""
    adjusted = stats.false_discovery_control(p_values.to_numpy(dtype=float), method="bh")
    return pd.Series(adjusted, index=p_values.index, name="q_value")


def select_de_genes(
    X: pd.DataFrame,
    y: pd.Series,
    min_variance: float = 1.0,
    alpha: float = 0.05,
    log_before_variance: bool = False,
) -> pd.DataFrame:
    """Variance filter -> Wilcoxon per gene -> BH correction -> flag q <= alpha.

    Returns ALL genes that survived the variance filter (not only the significant ones),
    sorted by q_value, with columns p_value, q_value, auc, direction, significant.
    """
    _check_aligned(X, y)

    if log_before_variance:
        variance_input = pd.DataFrame(np.log1p(X), index=X.index, columns=X.columns)
    else:
        variance_input = X

    kept = filter_variance(variance_input, min_variance).columns
    X = X[kept]

    table = wilcoxon_per_gene(X, y)
    table["q_value"] = adjust_pvalues(table["p_value"])
    table["significant"] = table["q_value"] < alpha
    table = table[["p_value", "q_value", "auc", "direction", "significant"]]
    return table.sort_values("q_value", kind="stable")


def _check_aligned(X: pd.DataFrame, y: pd.Series) -> None:
    if not X.index.equals(y.index):
        raise ValueError("X and y must have identical sample indexes in the same order")
    if not {0, 1} <= set(y.unique()):
        raise ValueError("y must contain both classes 0 and 1")
