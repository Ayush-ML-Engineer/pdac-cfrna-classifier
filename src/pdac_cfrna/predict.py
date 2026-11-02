"""Scoring raw count tables with a saved model bundle."""

from __future__ import annotations

import pandas as pd

from pdac_cfrna.bundle import ModelBundle
from pdac_cfrna.normalization import apply_normalizer


def predict_scores(counts: pd.DataFrame, bundle: ModelBundle) -> pd.Series:
    """PDAC score (probability between 0 and 1) for every sample of a raw counts table.

    Args:
        counts: raw counts, genes in rows and samples in columns. Every sample is scored on its
            own: the result does not depend on the other samples, and one sample is enough.
        bundle: a trained model bundle.

    Returns:
        A Series named "pdac_score", indexed by sample.

    Raises:
        ValueError: the table looks transposed, genes the model needs are missing, or the counts
            are invalid (negative, missing, repeated names, a sample with no counts, ...).
    """
    needed = bundle.normalizer.factors.index
    transposed = (
        isinstance(counts, pd.DataFrame)
        and not needed.isin(counts.index).any()
        and needed.isin(counts.columns).any()
    )
    if transposed:
        raise ValueError(
            "the table looks transposed: gene names were found in the columns, but genes "
            "must be in the rows and samples in the columns"
        )

    normalized = apply_normalizer(counts, bundle.normalizer)
    missing = [g for g in bundle.genes if g not in normalized.index]
    if missing:
        raise ValueError(f"{len(missing)} model gene(s) are missing, e.g. {missing[:5]}")

    X = normalized.loc[bundle.genes].T
    column = list(bundle.pipeline.classes_).index(1)
    scores = bundle.pipeline.predict_proba(X)[:, column]
    return pd.Series(scores, index=X.index, name="pdac_score")
