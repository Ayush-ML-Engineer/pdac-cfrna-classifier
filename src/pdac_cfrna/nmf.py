"""Constrained two-factor NMF used by the paper to learn intrinsic and batch gene profiles.

Port of intrinsic_factors.R. Tables are genes x samples: Y (n x p) is approximated by W @ H,
where W (n x f) holds one profile per factor and H (f x p) holds each sample's mixture weights,
which sum to 1 for every sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pdac_cfrna.normalization import cpm

PLATELET_GENES = ["PF4", "PPBP", "MT-ND4", "MT-CO3", "TAGLN2", "TM2D2", "CA2", "IFRD1"]
INTRINSIC_GENES = ["ALB", "APOB", "C3", "CP", "EEF1A1", "RPL26", "RPS21", "CTSB"]


def _update_h(H: np.ndarray, W: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """One multiplicative update of H using the new W (rows 2..f), then restore the constraint."""
    with np.errstate(divide="ignore", invalid="ignore"):
        H_new = H.copy()
        H_new[1:] = H[1:] * (W.T @ Y)[1:] / (W.T @ W @ H)[1:]
    H_new[np.isnan(H_new)] = 0
    total = H_new[1:].sum(axis=0)  # per sample: weight carried by rows 2..f
    too_big = total > 0.999
    H_new[1:, too_big] = 0.999 * H_new[1:, too_big] / total[too_big]
    H_new[0] = 1 - H_new[1:].sum(axis=0)  # row 1 is the remainder, so each column sums to 1
    return H_new


def nmf(
    Y: np.ndarray,
    n_factors: int = 2,
    reps: int = 5,
    seed: int = 0,
    max_iter: int = 50,
    tol: float = 0.002,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Factorize Y (genes x samples, non-negative) as W @ H with every column of H summing to 1.

    Runs `reps` random restarts and keeps the one with the smallest reconstruction error
    ||Y - W H|| / ||Y|| (Frobenius norm). Each restart stops when the relative change of W and H
    falls below `tol`, or after max_iter + 1 iterations.

    Returns:
        W (genes x factors), H (factors x samples), reconstruction error of the best restart.

    Raises:
        ValueError: Y is not a 2-D table of finite, non-negative numbers with some positive
            value, or n_factors < 2, reps < 1, max_iter < 0 or tol <= 0.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must be a 2-D table (genes x samples)")
    if not np.isfinite(Y).all() or (Y < 0).any():
        raise ValueError("Y must contain finite, non-negative numbers")
    if not (Y > 0).any():
        raise ValueError("Y must contain at least one positive value")
    if n_factors < 2 or reps < 1 or max_iter < 0 or tol <= 0:
        raise ValueError("need n_factors >= 2, reps >= 1, max_iter >= 0 and tol > 0")

    n, p = Y.shape
    rng = np.random.default_rng(seed)
    best: tuple[float, np.ndarray, np.ndarray] | None = None

    for _ in range(reps):
        W = rng.random((n, n_factors))
        W = 1e6 * W / W.sum(axis=0)  # every profile starts with a total of one million
        H = np.full((n_factors, p), 0.5)
        iteration = 0
        while True:
            iteration += 1
            with np.errstate(divide="ignore", invalid="ignore"):
                W_new = W * (Y @ H.T) / (W @ H @ H.T)
            W_new[np.isnan(W_new)] = 0  # genes that are zero everywhere give 0 / 0
            H_new = _update_h(H, W_new, Y)

            change = np.abs(W - W_new).sum() / W.sum() + np.abs(H - H_new).sum() / H.sum()
            W, H = W_new, H_new
            if change < tol or iteration > max_iter:
                break

        recon = float(np.linalg.norm(Y - W @ H) / np.linalg.norm(Y))
        if best is None or recon < best[0]:
            best = (recon, W, H)

    assert best is not None  # reps >= 1
    return best[1], best[2], best[0]


def fit_reference_factors(
    counts: pd.DataFrame,
    platelet_genes: list[str] = PLATELET_GENES,
    intrinsic_genes: list[str] = INTRINSIC_GENES,
    reps: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Learn each gene's intrinsic and extrinsic (batch) fraction from healthy reference samples.

    Runs the NMF on the CPM-normalized counts (genes x samples), turns W into per-gene fractions
    and names the two factors: the one carrying the smaller share of the platelet marker genes is
    "Intrinsic". Genes that are zero in every sample keep NaN rows, as in the paper's file.

    Returns:
        Table indexed by gene with columns "Intrinsic" and "Extrinsic" (rows sum to 1 or are NaN).

    Raises:
        ValueError: a marker gene is missing from the counts, the platelet markers do not
            separate the two factors, or the intrinsic markers contradict the platelet markers.
    """
    missing = [g for g in [*platelet_genes, *intrinsic_genes] if g not in counts.index]
    if missing:
        raise ValueError(f"marker genes missing from the counts: {missing}")

    W, _, _ = nmf(cpm(counts).to_numpy(), n_factors=2, reps=reps, seed=seed)

    with np.errstate(divide="ignore", invalid="ignore"):
        fractions = pd.DataFrame(W / W.sum(axis=1, keepdims=True), index=counts.index)

    platelet_share = fractions.loc[platelet_genes].sum()  # NaN rows count as 0
    if np.isclose(platelet_share.iloc[0], platelet_share.iloc[1]):
        raise ValueError("the platelet marker genes do not separate the two factors")

    fractions = fractions.iloc[:, np.argsort(platelet_share.to_numpy())]
    fractions.columns = pd.Index(["Intrinsic", "Extrinsic"])

    intrinsic_share = fractions.loc[intrinsic_genes].sum()
    if intrinsic_share["Intrinsic"] <= intrinsic_share["Extrinsic"]:
        raise ValueError("cannot tell which factor is which: the marker gene lists disagree")

    return fractions
