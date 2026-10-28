"""Constrained two-factor NMF used by the paper to learn intrinsic and batch gene profiles.

Port of intrinsic_factors.R. Tables are genes x samples: Y (n x p) is approximated by W @ H,
where W (n x f) holds one profile per factor and H (f x p) holds each sample's mixture weights,
which sum to 1 for every sample.
"""

from __future__ import annotations

import numpy as np


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
