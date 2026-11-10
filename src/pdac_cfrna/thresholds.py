"""Choosing a decision threshold for the PDAC score, and reporting what it does.

The classifier outputs a score between 0 and 1. A threshold turns it into a call: score >= threshold
means "PDAC". This module

* tabulates sensitivity, specificity, precision and F-scores over a grid of thresholds,
* picks one threshold by a stated rule (F-beta, a sensitivity target, or Youden),
* counts the confusion matrix with Wilson confidence intervals,
* estimates by cross-validation how a threshold chosen on some samples performs on others, and
* can produce out-of-fold scores on the training cohort (gene selection repeated in every fold).

A threshold is only honestly evaluated on samples that did not take part in choosing it.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from pdac_cfrna.experiment import ExperimentConfig
from pdac_cfrna.model_config import ModelConfig
from pdac_cfrna.models import build_paper_pipeline
from pdac_cfrna.selection import select_de_genes

GRID: np.ndarray = np.round(np.arange(0.01, 1.0, 0.01), 2)
Rule = Literal["fbeta", "sensitivity", "youden"]


def oof_scores(
    X: pd.DataFrame,
    y: pd.Series,
    exp: ExperimentConfig,
    model: ModelConfig,
    n_splits: int = 5,
    n_repeats: int = 5,
    fixed_genes: list[str] | None = None,
) -> tuple[pd.Series, list[int]]:
    """Out-of-fold PDAC scores for the rows of X, averaged over `n_repeats` repeated CV rounds.

    By default the genes are selected again in every fold from the fold's training rows only
    (variance filter, Wilcoxon, BH), then the pipeline (log1p, SMOTE, random forest) is fitted on
    those rows and scores the held-out rows. Each sample is therefore scored by models that never
    saw it, and the gene selection never saw its label. With few PDAC samples the selected genes
    change a lot between folds, and a fold can select no gene at all; such a fold is skipped (its
    gene count is recorded as 0) and its samples are scored by the other repeats.

    With `fixed_genes` the same genes are used in every fold. That is less noisy, but those genes
    were chosen using the held-out labels too, so the scores come out too good: use it only to
    see how much that matters.

    One thing is not repeated inside the folds: the frozen normalization constants. They are
    fitted without labels on the reference cohorts, so they carry no label information.

    Returns:
        The averaged score of every sample (Series "oof_score", indexed like X) and the number of
        genes used in each fold (0 for a skipped fold).

    Raises:
        ValueError: X and y are not aligned, or some sample was never scored.
    """
    if not X.index.equals(y.index):
        raise ValueError("X and y must have identical sample indexes in the same order")
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=exp.seed)
    total = np.zeros(len(X))
    n_scored = np.zeros(len(X))
    n_genes: list[int] = []
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        if fixed_genes is not None:
            genes = list(fixed_genes)
        else:
            table = select_de_genes(X_tr, y_tr, exp.variance_cutoff, exp.alpha, exp.log_variance)
            genes = list(table.index[table["significant"]])
        n_genes.append(len(genes))
        if not genes:
            continue
        pipe = build_paper_pipeline(int(y_tr.sum()), exp.seed + i, model, exp.n_synthetic_pdac)
        pipe.fit(X_tr[genes], y_tr)
        column = list(pipe.classes_).index(1)
        total[test_idx] += pipe.predict_proba(X.iloc[test_idx][genes])[:, column]
        n_scored[test_idx] += 1
    if (n_scored == 0).any():
        raise ValueError(f"{int((n_scored == 0).sum())} sample(s) were never scored")
    return pd.Series(total / n_scored, index=X.index, name="oof_score"), n_genes


def confusion_counts(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, int]:
    """tp, fn, tn, fp when every score >= threshold is called PDAC."""
    called = score >= threshold
    pos = y == 1
    return {
        "tp": int((pos & called).sum()),
        "fn": int((pos & ~called).sum()),
        "tn": int((~pos & ~called).sum()),
        "fp": int((~pos & called).sum()),
    }


def fbeta(precision: np.ndarray, recall: np.ndarray, beta: float) -> np.ndarray:
    """F-beta score (beta > 1 weights recall more than precision); 0 where it is undefined."""
    b2 = beta**2
    denom = b2 * precision + recall
    out = np.zeros_like(denom, dtype=float)
    np.divide((1 + b2) * precision * recall, denom, out=out, where=denom > 0)
    return out


def threshold_table(y: np.ndarray, score: np.ndarray, grid: np.ndarray = GRID) -> pd.DataFrame:
    """Confusion counts and rates for every threshold of the grid."""
    rows = [{"threshold": float(t), **confusion_counts(y, score, float(t))} for t in grid]
    t = pd.DataFrame(rows)
    pos, neg = t["tp"] + t["fn"], t["tn"] + t["fp"]
    called = t["tp"] + t["fp"]
    t["sensitivity"] = t["tp"] / pos
    t["specificity"] = t["tn"] / neg
    t["precision"] = (t["tp"] / called).where(called > 0, 0.0)
    t["f1"] = fbeta(t["precision"].to_numpy(), t["sensitivity"].to_numpy(), 1.0)
    t["f2"] = fbeta(t["precision"].to_numpy(), t["sensitivity"].to_numpy(), 2.0)
    t["youden"] = t["sensitivity"] + t["specificity"] - 1
    return t


def _middle_of_best(table: pd.DataFrame, values: np.ndarray) -> float:
    """The middle of the thresholds that tie for the best value (a plateau is not one point)."""
    best = np.flatnonzero(np.isclose(values, values.max(), atol=1e-12))
    return float(np.median(table["threshold"].to_numpy()[best]))


def choose_threshold(
    table: pd.DataFrame,
    rule: Rule = "fbeta",
    beta: float = 2.0,
    target_sensitivity: float = 0.90,
) -> float:
    """Pick one threshold from a threshold table.

    Rules:
        fbeta: the threshold with the highest F-beta score (beta=2 weights recall twice as much as
            precision, so missing a cancer costs more than a false alarm). A tie between
            neighbouring thresholds is resolved by taking the middle one.
        sensitivity: the highest threshold that still catches at least `target_sensitivity` of
            the PDAC samples (that is the one with the most specificity at that sensitivity).
        youden: the highest sensitivity + specificity - 1, treating both errors as equally bad.

    Raises:
        ValueError: unknown rule, or no threshold reaches the target sensitivity.
    """
    if rule == "fbeta":
        values = fbeta(table["precision"].to_numpy(), table["sensitivity"].to_numpy(), beta)
        return _middle_of_best(table, values)
    if rule == "youden":
        return _middle_of_best(table, table["youden"].to_numpy())
    if rule == "sensitivity":
        ok = table[table["sensitivity"] >= target_sensitivity]
        if ok.empty:
            raise ValueError(f"no threshold reaches sensitivity {target_sensitivity:.2f}")
        return float(ok["threshold"].max())
    raise ValueError(f"unknown rule {rule!r}; use fbeta, sensitivity or youden")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson confidence interval for a proportion k/n (NaN, NaN if n is 0)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(counts: dict[str, int]) -> dict[str, tuple[float, float, float]]:
    """Sensitivity, specificity, PPV and NPV as (value, CI low, CI high) from confusion counts."""
    pairs = {
        "sensitivity": (counts["tp"], counts["tp"] + counts["fn"]),
        "specificity": (counts["tn"], counts["tn"] + counts["fp"]),
        "ppv": (counts["tp"], counts["tp"] + counts["fp"]),
        "npv": (counts["tn"], counts["tn"] + counts["fn"]),
    }
    out: dict[str, tuple[float, float, float]] = {}
    for name, (k, n) in pairs.items():
        low, high = wilson_interval(k, n)
        out[name] = (k / n if n else float("nan"), low, high)
    return out


def cv_threshold_performance(
    y: np.ndarray,
    score: np.ndarray,
    rule: Rule = "fbeta",
    beta: float = 2.0,
    target_sensitivity: float = 0.90,
    n_splits: int = 5,
    n_repeats: int = 20,
    seed: int = 0,
) -> dict[str, object]:
    """How well does "choose a threshold on some samples" work on samples it was not chosen on?

    The scores are split into folds. In each fold the threshold is chosen on the other folds and
    applied to the held-out fold; the held-out confusion counts of the folds are pooled, and the
    whole procedure is repeated `n_repeats` times with new splits.

    Returns:
        thresholds: every threshold chosen (one per fold and repeat).
        sensitivity, specificity, ppv: mean over the repeats of the pooled held-out rate.
        counts: mean pooled held-out tp, fn, tn, fp over the repeats.
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    chosen: list[float] = []
    pooled: list[dict[str, int]] = []
    current = {"tp": 0, "fn": 0, "tn": 0, "fp": 0}
    for i, (train_idx, test_idx) in enumerate(cv.split(score.reshape(-1, 1), y)):
        table = threshold_table(y[train_idx], score[train_idx])
        try:
            t = choose_threshold(table, rule, beta, target_sensitivity)
        except ValueError:  # the target sensitivity was not reachable on this part
            t = float(GRID.min())
        chosen.append(t)
        for key, value in confusion_counts(y[test_idx], score[test_idx], t).items():
            current[key] += value
        if (i + 1) % n_splits == 0:
            pooled.append(current)
            current = {"tp": 0, "fn": 0, "tn": 0, "fp": 0}
    mean = {k: float(np.mean([p[k] for p in pooled])) for k in pooled[0]}
    called = mean["tp"] + mean["fp"]
    return {
        "thresholds": chosen,
        "counts": mean,
        "sensitivity": mean["tp"] / (mean["tp"] + mean["fn"]),
        "specificity": mean["tn"] / (mean["tn"] + mean["fp"]),
        "ppv": mean["tp"] / called if called else float("nan"),
    }
