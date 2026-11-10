"""Tests for threshold selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pdac_cfrna.experiment import ExperimentConfig
from pdac_cfrna.run_config import load_run_config
from pdac_cfrna.thresholds import (
    choose_threshold,
    confusion_counts,
    cv_threshold_performance,
    fbeta,
    oof_scores,
    summarize,
    threshold_table,
    wilson_interval,
)

Y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
S = np.array([0.9, 0.8, 0.4, 0.3, 0.6, 0.35, 0.2, 0.1, 0.1, 0.05])


def test_confusion_counts_uses_greater_or_equal() -> None:
    assert confusion_counts(Y, S, 0.4) == {"tp": 3, "fn": 1, "tn": 5, "fp": 1}
    assert confusion_counts(Y, S, 0.9) == {"tp": 1, "fn": 3, "tn": 6, "fp": 0}


def test_fbeta_known_values_and_zero_when_undefined() -> None:
    f2 = fbeta(np.array([0.5, 0.0]), np.array([1.0, 0.0]), 2.0)
    assert f2[0] == pytest.approx(5 * 0.5 * 1.0 / (4 * 0.5 + 1.0))
    assert f2[1] == 0.0


def test_threshold_table_rates() -> None:
    table = threshold_table(Y, S, np.array([0.4]))
    row = table.iloc[0]
    assert (row["tp"], row["fn"], row["tn"], row["fp"]) == (3, 1, 5, 1)
    assert row["sensitivity"] == pytest.approx(0.75)
    assert row["specificity"] == pytest.approx(5 / 6)
    assert row["precision"] == pytest.approx(0.75)


def test_f2_chooses_a_lower_threshold_than_precision_heavy_rule() -> None:
    table = threshold_table(Y, S)
    t_f2 = choose_threshold(table, "fbeta", beta=2.0)
    t_half = choose_threshold(table, "fbeta", beta=0.5)
    assert t_f2 <= t_half


def test_sensitivity_rule_returns_highest_threshold_meeting_target() -> None:
    table = threshold_table(Y, S)
    t = choose_threshold(table, "sensitivity", target_sensitivity=1.0)
    assert confusion_counts(Y, S, t)["fn"] == 0
    assert confusion_counts(Y, S, t + 0.01)["fn"] > 0


def test_unreachable_sensitivity_and_unknown_rule_raise() -> None:
    table = threshold_table(Y, S, np.array([0.99]))
    with pytest.raises(ValueError, match="no threshold"):
        choose_threshold(table, "sensitivity", target_sensitivity=0.9)
    with pytest.raises(ValueError, match="unknown rule"):
        choose_threshold(table, "magic")  # type: ignore[arg-type]


def test_wilson_interval_contains_estimate_and_handles_empty() -> None:
    low, high = wilson_interval(18, 20)
    assert low < 0.9 < high
    assert all(np.isnan(v) for v in wilson_interval(0, 0))
    assert wilson_interval(20, 20)[1] == 1.0


def test_summarize_rates() -> None:
    out = summarize({"tp": 3, "fn": 1, "tn": 5, "fp": 1})
    assert out["sensitivity"][0] == pytest.approx(0.75)
    assert out["ppv"][0] == pytest.approx(0.75)
    assert out["npv"][0] == pytest.approx(5 / 6)


def test_cv_threshold_performance_on_separable_scores() -> None:
    rng = np.random.default_rng(0)
    y = np.array([1] * 20 + [0] * 60)
    s = np.concatenate([rng.uniform(0.6, 1, 20), rng.uniform(0, 0.4, 60)])
    est = cv_threshold_performance(y, s, "fbeta", n_repeats=3)
    assert est["sensitivity"] == pytest.approx(1.0)
    assert est["specificity"] == pytest.approx(1.0)
    assert len(est["thresholds"]) == 15  # type: ignore[arg-type]


def _toy(n_pos: int = 12, n_neg: int = 36, n_genes: int = 20) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(1)
    y = np.array([1] * n_pos + [0] * n_neg)
    X = rng.normal(loc=6.0, scale=2.0, size=(len(y), n_genes))
    X[y == 1, :5] += 4.0  # five informative genes
    idx = [f"s{i}" for i in range(len(y))]
    return (
        pd.DataFrame(np.abs(X), index=idx, columns=[f"g{i}" for i in range(n_genes)]),
        pd.Series(y, index=idx),
    )


def test_oof_scores_cover_every_sample_and_separate_classes() -> None:
    X, y = _toy()
    exp = ExperimentConfig(name="toy", task="pdac_vs_all")
    scores, n_genes = oof_scores(X, y, exp, load_run_config().model, n_splits=3, n_repeats=2)
    assert scores.index.equals(X.index)
    assert scores.between(0, 1).all()
    assert len(n_genes) == 6
    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_oof_scores_rejects_misaligned_inputs() -> None:
    X, y = _toy()
    exp = ExperimentConfig(name="toy", task="pdac_vs_all")
    with pytest.raises(ValueError, match="identical sample indexes"):
        oof_scores(X, y.iloc[::-1], exp, load_run_config().model)
