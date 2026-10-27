import numpy as np
import pandas as pd
import pytest

from pdac_cfrna.selection import (
    adjust_pvalues,
    filter_variance,
    select_de_genes,
    wilcoxon_per_gene,
)


# ---------- Chunk A: variance filter ----------
def _variance_table() -> pd.DataFrame:
    # Sample variance (divide by n-1), worked by hand:
    #   constant [3,3,3] -> 0 | at_cutoff [1,2,3] -> 2/(3-1) = 1.0 exactly
    #   between [0,1.1,2.2] -> 1.21 (population variance would be 0.81)
    #   high [0,10,20] -> 100 | low [0,.5,1] -> 0.25
    return pd.DataFrame(
        {
            "constant": [3.0, 3.0, 3.0],
            "at_cutoff": [1.0, 2.0, 3.0],
            "between": [0.0, 1.1, 2.2],
            "high": [0.0, 10.0, 20.0],
            "low": [0.0, 0.5, 1.0],
        },
        index=["s1", "s2", "s3"],
    )


def test_filter_variance_keeps_only_genes_strictly_above_the_cutoff():
    out = filter_variance(_variance_table(), min_variance=1.0)
    assert list(out.columns) == ["between", "high"]  # at_cutoff (variance exactly 1) is removed


def test_filter_variance_uses_sample_variance_not_population():
    # Sample variance 1.21 passes; population variance 0.81 would wrongly remove it.
    assert "between" in filter_variance(_variance_table(), min_variance=1.0).columns


def test_filter_variance_keeps_original_column_order_and_samples():
    out = filter_variance(_variance_table(), min_variance=0.1)
    assert list(out.columns) == ["at_cutoff", "between", "high", "low"]  # only "constant" is gone
    assert list(out.index) == ["s1", "s2", "s3"]


def test_filter_variance_returns_empty_table_when_all_genes_fail():
    assert filter_variance(_variance_table(), min_variance=1000.0).shape == (3, 0)


# ---------- Chunk B: Wilcoxon per gene ----------
def _separation_table():
    # 5 class-1 and 5 class-0 samples
    y = pd.Series([1] * 5 + [0] * 5, index=[f"s{i}" for i in range(10)])
    X = pd.DataFrame(
        {
            "perfect_up": [10, 11, 12, 13, 14, 1, 2, 3, 4, 5],
            "perfect_down": [1, 2, 3, 4, 5, 10, 11, 12, 13, 14],
            "noise": [1, 6, 2, 7, 3, 4, 8, 5, 9, 10],
        },
        index=y.index,
        dtype=float,
    )
    return X, y


def test_wilcoxon_perfect_separation_has_smaller_p_than_noise():
    X, y = _separation_table()
    res = wilcoxon_per_gene(X, y)
    assert res.loc["perfect_up", "p_value"] < 0.05
    assert res.loc["noise", "p_value"] > 0.05
    assert res.loc["perfect_up", "p_value"] == pytest.approx(res.loc["perfect_down", "p_value"])


def test_wilcoxon_direction_and_auc():
    X, y = _separation_table()
    res = wilcoxon_per_gene(X, y)
    assert res.loc["perfect_up", "direction"] == "up"
    assert res.loc["perfect_up", "auc"] == pytest.approx(1.0)
    assert res.loc["perfect_down", "direction"] == "down"
    assert res.loc["perfect_down", "auc"] == pytest.approx(0.0)


def test_wilcoxon_is_unchanged_by_log_transform():
    X, y = _separation_table()
    a = wilcoxon_per_gene(X, y)["p_value"]
    b = wilcoxon_per_gene(np.log1p(X), y)["p_value"]
    pd.testing.assert_series_equal(a, b)


def test_wilcoxon_all_tied_gene_gets_p_of_one():
    X, y = _separation_table()
    X["flat"] = 7.0
    assert wilcoxon_per_gene(X, y).loc["flat", "p_value"] == 1.0


def test_wilcoxon_rejects_misaligned_inputs():
    X, y = _separation_table()
    with pytest.raises(ValueError, match="identical sample indexes"):
        wilcoxon_per_gene(X, y.iloc[::-1])


# ---------- Chunk C: Benjamini-Hochberg ----------
def test_bh_hand_worked_example_and_order_preserved():
    # sorted p: .001 .01 .03 .5 ; p*m/rank: .004 .02 .04 .5
    p = pd.Series([0.03, 0.001, 0.5, 0.01], index=list("abcd"))
    q = adjust_pvalues(p)
    assert list(q.index) == list("abcd")
    assert q.tolist() == pytest.approx([0.04, 0.004, 0.5, 0.02])


def test_bh_enforces_monotonicity():
    # p*m/rank = .04, .04, .04, .04 for sorted .01 .02 .03 .04
    q = adjust_pvalues(pd.Series([0.01, 0.02, 0.03, 0.04]))
    assert q.tolist() == pytest.approx([0.04] * 4)


def test_bh_never_smaller_than_raw_and_capped_at_one():
    p = pd.Series([0.2, 0.9, 0.95, 0.99])
    q = adjust_pvalues(p)
    assert (q >= p).all()
    assert (q <= 1.0).all()


# ---------- Chunk D: full selection table ----------
def _selection_inputs():
    X, y = _separation_table()
    X["constant"] = 3.0  # removed by the variance filter
    return X, y


def test_select_de_genes_table_structure():
    X, y = _selection_inputs()
    out = select_de_genes(X, y, min_variance=1.0, alpha=0.05)
    assert list(out.columns) == ["p_value", "q_value", "auc", "direction", "significant"]
    assert set(out.index) == {"perfect_up", "perfect_down", "noise"}  # 'constant' filtered out
    assert out["q_value"].is_monotonic_increasing  # sorted by q_value


def test_select_de_genes_flags_significance_on_adjusted_p():
    X, y = _selection_inputs()
    out = select_de_genes(X, y, alpha=0.05)
    assert out.loc["perfect_up", "significant"]
    assert not out.loc["noise", "significant"]
    assert (out["significant"] == (out["q_value"] < 0.05)).all()


def test_select_de_genes_with_no_surviving_genes_returns_empty_table():
    X, y = _selection_inputs()
    out = select_de_genes(X, y, min_variance=1e9)
    assert out.empty


def test_a_gene_whose_q_value_equals_alpha_is_not_significant():
    X, y = _selection_inputs()
    q = select_de_genes(X, y).loc["perfect_up", "q_value"]
    out = select_de_genes(X, y, alpha=q)  # alpha exactly equal to that gene's q_value
    assert not out.loc["perfect_up", "significant"]
    assert select_de_genes(X, y, alpha=q * 1.0001).loc["perfect_up", "significant"]
