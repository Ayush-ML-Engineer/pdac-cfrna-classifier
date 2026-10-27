import numpy as np
import pandas as pd
import pytest

from pdac_cfrna.normalization import cf_norm, normalize_counts, tmm


def _counts() -> pd.DataFrame:
    return pd.DataFrame(
        [[10, 20, 30], [40, 10, 20], [50, 70, 50]],
        index=["g1", "g2", "g3"],
        columns=["s1", "s2", "s3"],
    )


def _factors() -> pd.DataFrame:
    return pd.DataFrame(
        {"Intrinsic": [0.9, 0.5, 0.1], "Extrinsic": [0.1, 0.5, 0.9]},
        index=["g1", "g2", "g3"],
    )


def test_result_is_cf_norm_followed_by_tmm():
    expected = tmm(cf_norm(_counts(), _factors()))
    pd.testing.assert_frame_equal(normalize_counts(_counts(), _factors()), expected)


def test_output_has_the_shared_genes_in_factor_order_and_all_samples():
    counts = _counts().rename(index={"g2": "extra_in_counts"})
    out = normalize_counts(counts, _factors().iloc[::-1])
    assert list(out.index) == ["g3", "g1"]
    assert list(out.columns) == ["s1", "s2", "s3"]


def test_inputs_are_not_modified():
    counts, factors = _counts(), _factors()
    counts_before, factors_before = counts.copy(), factors.copy()
    normalize_counts(counts, factors)
    pd.testing.assert_frame_equal(counts, counts_before)
    pd.testing.assert_frame_equal(factors, factors_before)


def test_a_samples_result_depends_on_which_other_samples_are_in_the_batch():
    # The documented limitation: medians and the TMM reference come from the batch itself.
    both = normalize_counts(_counts()[["s1", "s2"]], _factors())["s1"].to_numpy()
    three = normalize_counts(_counts(), _factors())["s1"].to_numpy()
    assert not np.allclose(both, three)


def test_a_dataframe_is_required():
    with pytest.raises(TypeError, match="DataFrame"):
        normalize_counts(_counts().to_numpy(), _factors())  # type: ignore[arg-type]


def test_at_least_two_samples_are_required():
    with pytest.raises(ValueError, match="2 samples"):
        normalize_counts(_counts()[["s1"]], _factors())


def test_repeated_gene_or_sample_names_are_rejected():
    dup_genes = _counts().rename(index={"g2": "g1"})
    dup_samples = _counts().rename(columns={"s2": "s1"})
    with pytest.raises(ValueError, match="gene names"):
        normalize_counts(dup_genes, _factors())
    with pytest.raises(ValueError, match="sample names"):
        normalize_counts(dup_samples, _factors())


def test_non_numeric_missing_and_negative_counts_are_rejected():
    text = _counts().astype(object)
    text.iloc[0, 0] = "ten"
    missing = _counts().astype(float)
    missing.iloc[0, 0] = np.nan
    negative = _counts()
    negative.iloc[0, 0] = -1
    with pytest.raises(ValueError, match="numbers only"):
        normalize_counts(text, _factors())
    with pytest.raises(ValueError, match="missing"):
        normalize_counts(missing, _factors())
    with pytest.raises(ValueError, match="negative"):
        normalize_counts(negative, _factors())


def test_factors_without_the_required_columns_are_rejected():
    with pytest.raises(ValueError, match="Extrinsic"):
        normalize_counts(_counts(), _factors()[["Intrinsic"]])
