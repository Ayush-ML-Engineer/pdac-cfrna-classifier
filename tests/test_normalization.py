import numpy as np
import pandas as pd
import pytest

from pdac_cfrna.normalization import cf_norm


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


# Hand-worked: colSums(Ci) = [34, 30, 42] (median 34), colSums(Cb) = [66, 70, 58] (median 66).
# e.g. s2/g1 = 20 * (0.9 / (30/34) + 0.1 / (70/66)) = 22.2857
EXPECTED = pd.DataFrame(
    [[10.0, 22.2857, 25.2709], [40.0, 10.3810, 19.4745], [50.0, 67.3333, 55.2545]],
    index=["g1", "g2", "g3"],
    columns=["s1", "s2", "s3"],
)


def test_matches_the_hand_worked_output():
    pd.testing.assert_frame_equal(cf_norm(_counts(), _factors()), EXPECTED, atol=1e-4, rtol=0)


def test_every_column_sums_to_the_two_medians_added_up():
    out = cf_norm(_counts(), _factors())
    np.testing.assert_allclose(out.sum().to_numpy(), [100, 100, 100])


def test_the_median_sample_is_returned_unchanged():
    # s1 has the median intrinsic total (34) and the median batch total (66): both factors are 1.
    out = cf_norm(_counts(), _factors())
    np.testing.assert_allclose(out["s1"].to_numpy(), _counts()["s1"].to_numpy())


def test_a_batch_of_one_sample_is_returned_unchanged():
    one = _counts()[["s2"]]
    np.testing.assert_allclose(cf_norm(one, _factors()).to_numpy(), one.to_numpy())


def test_output_has_the_shared_genes_in_factor_table_order():
    counts = _counts().rename(index={"g2": "extra_in_counts"})
    factors = _factors().iloc[::-1]  # g3, g2, g1
    out = cf_norm(counts, factors)
    assert list(out.index) == ["g3", "g1"]
    assert list(out.columns) == ["s1", "s2", "s3"]


def test_inputs_are_not_modified():
    counts, factors = _counts(), _factors()
    counts_before, factors_before = counts.copy(), factors.copy()
    cf_norm(counts, factors)
    pd.testing.assert_frame_equal(counts, counts_before)
    pd.testing.assert_frame_equal(factors, factors_before)


def test_no_shared_genes_raises():
    with pytest.raises(ValueError, match="no genes"):
        cf_norm(_counts(), _factors().rename(index=lambda g: g + "_x"))


def test_a_sample_with_all_zero_counts_raises_and_is_named():
    counts = _counts()
    counts["s2"] = 0
    with pytest.raises(ValueError, match="s2"):
        cf_norm(counts, _factors())


def test_a_sample_whose_counts_fall_only_on_zero_weight_genes_raises():
    factors = _factors()
    factors.loc["g3"] = [0.0, 1.0]  # g3 has no intrinsic part
    counts = _counts()
    counts["s2"] = [0, 0, 70]  # s2 has counts only in g3
    with pytest.raises(ValueError, match="s2"):
        cf_norm(counts, factors)
