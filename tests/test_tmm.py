import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pdac_cfrna.config import load_env
from pdac_cfrna.normalization import (
    cf_norm,
    load_factors,
    tmm,
    tmm_reference_sample,
    tmm_scale_factors,
)

load_env()


def _random_counts(seed: int = 0) -> pd.DataFrame:
    """500 genes x 6 samples (a to f), similar composition in every sample."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.poisson(30, (500, 6)) + 1, columns=list("abcdef"), dtype=float)


def test_factors_have_the_sample_names_are_positive_and_have_geometric_mean_one():
    counts = _random_counts()
    f = tmm_scale_factors(counts)
    assert list(f.index) == list(counts.columns)
    assert (f > 0).all()
    assert np.exp(np.log(f).mean()) == pytest.approx(1.0)


def test_identical_samples_all_get_a_factor_of_one():
    column = _random_counts()["a"]
    same = pd.concat([column] * 4, axis=1, keys=list("wxyz"))
    np.testing.assert_allclose(tmm_scale_factors(same).to_numpy(), 1.0)


def test_a_gene_that_is_zero_everywhere_changes_nothing():
    counts = _random_counts()
    with_zero_gene = counts.copy()
    with_zero_gene.loc["all_zero"] = 0
    pd.testing.assert_series_equal(tmm_scale_factors(counts), tmm_scale_factors(with_zero_gene))


def test_reordering_the_samples_reorders_the_factors_the_same_way():
    counts = _random_counts()
    order = list("fcadbe")
    original = tmm_scale_factors(counts)
    shuffled = tmm_scale_factors(counts[order])
    np.testing.assert_allclose(shuffled.to_numpy(), original[order].to_numpy())


def test_a_sample_dominated_by_a_few_genes_gets_a_lower_factor():
    counts = _random_counts()
    counts.iloc[:50, 0] *= 6  # 50 of 500 genes inflated in sample "a"
    f = tmm_scale_factors(counts)
    assert f["a"] < f.drop("a").min()


def test_a_single_sample_gets_a_factor_of_one():
    f = tmm_scale_factors(_random_counts()[["a"]])
    np.testing.assert_allclose(f.to_numpy(), [1.0])


def test_reference_sample_is_the_one_closest_to_the_average_75th_percentile():
    # 75th percentile of each sample's proportions: A 0.30, B 0.40, C 0.45 (average 0.383).
    counts = pd.DataFrame(
        {"A": [20, 20, 30, 30], "B": [10, 10, 40, 40], "C": [5, 5, 45, 45]},
        index=["g1", "g2", "g3", "g4"],
    )
    assert tmm_reference_sample(counts) == "B"


def test_tmm_divides_each_sample_by_its_factor_and_keeps_shape_and_labels():
    counts = _random_counts()
    factors = tmm_scale_factors(counts)
    out = tmm(counts)
    assert out.shape == counts.shape
    assert list(out.index) == list(counts.index)
    assert list(out.columns) == list(counts.columns)
    for sample in counts.columns:
        np.testing.assert_allclose(
            out[sample].to_numpy() * factors[sample], counts[sample].to_numpy()
        )


def test_inputs_are_not_modified():
    counts = _random_counts()
    before = counts.copy()
    tmm(counts)
    pd.testing.assert_frame_equal(counts, before)


def test_negative_or_missing_counts_raise():
    negative = _random_counts()
    negative.iloc[0, 0] = -1
    missing = _random_counts()
    missing.iloc[0, 0] = np.nan
    for bad in (negative, missing):
        with pytest.raises(ValueError, match="non-negative"):
            tmm_scale_factors(bad)


def test_a_sample_without_any_counts_raises_and_is_named():
    counts = _random_counts()
    counts["c"] = 0
    with pytest.raises(ValueError, match="c"):
        tmm_scale_factors(counts)


def test_a_matrix_of_zeros_raises():
    with pytest.raises(ValueError, match="no gene"):
        tmm_scale_factors(pd.DataFrame(np.zeros((5, 3)), columns=list("abc")))


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("PDAC_DATA_ROOT"), reason="needs the real data folder")
def test_real_data_reference_sample_and_factors():
    root = Path(os.environ["PDAC_DATA_ROOT"])
    raw = pd.read_csv(root / "pdac_genecount.csv", index_col=0)
    normalized = cf_norm(raw, load_factors(root / "updated_intrinsic.csv"))
    assert tmm_reference_sample(normalized) == "PCX122"
    f = tmm_scale_factors(normalized)
    first_five = [0.8854, 0.8608, 0.9144, 0.9036, 0.9257]
    np.testing.assert_allclose(f.iloc[:5].to_numpy(), first_five, atol=1e-4)
    assert f.min() == pytest.approx(0.5397, abs=1e-4)
    assert f.max() == pytest.approx(1.8217, abs=1e-4)
