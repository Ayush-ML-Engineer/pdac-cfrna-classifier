import dataclasses

import numpy as np
import pandas as pd
import pytest

from pdac_cfrna.normalization import (
    apply_normalizer,
    cf_norm,
    fit_normalizer,
    normalize_counts,
    tmm_reference_sample,
)


def _batch(seed: int = 0, n_samples: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """200 genes x n_samples of raw counts, and factors for all 200 genes."""
    rng = np.random.default_rng(seed)
    genes = [f"g{i}" for i in range(200)]
    counts = pd.DataFrame(
        rng.poisson(30, (200, n_samples)) + 1,
        index=genes,
        columns=[f"s{seed}_{j}" for j in range(n_samples)],
    )
    fraction = np.random.default_rng(99).uniform(0.05, 0.95, 200)
    factors = pd.DataFrame({"Intrinsic": fraction, "Extrinsic": 1 - fraction}, index=genes)
    return counts, factors


def _hand_example() -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.DataFrame(
        [[10, 20, 30], [40, 10, 20], [50, 70, 50]],
        index=["g1", "g2", "g3"],
        columns=["s1", "s2", "s3"],
    )
    factors = pd.DataFrame(
        {"Intrinsic": [0.9, 0.5, 0.1], "Extrinsic": [0.1, 0.5, 0.9]}, index=["g1", "g2", "g3"]
    )
    return counts, factors


def test_applying_the_constants_to_the_reference_batch_reproduces_normalize_counts():
    counts, factors = _batch()
    constants = fit_normalizer(counts, factors)
    pd.testing.assert_frame_equal(
        apply_normalizer(counts, constants), normalize_counts(counts, factors), rtol=1e-9
    )


def test_a_samples_result_is_the_same_alone_and_inside_a_batch():
    reference, factors = _batch(seed=0)
    constants = fit_normalizer(reference, factors)
    new, _ = _batch(seed=1, n_samples=5)
    together = apply_normalizer(new, constants)
    for sample in new.columns:
        alone = apply_normalizer(new[[sample]], constants)
        np.testing.assert_allclose(
            alone[sample].to_numpy(), together[sample].to_numpy(), atol=1e-12
        )


def test_a_single_new_sample_is_allowed():
    reference, factors = _batch(seed=0)
    new, _ = _batch(seed=1, n_samples=1)
    out = apply_normalizer(new, fit_normalizer(reference, factors))
    assert out.shape == (200, 1)
    assert np.isfinite(out.to_numpy()).all()


def test_the_constants_hold_the_two_medians_of_the_reference_batch():
    counts, factors = _hand_example()  # intrinsic totals 34, 30, 42; batch totals 66, 70, 58
    constants = fit_normalizer(counts, factors)
    assert constants.median_intrinsic == pytest.approx(34.0)
    assert constants.median_batch == pytest.approx(66.0)


def test_the_tmm_reference_is_the_cf_normalized_reference_sample():
    counts, factors = _batch()
    constants = fit_normalizer(counts, factors)
    cf = cf_norm(counts, factors)
    expected = cf[tmm_reference_sample(cf)]
    np.testing.assert_allclose(constants.tmm_reference.to_numpy(), expected.to_numpy())
    assert list(constants.tmm_reference.index) == list(cf.index)


def test_extra_genes_are_ignored_and_the_output_follows_the_constants_gene_order():
    reference, factors = _batch(seed=0)
    constants = fit_normalizer(reference, factors)
    new, _ = _batch(seed=1, n_samples=3)
    extra = pd.DataFrame(np.ones((2, 3)), index=["extra1", "extra2"], columns=new.columns)
    shuffled = pd.concat([new, extra]).sample(frac=1, random_state=0)
    out = apply_normalizer(shuffled, constants)
    assert list(out.index) == list(constants.factors.index)
    pd.testing.assert_frame_equal(out, apply_normalizer(new, constants))


def test_missing_genes_raise_and_say_how_many():
    reference, factors = _batch(seed=0)
    constants = fit_normalizer(reference, factors)
    new, _ = _batch(seed=1, n_samples=3)
    with pytest.raises(ValueError, match="3 of the 200"):
        apply_normalizer(new.drop(index=["g1", "g2", "g3"]), constants)


def test_a_new_sample_without_counts_raises_and_is_named():
    reference, factors = _batch(seed=0)
    constants = fit_normalizer(reference, factors)
    new, _ = _batch(seed=1, n_samples=3)
    new[new.columns[1]] = 0
    with pytest.raises(ValueError, match=str(new.columns[1])):
        apply_normalizer(new, constants)


def test_inputs_and_constants_are_not_modified():
    reference, factors = _batch(seed=0)
    constants = fit_normalizer(reference, factors)
    new, _ = _batch(seed=1, n_samples=3)
    new_before, factors_before = new.copy(), constants.factors.copy()
    reference_before = constants.tmm_reference.copy()
    apply_normalizer(new, constants)
    pd.testing.assert_frame_equal(new, new_before)
    pd.testing.assert_frame_equal(constants.factors, factors_before)
    pd.testing.assert_series_equal(constants.tmm_reference, reference_before)


def test_the_constants_cannot_be_changed_afterwards():
    counts, factors = _batch()
    constants = fit_normalizer(counts, factors)
    with pytest.raises(dataclasses.FrozenInstanceError):
        constants.median_batch = 1.0  # type: ignore[misc]


def test_fitting_does_not_keep_a_view_of_the_input_factors():
    counts, factors = _batch()
    constants = fit_normalizer(counts, factors)
    factors.iloc[:, :] = 0.5  # changing the caller's table later must not change the constants
    assert not (constants.factors == 0.5).all().all()


def test_fitting_needs_two_samples_valid_counts_and_shared_genes():
    counts, factors = _batch()
    negative = counts.copy()
    negative.iloc[0, 0] = -1
    missing = counts.astype(float)
    missing.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="2 samples"):
        fit_normalizer(counts[[counts.columns[0]]], factors)
    with pytest.raises(ValueError, match="negative"):
        fit_normalizer(negative, factors)
    with pytest.raises(ValueError, match="missing"):
        fit_normalizer(missing, factors)
    with pytest.raises(TypeError, match="DataFrame"):
        fit_normalizer(counts.to_numpy(), factors)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Extrinsic"):
        fit_normalizer(counts, factors[["Intrinsic"]])
    with pytest.raises(ValueError, match="share no genes"):
        fit_normalizer(counts, factors.rename(index=lambda g: g + "_x"))
