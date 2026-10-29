import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pdac_cfrna.config import load_env
from pdac_cfrna.nmf import nmf

load_env()


def _synthetic() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """300 genes x 20 samples that are an exact mixture of two profiles."""
    rng = np.random.default_rng(1)
    w_true = np.column_stack([rng.gamma(2, 50, 300), rng.gamma(2, 50, 300)])
    w_true[:100, 1] *= 0.02  # 100 clearly "intrinsic" genes
    w_true[100:200, 0] *= 0.02  # 100 clearly "batch" genes
    a = np.linspace(0.15, 0.85, 20)
    h_true = np.vstack([a, 1 - a])
    return w_true @ h_true, w_true, h_true


def test_shapes_and_every_column_of_h_sums_to_one():
    Y, _, _ = _synthetic()
    W, H, _ = nmf(Y)
    assert W.shape == (300, 2)
    assert H.shape == (2, 20)
    np.testing.assert_allclose(H.sum(axis=0), 1.0)


def test_first_row_of_h_is_the_remainder_of_the_second():
    Y, _, _ = _synthetic()
    _, H, _ = nmf(Y)
    np.testing.assert_allclose(H[0], 1 - H[1])


def test_w_and_h_are_non_negative():
    Y, _, _ = _synthetic()
    W, H, _ = nmf(Y)
    assert (W >= 0).all() and (H >= 0).all()


def test_exactly_factorizable_data_is_reconstructed_closely():
    Y, _, _ = _synthetic()
    W, H, recon = nmf(Y)
    assert recon < 0.02
    assert recon == pytest.approx(np.linalg.norm(Y - W @ H) / np.linalg.norm(Y))


def test_recovered_gene_fractions_match_the_true_ones():
    Y, w_true, _ = _synthetic()
    W, _, _ = nmf(Y)
    found = W[:, 0] / W.sum(axis=1)
    true = w_true[:, 0] / w_true.sum(axis=1)
    assert abs(np.corrcoef(found, true)[0, 1]) > 0.99  # the two factors may come out swapped


def test_a_gene_that_is_zero_everywhere_gets_a_zero_row_and_no_nan():
    Y, _, _ = _synthetic()
    Y[7] = 0
    W, H, _ = nmf(Y)
    np.testing.assert_allclose(W[7], 0)
    assert not np.isnan(W).any() and not np.isnan(H).any()


def test_same_seed_gives_identical_results_and_another_seed_differs():
    Y, _, _ = _synthetic()
    a = nmf(Y, reps=1, seed=3)[0]
    b = nmf(Y, reps=1, seed=3)[0]
    c = nmf(Y, reps=1, seed=4)[0]
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_more_restarts_never_give_a_worse_error():
    Y, _, _ = _synthetic()
    assert nmf(Y, reps=5, seed=0)[2] <= nmf(Y, reps=1, seed=0)[2]


def test_more_iterations_do_not_increase_the_error():
    Y, _, _ = _synthetic()
    one = nmf(Y, reps=1, max_iter=0, tol=1e-12)[2]  # a single iteration
    many = nmf(Y, reps=1, max_iter=30, tol=1e-12)[2]
    assert many <= one


def test_returned_w_and_h_belong_to_the_returned_error():
    Y, _, _ = _synthetic()
    W, H, recon = nmf(Y, reps=3)
    assert recon == pytest.approx(np.linalg.norm(Y - W @ H) / np.linalg.norm(Y))


def test_input_is_not_modified():
    Y, _, _ = _synthetic()
    before = Y.copy()
    nmf(Y)
    np.testing.assert_array_equal(Y, before)


@pytest.mark.parametrize(
    "bad",
    [
        np.array([1.0, 2.0, 3.0]),  # not 2-D
        np.array([[1.0, -1.0], [2.0, 3.0]]),  # negative
        np.array([[1.0, np.nan], [2.0, 3.0]]),  # missing
        np.array([[1.0, np.inf], [2.0, 3.0]]),  # infinite
        np.zeros((3, 3)),  # nothing positive
    ],
)
def test_bad_input_is_rejected(bad):
    with pytest.raises(ValueError):
        nmf(bad)


@pytest.mark.parametrize("kwargs", [{"n_factors": 1}, {"reps": 0}, {"max_iter": -1}, {"tol": 0}])
def test_bad_settings_are_rejected(kwargs):
    Y, _, _ = _synthetic()
    with pytest.raises(ValueError):
        nmf(Y, **kwargs)


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("PDAC_DATA_ROOT"), reason="needs the real data folder")
def test_reference_data_error_matches_the_value_found_on_the_authors_data():
    path = Path(os.environ["PDAC_DATA_ROOT"]) / "reference_data_counts.csv"
    counts = pd.read_csv(path, index_col=0)
    cpm = (counts / counts.sum() * 1e6).to_numpy()
    for seed in (0, 1):
        W, H, recon = nmf(cpm, seed=seed)
        assert W.shape == (56609, 2) and H.shape == (2, 70)
        assert recon == pytest.approx(0.3818, abs=1e-3)
        np.testing.assert_allclose(H.sum(axis=0), 1.0)


# ---------- frozen rows ----------
def _fixed_example() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Y, w_true, _ = _synthetic()
    rows = np.arange(0, 60)
    return Y, rows, w_true[rows]  # the true profiles of the first 60 genes


def test_frozen_rows_stay_exactly_as_given():
    Y, rows, values = _fixed_example()
    W, H, _ = nmf(Y, reps=2, fixed_rows=rows, fixed_values=values)
    np.testing.assert_array_equal(W[rows], values)
    np.testing.assert_allclose(H.sum(axis=0), 1.0)


def test_freezing_the_true_profiles_of_some_genes_still_reconstructs_the_data():
    Y, rows, values = _fixed_example()
    _, _, recon = nmf(Y, fixed_rows=rows, fixed_values=values)
    assert recon < 0.03


def test_rows_and_values_must_be_given_together():
    Y, rows, values = _fixed_example()
    with pytest.raises(ValueError, match="together"):
        nmf(Y, fixed_rows=rows)
    with pytest.raises(ValueError, match="together"):
        nmf(Y, fixed_values=values)


@pytest.mark.parametrize(
    "rows, values",
    [
        (np.array([[0, 1]]), np.ones((2, 2))),  # not 1-D
        (np.array([0.0, 1.0]), np.ones((2, 2))),  # not integer positions
        (np.array([-1, 1]), np.ones((2, 2))),  # negative position
        (np.array([0, 300]), np.ones((2, 2))),  # out of range (300 genes)
        (np.array([1, 1]), np.ones((2, 2))),  # repeated
        (np.array([0, 1]), np.ones((3, 2))),  # wrong number of value rows
        (np.array([0, 1]), np.ones((2, 3))),  # wrong number of factors
        (np.array([0, 1]), -np.ones((2, 2))),  # negative values
        (np.array([0, 1]), np.full((2, 2), np.nan)),  # missing values
    ],
)
def test_invalid_frozen_rows_are_rejected(rows, values):
    Y, _, _ = _synthetic()
    with pytest.raises(ValueError):
        nmf(Y, fixed_rows=rows, fixed_values=values)
