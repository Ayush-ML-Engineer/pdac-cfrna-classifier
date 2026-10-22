import numpy as np
import pandas as pd
import pytest

from pdac_cfrna.evaluation import auc, bootstrap_auc_ci


def _scores(n: int, noise: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Half class 1, half class 0; class 1 scores are higher by 1, blurred by `noise`."""
    rng = np.random.default_rng(seed)
    y = np.array([1] * (n // 2) + [0] * (n // 2))
    return y, y + rng.normal(0, noise, size=n)


# ---------- auc ----------
def test_auc_perfect_separation_is_one():
    assert auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_auc_reversed_scores_is_zero():
    assert auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_auc_constant_scores_is_one_half():
    assert auc([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_auc_hand_worked_example():
    # pairs (pos, neg): 0.8>0.1 yes, 0.8>0.4 yes, 0.35>0.1 yes, 0.35>0.4 no -> 3/4
    assert auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)


def test_auc_accepts_pandas_series():
    y = pd.Series([0, 0, 1, 1], index=list("abcd"))
    s = pd.Series([0.1, 0.2, 0.8, 0.9], index=list("abcd"))
    assert auc(y, s) == 1.0


def test_auc_with_a_single_class_raises():
    with pytest.raises(ValueError):
        auc([1, 1, 1], [0.2, 0.5, 0.9])


# ---------- bootstrap_auc_ci ----------
def test_ci_of_perfectly_separated_scores_is_exactly_one_one():
    y, s = np.array([0, 0, 0, 1, 1, 1]), np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert bootstrap_auc_ci(y, s, n_boot=200) == (1.0, 1.0)


def test_ci_is_ordered_inside_zero_one_and_contains_the_point_estimate():
    y, s = _scores(60, noise=0.8)
    low, high = bootstrap_auc_ci(y, s, n_boot=500, seed=1)
    assert 0 <= low <= high <= 1
    assert low <= auc(y, s) <= high


def test_same_seed_gives_the_same_interval():
    y, s = _scores(60, noise=0.8)
    assert bootstrap_auc_ci(y, s, seed=3) == bootstrap_auc_ci(y, s, seed=3)


def test_more_samples_give_a_narrower_interval():
    small = bootstrap_auc_ci(*_scores(40, noise=0.8), n_boot=500)
    large = bootstrap_auc_ci(*_scores(400, noise=0.8), n_boot=500)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_higher_confidence_gives_a_wider_interval():
    y, s = _scores(60, noise=0.8)
    narrow = bootstrap_auc_ci(y, s, confidence=0.5, seed=2)
    wide = bootstrap_auc_ci(y, s, confidence=0.95, seed=2)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_works_with_pandas_inputs():
    y, s = _scores(40, noise=0.8)
    low, high = bootstrap_auc_ci(pd.Series(y), pd.Series(s), n_boot=100)
    assert low <= high


def test_all_one_class_raises_a_clear_error_instead_of_returning_nan():
    with pytest.raises(ValueError, match="both classes"):
        bootstrap_auc_ci([1, 1, 1, 1], [0.1, 0.2, 0.3, 0.4], n_boot=20)


@pytest.mark.parametrize("kwargs", [{"n_boot": 0}, {"confidence": 0}, {"confidence": 1}])
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        bootstrap_auc_ci([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], **kwargs)


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="length"):
        bootstrap_auc_ci([0, 1, 0], [0.1, 0.9])
