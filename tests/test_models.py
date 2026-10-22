import numpy as np
import pandas as pd
import pytest

from pdac_cfrna.model_config import ModelConfig
from pdac_cfrna.models import build_paper_pipeline, score_task

N_POS, N_NEG = 8, 16


def _small_model() -> ModelConfig:
    # Few trees keep the tests fast; the paper's real values are checked in test_model_config.
    return ModelConfig(
        name="small", n_estimators=10, max_features=1, min_samples_leaf=1, bootstrap=True
    )


def _toy_data(n_pos: int = N_POS, n_neg: int = N_NEG) -> tuple[pd.DataFrame, pd.Series]:
    """Toy samples with 3 genes; class 1 has higher g1. SMOTE's k=5 needs > 5 training positives."""
    rng = np.random.default_rng(0)
    index = [f"s{i}" for i in range(n_pos + n_neg)]
    X = pd.DataFrame(
        rng.poisson(20, size=(n_pos + n_neg, 3)),
        columns=["g1", "g2", "g3"],
        index=index,
    )
    y = pd.Series([1] * n_pos + [0] * n_neg, index=index)
    X.loc[y == 1, "g1"] += 15
    return X.astype(float), y


def test_pipeline_has_the_three_steps_in_order():
    pipe = build_paper_pipeline(N_POS, seed=1, model=_small_model())
    assert list(pipe.named_steps) == ["log", "smote", "rf"]


def test_forest_uses_the_values_from_the_model_config():
    pipe = build_paper_pipeline(N_POS, seed=1, model=_small_model())
    rf = pipe.named_steps["rf"]
    assert rf.n_estimators == 10  # from the config, not a hardcoded 250
    assert rf.max_features == 1
    assert rf.min_samples_leaf == 1
    assert rf.bootstrap is True
    assert rf.random_state == 1


def test_log_step_applies_log1p():
    X, _ = _toy_data()
    pipe = build_paper_pipeline(N_POS, seed=1, model=_small_model())
    np.testing.assert_allclose(pipe.named_steps["log"].transform(X), np.log1p(X.to_numpy()))


def test_smote_adds_exactly_n_synthetic_positives_and_no_negatives():
    X, y = _toy_data()
    pipe = build_paper_pipeline(N_POS, seed=1, model=_small_model(), n_synthetic=4)
    _, y_res = pipe.named_steps["smote"].fit_resample(X, y)
    assert int(y_res.sum()) == N_POS + 4
    assert int((y_res == 0).sum()) == N_NEG


def test_fit_and_predict_proba_through_the_whole_pipeline():
    X, y = _toy_data()
    pipe = build_paper_pipeline(N_POS, seed=1, model=_small_model(), n_synthetic=4)
    pipe.fit(X, y)
    assert list(pipe.classes_) == [0, 1]
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert ((proba >= 0) & (proba <= 1)).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_scoring_five_rows_returns_five_rows_so_smote_is_not_applied_to_test_data():
    X, y = _toy_data()
    pipe = build_paper_pipeline(N_POS, seed=1, model=_small_model(), n_synthetic=4)
    pipe.fit(X, y)
    assert pipe.predict_proba(X.iloc[:5]).shape == (5, 2)


def test_same_seed_gives_identical_scores_from_two_separate_pipelines():
    X, y = _toy_data()
    scores = []
    for _ in range(2):
        pipe = build_paper_pipeline(N_POS, seed=7, model=_small_model(), n_synthetic=4)
        scores.append(pipe.fit(X, y).predict_proba(X)[:, 1])
    np.testing.assert_array_equal(scores[0], scores[1])


def test_fitting_does_not_modify_the_input_table():
    X, y = _toy_data()
    before = X.copy()
    build_paper_pipeline(N_POS, seed=1, model=_small_model()).fit(X, y)
    pd.testing.assert_frame_equal(X, before)


def test_zero_synthetic_samples_leaves_the_training_set_unchanged():
    X, y = _toy_data()
    pipe = build_paper_pipeline(N_POS, seed=1, model=_small_model(), n_synthetic=0)
    _, y_res = pipe.named_steps["smote"].fit_resample(X, y)
    assert len(y_res) == len(y)


def test_too_few_positives_for_smote_neighbours_raises():
    X, y = _toy_data()
    keep = y.index[(y == 0) | (y.index.isin(["s0", "s1", "s2"]))]  # only 3 positives
    X3, y3 = X.loc[keep], y.loc[keep]
    pipe = build_paper_pipeline(3, seed=1, model=_small_model(), n_synthetic=4)
    with pytest.raises(ValueError):
        pipe.fit(X3, y3)


# ---------- score_task ----------
def _split_toy():
    """12 positives + 24 negatives; hold out 3 positives and 6 negatives as the test set."""
    X, y = _toy_data(12, 24)
    test_ids = ["s0", "s1", "s2", "s12", "s13", "s14", "s15", "s16", "s17"]
    is_test = X.index.isin(test_ids)
    return X[~is_test], y[~is_test], X[is_test], y[is_test]


def test_score_task_returns_one_named_score_per_test_sample_in_zero_one():
    X_train, y_train, X_test, _ = _split_toy()
    scores = score_task(X_train, y_train, X_test, seed=1, model=_small_model(), n_synthetic=4)
    assert isinstance(scores, pd.Series)
    assert scores.name == "pdac_score"
    assert list(scores.index) == list(X_test.index)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_score_task_respects_the_seed_argument():
    X_train, y_train, X_test, _ = _split_toy()
    a = score_task(X_train, y_train, X_test, seed=1, model=_small_model(), n_synthetic=4)
    b = score_task(X_train, y_train, X_test, seed=1, model=_small_model(), n_synthetic=4)
    c = score_task(X_train, y_train, X_test, seed=2, model=_small_model(), n_synthetic=4)
    pd.testing.assert_series_equal(a, b)  # same seed -> identical
    assert not a.equals(c)  # different seed -> different (a hardcoded seed would fail this)


def test_score_task_matches_a_manually_built_pipeline():
    X_train, y_train, X_test, _ = _split_toy()
    pipe = build_paper_pipeline(int(y_train.sum()), seed=3, model=_small_model(), n_synthetic=4)
    expected = pipe.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    got = score_task(X_train, y_train, X_test, seed=3, model=_small_model(), n_synthetic=4)
    np.testing.assert_array_equal(got.to_numpy(), expected)


def test_score_task_gives_positives_higher_scores_on_average():
    X_train, y_train, X_test, y_test = _split_toy()
    scores = score_task(X_train, y_train, X_test, seed=1, model=_small_model(), n_synthetic=4)
    assert scores[y_test == 1].mean() > scores[y_test == 0].mean()
