import numpy as np
import pandas as pd

from pdac_cfrna.model_config import ModelConfig
from pdac_cfrna.models import build_paper_pipeline, score_task

N_POS, N_NEG = 8, 16


def _small_model() -> ModelConfig:
    return ModelConfig(
        name="small", n_estimators=10, max_features=1, min_samples_leaf=1, bootstrap=True
    )


def _toy_data(n_pos: int = N_POS, n_neg: int = N_NEG) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    index = [f"s{i}" for i in range(n_pos + n_neg)]
    X = pd.DataFrame(
        rng.poisson(20, size=(n_pos + n_neg, 3)), columns=["g1", "g2", "g3"], index=index
    )
    y = pd.Series([1] * n_pos + [0] * n_neg, index=index)
    X.loc[y == 1, "g1"] += 15
    return X.astype(float), y


def test_none_balances_the_classes_exactly():
    X, y = _toy_data()
    pipe = build_paper_pipeline(N_POS, seed=1, model=_small_model(), n_synthetic=None)
    _, y_res = pipe.named_steps["smote"].fit_resample(X, y)
    assert int(y_res.sum()) == N_NEG  # PDAC raised to the size of the other class
    assert int((y_res == 0).sum()) == N_NEG


def test_an_integer_still_adds_exactly_that_many_samples():
    X, y = _toy_data()
    pipe = build_paper_pipeline(N_POS, seed=1, model=_small_model(), n_synthetic=4)
    _, y_res = pipe.named_steps["smote"].fit_resample(X, y)
    assert int(y_res.sum()) == N_POS + 4
    assert int((y_res == 0).sum()) == N_NEG


def test_the_balancing_pipeline_fits_and_scores_only_the_rows_it_is_given():
    X, y = _toy_data()
    pipe = build_paper_pipeline(N_POS, seed=1, model=_small_model(), n_synthetic=None)
    pipe.fit(X, y)
    assert pipe.predict_proba(X.iloc[:5]).shape == (5, 2)


def test_score_task_accepts_none_for_n_synthetic():
    X, y = _toy_data(12, 24)
    scores = score_task(X.iloc[6:], y.iloc[6:], X.iloc[:6], 1, _small_model(), n_synthetic=None)
    assert scores.name == "pdac_score"
    assert ((scores >= 0) & (scores <= 1)).all()
