"""The paper's classifier pipeline: log(x+1) -> SMOTE -> random forest."""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer

from pdac_cfrna.model_config import ModelConfig


def build_paper_pipeline(
    n_pos: int,
    seed: int,
    model: ModelConfig,
    n_synthetic: int | None = 100,
) -> Pipeline:
    """Build the three-step pipeline used in the paper.

    Args:
        n_pos: Number of PDAC (class 1) samples in the TRAINING data. SMOTE needs it to
            turn "add n_synthetic samples" into a target count.
        seed: Random seed for SMOTE and the random forest.
        model: Random-forest settings, normally loaded from configs/models/.
        n_synthetic: Number of synthetic PDAC samples SMOTE adds (only while fitting), or None
            to balance the classes exactly (PDAC is raised to the size of the other class).

    Returns:
        An imblearn Pipeline with steps named "log", "smote" and "rf".
    """
    sampling_strategy: str | dict[int, int] = (
        "auto" if n_synthetic is None else {1: n_pos + n_synthetic}
    )
    return Pipeline(
        [
            ("log", FunctionTransformer(np.log1p)),
            (
                "smote",
                SMOTE(sampling_strategy=sampling_strategy, k_neighbors=5, random_state=seed),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=model.n_estimators,
                    max_features=model.max_features,
                    min_samples_leaf=model.min_samples_leaf,
                    bootstrap=model.bootstrap,
                    random_state=seed,
                ),
            ),
        ]
    )


def score_task(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    seed: int,
    model: ModelConfig,
    n_synthetic: int | None = 100,
) -> pd.Series:
    """Fit the paper pipeline on the training data and return the PDAC score of each test sample.

    Returns:
        A Series named "pdac_score", indexed like X_test.
    """
    n_pos = int(y_train.sum())
    pipe = build_paper_pipeline(n_pos=n_pos, seed=seed, model=model, n_synthetic=n_synthetic)
    pipe.fit(X_train, y_train)
    class_one_column = list(pipe.classes_).index(1)
    scores = pipe.predict_proba(X_test)[:, class_one_column]
    return pd.Series(scores, index=X_test.index, name="pdac_score")
