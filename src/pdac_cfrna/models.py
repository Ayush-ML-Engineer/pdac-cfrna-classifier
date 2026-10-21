"""The paper's classifier pipeline: log(x+1) -> SMOTE -> random forest."""

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer

from pdac_cfrna.model_config import ModelConfig


def build_paper_pipeline(
    n_pos: int,
    seed: int,
    model: ModelConfig,
    n_synthetic: int = 100,
) -> Pipeline:
    """Build the three-step pipeline used in the paper.

    Args:
        n_pos: Number of PDAC (class 1) samples in the TRAINING data. SMOTE needs it to
            turn "add n_synthetic samples" into a target count.
        seed: Random seed for SMOTE and the random forest.
        model: Random-forest settings, normally loaded from configs/models/.
        n_synthetic: Number of synthetic PDAC samples SMOTE adds (only while fitting).

    Returns:
        An imblearn Pipeline with steps named "log", "smote" and "rf".
    """
    return Pipeline(
        [
            ("log", FunctionTransformer(np.log1p)),
            (
                "smote",
                SMOTE(sampling_strategy={1: n_pos + n_synthetic}, k_neighbors=5, random_state=seed),
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
