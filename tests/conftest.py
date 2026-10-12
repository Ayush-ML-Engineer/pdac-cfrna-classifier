from pathlib import Path

import pytest

from pdac_cfrna.config import DatasetConfig

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def toy_cfg() -> DatasetConfig:
    return DatasetConfig.model_validate(
        {
            "name": "toy",
            "matrix": {"path": str(FIXTURES / "toy_counts.csv")},
            "metadata": {
                "path": str(FIXTURES / "toy_meta.csv"),
                "sample_id_column": "SeqID",
                "label_column": "Dx",
                "cohort_column": "Cohort",
            },
            "class_map": {"PDAC": 1, "Benign Pancreas": 0, "Pancreatitis": 0, "IPMN": 0},
            "split": {"train_cohorts": ["CEDAR"], "validation_cohorts": ["BCC"]},
        }
    )
