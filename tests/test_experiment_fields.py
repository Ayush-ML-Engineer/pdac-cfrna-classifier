import pytest

from pdac_cfrna.experiment import ExperimentConfig


def _valid() -> dict:
    return {"name": "my_exp", "task": "pdac_vs_benign"}


def test_n_synthetic_pdac_null_means_balance_exactly():
    cfg = ExperimentConfig.model_validate({**_valid(), "n_synthetic_pdac": None})
    assert cfg.n_synthetic_pdac is None


def test_n_synthetic_pdac_defaults_to_100_and_rejects_negative_numbers():
    assert ExperimentConfig(**_valid()).n_synthetic_pdac == 100
    with pytest.raises(ValueError):
        ExperimentConfig.model_validate({**_valid(), "n_synthetic_pdac": -1})
