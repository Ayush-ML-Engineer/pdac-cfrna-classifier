import pytest

from pdac_cfrna.model_config import ModelConfig
from pdac_cfrna.run_config import load_run_config


def _valid() -> dict:
    return {
        "name": "m",
        "n_estimators": 10,
        "max_features": 1,
        "min_samples_leaf": 1,
        "bootstrap": True,
    }


def test_shipped_paper_config_has_the_papers_settings():
    cfg = load_run_config().model
    assert cfg.n_estimators == 250
    assert cfg.max_features == 1
    assert cfg.min_samples_leaf == 1
    assert cfg.bootstrap is True


def test_valid_file_loads(tmp_path):
    cfg = ModelConfig.model_validate(_valid())
    assert cfg.n_estimators == 10


def test_seed_is_rejected_because_it_belongs_to_the_experiment(tmp_path):
    with pytest.raises(ValueError, match="seed"):
        ModelConfig.model_validate({**_valid(), "seed": 42})


@pytest.mark.parametrize(
    "bad",
    [{"n_estimators": 0}, {"max_features": 0}, {"min_samples_leaf": 0}, {"name": "has space"}],
)
def test_out_of_range_values_are_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        ModelConfig.model_validate({**_valid(), **bad})


def test_missing_required_key_is_rejected(tmp_path):
    content = _valid()
    del content["n_estimators"]
    with pytest.raises(ValueError, match="n_estimators"):
        ModelConfig.model_validate(content)
