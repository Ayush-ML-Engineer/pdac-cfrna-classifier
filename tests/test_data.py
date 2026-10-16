import pytest

from pdac_cfrna.config import DatasetConfig, load_dataset_config
from pdac_cfrna.data import load_dataset


def test_loads_and_transposes(toy_cfg):
    ds = load_dataset(toy_cfg)
    assert ds.X.shape == (12, 6)  # samples x genes
    assert set(ds.y.unique()) == {0, 1}
    assert list(ds.X.index) == list(ds.y.index)


def test_cohort_masks_disjoint(toy_cfg):
    ds = load_dataset(toy_cfg)
    assert ds.train_mask.sum() == 8
    assert ds.validation_mask.sum() == 4
    assert not (ds.train_mask & ds.validation_mask).any()


def test_unmapped_label_errors(toy_cfg):
    bad = DatasetConfig.model_validate(
        {**toy_cfg.model_dump(), "class_map": {"PDAC": 1, "Benign Pancreas": 0}}
    )
    with pytest.raises(ValueError, match="not in class_map"):
        load_dataset(bad)


def test_unmapped_label_drop(toy_cfg):
    cfg = DatasetConfig.model_validate(
        {**toy_cfg.model_dump(), "class_map": {"PDAC": 1, "Benign Pancreas": 0}, "unmapped": "drop"}
    )
    assert load_dataset(cfg).X.shape[0] == 6


def test_unknown_yaml_key_rejected(toy_cfg):
    with pytest.raises(ValueError):
        DatasetConfig.model_validate({**toy_cfg.model_dump(), "typo_key": 1})


def test_shipped_configs_parse(monkeypatch):
    monkeypatch.setenv("PDAC_DATA_ROOT", "/tmp")
    for name in ("moore2025_cfnorm", "moore2025_raw"):
        assert load_dataset_config(name).positive_class == 1


def test_unset_env_var_gives_clear_error(toy_cfg, monkeypatch):
    monkeypatch.delenv("PDAC_DATA_ROOT", raising=False)
    cfg = DatasetConfig.model_validate(
        {**toy_cfg.model_dump(), "matrix": {"path": "${PDAC_DATA_ROOT}/x.csv"}}
    )
    with pytest.raises(ValueError, match="PDAC_DATA_ROOT"):
        load_dataset(cfg)


def test_missing_file_gives_clear_error(toy_cfg, tmp_path):
    cfg = DatasetConfig.model_validate(
        {**toy_cfg.model_dump(), "matrix": {"path": str(tmp_path / "nope.csv")}}
    )
    with pytest.raises(FileNotFoundError, match="nope.csv"):
        load_dataset(cfg)


def test_apply_task_overrides_class_map_and_drops_other_labels(toy_cfg):
    from pdac_cfrna.data import apply_task

    cfg = apply_task(toy_cfg, {"PDAC": 1, "Benign Pancreas": 0}, "drop")
    ds = load_dataset(cfg)
    assert set(ds.y.unique()) == {0, 1}
    assert ds.X.shape[0] == 6  # 3 PDAC + 3 benign of the 12 toy samples
    assert toy_cfg.class_map != cfg.class_map  # original config is untouched
