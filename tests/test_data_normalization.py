import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pdac_cfrna.config import (
    DATASET_CONFIG_DIR,
    DatasetConfig,
    NormalizationSpec,
    load_dataset_config,
)
from pdac_cfrna.data import apply_task, load_dataset
from pdac_cfrna.normalization import normalize_counts


def _write_toy_files(tmp_path: Path, orientation: str = "genes_by_samples") -> DatasetConfig:
    """40 genes x 12 samples of raw counts, factors for 35 of the genes, and metadata."""
    rng = np.random.default_rng(0)
    genes = [f"g{i}" for i in range(40)]
    samples = [f"s{j}" for j in range(12)]
    counts = pd.DataFrame(rng.poisson(30, (40, 12)) + 1, index=genes, columns=samples)
    fraction = rng.uniform(0.05, 0.95, 35)
    factors = pd.DataFrame({"Intrinsic": fraction, "Extrinsic": 1 - fraction}, index=genes[:35])
    meta = pd.DataFrame(
        {
            "SeqID": samples,
            "Cohort": ["CEDAR"] * 8 + ["BCC"] * 4,
            "Dx": ["PDAC", "Benign Pancreas", "Pancreatitis", "IPMN"] * 3,
        }
    )
    (counts if orientation == "genes_by_samples" else counts.T).to_csv(tmp_path / "counts.csv")
    factors.to_csv(tmp_path / "factors.csv")
    meta.to_csv(tmp_path / "meta.csv")
    return DatasetConfig.model_validate(
        {
            "name": "toy_raw",
            "matrix": {"path": str(tmp_path / "counts.csv"), "orientation": orientation},
            "normalization": {"factors": str(tmp_path / "factors.csv")},
            "metadata": {
                "path": str(tmp_path / "meta.csv"),
                "sample_id_column": "SeqID",
                "label_column": "Dx",
                "cohort_column": "Cohort",
            },
            "class_map": {"PDAC": 1, "Benign Pancreas": 0, "Pancreatitis": 0, "IPMN": 0},
            "split": {"train_cohorts": ["CEDAR"], "validation_cohorts": ["BCC"]},
        }
    )


def _expected_normalization(tmp_path: Path) -> pd.DataFrame:
    counts = pd.read_csv(tmp_path / "counts.csv", index_col=0)
    factors = pd.read_csv(tmp_path / "factors.csv", index_col=0)
    return normalize_counts(counts, factors)  # genes x samples


def test_the_loader_normalizes_the_raw_counts(tmp_path):
    ds = load_dataset(_write_toy_files(tmp_path))
    expected = _expected_normalization(tmp_path).T
    assert ds.X.shape == (12, 35)  # 5 genes without factors are gone
    pd.testing.assert_frame_equal(ds.X, expected, check_names=False)


def test_normalization_sees_all_samples_before_any_are_dropped(tmp_path):
    cfg = apply_task(_write_toy_files(tmp_path), {"PDAC": 1, "Benign Pancreas": 0}, "drop")
    ds = load_dataset(cfg)  # keeps 6 of the 12 samples
    everyone = _expected_normalization(tmp_path).T
    counts = pd.read_csv(tmp_path / "counts.csv", index_col=0)
    factors = pd.read_csv(tmp_path / "factors.csv", index_col=0)
    kept_only = normalize_counts(counts[list(ds.X.index)], factors).T

    np.testing.assert_allclose(ds.X.to_numpy(), everyone.loc[ds.X.index].to_numpy())
    assert not np.allclose(ds.X.to_numpy(), kept_only.to_numpy())  # normalizing the subset differs


def test_samples_by_genes_orientation_gives_the_same_result(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = load_dataset(_write_toy_files(tmp_path / "a"))
    second = load_dataset(_write_toy_files(tmp_path / "b", orientation="samples_by_genes"))
    pd.testing.assert_frame_equal(first.X, second.X)


def test_a_missing_factor_file_is_reported(tmp_path):
    cfg = _write_toy_files(tmp_path)
    (tmp_path / "factors.csv").unlink()
    with pytest.raises(FileNotFoundError, match="Factor file"):
        load_dataset(cfg)


def test_the_normalization_block_survives_apply_task(tmp_path):
    cfg = apply_task(_write_toy_files(tmp_path), {"PDAC": 1, "Benign Pancreas": 0}, "drop")
    assert cfg.normalization is not None
    assert cfg.normalization.factors.endswith("factors.csv")


def test_a_dataset_without_normalization_is_left_alone(toy_cfg):
    assert toy_cfg.normalization is None


def test_unknown_key_in_the_normalization_block_is_rejected():
    with pytest.raises(ValueError):
        NormalizationSpec.model_validate({"factors": "x.csv", "typo_key": 1})


def test_factors_path_expands_environment_variables(monkeypatch):
    monkeypatch.setenv("PDAC_DATA_ROOT", "/data")
    spec = NormalizationSpec(factors="${PDAC_DATA_ROOT}/f.csv")
    assert spec.resolved_factors_path() == Path("/data/f.csv")
    monkeypatch.delenv("PDAC_DATA_ROOT")
    with pytest.raises(ValueError, match="PDAC_DATA_ROOT"):
        spec.resolved_factors_path()


def test_every_shipped_dataset_config_parses():
    paths = sorted(DATASET_CONFIG_DIR.glob("*.yaml"))
    assert paths
    for path in paths:
        assert load_dataset_config(path).name == path.stem


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("PDAC_DATA_ROOT"), reason="needs the real data folder")
def test_real_raw_dataset_matches_the_shipped_normalized_dataset():
    raw_ds = load_dataset(load_dataset_config("moore2025_raw"))
    shipped_ds = load_dataset(load_dataset_config("moore2025_cfnorm"))
    assert raw_ds.X.shape == (248, 39984)
    assert list(raw_ds.X.index) == list(shipped_ds.X.index)
    shared = raw_ds.X.columns.intersection(shipped_ds.X.columns)
    assert len(shared) == 39959
    np.testing.assert_allclose(
        raw_ds.X[shared].to_numpy(), shipped_ds.X[shared].to_numpy(), rtol=1e-4, atol=1e-6
    )
