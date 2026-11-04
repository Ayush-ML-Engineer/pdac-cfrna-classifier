import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pdac_cfrna.bundle import ModelBundle, load_bundle, save_bundle, scores_to_frame
from pdac_cfrna.config import DatasetConfig, load_env
from pdac_cfrna.data import load_dataset
from pdac_cfrna.evaluation import auc
from pdac_cfrna.experiment import ExperimentConfig
from pdac_cfrna.model_config import ModelConfig
from pdac_cfrna.predict import predict_scores
from pdac_cfrna.run_config import RunConfig, load_run_config
from pdac_cfrna.training import load_for_task, select_experiment_table, train_bundle

load_env()

N_GENES = 120
SIGNAL_GENES = [f"g{i}" for i in range(10)]
TEST_SAMPLES = [f"s{j}" for j in range(30, 40)]


def _toy_world(tmp_path: Path, mode: str = "frozen") -> tuple[DatasetConfig, pd.DataFrame]:
    """40 samples (30 training with 10 PDAC, 10 validation), 120 genes; PDAC raises g0-g9."""
    rng = np.random.default_rng(0)
    genes = [f"g{i}" for i in range(N_GENES)]
    samples = [f"s{j}" for j in range(40)]
    is_pdac = np.array(([1] + [0, 0]) * 10 + ([1, 0]) * 5, dtype=bool)  # 10/30 and 5/10
    counts = pd.DataFrame(rng.poisson(30, (N_GENES, 40)) + 1, index=genes, columns=samples)
    counts.iloc[:10, np.flatnonzero(is_pdac)] *= 3
    fraction = rng.uniform(0.05, 0.95, N_GENES)
    factors = pd.DataFrame({"Intrinsic": fraction, "Extrinsic": 1 - fraction}, index=genes)
    meta = pd.DataFrame(
        {
            "SeqID": samples,
            "Cohort": ["CEDAR"] * 30 + ["BCC"] * 10,
            "Dx": np.where(is_pdac, "PDAC", "Benign Pancreas"),
        }
    )
    counts.to_csv(tmp_path / "counts.csv")
    factors.to_csv(tmp_path / "factors.csv")
    meta.to_csv(tmp_path / "meta.csv")
    cfg = DatasetConfig.model_validate(
        {
            "name": "toy_world",
            "matrix": {"path": str(tmp_path / "counts.csv")},
            "normalization": {"factors": str(tmp_path / "factors.csv"), "mode": mode},
            "metadata": {
                "path": str(tmp_path / "meta.csv"),
                "sample_id_column": "SeqID",
                "label_column": "Dx",
                "cohort_column": "Cohort",
            },
            "class_map": {"PDAC": 1, "Benign Pancreas": 0},
            "split": {"train_cohorts": ["CEDAR"], "validation_cohorts": ["BCC"]},
        }
    )
    return cfg, counts


# tests/test_bundle_predict.py
def _run_config(dataset: DatasetConfig) -> RunConfig:
    return RunConfig(
        dataset=dataset,
        experiment=ExperimentConfig(
            name="toy_model", task="pdac_vs_benign", seed=1, n_synthetic_pdac=5
        ),
        model=ModelConfig(
            name="random_forest_paper",
            n_estimators=250,
            max_features=1,
            min_samples_leaf=1,
            bootstrap=True,
        ),
    )


FROZEN_ON_CEDAR = [
    "dataset.normalization.mode=frozen",
    "dataset.normalization.reference_cohorts=[CEDAR]",
]


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("PDAC_DATA_ROOT"), reason="needs the real data folder")
def test_real_data_bundle_scores_the_validation_cohort_from_raw_counts():
    root = Path(os.environ["PDAC_DATA_ROOT"])
    cfg = load_run_config(FROZEN_ON_CEDAR, name="model_frozen_vs_all")
    table = select_experiment_table(cfg)
    genes = list(table.index[table["significant"]])
    ds = load_for_task(cfg.dataset, cfg.experiment.task)
    bundle = train_bundle(ds, cfg, genes)
    raw = pd.read_csv(root / "pdac_genecount.csv", index_col=0)
    bcc = list(ds.X.index[ds.validation_mask])

    scores = predict_scores(raw[bcc], bundle)  # raw counts in, scores out
    assert len(genes) == 36
    assert auc(ds.y[bcc], scores) == pytest.approx(0.879, abs=0.01)

    alone = predict_scores(raw[[bcc[0]]], bundle)  # one sample, no batch
    assert alone.iloc[0] == pytest.approx(scores[bcc[0]], abs=1e-12)


@pytest.fixture
def world(tmp_path):
    cfg, counts = _toy_world(tmp_path)
    ds = load_dataset(cfg)
    bundle = train_bundle(ds, _run_config(cfg), SIGNAL_GENES)
    return ds, counts, bundle


def test_the_bundle_holds_a_fitted_model_the_genes_the_constants_and_the_metadata(world):
    ds, _, bundle = world
    assert isinstance(bundle, ModelBundle)
    assert bundle.genes == SIGNAL_GENES
    assert bundle.normalizer is ds.normalizer
    assert bundle.pipeline.predict_proba(ds.X.loc[:, SIGNAL_GENES]).shape == (40, 2)
    required = {
        "format_version",
        "created_at",
        "experiment",
        "dataset",
        "task",
        "genes",
        "n_train",
        "n_train_pdac",
        "seed",
        "n_synthetic_pdac",
        "model",
        "library_versions",
    }
    assert required <= set(bundle.metadata)
    assert bundle.metadata["n_train"] == 30
    assert bundle.metadata["n_train_pdac"] == 10
    assert bundle.metadata["genes"] == SIGNAL_GENES


def test_scores_are_probabilities_named_and_indexed_by_sample(world):
    _, counts, bundle = world
    scores = predict_scores(counts[TEST_SAMPLES], bundle)
    assert scores.name == "pdac_score"
    assert list(scores.index) == TEST_SAMPLES
    assert ((scores >= 0) & (scores <= 1)).all()


def test_predicting_from_raw_counts_equals_the_training_time_path(world):
    ds, counts, bundle = world
    expected = bundle.pipeline.predict_proba(ds.X.loc[TEST_SAMPLES, SIGNAL_GENES])[:, 1]
    got = predict_scores(counts[TEST_SAMPLES], bundle)
    np.testing.assert_allclose(got.to_numpy(), expected, atol=1e-12)


def test_the_model_separates_the_validation_samples_of_the_toy_world(world):
    ds, counts, bundle = world
    scores = predict_scores(counts[TEST_SAMPLES], bundle)
    assert auc(ds.y[TEST_SAMPLES], scores) > 0.9


def test_a_samples_score_is_the_same_alone_and_in_a_batch_and_one_sample_is_enough(world):
    _, counts, bundle = world
    together = predict_scores(counts[TEST_SAMPLES], bundle)
    for sample in TEST_SAMPLES[:3]:
        alone = predict_scores(counts[[sample]], bundle)
        assert alone.iloc[0] == pytest.approx(together[sample], abs=1e-12)


def test_the_order_of_the_genes_in_the_input_does_not_matter(world):
    _, counts, bundle = world
    shuffled = counts[TEST_SAMPLES].sample(frac=1, random_state=3)
    pd.testing.assert_series_equal(
        predict_scores(shuffled, bundle), predict_scores(counts[TEST_SAMPLES], bundle)
    )


def test_the_input_is_not_modified(world):
    _, counts, bundle = world
    new = counts[TEST_SAMPLES].copy()
    before = new.copy()
    predict_scores(new, bundle)
    pd.testing.assert_frame_equal(new, before)


def test_missing_genes_are_reported(world):
    _, counts, bundle = world
    with pytest.raises(ValueError, match="missing"):
        predict_scores(counts[TEST_SAMPLES].drop(index=SIGNAL_GENES[:2]), bundle)


def test_a_transposed_table_is_detected(world):
    _, counts, bundle = world
    with pytest.raises(ValueError, match="transposed"):
        predict_scores(counts[TEST_SAMPLES].T, bundle)


def test_save_and_load_give_identical_scores_and_write_a_readable_sidecar(world, tmp_path):
    _, counts, bundle = world
    path = save_bundle(bundle, tmp_path / "out" / "model.joblib")
    loaded = load_bundle(path)
    pd.testing.assert_series_equal(
        predict_scores(counts[TEST_SAMPLES], loaded), predict_scores(counts[TEST_SAMPLES], bundle)
    )
    sidecar = json.loads((tmp_path / "out" / "model.json").read_text())
    assert sidecar["genes"] == SIGNAL_GENES
    assert "scikit-learn" in sidecar["library_versions"]


def test_a_bundle_saved_with_another_library_version_is_refused(world, tmp_path):
    _, _, bundle = world
    bundle.metadata["library_versions"]["scikit-learn"] = "0.1.0"
    path = save_bundle(bundle, tmp_path / "model.joblib")
    with pytest.raises(ValueError, match="scikit-learn"):
        load_bundle(path)


def test_loading_something_that_is_not_a_bundle_or_does_not_exist_fails(tmp_path):
    import joblib

    joblib.dump({"not": "a bundle"}, tmp_path / "x.joblib")
    with pytest.raises(ValueError, match="not a model bundle"):
        load_bundle(tmp_path / "x.joblib")
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "missing.joblib")


def test_training_needs_frozen_normalization(tmp_path):
    cfg, _ = _toy_world(tmp_path, mode="joint")
    with pytest.raises(ValueError, match="frozen"):
        train_bundle(load_dataset(cfg), _run_config(cfg), SIGNAL_GENES)


def test_training_rejects_genes_that_are_not_in_the_dataset(tmp_path):
    cfg, _ = _toy_world(tmp_path)
    with pytest.raises(ValueError, match="not in the dataset"):
        train_bundle(load_dataset(cfg), _run_config(cfg), [*SIGNAL_GENES, "not_a_gene"])


def test_scores_to_frame_has_one_row_per_sample():
    frame = scores_to_frame(pd.Series([0.2, 0.9], index=["a", "b"], name="pdac_score"))
    assert list(frame.columns) == ["sample_id", "pdac_score"]
    assert list(frame["sample_id"]) == ["a", "b"]
