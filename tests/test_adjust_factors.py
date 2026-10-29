import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pdac_cfrna.config import load_env
from pdac_cfrna.nmf import adjust_factors, fit_reference_factors
from pdac_cfrna.normalization import normalize_counts

load_env()


def _patient_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """400 genes x 20 samples (exact mixtures of two profiles) and the true fraction of each gene.

    `fixed` knows genes g0-g299: genes g250-g299 have NaN factors (unseen in the reference) and
    genes g300-g399 are missing from it entirely.
    """
    rng = np.random.default_rng(2)
    w = np.column_stack([rng.gamma(2, 50, 400), rng.gamma(2, 50, 400)])
    w[:130, 1] *= 0.02
    w[130:260, 0] *= 0.02
    w = w / w.sum(axis=0) * 1e6
    a = np.linspace(0.15, 0.85, 20)
    genes = [f"g{i}" for i in range(400)]
    counts = pd.DataFrame(
        w @ np.vstack([a, 1 - a]), index=genes, columns=[f"s{j}" for j in range(20)]
    )
    true_fraction = w[:, 0] / w.sum(axis=1)
    fixed = pd.DataFrame(
        {"Intrinsic": true_fraction[:300], "Extrinsic": 1 - true_fraction[:300]}, index=genes[:300]
    )
    fixed.iloc[250:] = np.nan
    return counts, fixed, true_fraction


def test_genes_with_factors_keep_exactly_their_fractions():
    counts, fixed, _ = _patient_data()
    out = adjust_factors(counts, fixed)
    known = fixed.dropna()
    np.testing.assert_allclose(out.loc[known.index, "Intrinsic"], known["Intrinsic"], atol=1e-9)


def test_genes_missing_from_fixed_are_learned_and_are_valid_fractions():
    counts, fixed, true_fraction = _patient_data()
    out = adjust_factors(counts, fixed)
    new = out.loc[[f"g{i}" for i in range(300, 400)]]
    assert new.notna().all().all()
    np.testing.assert_allclose(new.sum(axis=1).to_numpy(), 1.0)
    found, true = new["Intrinsic"].to_numpy(), true_fraction[300:]
    assert np.corrcoef(found, true)[0, 1] > 0.95


def test_reference_genes_with_nan_factors_are_learned_not_frozen_at_zero():
    counts, fixed, true_fraction = _patient_data()
    out = adjust_factors(counts, fixed)
    unseen = out.loc[[f"g{i}" for i in range(250, 300)]]
    assert unseen.notna().all().all()
    assert np.corrcoef(unseen["Intrinsic"].to_numpy(), true_fraction[250:300])[0, 1] > 0.95


def test_output_has_the_counts_genes_and_the_standard_columns():
    counts, fixed, _ = _patient_data()
    out = adjust_factors(counts, fixed)
    assert list(out.index) == list(counts.index)
    assert list(out.columns) == ["Intrinsic", "Extrinsic"]


def test_a_gene_that_is_zero_in_every_sample_gets_a_nan_row():
    counts, fixed, _ = _patient_data()
    counts.loc["g10"] = 0  # a fixed gene absent from this dataset
    counts.loc["g350"] = 0  # a new gene absent from this dataset
    out = adjust_factors(counts, fixed)
    assert out.loc[["g10", "g350"]].isna().all().all()


def test_fixed_genes_that_are_not_in_the_counts_are_ignored():
    counts, fixed, _ = _patient_data()
    extra = pd.DataFrame({"Intrinsic": [0.5], "Extrinsic": [0.5]}, index=["not_in_counts"])
    out = adjust_factors(counts, pd.concat([fixed, extra]))
    assert "not_in_counts" not in out.index


def test_inputs_are_not_modified_and_the_result_is_reproducible():
    counts, fixed, _ = _patient_data()
    counts_before, fixed_before = counts.copy(), fixed.copy()
    first = adjust_factors(counts, fixed, seed=3)
    second = adjust_factors(counts, fixed, seed=3)
    pd.testing.assert_frame_equal(counts, counts_before)
    pd.testing.assert_frame_equal(fixed, fixed_before)
    pd.testing.assert_frame_equal(first, second)


def test_fixed_must_have_the_factor_columns_and_share_genes_with_counts():
    counts, fixed, _ = _patient_data()
    with pytest.raises(ValueError, match="Extrinsic"):
        adjust_factors(counts, fixed.rename(columns={"Extrinsic": "Batch"}))
    with pytest.raises(ValueError, match="shares no gene"):
        adjust_factors(counts, fixed.rename(index=lambda g: g + "_x"))


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("PDAC_DATA_ROOT"), reason="needs the real data folder")
def test_real_data_reproduces_the_authors_updated_factors_and_normalization():
    root = Path(os.environ["PDAC_DATA_ROOT"])
    reference = pd.read_csv(root / "reference_data_counts.csv", index_col=0)
    raw = pd.read_csv(root / "pdac_genecount.csv", index_col=0)
    theirs = pd.read_csv(root / "updated_intrinsic.csv", index_col=0).dropna()
    shipped = pd.read_csv(root / "normalized_pdac_genecount.csv", index_col=0)

    mine = adjust_factors(raw, fit_reference_factors(reference)).dropna()
    assert len(mine) == 39984
    assert set(mine.index) == set(theirs.index)
    diff = (mine["Intrinsic"] - theirs.loc[mine.index, "Intrinsic"]).abs()
    assert np.corrcoef(mine["Intrinsic"], theirs.loc[mine.index, "Intrinsic"])[0, 1] > 0.98
    assert diff.median() < 1e-3

    # The test that matters: our own factors reproduce the paper's normalized values.
    out = normalize_counts(raw, mine)
    shared = out.index.intersection(shipped.index)
    a = out.loc[shared, shipped.columns].to_numpy()
    b = shipped.loc[shared].to_numpy()
    big = b > 5
    rel = np.abs(a - b)[big] / b[big]
    assert np.median(rel) < 5e-4
    assert np.percentile(rel, 99) < 5e-3


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("PDAC_DATA_ROOT"), reason="needs the real data folder")
def test_fitting_on_cedar_samples_only_gives_a_smaller_table():
    root = Path(os.environ["PDAC_DATA_ROOT"])
    reference = pd.read_csv(root / "reference_data_counts.csv", index_col=0)
    raw = pd.read_csv(root / "pdac_genecount.csv", index_col=0)
    meta = pd.read_csv(root / "pdac_meta.csv", index_col=0)
    cedar = meta.loc[meta["Cohort"] == "CEDAR", "SeqID"].tolist()
    table = adjust_factors(raw[cedar], fit_reference_factors(reference)).dropna()
    assert len(table) == 34407
