import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pdac_cfrna.config import load_env
from pdac_cfrna.nmf import INTRINSIC_GENES, PLATELET_GENES, fit_reference_factors
from pdac_cfrna.normalization import load_factors

load_env()


def _reference_counts(swap: bool = False) -> pd.DataFrame:
    """300 genes x 20 samples: an exact mixture of an intrinsic and a batch profile.

    Rows 0-7 are the intrinsic marker genes (high in the intrinsic profile), rows 100-107 the
    platelet marker genes (high in the batch profile). Every profile sums to 1e6, so every
    sample has a total of 1e6 and CPM leaves the table unchanged.
    """
    rng = np.random.default_rng(1)
    w = np.column_stack([rng.gamma(2, 50, 300), rng.gamma(2, 50, 300)])
    w[:100, 1] *= 0.02
    w[100:200, 0] *= 0.02
    w = w / w.sum(axis=0) * 1e6
    if swap:
        w = w[:, ::-1]  # the NMF may find the two profiles in either order
    a = np.linspace(0.15, 0.85, 20)
    counts = w @ np.vstack([a, 1 - a])
    genes = [f"g{i}" for i in range(300)]
    genes[0:8] = INTRINSIC_GENES
    genes[100:108] = PLATELET_GENES
    return pd.DataFrame(counts, index=genes, columns=[f"s{j}" for j in range(20)])


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_factors_are_named_by_the_marker_genes_whatever_order_the_nmf_returns(swap, seed):
    out = fit_reference_factors(_reference_counts(swap), seed=seed)
    assert out.loc[PLATELET_GENES, "Extrinsic"].mean() > 0.5
    assert out.loc[INTRINSIC_GENES, "Intrinsic"].mean() > 0.5


def test_columns_and_row_sums():
    out = fit_reference_factors(_reference_counts())
    assert list(out.columns) == ["Intrinsic", "Extrinsic"]
    assert list(out.index) == list(_reference_counts().index)
    np.testing.assert_allclose(out.sum(axis=1).to_numpy(), 1.0)


def test_a_gene_that_is_zero_everywhere_gets_a_nan_row():
    counts = _reference_counts()
    counts.loc["g50"] = 0
    out = fit_reference_factors(counts)
    assert out.loc["g50"].isna().all()
    assert out.drop(index="g50").notna().all().all()


def test_a_missing_marker_gene_raises_and_is_named():
    counts = _reference_counts().drop(index="PF4")
    with pytest.raises(ValueError, match="PF4"):
        fit_reference_factors(counts)


def test_marker_lists_that_contradict_each_other_are_detected():
    # The "intrinsic" markers are really platelet genes: they load on the other factor.
    with pytest.raises(ValueError, match="cannot tell"):
        fit_reference_factors(_reference_counts(), intrinsic_genes=PLATELET_GENES)


def test_swapping_both_lists_consistently_just_inverts_the_labels():
    # Consistent lists cannot be told apart from a correct pair: the naming simply flips.
    normal = fit_reference_factors(_reference_counts())
    flipped = fit_reference_factors(
        _reference_counts(), platelet_genes=INTRINSIC_GENES, intrinsic_genes=PLATELET_GENES
    )
    np.testing.assert_allclose(
        flipped["Intrinsic"].to_numpy(), normal["Extrinsic"].to_numpy(), atol=1e-6
    )


def test_input_is_not_modified_and_the_result_is_reproducible():
    counts = _reference_counts()
    before = counts.copy()
    first = fit_reference_factors(counts, seed=4)
    second = fit_reference_factors(counts, seed=4)
    pd.testing.assert_frame_equal(counts, before)
    pd.testing.assert_frame_equal(first, second)


def test_the_result_can_be_saved_and_loaded_by_load_factors(tmp_path):
    out = fit_reference_factors(_reference_counts())
    path = tmp_path / "factors.csv"
    out.to_csv(path)
    assert len(load_factors(path)) == len(out)


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("PDAC_DATA_ROOT"), reason="needs the real data folder")
def test_real_reference_factors_match_the_authors_spin_data_intrinsic_file():
    root = Path(os.environ["PDAC_DATA_ROOT"])
    mine = fit_reference_factors(pd.read_csv(root / "reference_data_counts.csv", index_col=0))
    theirs = pd.read_csv(root / "spin_data_intrinsic.csv", index_col=0)  # columns: Intrinsic, Batch

    assert mine["Intrinsic"].isna().sum() == 28860
    assert set(mine.index[mine["Intrinsic"].isna()]) == set(
        theirs.index[theirs["Intrinsic"].isna()]
    )
    genes = mine.index[mine["Intrinsic"].notna()].intersection(theirs.index)
    assert len(genes) == 27749
    diff = (mine.loc[genes, "Intrinsic"] - theirs.loc[genes, "Intrinsic"]).abs()
    assert np.corrcoef(mine.loc[genes, "Intrinsic"], theirs.loc[genes, "Intrinsic"])[0, 1] > 0.999
    assert diff.median() < 1e-3
    assert diff.quantile(0.99) < 0.01
    assert diff.max() < 0.06
