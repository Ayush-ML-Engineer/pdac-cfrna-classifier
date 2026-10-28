import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pdac_cfrna.config import load_env
from pdac_cfrna.normalization import cf_norm, cpm, load_factors

load_env()  # reads .env so PDAC_DATA_ROOT is known to the slow test below

GOOD = ",Intrinsic,Extrinsic\ngA,0.9,0.1\ngB,,\ngC,0.25,0.75\n"


def _counts() -> pd.DataFrame:
    return pd.DataFrame(
        [[10, 20, 30], [40, 10, 20], [50, 70, 50]],
        index=["g1", "g2", "g3"],
        columns=["s1", "s2", "s3"],
    )


def _factors() -> pd.DataFrame:
    return pd.DataFrame(
        {"Intrinsic": [0.9, 0.5, 0.1], "Extrinsic": [0.1, 0.5, 0.9]},
        index=["g1", "g2", "g3"],
    )


# Hand-worked: colSums(Ci) = [34, 30, 42] (median 34), colSums(Cb) = [66, 70, 58] (median 66).
# e.g. s2/g1 = 20 * (0.9 / (30/34) + 0.1 / (70/66)) = 22.2857
EXPECTED = pd.DataFrame(
    [[10.0, 22.2857, 25.2709], [40.0, 10.3810, 19.4745], [50.0, 67.3333, 55.2545]],
    index=["g1", "g2", "g3"],
    columns=["s1", "s2", "s3"],
)


def test_matches_the_hand_worked_output():
    pd.testing.assert_frame_equal(cf_norm(_counts(), _factors()), EXPECTED, atol=1e-4, rtol=0)


def test_every_column_sums_to_the_two_medians_added_up():
    out = cf_norm(_counts(), _factors())
    np.testing.assert_allclose(out.sum().to_numpy(), [100, 100, 100])


def test_the_median_sample_is_returned_unchanged():
    # s1 has the median intrinsic total (34) and the median batch total (66): both factors are 1.
    out = cf_norm(_counts(), _factors())
    np.testing.assert_allclose(out["s1"].to_numpy(), _counts()["s1"].to_numpy())


def test_a_batch_of_one_sample_is_returned_unchanged():
    one = _counts()[["s2"]]
    np.testing.assert_allclose(cf_norm(one, _factors()).to_numpy(), one.to_numpy())


def test_output_has_the_shared_genes_in_factor_table_order():
    counts = _counts().rename(index={"g2": "extra_in_counts"})
    factors = _factors().iloc[::-1]  # g3, g2, g1
    out = cf_norm(counts, factors)
    assert list(out.index) == ["g3", "g1"]
    assert list(out.columns) == ["s1", "s2", "s3"]


def test_inputs_are_not_modified():
    counts, factors = _counts(), _factors()
    counts_before, factors_before = counts.copy(), factors.copy()
    cf_norm(counts, factors)
    pd.testing.assert_frame_equal(counts, counts_before)
    pd.testing.assert_frame_equal(factors, factors_before)


def test_no_shared_genes_raises():
    with pytest.raises(ValueError, match="no genes"):
        cf_norm(_counts(), _factors().rename(index=lambda g: g + "_x"))


def test_a_sample_with_all_zero_counts_raises_and_is_named():
    counts = _counts()
    counts["s2"] = 0
    with pytest.raises(ValueError, match="s2"):
        cf_norm(counts, _factors())


def test_a_sample_whose_counts_fall_only_on_zero_weight_genes_raises():
    factors = _factors()
    factors.loc["g3"] = [0.0, 1.0]  # g3 has no intrinsic part
    counts = _counts()
    counts["s2"] = [0, 0, 70]  # s2 has counts only in g3
    with pytest.raises(ValueError, match="s2"):
        cf_norm(counts, factors)


# ---------- load_factors ----------
def _write_factors(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "factors.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_factors_drops_empty_rows_and_keeps_order_and_values(tmp_path):
    out = load_factors(_write_factors(tmp_path, GOOD))
    assert list(out.index) == ["gA", "gC"]
    assert list(out.columns) == ["Intrinsic", "Extrinsic"]
    np.testing.assert_allclose(out["Intrinsic"].to_numpy(), [0.9, 0.25])
    np.testing.assert_allclose(out["Extrinsic"].to_numpy(), [0.1, 0.75])


def test_load_factors_ignores_extra_columns(tmp_path):
    out = load_factors(_write_factors(tmp_path, ",Intrinsic,Extrinsic,spin_cpm\ngA,0.9,0.1,5.0\n"))
    assert list(out.columns) == ["Intrinsic", "Extrinsic"]


def test_load_factors_rejects_the_reference_only_file_layout(tmp_path):
    # spin_data_intrinsic.csv calls the second column "Batch", not "Extrinsic".
    with pytest.raises(ValueError, match="Extrinsic"):
        load_factors(_write_factors(tmp_path, ",Intrinsic,Batch\ngA,0.9,0.1\n"))


def test_load_factors_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_factors(tmp_path / "nope.csv")


def test_load_factors_rejects_rows_that_do_not_sum_to_one(tmp_path):
    with pytest.raises(ValueError, match="sum to 1"):
        load_factors(_write_factors(tmp_path, ",Intrinsic,Extrinsic\ngA,0.9,0.2\n"))


def test_load_factors_rejects_values_outside_zero_one(tmp_path):
    with pytest.raises(ValueError, match="between 0 and 1"):
        load_factors(_write_factors(tmp_path, ",Intrinsic,Extrinsic\ngA,1.2,-0.2\n"))


def test_load_factors_rejects_duplicate_gene_names(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        load_factors(_write_factors(tmp_path, ",Intrinsic,Extrinsic\ngA,0.9,0.1\ngA,0.5,0.5\n"))


def test_load_factors_rejects_a_gene_with_only_one_factor(tmp_path):
    with pytest.raises(ValueError, match="no Extrinsic"):
        load_factors(_write_factors(tmp_path, ",Intrinsic,Extrinsic\ngA,0.9,\n"))


def test_load_factors_rejects_a_file_without_any_factors(tmp_path):
    with pytest.raises(ValueError, match="no gene has factors"):
        load_factors(_write_factors(tmp_path, ",Intrinsic,Extrinsic\ngA,,\ngB,,\n"))


def test_load_factors_rejects_non_numeric_values(tmp_path):
    with pytest.raises(ValueError, match="numeric"):
        load_factors(_write_factors(tmp_path, ",Intrinsic,Extrinsic\ngA,high,low\n"))


def test_loaded_factors_can_be_used_by_cf_norm(tmp_path):
    path = _write_factors(tmp_path, ",Intrinsic,Extrinsic\ng1,0.9,0.1\ng2,0.5,0.5\ng3,0.1,0.9\n")
    out = cf_norm(_counts(), load_factors(path))
    pd.testing.assert_frame_equal(out, EXPECTED, atol=1e-4, rtol=0)


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("PDAC_DATA_ROOT"), reason="needs the real data folder")
def test_real_updated_intrinsic_file_has_39984_usable_genes():
    out = load_factors(Path(os.environ["PDAC_DATA_ROOT"]) / "updated_intrinsic.csv")
    assert len(out) == 39984  # 56,607 rows minus 16,623 genes without factors


def _cpm_data() -> pd.DataFrame:
    return pd.DataFrame([[1, 3], [3, 1]], index=["g1", "g2"], columns=["s1", "s2"])


def test_cpm_matches_the_hand_worked_example():
    expected = pd.DataFrame(
        [[250000.0, 750000.0], [750000.0, 250000.0]], index=["g1", "g2"], columns=["s1", "s2"]
    )
    pd.testing.assert_frame_equal(cpm(_cpm_data()), expected)


def test_every_sample_sums_to_one_million():
    rng = np.random.default_rng(0)
    counts = pd.DataFrame(rng.poisson(20, (100, 5)) + 1)
    np.testing.assert_allclose(cpm(counts).sum().to_numpy(), 1e6)


def test_a_samples_cpm_does_not_depend_on_its_sequencing_depth():
    counts = _cpm_data()
    deeper = counts.copy()
    deeper["s1"] = deeper["s1"] * 10
    pd.testing.assert_series_equal(cpm(counts)["s1"], cpm(deeper)["s1"])


def test_cpm_keeps_labels_and_does_not_modify_the_input():
    counts = _cpm_data()
    before = counts.copy()
    out = cpm(counts)
    assert list(out.index) == ["g1", "g2"] and list(out.columns) == ["s1", "s2"]
    pd.testing.assert_frame_equal(counts, before)


def test_cpm_of_a_sample_with_a_total_of_zero_raises_and_is_named():
    counts = _cpm_data()
    counts["s2"] = 0
    with pytest.raises(ValueError, match="s2"):
        cpm(counts)


def test_cpm_rejects_negative_and_missing_counts():
    negative = _cpm_data()
    negative.iloc[0, 0] = -1
    missing = _cpm_data().astype(float)
    missing.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="negative"):
        cpm(negative)
    with pytest.raises(ValueError, match="missing"):
        cpm(missing)
