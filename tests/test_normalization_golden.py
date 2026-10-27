import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pdac_cfrna.config import load_env
from pdac_cfrna.normalization import load_factors, normalize_counts

load_env()

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not os.environ.get("PDAC_DATA_ROOT"), reason="needs the real data folder"),
]


def _load_real_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(os.environ["PDAC_DATA_ROOT"])
    raw = pd.read_csv(root / "pdac_genecount.csv", index_col=0)
    shipped = pd.read_csv(root / "normalized_pdac_genecount.csv", index_col=0)
    mine = normalize_counts(raw, load_factors(root / "updated_intrinsic.csv"))
    return mine, shipped


def test_reproduces_the_shipped_normalized_file_on_the_shared_genes():
    mine, shipped = _load_real_data()
    assert mine.shape == (39984, 248)
    assert list(mine.columns) == list(shipped.columns)
    shared = mine.index.intersection(shipped.index)
    assert len(shared) == 39959
    np.testing.assert_allclose(
        mine.loc[shared].to_numpy(),
        shipped.loc[shared, mine.columns].to_numpy(),
        rtol=1e-4,
        atol=1e-6,
    )


def test_the_25_unshared_gene_names_are_excel_date_conversions():
    # Raw file and factors say "1-Dec", "10-Mar", ...; the shipped file renamed those genes.
    mine, shipped = _load_real_data()
    only_mine = mine.index.difference(shipped.index)
    only_shipped = shipped.index.difference(mine.index)
    assert len(only_mine) == len(only_shipped) == 25
    assert all(re.fullmatch(r"\d+-[A-Za-z]{3}", gene) for gene in only_mine)
