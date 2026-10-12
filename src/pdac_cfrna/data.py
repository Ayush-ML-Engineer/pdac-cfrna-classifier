"""Dataset loading driven entirely by a DatasetConfig."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pdac_cfrna.config import DatasetConfig


@dataclass(frozen=True)
class Dataset:
    X: pd.DataFrame  # samples x genes
    y: pd.Series  # int class ids, index = sample ids
    meta: pd.DataFrame  # aligned to X
    config: DatasetConfig

    @property
    def train_mask(self) -> pd.Series:
        return self._cohort_mask(self.config.split.train_cohorts if self.config.split else None)

    @property
    def validation_mask(self) -> pd.Series:
        s = self.config.split
        return self._cohort_mask(s.validation_cohorts if s else None, empty_if_none=True)

    def _cohort_mask(self, cohorts: list[str] | None, empty_if_none: bool = False) -> pd.Series:
        col = self.config.metadata.cohort_column
        if not cohorts or col is None:
            return pd.Series(not empty_if_none, index=self.X.index)
        return self.meta[col].isin(cohorts)


def _require_file(path: Path, template: str, what: str) -> None:
    """Stop with a helpful message if `path` is not an existing file."""
    if path.is_file():
        return
    raise FileNotFoundError(
        f"{what} not found at: {path}\n"
        f"  (built from the config path {template!r})\n"
        "  Check that the folder variable in your .env points at the folder that holds "
        "the file, and that the filename matches exactly "
        "(watch for hidden double extensions such as .csv.csv)."
    )


def load_dataset(cfg: DatasetConfig) -> Dataset:
    mspec, metaspec = cfg.matrix, cfg.metadata

    # Step 1: resolve and check BOTH files before reading anything.
    matrix_path = mspec.resolved_path()
    meta_path = metaspec.resolved_path()
    _require_file(matrix_path, mspec.path, "Expression matrix")
    _require_file(meta_path, metaspec.path, "Metadata file")

    # Step 2: expression matrix, flipped so rows are samples.
    X = pd.read_csv(matrix_path, index_col=mspec.index_col)
    if mspec.orientation == "genes_by_samples":
        X = X.T
    X.index = X.index.astype(str)
    X.index.name = "sample_id"

    # Step 3: metadata, indexed by the sample-id column.
    meta = pd.read_csv(
        meta_path,
        index_col=metaspec.index_col,
        na_values=metaspec.na_values,
        keep_default_na=True,
    )
    meta = meta.set_index(metaspec.sample_id_column)
    meta.index = meta.index.astype(str)

    # Step 4: every matrix sample must have metadata; reorder metadata to match.
    missing = X.index.difference(meta.index)
    if len(missing):
        raise ValueError(f"{len(missing)} matrix samples lack metadata, e.g. {list(missing[:5])}")
    meta = meta.loc[X.index]

    # Step 5: translate text diagnoses into class numbers.
    labels = meta[metaspec.label_column].astype(str).str.strip()
    unknown = sorted(set(labels) - set(cfg.class_map))
    if unknown and cfg.unmapped == "error":
        raise ValueError(f"Labels not in class_map: {unknown}")

    keep = labels.isin(cfg.class_map)
    X, meta, labels = X.loc[keep], meta.loc[keep], labels.loc[keep]
    y = labels.map(cfg.class_map).astype(int).rename("y")

    return Dataset(X=X, y=y, meta=meta, config=cfg)
