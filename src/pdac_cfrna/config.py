"""Typed, validated configuration.

A new dataset means a new YAML in ``configs/dataset.yaml``, not new code.
The parameters are changed per run with the overrides in run_config.py
Paths may contain ``${ENV_VAR}`` references (e.g. ``${PDAC_DATA_ROOT}``),
expanded at load time so no machine-specific path lives in git.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]

_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def resolve_path(template: str) -> Path:
    """Turn a config path like '${PDAC_DATA_ROOT}/x.csv' into a real Path."""
    expanded = os.path.expandvars(template)
    unresolved = _ENV_VAR_PATTERN.findall(expanded)
    if unresolved:
        names = ", ".join(sorted(set(unresolved)))
        raise ValueError(
            f"Environment variable(s) not set: {names}. "
            f"The config path was {template!r}. "
            "Set them in the .env file in the repo root (copy .env.example first)."
        )
    return Path(expanded)


class _Strict(BaseModel):
    """Reject unknown keys so YAML typos fail loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class MatrixSpec(_Strict):
    """One expression matrix file."""

    path: str
    # "genes_by_samples": rows are genes, columns are samples (needs transposing).
    orientation: Literal["genes_by_samples", "samples_by_genes"] = "genes_by_samples"
    index_col: int = 0

    def resolved_path(self) -> Path:
        return resolve_path(self.path)


class MetadataSpec(_Strict):
    path: str
    sample_id_column: str
    label_column: str
    cohort_column: str | None = None
    index_col: int | None = 0
    na_values: list[str] = ["n/a", "NA", "np", ""]

    def resolved_path(self) -> Path:
        return resolve_path(self.path)


class SplitSpec(_Strict):
    """Which cohort values play which role (paper: CEDAR = discovery, BCC = validation)."""

    train_cohorts: list[str]
    validation_cohorts: list[str] = []


class NormalizationSpec(_Strict):
    """Normalize the raw counts when the dataset is loaded (cf-normalization, then TMM).

    mode "joint": all samples are normalized together, as in the paper.
    mode "frozen": the constants are fitted on the samples of `reference_cohorts` only (default:
    the split's training cohorts) and then applied to every sample on its own.
    """

    factors: str  # path template of a factor CSV such as updated_intrinsic.csv
    mode: Literal["joint", "frozen"] = "joint"
    reference_cohorts: list[str] | None = None

    def resolved_factors_path(self) -> Path:
        return resolve_path(self.factors)

    @model_validator(mode="after")
    def _reference_cohorts_only_when_frozen(self) -> NormalizationSpec:
        if self.mode == "joint" and self.reference_cohorts is not None:
            raise ValueError("reference_cohorts only makes sense with mode: frozen")
        return self


class DatasetConfig(_Strict):
    name: str
    description: str = ""
    matrix: MatrixSpec
    metadata: MetadataSpec
    # raw label -> class id. Labels not listed are handled by `unmapped`.
    normalization: NormalizationSpec | None = None  # None: the matrix is already normalized
    class_map: dict[str, int]
    positive_class: int = 1
    unmapped: Literal["drop", "error"] = "error"
    split: SplitSpec | None = None

    @field_validator("class_map")
    @classmethod
    def _non_empty(cls, v: dict[str, int]) -> dict[str, int]:
        if len(set(v.values())) < 2:
            raise ValueError("class_map must define at least two classes")
        return v

    @model_validator(mode="after")
    def _positive_in_map(self) -> DatasetConfig:
        if self.positive_class not in self.class_map.values():
            raise ValueError("positive_class must appear in class_map values")
        return self

    @model_validator(mode="after")
    def _frozen_normalization_needs_cohorts(self) -> DatasetConfig:
        spec = self.normalization
        if spec is not None and spec.mode == "frozen":
            if self.metadata.cohort_column is None:
                raise ValueError("frozen normalization needs metadata.cohort_column")
            if spec.reference_cohorts is None and self.split is None:
                raise ValueError("frozen normalization needs reference_cohorts or a split")
        return self


def load_env() -> None:
    """Load .env from the repo root if present (never overrides real env vars)."""
    load_dotenv(REPO_ROOT / ".env", override=False)
