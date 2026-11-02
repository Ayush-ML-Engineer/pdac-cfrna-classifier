"""The model bundle: everything needed to go from a raw count table to a score, in one file."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from imblearn.pipeline import Pipeline

from pdac_cfrna.normalization import NormalizationConstants

BUNDLE_FORMAT_VERSION = 1
# A pickled model only loads reliably with the same major.minor version of these libraries.
VERSION_CHECKED_PACKAGES = ("scikit-learn", "imbalanced-learn")
VERSION_REPORTED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "imbalanced-learn",
    "joblib",
)


@dataclass
class ModelBundle:
    pipeline: Pipeline  # fitted: log1p -> SMOTE (fit only) -> random forest
    genes: list[str]  # the genes the classifier was trained on, in column order
    normalizer: NormalizationConstants  # frozen constants for new raw counts
    metadata: dict[str, Any]  # who/what/when: experiment, task, sizes, seed, versions, ...


def library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in VERSION_REPORTED_PACKAGES:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(".json")


def save_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    """Write the bundle with joblib, plus a readable JSON file next to it (same name, .json)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    _sidecar_path(path).write_text(
        json.dumps(bundle.metadata, indent=2, default=str), encoding="utf-8"
    )
    return path


def _major_minor(version: str | None) -> str | None:
    return ".".join(version.split(".")[:2]) if version else None


def load_bundle(path: str | Path) -> ModelBundle:
    """Load a bundle written by save_bundle.

    SECURITY: joblib files are pickles and can run code when loaded. Only load bundles that you
    made yourself or that come from someone you trust.

    Raises:
        FileNotFoundError: the file does not exist.
        ValueError: the file is not a model bundle, or it was saved with a different major.minor
            version of scikit-learn or imbalanced-learn than the one installed.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model bundle not found: {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise ValueError(f"{path} is not a model bundle (found {type(bundle).__name__})")

    saved = bundle.metadata.get("library_versions", {})
    installed = library_versions()
    for package in VERSION_CHECKED_PACKAGES:
        if _major_minor(saved.get(package)) != _major_minor(installed.get(package)):
            raise ValueError(
                f"{path.name} was saved with {package} {saved.get(package)}, but "
                f"{installed.get(package)} is installed; retrain the model or install "
                f"{package} {saved.get(package)}"
            )
    return bundle


def scores_to_frame(scores: pd.Series) -> pd.DataFrame:
    """The layout of the prediction file: one row per sample."""
    return scores.rename_axis("sample_id").reset_index()
