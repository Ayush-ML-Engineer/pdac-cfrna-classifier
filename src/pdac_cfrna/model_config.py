"""Model configuration: the rules (schema) for files in configs/models/.

Same pattern as datasets and experiments: the YAML holds the values, this module checks them.
The seed is NOT here: it belongs to the experiment, because one model config is reused
with many seeds.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from pdac_cfrna.config import REPO_ROOT

MODEL_CONFIG_DIR = REPO_ROOT / "configs" / "models"


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    description: str = ""
    n_estimators: int = Field(ge=1)
    max_features: int = Field(ge=1)
    min_samples_leaf: int = Field(ge=1)
    bootstrap: bool


def resolve_model_path(name_or_path: str | Path) -> Path:
    """'my_model' -> configs/models/my_model.yaml ; an explicit path is used as given."""
    p = Path(name_or_path)
    if not p.suffix:
        p = MODEL_CONFIG_DIR / f"{name_or_path}.yaml"
    if not p.is_file():
        raise FileNotFoundError(f"Model config not found: {p}")
    return p


def load_model_config(name_or_path: str | Path) -> ModelConfig:
    path = resolve_model_path(name_or_path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain key: value pairs, got {type(raw).__name__}")
    return ModelConfig.model_validate(raw)
