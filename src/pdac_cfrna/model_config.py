"""Random-forest settings (configs/model.yaml)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    description: str = ""
    n_estimators: int = Field(ge=1)
    max_features: int = Field(ge=1)
    min_samples_leaf: int = Field(ge=1)
    bootstrap: bool
