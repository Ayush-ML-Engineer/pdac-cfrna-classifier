"""Experiment configuration: the rules (schema) for files in configs/experiments/.

The YAML file holds the VALUES for one experiment; this module holds the RULES that
decide whether those values are legal, and turns the file into a Python object.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from pdac_cfrna.config import REPO_ROOT

EXPERIMENT_CONFIG_DIR = REPO_ROOT / "configs" / "experiments"


@dataclass(frozen=True)
class Task:
    """A binary task: how diagnosis labels become 0/1, and what to do with the other labels."""

    class_map: dict[str, int]
    unmapped: Literal["drop", "error"]


# Single source of truth for the allowed task names. Adding a task = adding one entry here.
TASKS: dict[str, Task] = {
    "pdac_vs_benign": Task(
        class_map={"PDAC": 1, "Benign Pancreas": 0},
        unmapped="drop",
    ),
    "pdac_vs_non_cancer": Task(
        class_map={"PDAC": 1, "Benign Pancreas": 0, "Pancreatitis": 0, "IPMN": 0},
        unmapped="drop",  # islet cell tumours and other cancers are left out
    ),
    "pdac_vs_all": Task(
        class_map={
            "PDAC": 1,
            "Benign Pancreas": 0,
            "Pancreatitis": 0,
            "IPMN": 0,
            "Islet Cell Tumor": 0,
            "Cancer_other": 0,
        },
        unmapped="error",
    ),
}


class ExperimentConfig(BaseModel):
    # Same strictness as DatasetConfig: unknown keys are errors, objects are read-only.
    model_config = ConfigDict(extra="forbid", frozen=True)

    # The name becomes part of a folder name, so only filesystem-safe characters are allowed.
    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    description: str = ""
    dataset: str  # name of a file in configs/datasets/, without .yaml
    task: str
    variance_cutoff: float = Field(default=1.0, ge=0)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    log_variance: bool = False
    seed: int = Field(default=42, ge=0)

    @field_validator("task")
    @classmethod
    def _task_must_exist(cls, v: str) -> str:
        if v not in TASKS:
            raise ValueError(f"Unknown task {v!r}. Allowed: {sorted(TASKS)}")
        return v

    @property
    def resolved_task(self) -> Task:
        return TASKS[self.task]


def resolve_experiment_path(name_or_path: str | Path) -> Path:
    """'my_exp' -> configs/experiments/my_exp.yaml ; an explicit path is used as given."""
    p = Path(name_or_path)
    if not p.suffix:
        p = EXPERIMENT_CONFIG_DIR / f"{name_or_path}.yaml"
    if not p.is_file():
        raise FileNotFoundError(f"Experiment config not found: {p}")
    return p


def load_experiment_config(path: Path) -> ExperimentConfig:
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain key: value pairs, got {type(raw).__name__}")
    return ExperimentConfig.model_validate(raw)
