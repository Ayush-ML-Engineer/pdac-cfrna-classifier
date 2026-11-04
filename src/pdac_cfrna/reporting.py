"""Summarize finished runs (runs/*/run_metadata.json) into a table for the report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

COLUMNS = [
    "experiment",
    "task",
    "changes",
    "genes",
    "mean_auc",
    "std_auc",
    "seed_averaged_auc",
    "ci_low",
    "ci_high",
    "run_folder",
]


def runs_table(runs_root: str | Path) -> pd.DataFrame:
    """One row per experiment: its newest successful run that reports an AUC.

    Run folders are named <timestamp>_<experiment>, so sorting the names puts newer runs last.
    Runs that failed, or that have no mean AUC (for example a model-training run), are skipped.
    """
    rows: dict[str, dict[str, object]] = {}
    for folder in sorted(Path(runs_root).glob("*/")):
        path = folder / "run_metadata.json"
        if not path.is_file():
            continue
        meta = json.loads(path.read_text(encoding="utf-8"))
        results = meta.get("results", {})
        if meta.get("status") != "success" or "mean_auc" not in results:
            continue
        changes = meta.get("overrides")
        rows[meta["experiment_name"]] = {
            "experiment": meta["experiment_name"],
            "task": meta["task"],
            "changes": "; ".join(c.removeprefix("--set ") for c in changes) if changes else "",
            "genes": results.get("n_genes"),
            "mean_auc": results["mean_auc"],
            "std_auc": results.get("std_auc"),
            "seed_averaged_auc": results.get("seed_averaged_auc"),
            "ci_low": results.get("seed_averaged_ci_low"),
            "ci_high": results.get("seed_averaged_ci_high"),
            "run_folder": folder.name,
        }
    table = pd.DataFrame(list(rows.values()), columns=COLUMNS)
    return table.sort_values("experiment").reset_index(drop=True)


def _cell(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("|", "\\|")


def to_markdown(table: pd.DataFrame) -> str:
    """A GitHub-style markdown table (no extra dependencies)."""
    header = "| " + " | ".join(table.columns) + " |"
    divider = "|" + "|".join(" --- " for _ in table.columns) + "|"
    lines = [header, divider]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
    return "\n".join(lines)
