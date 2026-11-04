import json
from pathlib import Path

import pandas as pd

from pdac_cfrna.reporting import COLUMNS, runs_table, to_markdown


def _make_run(
    root: Path,
    folder: str,
    experiment: str,
    status: str = "success",
    overrides: list[str] | None = None,
    **results,
) -> None:
    path = root / folder
    path.mkdir(parents=True)
    meta = {
        "experiment_name": experiment,
        "task": "pdac_vs_all",
        "overrides": overrides or [],
        "status": status,
        "results": results,
    }
    (path / "run_metadata.json").write_text(json.dumps(meta), encoding="utf-8")


def test_the_newest_successful_run_of_each_experiment_is_used(tmp_path):
    _make_run(tmp_path, "2026-10-01_000000_a", "a", mean_auc=0.8, n_genes=29)
    _make_run(tmp_path, "2026-10-02_000000_a", "a", mean_auc=0.9, n_genes=29, std_auc=0.01)
    table = runs_table(tmp_path)
    assert list(table["experiment"]) == ["a"]
    assert table.loc[0, "mean_auc"] == 0.9
    assert table.loc[0, "run_folder"] == "2026-10-02_000000_a"


def test_failed_runs_and_runs_without_an_auc_are_skipped(tmp_path):
    _make_run(tmp_path, "2026-10-01_000000_a", "a", mean_auc=0.8)
    _make_run(tmp_path, "2026-10-02_000000_a", "a", status="failed", mean_auc=0.99)
    _make_run(tmp_path, "2026-10-03_000000_model", "model", n_genes=36)  # a training run
    table = runs_table(tmp_path)
    assert list(table["experiment"]) == ["a"]
    assert table.loc[0, "mean_auc"] == 0.8


def test_the_table_is_sorted_by_experiment_and_has_the_documented_columns(tmp_path):
    _make_run(tmp_path, "2026-10-01_000000_b", "b", mean_auc=0.7)
    _make_run(tmp_path, "2026-10-01_000001_a", "a", mean_auc=0.6)
    table = runs_table(tmp_path)
    assert list(table["experiment"]) == ["a", "b"]
    assert list(table.columns) == COLUMNS


def test_an_empty_folder_gives_an_empty_table_with_the_columns(tmp_path):
    table = runs_table(tmp_path)
    assert table.empty and list(table.columns) == COLUMNS


def test_markdown_has_a_header_a_divider_and_one_line_per_run(tmp_path):
    _make_run(tmp_path, "2026-10-01_000000_a", "a", mean_auc=0.8123, std_auc=0.0101, n_genes=29)
    lines = to_markdown(runs_table(tmp_path)).splitlines()
    assert lines[0].startswith("| experiment |")
    assert set(lines[1].replace("|", "").split()) == {"---"}
    assert len(lines) == 3
    assert "0.812" in lines[2] and "0.010" in lines[2]


def test_missing_values_become_empty_cells():
    table = pd.DataFrame([["a", None]], columns=["x", "y"])
    assert to_markdown(table).splitlines()[2] == "| a |  |"


def test_the_changes_column_lists_the_overrides_without_the_flag_name(tmp_path):
    _make_run(
        tmp_path,
        "2026-10-01_000000_a",
        "a",
        overrides=["--set experiment.task=pdac_vs_benign", "--patch p.yaml"],
        mean_auc=0.8,
    )
    table = runs_table(tmp_path)
    assert table.loc[0, "changes"] == "experiment.task=pdac_vs_benign; --patch p.yaml"
