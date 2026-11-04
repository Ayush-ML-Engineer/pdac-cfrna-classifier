"""Tests for the gene comparison against the paper's gene list."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pdac_cfrna.gene_comparison import (
    PaperGenes,
    collect_runs,
    compare,
    latest_per_name,
    load_paper_genes,
    read_run_genes,
    render_run_report,
    render_summary,
)

PAPER = PaperGenes(genes=["A", "B", "C"], reported_total=4, source="test")


def _run(root: Path, name: str, genes: list[str], *, kind: str = "csv", status: str = "success"):
    folder = root / f"2026-01-01_000000_{name}"
    folder.mkdir(parents=True)
    meta = {"status": status, "experiment_name": name, "task": "pdac_vs_all", "overrides": []}
    (folder / "run_metadata.json").write_text(json.dumps(meta))
    if kind == "csv":
        everything = [*genes, "Z"]  # Z is tested but not significant
        table = pd.DataFrame(
            {
                "gene": everything,
                "p_value": 0.001,
                "q_value": 0.01,
                "auc": 0.8,
                "direction": "up",
                "significant": [g in genes for g in everything],
            }
        )
        table.to_csv(folder / "de_genes.csv", index=False)
    else:
        (folder / "model.json").write_text(json.dumps({"genes": genes}))
    return folder


def test_shipped_paper_genes_file_is_valid() -> None:
    paper = load_paper_genes()
    assert paper.reported_total == 29
    assert len(paper.genes) == 25
    assert paper.unnamed == 4
    assert "AC139099.4" in paper.genes


def test_load_paper_genes_rejects_duplicates_and_small_total(tmp_path: Path) -> None:
    bad = tmp_path / "p.yaml"
    bad.write_text("genes: [A, A]\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_paper_genes(bad)
    bad.write_text("reported_total: 1\ngenes: [A, B]\n")
    with pytest.raises(ValueError, match="below"):
        load_paper_genes(bad)


def test_compare_splits_matched_extra_missing() -> None:
    cmp = compare(["B", "X", "A"], PAPER)
    assert cmp.matched == ["B", "A"]
    assert cmp.extra == ["X"]
    assert cmp.missing == ["C"]


def test_read_run_genes_from_csv_and_model_json(tmp_path: Path) -> None:
    csv_run = read_run_genes(_run(tmp_path, "one", ["A", "B"]))
    json_run = read_run_genes(_run(tmp_path, "two", ["A", "C"], kind="json"))
    assert csv_run is not None and csv_run.genes == ["A", "B"] and csv_run.table is not None
    assert json_run is not None and json_run.genes == ["A", "C"] and json_run.table is None


def test_failed_run_is_skipped(tmp_path: Path) -> None:
    _run(tmp_path, "bad", ["A"], status="failed")
    assert collect_runs(tmp_path) == []


def test_run_report_lists_all_three_groups(tmp_path: Path) -> None:
    run = read_run_genes(_run(tmp_path, "one", ["A", "X"]))
    assert run is not None
    text = render_run_report(run, compare(run.genes, PAPER), PAPER)
    assert "Selected and named in the paper (1)" in text
    assert "Selected but not named in the paper (1)" in text
    assert "Named in the paper but not selected (2)" in text
    assert "not tested" in text  # B and C are missing from the table


def test_summary_finds_stable_genes(tmp_path: Path) -> None:
    _run(tmp_path, "one", ["A", "B", "X"])
    _run(tmp_path, "two", ["A", "X"], kind="json")
    text = render_summary(latest_per_name(collect_runs(tmp_path)), PAPER)
    assert "Stable genes: selected in every run (2)" in text
    assert "Named in the paper but selected by no run (1)" in text  # C
