"""Compare the genes each run selected with the genes the paper reports.

A run folder holds its genes either in ``de_genes.csv`` (the gene-selection table written by
``run_paper_replication.py``; the genes are the ``significant`` rows) or in ``model.json`` (the
genes saved with a model bundle by ``train_model.py``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml

from pdac_cfrna.config import REPO_ROOT

PAPER_GENES_PATH = REPO_ROOT / "reference" / "paper_genes.yaml"
REPORT_NAME = "gene_comparison.md"
DEFAULT_LIMIT = 60  # longest gene list printed in full in a report


@dataclass(frozen=True)
class PaperGenes:
    genes: list[str]  # the genes the paper's text names
    reported_total: int  # how many genes the paper says it found
    source: str

    @property
    def unnamed(self) -> int:
        """How many of the paper's genes are not named in the text."""
        return self.reported_total - len(self.genes)


@dataclass(frozen=True)
class RunGenes:
    folder: Path
    name: str
    task: str | None
    overrides: list[str]
    genes: list[str]  # in selection order
    table: pd.DataFrame | None  # the full selection table (index = gene), if the run saved it
    source: str  # which file the genes were read from


@dataclass(frozen=True)
class Comparison:
    matched: list[str]  # selected and named in the paper (in selection order)
    extra: list[str]  # selected but not named in the paper
    missing: list[str]  # named in the paper but not selected (in paper order)


def load_paper_genes(path: Path = PAPER_GENES_PATH) -> PaperGenes:
    """Read the paper's gene list.

    Raises:
        FileNotFoundError: the file does not exist.
        ValueError: the file is malformed (no gene list, duplicates, or a total below the number of
            named genes).
    """
    if not path.is_file():
        raise FileNotFoundError(f"Paper gene list not found: {path}")
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict) or not isinstance(raw.get("genes"), list):
        raise ValueError(f"{path} must contain a 'genes' list")
    genes = [str(g) for g in raw["genes"]]
    if len(set(genes)) != len(genes):
        raise ValueError(f"{path}: duplicate gene names")
    total = int(raw.get("reported_total", len(genes)))
    if total < len(genes):
        raise ValueError(f"{path}: reported_total {total} is below the {len(genes)} named genes")
    return PaperGenes(genes=genes, reported_total=total, source=str(raw.get("source", "")))


def read_run_genes(run_dir: Path) -> RunGenes | None:
    """The genes of one run folder, or None if the run failed or saved no genes."""
    meta_path = run_dir / "run_metadata.json"
    if not meta_path.is_file():
        return None
    meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") != "success":
        return None

    table: pd.DataFrame | None = None
    de_path, model_path = run_dir / "de_genes.csv", run_dir / "model.json"
    if de_path.is_file():
        table = pd.read_csv(de_path, index_col="gene")
        genes = [str(g) for g in table.index[table["significant"].astype(bool)]]
        source = de_path.name
    elif model_path.is_file():
        genes = [str(g) for g in json.loads(model_path.read_text(encoding="utf-8"))["genes"]]
        source = model_path.name
    else:
        return None
    return RunGenes(
        folder=run_dir,
        name=str(meta.get("experiment_name", run_dir.name)),
        task=meta.get("task"),
        overrides=list(meta.get("overrides") or []),
        genes=genes,
        table=table,
        source=source,
    )


def compare(selected: list[str], paper: PaperGenes) -> Comparison:
    """Split the selected genes into those the paper names and those it does not."""
    named = set(paper.genes)
    chosen = set(selected)
    return Comparison(
        matched=[g for g in selected if g in named],
        extra=[g for g in selected if g not in named],
        missing=[g for g in paper.genes if g not in chosen],
    )


def _q_value(table: pd.DataFrame, gene: str) -> float:
    """The adjusted p-value of one gene from the selection table."""
    return float(cast(Any, table.loc[gene, "q_value"]))


def explain_missing(gene: str, table: pd.DataFrame | None) -> str:
    """Why a paper gene was not selected, as far as the saved selection table can tell."""
    if table is None:
        return "this run saved no selection table"
    if gene not in table.index:
        return "not tested: removed by the variance filter or absent from the matrix"
    return f"tested, adjusted p = {_q_value(table, gene):.3g}, not below the cutoff"


def collect_runs(runs_root: Path, only: list[Path] | None = None) -> list[RunGenes]:
    """Every run folder that has genes, oldest first (folder names start with a timestamp)."""
    folders = sorted(p for p in Path(runs_root).glob("*/") if p.is_dir())
    if only is not None:
        wanted = {p.resolve() for p in only}
        folders = [p for p in folders if p.resolve() in wanted]
    runs = [read_run_genes(p) for p in folders]
    return [r for r in runs if r is not None]


def latest_per_name(runs: list[RunGenes]) -> list[RunGenes]:
    """The newest run of each experiment name (later folders overwrite earlier ones)."""
    newest: dict[str, RunGenes] = {}
    for run in runs:
        newest[run.name] = run
    return sorted(newest.values(), key=lambda r: r.name)


# ---------------------------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------------------------


def _changes(overrides: list[str]) -> str:
    return ", ".join(f"`{o}`" for o in overrides) if overrides else "none (default settings)"


def _names(genes: list[str], limit: int) -> str:
    shown = ", ".join(genes[:limit])
    return shown + (f", ... and {len(genes) - limit} more" if len(genes) > limit else "")


def _gene_table(genes: list[str], table: pd.DataFrame | None, limit: int) -> list[str]:
    """A markdown table (or a plain list when no selection table exists), shortest q first."""
    if not genes:
        return ["None."]
    if table is None:
        return [_names(genes, limit)]
    ordered = sorted(genes, key=lambda g: _q_value(table, g))
    lines = ["| gene | direction | adjusted p |", "| --- | --- | --- |"]
    for gene in ordered[:limit]:
        lines.append(f"| {gene} | {table.loc[gene, 'direction']} | {_q_value(table, gene):.3g} |")
    if len(ordered) > limit:
        lines.append(f"\n... and {len(ordered) - limit} more (all of them are in `de_genes.csv`).")
    return lines


def render_run_report(
    run: RunGenes, cmp: Comparison, paper: PaperGenes, limit: int = DEFAULT_LIMIT
) -> str:
    """The gene_comparison.md text of one run."""
    named = len(paper.genes)
    lines = [
        f"# Gene comparison: {run.name}",
        "",
        f"- Run folder: `{run.folder.name}`",
        f"- Task: {run.task or 'unknown'}",
        f"- Changes from the defaults: {_changes(run.overrides)}",
        f"- Genes read from: `{run.source}`",
        "",
        "## Summary",
        "",
        "| | genes |",
        "| --- | --- |",
        f"| Selected by this run | {len(run.genes)} |",
        f"| Selected and named in the paper | {len(cmp.matched)} of {named} |",
        f"| Selected but not named in the paper | {len(cmp.extra)} |",
        f"| Named in the paper but not selected | {len(cmp.missing)} |",
        "",
        f"The paper reports {paper.reported_total} genes; its text names {named} of them "
        f"(see `reference/paper_genes.yaml`).",
        "",
        "## Selected genes",
        "",
        _names(run.genes, limit),
        "",
        f"## Selected and named in the paper ({len(cmp.matched)})",
        "",
        *_gene_table(cmp.matched, run.table, limit),
        "",
        f"## Selected but not named in the paper ({len(cmp.extra)})",
        "",
    ]
    if cmp.extra and paper.unnamed:
        lines += [
            f"Up to {paper.unnamed} of these may be paper genes whose names the text does not "
            'give, so read "not named" as "cannot be confirmed", not as "wrong".',
            "",
        ]
    lines += _gene_table(cmp.extra, run.table, limit)
    lines += ["", f"## Named in the paper but not selected ({len(cmp.missing)})", ""]
    if cmp.missing:
        lines += ["| gene | what happened |", "| --- | --- |"]
        lines += [f"| {g} | {explain_missing(g, run.table)} |" for g in cmp.missing]
    else:
        lines.append("None: every gene the paper names was selected.")
    return "\n".join(lines) + "\n"


def render_summary(runs: list[RunGenes], paper: PaperGenes, limit: int = DEFAULT_LIMIT) -> str:
    """One markdown page comparing all runs: which genes are stable across them."""
    labels = [f"R{i + 1}" for i in range(len(runs))]
    sets = [set(r.genes) for r in runs]
    comparisons = [compare(r.genes, paper) for r in runs]
    named = set(paper.genes)
    union = list(dict.fromkeys(g for r in runs for g in r.genes))  # first-seen order
    counts = {g: sum(g in s for s in sets) for g in union}

    lines = [
        "# Gene overlap across runs",
        "",
        f"Compared with the {len(paper.genes)} genes the paper's text names "
        f"(it reports {paper.reported_total}).",
        "",
        "| label | run | task | genes | match paper | not named in paper "
        "| paper genes not selected | changes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label, run, cmp in zip(labels, runs, comparisons, strict=True):
        lines.append(
            f"| {label} | {run.name} | {run.task or ''} | {len(run.genes)} | {len(cmp.matched)} "
            f"| {len(cmp.extra)} | {len(cmp.missing)} | {_changes(run.overrides)} |"
        )

    stable = [g for g in union if counts[g] == len(runs)]
    lines += [
        "",
        f"## Stable genes: selected in every run ({len(stable)})",
        "",
        f"In the paper's named list ({len([g for g in stable if g in named])}): "
        f"{_names([g for g in stable if g in named], limit) or 'none'}",
        "",
        f"Not named in the paper ({len([g for g in stable if g not in named])}): "
        f"{_names([g for g in stable if g not in named], limit) or 'none'}",
    ]

    partial = [g for g in union if counts[g] < len(runs) and (g in named or counts[g] >= 2)]
    partial.sort(key=lambda g: (-counts[g], g))
    lines += ["", f"## Genes that some runs select and others do not ({len(partial)})", ""]
    if partial:
        lines += [
            "| gene | in paper | " + " | ".join(labels) + " | runs |",
            "| --- | --- | " + " | ".join("---" for _ in labels) + " | --- |",
        ]
        for gene in partial[:limit]:
            marks = " | ".join("✓" if gene in s else "" for s in sets)
            in_paper = "yes" if gene in named else ""
            lines.append(f"| {gene} | {in_paper} | {marks} | {counts[gene]} of {len(runs)} |")
        if len(partial) > limit:
            lines.append(f"\n... and {len(partial) - limit} more.")
    else:
        lines.append("None.")

    only_one = [g for g in union if counts[g] == 1 and g not in named and len(runs) > 1]
    if only_one:
        lines += [
            "",
            f"{len(only_one)} further genes were selected by a single run only and are not listed.",
        ]
    never = [g for g in paper.genes if not any(g in s for s in sets)]
    lines += [
        "",
        f"## Named in the paper but selected by no run ({len(never)})",
        "",
        ", ".join(never) if never else "None.",
    ]
    return "\n".join(lines) + "\n"
