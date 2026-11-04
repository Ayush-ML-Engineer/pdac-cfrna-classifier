"""Compare the genes selected by each run with the genes the paper reports.

Run from the repo root (after some runs exist in runs/):
    uv run python scripts/compare_genes.py
    uv run python scripts/compare_genes.py --run runs/<run folder>

For every run folder with genes it prints the gene count and names, which genes match the paper,
which are not named in the paper and which paper genes were not selected, and writes
`gene_comparison.md` into the run folder. It also writes one summary page across all runs
(`reports/gene_overlap.md`) that shows which genes are stable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pdac_cfrna.config import REPO_ROOT
from pdac_cfrna.gene_comparison import (
    PAPER_GENES_PATH,
    REPORT_NAME,
    collect_runs,
    compare,
    latest_per_name,
    load_paper_genes,
    render_run_report,
    render_summary,
)
from pdac_cfrna.runs import RUNS_ROOT

CONSOLE_LIMIT = 40  # genes printed per list in the console; the markdown files are fuller


def _shown(genes: list[str]) -> str:
    if not genes:
        return "none"
    text = ", ".join(genes[:CONSOLE_LIMIT])
    return text + (
        f", ... +{len(genes) - CONSOLE_LIMIT} more" if len(genes) > CONSOLE_LIMIT else ""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", type=Path, default=RUNS_ROOT, help="folder holding the run folders"
    )
    parser.add_argument(
        "--run", type=Path, action="append", help="only this run folder (repeatable)"
    )
    parser.add_argument("--paper-genes", type=Path, default=PAPER_GENES_PATH)
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT / "reports" / "gene_overlap.md",
        help="where to write the page comparing all runs",
    )
    parser.add_argument("--no-summary", action="store_true", help="skip the page across runs")
    args = parser.parse_args()

    paper = load_paper_genes(args.paper_genes)
    runs = collect_runs(args.runs, args.run)
    if not runs:
        parser.error(f"no successful run with genes found in {args.runs}")

    print(f"Paper: {paper.reported_total} genes reported, {len(paper.genes)} named in the text.\n")
    for run in runs:
        cmp = compare(run.genes, paper)
        (run.folder / REPORT_NAME).write_text(render_run_report(run, cmp, paper), encoding="utf-8")
        print(f"== {run.folder.name}")
        print(f"   total genes: {len(run.genes)}  (read from {run.source})")
        print(f"   genes: {_shown(run.genes)}")
        print(f"   match the paper ({len(cmp.matched)}): {_shown(cmp.matched)}")
        print(f"   not named in the paper ({len(cmp.extra)}): {_shown(cmp.extra)}")
        print(f"   in the paper but not selected ({len(cmp.missing)}): {_shown(cmp.missing)}")
        print(f"   written: {run.folder / REPORT_NAME}\n")

    if not args.no_summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(render_summary(latest_per_name(runs), paper), encoding="utf-8")
        print(f"Summary across runs: {args.summary}")


if __name__ == "__main__":
    main()
