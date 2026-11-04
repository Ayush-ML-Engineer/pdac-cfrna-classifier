"""Print a markdown table of the newest successful run of every experiment.

Run from the repo root:
    uv run python scripts/summarize_runs.py
    uv run python scripts/summarize_runs.py --out reports/results_table.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pdac_cfrna.reporting import runs_table, to_markdown
from pdac_cfrna.runs import RUNS_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default=str(RUNS_ROOT), help="folder holding the run folders")
    parser.add_argument("--out", help="also write the table to this file")
    args = parser.parse_args()

    text = to_markdown(runs_table(args.runs))

    print("Summarization Completed")
    # print(text)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
