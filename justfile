# Windows-friendly. Install just: `winget install Casey.Just`
set shell := ["powershell.exe", "-NoLogo", "-Command"]

default:
    @just --list

setup:
    uv sync
    uv run pre-commit install

test:
    uv run pytest

lint:
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy src

fmt:
    uv run ruff check --fix .
    uv run ruff format .

# usage: just run                      (defaults)
#        just run --set experiment.task=pdac_vs_benign
run *args:
    uv run python scripts/run_paper_replication.py {{args}}
