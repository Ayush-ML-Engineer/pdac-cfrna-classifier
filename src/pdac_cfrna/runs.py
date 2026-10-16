"""Run folders: one self-describing folder per experiment execution under runs/."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from filelock import FileLock

from pdac_cfrna.config import REPO_ROOT
from pdac_cfrna.experiment import ExperimentConfig

RUNS_ROOT = REPO_ROOT / "runs"
TRACKED_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn")


def timestamp(now: datetime) -> str:
    """Fixed-width, zero-padded, no colons: sorts chronologically and is legal on Windows."""
    return now.strftime("%Y-%m-%d_%H%M%S")


def create_run_dir(exp_name: str, runs_root: Path, now: datetime) -> Path:
    """Create runs/<timestamp>_<name>/; add -2, -3 ... if that exact folder already exists."""
    base = f"{timestamp(now)}_{exp_name}"
    candidate, counter = runs_root / base, 1
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)  # fails instead of reusing a folder
            return candidate
        except FileExistsError:
            counter += 1
            candidate = runs_root / f"{base}-{counter}"


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError:  # git is not installed
        return None
    return out.stdout if out.returncode == 0 else None


def git_info(repo_root: Path) -> dict[str, Any]:
    """Which code produced this run: commit, branch, uncommitted changes and their diff."""
    commit = _git(["rev-parse", "HEAD"], repo_root)
    if commit is None:
        return {"commit": None, "branch": None, "dirty": None, "untracked": [], "diff": ""}
    status = (_git(["status", "--porcelain"], repo_root) or "").splitlines()
    changed_tracked = [line for line in status if not line.startswith("??")]
    untracked = [line[3:] for line in status if line.startswith("??")]
    branch = (_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root) or "").strip()
    return {
        "commit": commit.strip(),
        "branch": branch,
        "dirty": bool(changed_tracked),
        "untracked": untracked,
        "diff": _git(["diff", "HEAD"], repo_root) or "",
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def run_lock(runs_root: Path, timeout: float = -1) -> FileLock:
    """One run at a time on this machine. timeout=-1 waits forever."""
    runs_root.mkdir(parents=True, exist_ok=True)
    return FileLock(str(runs_root / ".run.lock"), timeout=timeout)


@dataclass
class Run:
    dir: Path
    extra: dict[str, Any] = field(default_factory=dict)  # e.g. result counts, added by the caller


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


@contextmanager
def tracked_run(
    exp_cfg: ExperimentConfig,
    exp_yaml_path: Path,
    runs_root: Path = RUNS_ROOT,
    repo_root: Path = REPO_ROOT,
    lock_timeout: float = -1,
) -> Iterator[Run]:
    """Hold the lock, create the run folder, record inputs, and always record the outcome."""
    with run_lock(runs_root, lock_timeout):
        started = datetime.now().astimezone()
        run_dir = create_run_dir(exp_cfg.name, runs_root, started)
        shutil.copy2(exp_yaml_path, run_dir / "experiment_config.yaml")

        git = git_info(repo_root)
        diff = git.pop("diff")
        if git["dirty"]:
            (run_dir / "code_changes.diff").write_text(diff, encoding="utf-8")

        meta: dict[str, Any] = {
            "experiment_name": exp_cfg.name,
            "experiment_yaml": _relative_to_repo(exp_yaml_path, repo_root),
            "task": exp_cfg.task,
            "seed": exp_cfg.seed,
            "config": exp_cfg.model_dump(),
            "git": git,
            "started_at": started.isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "packages": package_versions(),
            "status": "running",
        }
        run = Run(dir=run_dir)
        t0 = time.perf_counter()
        _write_metadata(run_dir, meta)
        try:
            yield run
        except BaseException as exc:  # includes Ctrl+C, so interrupted runs are recorded too
            meta["status"] = "failed"
            meta["error"] = f"{type(exc).__name__}: {exc}"
            raise
        else:
            meta["status"] = "success"
        finally:
            meta["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            meta["duration_seconds"] = round(time.perf_counter() - t0, 2)
            meta["results"] = run.extra
            _write_metadata(run_dir, meta)


def _write_metadata(run_dir: Path, meta: dict[str, Any]) -> None:
    (run_dir / "run_metadata.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
