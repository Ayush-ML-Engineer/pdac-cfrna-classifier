import json
import subprocess
from datetime import datetime
from pathlib import Path

import pytest
import yaml
from filelock import Timeout

from pdac_cfrna.config import DatasetConfig
from pdac_cfrna.experiment import ExperimentConfig
from pdac_cfrna.model_config import ModelConfig
from pdac_cfrna.run_config import RunConfig
from pdac_cfrna.runs import create_run_dir, git_info, run_lock, timestamp, tracked_run

NOW = datetime(2026, 10, 16, 2, 3, 22)


def _cfg() -> RunConfig:
    dataset = DatasetConfig.model_validate(
        {
            "name": "d",
            "matrix": {"path": "x.csv"},
            "metadata": {"path": "m.csv", "sample_id_column": "id", "label_column": "dx"},
            "class_map": {"PDAC": 1, "Benign Pancreas": 0},
        }
    )
    experiment = ExperimentConfig(name="my_exp", task="pdac_vs_benign", seed=5)
    model = ModelConfig(
        name="m", n_estimators=1, max_features=1, min_samples_leaf=1, bootstrap=True
    )
    return RunConfig(
        dataset=dataset, experiment=experiment, model=model, overrides=["--set experiment.seed=5"]
    )


def test_timestamp_is_zero_padded_windows_safe_and_sorts_chronologically():
    a = timestamp(datetime(2026, 10, 16, 2, 3, 22))
    b = timestamp(datetime(2026, 10, 16, 2, 3, 23))
    c = timestamp(datetime(2026, 10, 16, 10, 5, 1))
    assert a == "2026-10-16_020322"
    assert ":" not in a
    assert a < b < c  # plain alphabetical order is time order


def test_create_run_dir_never_reuses_an_existing_folder(tmp_path):
    first = create_run_dir("exp", tmp_path, NOW)
    second = create_run_dir("exp", tmp_path, NOW)
    assert first != second
    assert first.name == "2026-10-16_020322_exp"
    assert second.name == "2026-10-16_020322_exp-2"


def test_lock_blocks_a_second_run_while_the_first_is_active(tmp_path):
    with run_lock(tmp_path), pytest.raises(Timeout), run_lock(tmp_path, timeout=0.2):
        pass
    with run_lock(tmp_path, timeout=0.2):  # free again after the first is released
        pass


def test_successful_run_records_inputs_and_outcome(tmp_path):
    runs, repo = tmp_path / "runs", tmp_path / "repo"
    repo.mkdir()
    with tracked_run(_cfg(), runs_root=runs, repo_root=repo) as run:
        run.extra["n_significant"] = 29
    meta = json.loads((run.dir / "run_metadata.json").read_text())
    assert meta["status"] == "success"
    assert meta["task"] == "pdac_vs_benign"
    assert meta["seed"] == 5
    assert meta["results"] == {"n_significant": 29}
    assert meta["overrides"] == ["--set experiment.seed=5"]
    assert meta["config"]["experiment"]["name"] == "my_exp"
    assert meta["duration_seconds"] >= 0
    saved = yaml.safe_load((run.dir / "resolved_config.yaml").read_text())
    assert saved["experiment"]["seed"] == 5
    assert saved["dataset"]["name"] == "d"


def test_failed_run_is_recorded_and_the_error_still_propagates(tmp_path):
    runs, repo = tmp_path / "runs", tmp_path / "repo"
    repo.mkdir()
    with (
        pytest.raises(RuntimeError, match="boom"),
        tracked_run(_cfg(), runs_root=runs, repo_root=repo) as run,
    ):
        raise RuntimeError("boom")
    meta = json.loads((run.dir / "run_metadata.json").read_text())
    assert meta["status"] == "failed"
    assert "boom" in meta["error"]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def test_git_info_reports_commit_clean_then_dirty_with_diff(tmp_path):
    _git(tmp_path, "init", "-b", "main")
    (tmp_path / "a.txt").write_text("one\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "first")

    clean = git_info(tmp_path)
    assert len(clean["commit"]) == 40
    assert clean["dirty"] is False

    (tmp_path / "a.txt").write_text("two\n")
    (tmp_path / "new.txt").write_text("x")
    dirty = git_info(tmp_path)
    assert dirty["dirty"] is True
    assert "+two" in dirty["diff"]
    assert dirty["untracked"] == ["new.txt"]


def test_git_info_outside_a_repo_returns_none_commit(tmp_path):
    assert git_info(tmp_path)["commit"] is None


def test_dirty_run_saves_the_diff_file(tmp_path):
    repo, runs = tmp_path / "repo", tmp_path / "runs"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    (repo / "a.txt").write_text("one\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "first")
    (repo / "a.txt").write_text("two\n")
    with tracked_run(_cfg(), runs_root=runs, repo_root=repo) as run:
        pass
    assert "+two" in (run.dir / "code_changes.diff").read_text()
