from pathlib import Path

import pytest
import yaml

from pdac_cfrna.experiment import (
    TASKS,
    ExperimentConfig,
    load_experiment_config,
    resolve_experiment_path,
)


def _write_yaml(tmp_path: Path, content: dict) -> Path:
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return path


def _valid() -> dict:
    return {"name": "my_exp", "dataset": "moore2025_cfnorm", "task": "pdac_vs_benign"}


def test_valid_file_loads_with_correct_values_and_defaults(tmp_path):
    path = _write_yaml(tmp_path, {**_valid(), "alpha": 0.01, "seed": 7})
    cfg = load_experiment_config(path)
    assert cfg.name == "my_exp"
    assert cfg.dataset == "moore2025_cfnorm"
    assert cfg.alpha == 0.01
    assert cfg.seed == 7
    assert cfg.variance_cutoff == 1.0  # default
    assert cfg.log_variance is False  # default


def test_unknown_key_is_rejected(tmp_path):
    path = _write_yaml(tmp_path, {**_valid(), "alpah": 0.01})  # typo of 'alpha'
    with pytest.raises(ValueError, match="alpah"):
        load_experiment_config(path)


def test_unknown_task_is_rejected_and_lists_allowed_names(tmp_path):
    path = _write_yaml(tmp_path, {**_valid(), "task": "pdac_vs_everything_else"})
    with pytest.raises(ValueError, match="pdac_vs_benign"):
        load_experiment_config(path)


@pytest.mark.parametrize(
    "bad",
    [
        {"alpha": 0},
        {"alpha": 1},
        {"alpha": 1.5},
        {"variance_cutoff": -1},
        {"seed": -3},
        {"name": "has space"},
        {"name": "bad/slash"},
        {"name": ""},
    ],
)
def test_out_of_range_values_are_rejected(tmp_path, bad):
    path = _write_yaml(tmp_path, {**_valid(), **bad})
    with pytest.raises(ValueError):
        load_experiment_config(path)


def test_missing_required_key_is_rejected(tmp_path):
    content = _valid()
    del content["task"]
    with pytest.raises(ValueError, match="task"):
        load_experiment_config(_write_yaml(tmp_path, content))


def test_empty_or_non_mapping_file_is_rejected(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="key: value"):
        load_experiment_config(path)


@pytest.mark.parametrize("task_name", sorted(TASKS))
def test_every_task_is_binary_and_contains_pdac_as_positive(task_name):
    task = TASKS[task_name]
    assert set(task.class_map.values()) == {0, 1}
    assert task.class_map["PDAC"] == 1


def test_pdac_vs_benign_resolves_to_two_labels_and_drops_the_rest():
    cfg = ExperimentConfig(**_valid())
    assert cfg.resolved_task.class_map == {"PDAC": 1, "Benign Pancreas": 0}
    assert cfg.resolved_task.unmapped == "drop"


def test_resolve_experiment_path_finds_shipped_configs_and_rejects_missing():
    found = resolve_experiment_path("de_selection_pdac_vs_all")
    assert found.name == "de_selection_pdac_vs_all.yaml"
    with pytest.raises(FileNotFoundError):
        resolve_experiment_path("no_such_experiment")


def test_shipped_experiment_files_are_valid():
    for name in ("de_selection_pdac_vs_benign", "de_selection_pdac_vs_all"):
        cfg = load_experiment_config(resolve_experiment_path(name))
        assert cfg.name == name
