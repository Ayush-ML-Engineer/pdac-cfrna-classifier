import pytest

from pdac_cfrna.experiment import TASKS, ExperimentConfig


def _valid() -> dict:
    return {"name": "my_exp", "task": "pdac_vs_benign"}


def _load(content: dict) -> ExperimentConfig:
    return ExperimentConfig.model_validate(content)


@pytest.mark.parametrize("old_field", ["dataset", "model"])
def test_the_dataset_and_model_now_live_in_their_own_files(old_field):
    with pytest.raises(ValueError, match=old_field):
        _load({**_valid(), old_field: "something"})


def test_valid_file_loads_with_correct_values_and_defaults():
    cfg = _load({**_valid(), "alpha": 0.01, "seed": 7})
    assert cfg.name == "my_exp"
    assert cfg.alpha == 0.01
    assert cfg.seed == 7
    assert cfg.variance_cutoff == 1.0  # default
    assert cfg.log_variance is False  # default


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="alpah"):
        _load({**_valid(), "alpah": 0.01})


def test_unknown_task_is_rejected_and_lists_allowed_names():
    with pytest.raises(ValueError, match="pdac_vs_benign"):
        _load({**_valid(), "task": "pdac_vs_everything_else"})


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
        {"gene_selection_task": "random task"},
        {"n_synthetic_pdac": -1},
        {"n_seeds": 0},
    ],
)
def test_out_of_range_values_are_rejected(bad):
    with pytest.raises(ValueError):
        _load({**_valid(), **bad})


def test_missing_required_key_is_rejected(tmp_path):
    content = _valid()
    del content["task"]
    with pytest.raises(ValueError, match="task"):
        _load(content)


@pytest.mark.parametrize("task_name", sorted(TASKS))
def test_every_task_is_binary_and_contains_pdac_as_positive(task_name):
    task = TASKS[task_name]
    assert set(task.class_map.values()) == {0, 1}
    assert task.class_map["PDAC"] == 1


def test_pdac_vs_benign_resolves_to_two_labels_and_drops_the_rest():
    cfg = ExperimentConfig(**_valid())
    assert cfg.resolved_task.class_map == {"PDAC": 1, "Benign Pancreas": 0}
    assert cfg.resolved_task.unmapped == "drop"


def test_new_fields_have_the_documented_defaults():
    cfg = ExperimentConfig(**_valid())
    assert cfg.gene_selection_task == "pdac_vs_all"
    assert cfg.n_synthetic_pdac == 100
    assert cfg.n_seeds == 10


def test_new_fields_accept_valid_values_including_zero_synthetic_samples():
    cfg = ExperimentConfig(
        **_valid(), n_synthetic_pdac=0, n_seeds=1, gene_selection_task="pdac_vs_benign"
    )
    assert cfg.n_synthetic_pdac == 0
    assert cfg.n_seeds == 1
    assert cfg.gene_selection_task == "pdac_vs_benign"


def test_unknown_gene_selection_task_lists_the_allowed_names(tmp_path):
    with pytest.raises(ValueError, match="pdac_vs_benign"):
        _load({**_valid(), "gene_selection_task": "random task"})
