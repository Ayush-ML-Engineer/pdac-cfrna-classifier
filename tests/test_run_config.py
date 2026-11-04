import argparse
from pathlib import Path

import pytest
import yaml

from pdac_cfrna.run_config import (
    CONFIG_DIR,
    SECTIONS,
    RunConfig,
    add_config_arguments,
    config_from_args,
    deep_merge,
    dump_resolved,
    load_run_config,
    parse_set,
)

DATA = "${PDAC_DATA_ROOT}"


@pytest.fixture
def toy_dir(tmp_path: Path) -> Path:
    """A tiny config folder, independent of the shipped defaults."""
    folder = tmp_path / "configs"
    folder.mkdir()
    (folder / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "toy",
                "matrix": {"path": "x.csv"},
                "normalization": {"factors": "f.csv", "mode": "joint"},
                "metadata": {
                    "path": "m.csv",
                    "sample_id_column": "id",
                    "label_column": "dx",
                    "cohort_column": "cohort",
                },
                "class_map": {"PDAC": 1, "Benign Pancreas": 0},
                "split": {"train_cohorts": ["A"], "validation_cohorts": ["B"]},
            }
        )
    )
    (folder / "experiment.yaml").write_text(
        yaml.safe_dump({"name": "base", "task": "pdac_vs_all", "seed": 0})
    )
    (folder / "model.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "rf",
                "n_estimators": 10,
                "max_features": 1,
                "min_samples_leaf": 1,
                "bootstrap": True,
            }
        )
    )
    return folder


# ---------- the shipped defaults ----------


def test_the_shipped_default_files_are_valid_together():
    cfg = load_run_config()
    assert isinstance(cfg, RunConfig)
    assert cfg.name == "paper_replication"
    assert cfg.overrides == []
    assert cfg.dataset.normalization is not None
    assert cfg.dataset.normalization.mode == "joint"
    assert cfg.experiment.n_synthetic_pdac == 100
    assert cfg.model.n_estimators == 250


def test_there_is_exactly_one_default_file_per_section():
    assert {p.name for p in CONFIG_DIR.glob("*.yaml")} == {f"{s}.yaml" for s in SECTIONS}


# ---------- parsing and merging ----------


@pytest.mark.parametrize(
    ("arg", "parts", "value"),
    [
        ("experiment.seed=3", ["experiment", "seed"], 3),
        ("experiment.alpha=0.01", ["experiment", "alpha"], 0.01),
        ("experiment.n_synthetic_pdac=null", ["experiment", "n_synthetic_pdac"], None),
        ("experiment.n_synthetic_pdac=", ["experiment", "n_synthetic_pdac"], None),
        ("experiment.log_variance=true", ["experiment", "log_variance"], True),
        ("experiment.task=pdac_vs_benign", ["experiment", "task"], "pdac_vs_benign"),
        ("experiment.exclude_genes=[A, B]", ["experiment", "exclude_genes"], ["A", "B"]),
        ("dataset.normalization=null", ["dataset", "normalization"], None),
        (f"dataset.matrix.path={DATA}/x.csv", ["dataset", "matrix", "path"], f"{DATA}/x.csv"),
    ],
)
def test_parse_set_reads_the_value_as_yaml(arg, parts, value):
    assert parse_set(arg) == (parts, value)


@pytest.mark.parametrize(
    ("arg", "message"),
    [
        ("experiment.seed", "KEY=VALUE"),
        ("seed=3", "must start with one of"),
        ("trainer.seed=3", "must start with one of"),
        ("experiment=3", "needs a field"),
        ("experiment..seed=3", "empty part"),
    ],
)
def test_parse_set_rejects_malformed_keys_with_a_helpful_message(arg, message):
    with pytest.raises(ValueError, match=message):
        parse_set(arg)


def test_deep_merge_merges_blocks_replaces_values_and_does_not_touch_the_inputs():
    base = {"a": {"x": 1, "y": 2}, "list": [1, 2], "keep": 5}
    patch = {"a": {"y": 20, "z": 30}, "list": [9]}
    merged = deep_merge(base, patch)
    assert merged == {"a": {"x": 1, "y": 20, "z": 30}, "list": [9], "keep": 5}
    assert base == {"a": {"x": 1, "y": 2}, "list": [1, 2], "keep": 5}


def test_deep_merge_lets_null_switch_a_block_off():
    assert deep_merge({"a": {"x": 1}}, {"a": None}) == {"a": None}


# ---------- loading with changes ----------


def test_no_changes_keeps_the_base_name_and_values(toy_dir):
    cfg = load_run_config(config_dir=toy_dir)
    assert cfg.name == "base"
    assert cfg.overrides == []
    assert cfg.experiment.task == "pdac_vs_all"


def test_set_changes_a_value_and_records_the_change(toy_dir):
    cfg = load_run_config(
        ["experiment.task=pdac_vs_benign", "model.n_estimators=50"], config_dir=toy_dir
    )
    assert cfg.experiment.task == "pdac_vs_benign"
    assert cfg.model.n_estimators == 50
    assert cfg.overrides == ["--set experiment.task=pdac_vs_benign", "--set model.n_estimators=50"]


def test_a_change_gives_a_derived_name_and_the_same_settings_give_the_same_name(toy_dir):
    a = load_run_config(["experiment.seed=4"], config_dir=toy_dir)
    b = load_run_config(["experiment.seed=4"], config_dir=toy_dir)
    c = load_run_config(["experiment.seed=5"], config_dir=toy_dir)
    assert a.name == b.name
    assert a.name != c.name
    assert a.name.startswith("base-")


def test_the_same_settings_written_differently_get_the_same_name(toy_dir, tmp_path):
    patch = tmp_path / "p.yaml"
    patch.write_text(yaml.safe_dump({"experiment": {"seed": 4}}))
    via_set = load_run_config(["experiment.seed=4"], config_dir=toy_dir)
    via_patch = load_run_config(patch_files=[patch], config_dir=toy_dir)
    assert via_set.name == via_patch.name


def test_an_explicit_name_wins_however_it_is_given(toy_dir):
    assert load_run_config(["experiment.seed=4"], name="mine", config_dir=toy_dir).name == "mine"
    assert (
        load_run_config(["experiment.seed=4", "experiment.name=mine2"], config_dir=toy_dir).name
        == "mine2"
    )


def test_a_bad_run_name_is_rejected(toy_dir):
    with pytest.raises(ValueError, match="name"):
        load_run_config(name="has space", config_dir=toy_dir)


def test_a_misspelled_field_is_rejected_and_named(toy_dir):
    with pytest.raises(ValueError, match="alpah"):
        load_run_config(["experiment.alpah=0.01"], config_dir=toy_dir)


def test_a_wrong_value_is_rejected(toy_dir):
    with pytest.raises(ValueError, match="n_seeds"):
        load_run_config(["experiment.n_seeds=abc"], config_dir=toy_dir)
    with pytest.raises(ValueError, match="task"):
        load_run_config(["experiment.task=everything"], config_dir=toy_dir)


def test_null_switches_the_normalization_block_off(toy_dir):
    cfg = load_run_config(["dataset.normalization=null"], config_dir=toy_dir)
    assert cfg.dataset.normalization is None


def test_setting_a_field_inside_a_switched_off_block_needs_its_required_fields(toy_dir):
    with pytest.raises(ValueError, match="factors"):
        load_run_config(
            ["dataset.normalization=null", "dataset.normalization.mode=frozen"], config_dir=toy_dir
        )


def test_setting_a_field_inside_a_value_that_is_not_a_block_is_reported(toy_dir):
    with pytest.raises(ValueError, match="not a block"):
        load_run_config(["dataset.matrix.path.deeper=1"], config_dir=toy_dir)


def test_invalid_combinations_are_caught_by_the_validators(toy_dir):
    with pytest.raises(ValueError, match="frozen"):  # reference_cohorts needs mode: frozen
        load_run_config(["dataset.normalization.reference_cohorts=[A]"], config_dir=toy_dir)


def test_a_patch_file_is_applied_and_a_later_set_wins(toy_dir, tmp_path):
    patch = tmp_path / "p.yaml"
    patch.write_text(yaml.safe_dump({"experiment": {"seed": 4, "n_seeds": 2}}))
    cfg = load_run_config(["experiment.seed=9"], patch_files=[patch], config_dir=toy_dir)
    assert (cfg.experiment.seed, cfg.experiment.n_seeds) == (9, 2)
    assert cfg.overrides[0] == "--patch p.yaml"


def test_patch_files_must_use_the_known_sections(toy_dir, tmp_path):
    patch = tmp_path / "p.yaml"
    patch.write_text(yaml.safe_dump({"trainer": {"seed": 4}}))
    with pytest.raises(ValueError, match="unknown top-level"):
        load_run_config(patch_files=[patch], config_dir=toy_dir)


def test_missing_files_are_reported(toy_dir, tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.yaml"):
        load_run_config(patch_files=[tmp_path / "nope.yaml"], config_dir=toy_dir)
    (toy_dir / "model.yaml").unlink()
    with pytest.raises(FileNotFoundError, match="model.yaml"):
        load_run_config(config_dir=toy_dir)


def test_the_saved_resolved_config_repeats_the_run(toy_dir, tmp_path):
    first = load_run_config(
        ["experiment.task=pdac_vs_benign", "model.n_estimators=7"], config_dir=toy_dir
    )
    saved = tmp_path / "resolved_config.yaml"
    dump_resolved(first, saved)
    again = load_run_config(patch_files=[saved], config_dir=toy_dir)
    assert again.dataset == first.dataset
    assert again.experiment == first.experiment  # same settings AND the same run name
    assert again.model == first.model


# ---------- command line ----------


def test_the_command_line_helpers_build_the_same_config(toy_dir):
    parser = argparse.ArgumentParser()
    add_config_arguments(parser)
    args = parser.parse_args(
        [
            "--set",
            "experiment.seed=2",
            "--set",
            "model.n_estimators=5",
            "--name",
            "cli_run",
            "--config-dir",
            str(toy_dir),
        ]
    )
    cfg = config_from_args(args)
    assert (cfg.name, cfg.experiment.seed, cfg.model.n_estimators) == ("cli_run", 2, 5)


# ---------- the documented recipes (docs/configuration.md) must keep working ----------

RECIPES = {
    "shipped_normalized_matrix": (
        [f"dataset.matrix.path={DATA}/normalized_pdac_genecount.csv", "dataset.normalization=null"],
        lambda c: c.dataset.normalization is None,
    ),
    "frozen_on_cedar": (
        ["dataset.normalization.mode=frozen", "dataset.normalization.reference_cohorts=[CEDAR]"],
        lambda c: c.dataset.normalization.mode == "frozen",
    ),
    "own_factors": (
        [f"dataset.normalization.factors={DATA}/our_updated_factors.csv"],
        lambda c: c.dataset.normalization.factors.endswith("our_updated_factors.csv"),
    ),
    "balanced_smote": (
        ["experiment.n_synthetic_pdac=null"],
        lambda c: c.experiment.n_synthetic_pdac is None,
    ),
    "other_task": (
        ["experiment.task=pdac_vs_benign"],
        lambda c: c.experiment.task == "pdac_vs_benign",
    ),
    "final_model_on_everything": (
        [
            "dataset.normalization.mode=frozen",
            "dataset.normalization.reference_cohorts=[CEDAR, BCC]",
            "dataset.split.train_cohorts=[CEDAR, BCC]",
            "dataset.split.validation_cohorts=[]",
        ],
        lambda c: c.dataset.split.validation_cohorts == [],
    ),
    "normalizer_on_both_train_on_cedar": (
        [
            "dataset.normalization.mode=frozen",
            "dataset.normalization.reference_cohorts=[CEDAR, BCC]",
        ],
        lambda c: c.dataset.split.train_cohorts == ["CEDAR"],
    ),
}


@pytest.mark.parametrize("recipe", sorted(RECIPES))
def test_every_documented_recipe_is_valid_on_the_shipped_defaults(recipe):
    set_args, check = RECIPES[recipe]
    cfg = load_run_config(set_args)
    assert check(cfg)
