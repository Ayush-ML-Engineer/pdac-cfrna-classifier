"""One config file per kind, plus overrides.

``configs/dataset.yaml``, ``configs/experiment.yaml`` and ``configs/model.yaml`` hold the default
settings (the paper replication). A particular run changes only what it needs:

    --patch FILE.yaml        a small YAML with just the differences (same structure, optional)
    --set section.key=value  one change on the command line (YAML syntax for the value)
    --name NAME              the run name (otherwise derived from the changes)

The three sections and the changes are merged into one validated ``RunConfig`` that is saved in
the run folder, so any run can be repeated from that one file.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from pdac_cfrna.config import REPO_ROOT, DatasetConfig, load_env
from pdac_cfrna.experiment import ExperimentConfig
from pdac_cfrna.model_config import ModelConfig

CONFIG_DIR = REPO_ROOT / "configs"
SECTIONS = ("dataset", "experiment", "model")


class RunConfig(BaseModel):
    """Everything one run needs, fully validated."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: DatasetConfig
    experiment: ExperimentConfig
    model: ModelConfig
    overrides: list[str] = []  # provenance only: the --patch / --set arguments that were applied

    @property
    def name(self) -> str:
        return self.experiment.name


# ---------------------------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------------------------


def read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Read a YAML file that must contain key: value pairs."""
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain key: value pairs, got {type(raw).__name__}")
    return raw


def deep_merge(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    """Return `base` with `patch` applied: mappings merge key by key, anything else replaces.

    A patch value of None (YAML ``null``) replaces the base value, which is how a block such as
    ``normalization`` is switched off.
    """
    merged = copy.deepcopy(dict(base))
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_set(arg: str) -> tuple[list[str], Any]:
    """Split ``section.key.subkey=value`` into (["section", "key", "subkey"], parsed value).

    The value is read as YAML, so ``null``, ``true``, ``3``, ``0.01`` and ``[A, B]`` work, and a
    plain word stays a string.
    """
    if "=" not in arg:
        raise ValueError(f"--set expects KEY=VALUE, got {arg!r}")
    key, _, text = arg.partition("=")
    parts = [p.strip() for p in key.split(".")]
    if not all(parts):
        raise ValueError(f"--set key {key!r} has an empty part")
    if parts[0] not in SECTIONS:
        raise ValueError(
            f"--set key {key!r} must start with one of {list(SECTIONS)} "
            f"(for example experiment.task={text or 'pdac_vs_benign'})"
        )
    if len(parts) < 2:
        raise ValueError(f"--set key {key!r} needs a field, e.g. {parts[0]}.<field>=<value>")
    try:
        value = yaml.safe_load(text) if text.strip() else None
    except yaml.YAMLError as exc:
        raise ValueError(f"--set {arg!r}: the value is not valid YAML: {exc}") from exc
    return parts, value


def _set_path(tree: dict[str, Any], parts: list[str], value: Any) -> None:
    node = tree
    for i, part in enumerate(parts[:-1]):
        child = node.get(part)
        if child is None:
            child = node[part] = {}
        if not isinstance(child, dict):
            raise ValueError(
                f"cannot set {'.'.join(parts)}: {'.'.join(parts[: i + 1])} is a "
                f"{type(child).__name__}, not a block with fields"
            )
        node = child
    node[parts[-1]] = value


def _auto_name(base_name: str, tree: Mapping[str, Any]) -> str:
    """base name + 8 hex digits of the settings: the same settings always get the same name."""
    payload = {s: {k: v for k, v in tree[s].items() if k != "name"} for s in SECTIONS}
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return f"{base_name}-{digest[:8]}"


def load_run_config(
    set_args: Sequence[str] = (),
    patch_files: Sequence[str | Path] = (),
    name: str | None = None,
    config_dir: Path = CONFIG_DIR,
) -> RunConfig:
    """Defaults from the three YAML files, then the patch files, then the --set changes.

    Raises:
        ValueError: a bad override, an unknown field, or a combination that fails validation.
        FileNotFoundError: a config or patch file is missing.
    """
    load_env()
    tree: dict[str, Any] = {s: read_yaml_mapping(config_dir / f"{s}.yaml") for s in SECTIONS}
    applied: list[str] = []
    name_given = name is not None

    for patch_file in patch_files:
        patch_path = Path(patch_file)
        patch = read_yaml_mapping(patch_path)
        patch.pop("overrides", None)  # a saved resolved_config.yaml can be used as a patch
        unknown = sorted(set(patch) - set(SECTIONS))
        if unknown:
            raise ValueError(f"{patch_path}: unknown top-level key(s) {unknown}; use {SECTIONS}")
        for section, values in patch.items():
            if not isinstance(values, dict):
                raise ValueError(f"{patch_path}: {section!r} must be a block with fields")
            name_given = name_given or "name" in values and section == "experiment"
            tree[section] = deep_merge(tree[section], values)
        applied.append(f"--patch {patch_path.name}")

    for arg in set_args:
        parts, value = parse_set(arg)
        _set_path(tree, parts, value)
        name_given = name_given or parts[:2] == ["experiment", "name"]
        applied.append(f"--set {arg}")

    if name is not None:
        tree["experiment"]["name"] = name
    elif applied and not name_given:
        tree["experiment"]["name"] = _auto_name(str(tree["experiment"].get("name", "run")), tree)

    return RunConfig(**tree, overrides=applied)


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add --patch, --set, --name and --config-dir to a script's parser."""
    parser.add_argument(
        "--patch",
        action="append",
        default=[],
        metavar="FILE",
        help="YAML file with only the settings to change (repeatable; applied in order)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="set_args",
        metavar="SECTION.KEY=VALUE",
        help="change one setting, e.g. experiment.task=pdac_vs_benign (repeatable)",
    )
    parser.add_argument("--name", help="run name (default: derived from the changes)")
    parser.add_argument(
        "--config-dir", type=Path, default=CONFIG_DIR, help="folder with the three default YAMLs"
    )


def config_from_args(args: argparse.Namespace) -> RunConfig:
    return load_run_config(
        set_args=args.set_args,
        patch_files=args.patch,
        name=args.name,
        config_dir=args.config_dir,
    )


def dump_resolved(cfg: RunConfig, path: Path) -> None:
    """Write the fully resolved config; it works as a --patch file to repeat the run."""
    path.write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
