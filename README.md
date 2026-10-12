# pdac-cfrna-classifier

Replication and improvement of the plasma cell-free RNA classifier for pancreatic cancer (PDAC) from
Moore et al., *Nature Communications* 16:7345 (2025), [doi:10.1038/s41467-025-62685-y](https://doi.org/10.1038/s41467-025-62685-y).

**Goal:** (1) reproduce the paper's pipeline as published, then (2) beat its external-validation
AUCs (BCC cohort: 0.904 PDAC-vs-benign, 0.937 vs non-cancer, 0.896 vs all) with leakage-free,
properly nested validation and stronger models.

## Design

- **Config over code.** A new CSV is a new `configs/datasets/<name>.yaml` (paths, orientation,
  label column, class map, cohort split). Pipelines are `configs/experiments/*.yaml`.
- **Reproducible runs.** Every run writes its config copy, git commit, seed, metrics and ROC-AUC to
  `runs/<timestamp>_<name>/`.
- **No data in git.** Clinical data lives outside the repo; point `PDAC_DATA_ROOT` at it.

## Quickstart

```powershell
winget install astral-sh.uv Casey.Just
copy .env.example .env      # then edit PDAC_DATA_ROOT
just setup
just test
```

## Layout

```
configs/datasets/     dataset YAMLs (what the data is)
configs/experiments/  experiment YAMLs (what to run)
src/pdac_cfrna/       config.py (validated configs), data.py (loader), ...
tests/                synthetic fixtures, no real data needed
runs/                 per-run outputs (gitignored)
reports/figures/      curated final figures only
```

## Data

See `docs/data_notes.md` for what each CSV is and known quirks.

## Citation

Moore, T. W. et al. Cell free RNA detection of pancreatic cancer in pre diagnostic high risk and
symptomatic patients. *Nat Commun* 16, 7345 (2025).
