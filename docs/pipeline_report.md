# Pipeline report: cfRNA PDAC classifier

## 1. What this is

A Python replication of the classifier from Moore et al. 2025 (Nat Commun 16:7345). Raw cfRNA
counts go through cf-normalization and TMM. A set of genes is selected on the training cohort
(variance filter, Wilcoxon test, Benjamini-Hochberg FDR). A log1p, SMOTE and random-forest
pipeline produces a PDAC score between 0 and 1. The model is trained on CEDAR (153 samples, 20 PDAC)
and validated on BCC (95 samples, 21 PDAC).

## 2. Replication results (validation cohort, 10 seeds)

| Task | Paper (BCC AUC) | This repo (joint normalization) |
|---|---|---|
| PDAC vs benign | 0.904 | 0.927 ± 0.007 |
| PDAC vs non-cancer | 0.937 | 0.932 ± 0.008 |
| PDAC vs all | 0.896 | 0.891 ± 0.011 |

The ± is seed-to-seed variation. The uncertainty from the small validation cohort is about ±0.06
(bootstrap 95% interval), so these numbers agree with the paper within noise. The "vs benign"
difference is the largest and has only 44 test samples. Regenerate the table for any set of runs with
`scripts/summarize_runs.py`.

All 24 genes the paper names appear in our 29-gene list.

## 3. Normalization modes

| Mode | What it does | PDAC vs all AUC |
|---|---|---|
| joint (paper) | all samples normalized together | 0.891 |
| frozen | constants fitted on CEDAR only, applied to each sample alone | 0.889 (36 genes) |
| separate batches | each cohort normalized on its own | about 0.79 (do not use) |

Our own NMF-fitted gene factors reproduce the authors' results (0.892 for PDAC vs all). Frozen
normalization costs almost nothing in AUC and makes a score independent of the other samples in
the file, which a deployed model needs.

## 4. Sensitivity analysis: SMOTE

The paper is ambiguous about whether SMOTE adds a fixed 100 synthetic samples or balances the
classes. Both are implemented. Balancing exactly gives AUCs of 0.922 / 0.928 / 0.893 (benign /
non-cancer / all), within 0.005 of the default.

## 5. Method notes and caveats

- **Gene selection and cross-validation.** The paper chooses genes before cross-validating, which
  lets information from the held-out folds shape the model. This repo selects genes on the training
  cohort only and evaluates on a separate cohort.
- **Normalization factors** in the paper use all 248 samples. Our frozen mode avoids this for the
  constants. The gene factors themselves are still an input fitted on all samples.
- **Gene list size** is 29 with joint normalization and 36 with frozen normalization, because
  frozen normalization shifts the values slightly.
- **One external cohort.** There is only one validation cohort.

## 6. What to expect

Roughly 0.85 to 0.90 AUC on new data from a similar population, under frozen normalization. The
pipeline gives a ranking score only: no calibrated probability and no chosen threshold.
Positive predictive value depends strongly on how common the disease is in the tested population,
and screening populations have far fewer PDAC cases than these case-enriched cohorts.

A final model trained on CEDAR and BCC together selected 872 genes in our run, against 29 to 36 from
CEDAR alone. We have no held-out data for it, and the larger list may partly reflect differences
between the two cohorts. It should not be adopted until the selection rule has been reviewed.

## 7. Not done

Clinical validation, other populations or sequencing platforms, threshold selection, calibration,
regulatory work, and any removal of sex-linked genes (a possible later analysis, because most of
the CEDAR PDAC patients are male).

## 8. Reproduce

```bash
uv sync
uv run pytest
uv run python scripts/run_paper_replication.py --name paper_vs_all
uv run python scripts/summarize_runs.py --out reports/results_table.md
```
