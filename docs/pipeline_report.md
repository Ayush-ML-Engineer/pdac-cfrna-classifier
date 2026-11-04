# PDAC cfRNA classifier: pipeline report

## 1. What this is
A from-scratch Python replication of Moore et al. 2025 (Nat Commun 16:7345): raw cfRNA
counts -> cf-normalization + TMM -> 29-gene selection (variance filter, Wilcoxon, BH FDR)
-> log1p + SMOTE + random forest -> PDAC score between 0 and 1. Trained on CEDAR (153 total samples, 20 PDAC), validated on BCC (95 total samples, 21 PDAC).

## 2. Replication results (validation cohort, 10 seeds)
| Task | Our AUC | CI note |
|---|---|---|
| PDAC vs benign | 0.927 ± 0.007 | 95% bootstrap CI on the validation cohort is about ±0.06 |
| PDAC vs non-cancer | 0.932 ± 0.008 | " |
| PDAC vs all | 0.891 ± 0.011 | " |

The ± is seed spread. The CI reflects how few patients are in the validation cohort and is the
uncertainty to quote. Regenerate this table with `scripts/summarize_runs.py`.

## 3. Normalization modes
| Mode | What it does | PDAC vs all AUC |
|---|---|---|
| joint (paper) | all samples normalized together | 0.891 (29 genes) |
| frozen | constants fitted on CEDAR only, applied per sample (batch-independent, deployable) | 0.889 (36 genes) |
| separate batches | each cohort normalized on its own | about 0.79 (do not use) |
Our own NMF-fitted factors reproduce the authors' results (0.892 vs all).

## 4. Sensitivity analyses
- SMOTE ambiguity (+100 samples vs exact balance): 0.893 vs 0.891 for pdac_vs_all, no meaningful difference.

## 5. Method notes and caveats
- The paper chooses genes before cross-validation, which leaks information. We select genes on
  the training cohort only.
- Normalization factors in the paper use all 248 samples; the frozen mode avoids this.
- Our selection gives 29 genes with joint normalization and 36 with frozen normalization.
- Only one external cohort (BCC); no second independent validation.

## 6. What to expect
Realistically about 0.85–0.90 AUC for new data under frozen normalization. The pipeline gives
a ranking score only: no calibrated probability and no chosen threshold. Positive predictive
value depends heavily on prevalence, and screening populations have far fewer PDAC cases than
this case-enriched cohort.

## 7. Not done
Clinical validation, other populations or platforms, threshold selection, regulatory work.

## 8. Reproduce
```
uv sync
uv run pytest
uv run python scripts/run_paper_replication.py paper_replication_raw_pdac_vs_all
uv run python scripts/summarize_runs.py --out reports/results_table.md
```
