# Gene overlap across runs

Compared with the 25 genes the paper's text names (it reports 29).

| label | run | task | genes | match paper | not named in paper | paper genes not selected | changes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | balanced_pdac_vs_all | pdac_vs_all | 29 | 25 | 4 | 0 | `--set experiment.n_synthetic_pdac=null` |
| R2 | frozen_pdac_vs_all | pdac_vs_all | 36 | 25 | 11 | 0 | `--set dataset.normalization.mode=frozen`, `--set dataset.normalization.reference_cohorts=[CEDAR]` |
| R3 | pdac_vs_all | pdac_vs_all | 29 | 25 | 4 | 0 | none (default settings) |
| R4 | pdac_vs_benign | pdac_vs_benign | 29 | 25 | 4 | 0 | `--set experiment.task=pdac_vs_benign` |
| R5 | pdac_vs_non_cancer | pdac_vs_non_cancer | 29 | 25 | 4 | 0 | `--set experiment.task=pdac_vs_non_cancer` |

## Stable genes: selected in every run (29)

In the paper's named list (25): GCLC, CD37, USP9Y, USP9X, CRP, MYC, RAB13, FOXO1, TNFRSF13C, RPS14, SMAD3, PAX5, RPS4X, FAM117A, RPS13, LARP1B, ACOT12, UTY, MS4A1, RIOX1, AFMID, AC139099.4, THAP12, OR4K13, SERPINA1

Not named in the paper (4): AC104109.4, SOX9-AS1, LINC00926, AC103740.1

## Genes that some runs select and others do not (0)

None.

7 further genes were selected by a single run only and are not listed.

## Named in the paper but selected by no run (0)

None.
