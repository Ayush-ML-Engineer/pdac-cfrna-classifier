| experiment | task | changes | genes | mean_auc | std_auc | seed_averaged_auc | ci_low | ci_high | run_folder |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced_pdac_vs_all | pdac_vs_all | experiment.n_synthetic_pdac=null | 29 | 0.893 | 0.010 | 0.901 | 0.831 | 0.956 | 2026-11-04_135400_balanced_pdac_vs_all |
| frozen_pdac_vs_all | pdac_vs_all | dataset.normalization.mode=frozen; dataset.normalization.reference_cohorts=[CEDAR] | 36 | 0.889 | 0.008 | 0.895 | 0.815 | 0.956 | 2026-11-04_140230_frozen_pdac_vs_all |
| pdac_vs_all | pdac_vs_all |  | 29 | 0.891 | 0.011 | 0.898 | 0.827 | 0.955 | 2026-11-04_135228_pdac_vs_all |
| pdac_vs_benign | pdac_vs_benign | experiment.task=pdac_vs_benign | 29 | 0.927 | 0.007 | 0.932 | 0.851 | 0.992 | 2026-11-04_135256_pdac_vs_benign |
| pdac_vs_non_cancer | pdac_vs_non_cancer | experiment.task=pdac_vs_non_cancer | 29 | 0.932 | 0.008 | 0.940 | 0.888 | 0.982 | 2026-11-04_135319_pdac_vs_non_cancer |
