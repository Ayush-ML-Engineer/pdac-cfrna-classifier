# Data notes (checked 2026-10-12)

| File | Shape | What it is |
|---|---|---|
| `pdac_genecount.csv` | 56,607 genes x 248 samples | Raw integer counts (~16.6k genes all-zero) |
| `normalized_pdac_genecount.csv` | 39,984 genes x 248 samples | cf-normalized counts (no all-zero genes) |
| `pdac_meta.csv` | 248 rows | CEDAR 153 + BCC 95; matches the paper's Table 1 |
| `expr_ml.csv` | 13,981 genes x 50 samples | **Not part of this paper's data.** Sample names (`TANDA1/2`, `PRUEBA`, `Controles`, `PDAC`) are Spanish-lab IDs, 25 control + 25 PDAC, log-scale values; 0 overlap with `pdac_meta.csv`. Not BCC. |

## Not present

The healthy-donor reference set (SRA PRJNA1199029, 70 samples / 10 donors with 7 handling conditions)
used to fit the NMF intrinsic/extrinsic factors is **not** in these files. All 248 columns are patient
samples. cf-normalization therefore cannot be re-fitted from scratch; use the provided normalized
matrix (or download the reference from SRA to re-implement it).

## Metadata quirks

- For PDAC rows the columns are shifted by one: `Grade` actually holds the **stage** (1-4),
  `Met_Site` holds the **grade**, `Outcome..NED..AWD..DOD` holds the **metastasis site**, and `X` holds the
  **outcome**. `Stage` is "n/a" for all PDAC. Non-PDAC rows look unshifted. Fix before any stage or survival work.
- `Pancreatitis_type` holds the diagnosis for every row (the label column).
- `CA19_9` has 126 CEDAR / 86 BCC values, matching the paper.
