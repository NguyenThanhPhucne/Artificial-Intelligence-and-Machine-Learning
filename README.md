# Multi-class Classification for Cloud Network Intrusion Detection on AWS Using CSE-CIC-IDS2018

CMP6202 Artificial Intelligence and Machine Learning, Birmingham City University, 2026–27.

## Data

The raw file is **not** stored in this repository (358 MB). Download `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv`
from the CSE-CIC-IDS2018 bucket on the Registry of Open Data on AWS
(<https://registry.opendata.aws/cse-cic-ids2018>), or `02-14-2018.csv` from the Kaggle mirror
*IDS 2018 Intrusion CSVs (CSE-CIC-IDS2018)*, and place it at:

```
IDS 2018 Intrusion CSVs (CSE-CIC-IDS2018)/02-14-2018.csv
```

### Licence and citation

- **Licence:** CC BY 4.0 (Attribution 4.0 International), <https://creativecommons.org/licenses/by/4.0/>, as listed
  for the Kaggle mirror *IDS 2018 Intrusion CSVs (CSE-CIC-IDS2018)* used in this project.
- **Citation required by the Canadian Institute for Cybersecurity (CIC):** “any use or redistribution of the data must
  include a citation to the CSE-CIC-IDS2018 dataset and a link to this page in AWS”
  (<https://www.unb.ca/cic/datasets/ids-2018.html>, section 5). In-text citation: (Sharafaldin, Lashkari and Ghorbani, 2018);
  AWS page: <https://registry.opendata.aws/cse-cic-ids2018>.
- **Changes made** (CC BY 4.0 requires indicating them): `data/processed/sample_75k_dedup.parquet` is derived from
  `02-14-2018.csv`. I removed rows with Infinity/NaN or impossible negative values, removed duplicate and
  contradictory flows, and drew a stratified sample of 75,000 rows (seed 42). The steps and counts are in
  `notebooks/01_data_preparation.ipynb` and `data/processed/manifest.json`.

Reference (BCU Harvard):

> Sharafaldin, I., Lashkari, A.H. and Ghorbani, A.A. (2018) ‘Toward generating a new intrusion detection dataset and
> intrusion traffic characterization’, *Proceedings of the 4th International Conference on Information Systems Security
> and Privacy (ICISSP 2018)*. SciTePress, pp. 108–116. Available at: https://doi.org/10.5220/0006639801080116.

## Setup (Apple Silicon, Python 3.12 arm64)

```bash
uv sync                      # creates .venv from uv.lock
uv run pytest                # data, leakage, model and notebook checks
uv run jupyter notebook      # open the notebooks below and run them in order
```

| Notebook | Report sections | Runtime (M1 Max) |
|---|---|---|
| `01_data_preparation.ipynb` | 1, 1.1, 1.2, 3.3 – cleaning, de-duplication, 75,000-flow sample | ~15 s |
| `02_exploratory_analysis.ipynb` | 2.1–2.4 – stratified split, leakage check, summary statistics, distributions, correlation, EDA on the training set | ~25 s |
| `03_modelling.ipynb` | 3.1, 3.2, 3.4, 4.1, 4.2 – eight algorithms, KFold vs StratifiedKFold, scaling, baselines, randomised and grid search | ~35 min first run, ~1.5 min cached |
| `04_evaluation.ipynb` | 5 (template steps 1–7), 6.1, 6.2 – test set, L1 vs L2, ablations, learning curves, model choice | ~25 min first run, ~2 min cached |

To reproduce everything from a clean kernel, including refitting every model instead of loading the cached searches
from `models/`:

```bash
for nb in 01_data_preparation 02_exploratory_analysis 03_modelling 04_evaluation; do
  NIDS_FORCE_RETRAIN=1 uv run jupyter nbconvert --to notebook --execute --inplace notebooks/$nb.ipynb
done
```

All randomness is seeded (seed 42); a full refit reproduces every reported table exactly (only timings vary).

## Run on Google Colab

The notebooks also run on Google Colab; the first code cell of each notebook does the setup and does nothing locally.

1. Upload the project folder to Google Drive so that it sits at `MyDrive/CMP6202_NIDS/Project/`. It needs `src/`,
   `notebooks/`, `data/`, `reports/`, `pyproject.toml` and `requirements-colab.txt`. Notebook 01 and Step 6 (a) of
   notebook 04 also need the raw CSV at `Project/IDS 2018 Intrusion CSVs (CSE-CIC-IDS2018)/02-14-2018.csv` (358 MB);
   notebooks 02–04 otherwise read only the processed sample in `data/processed/`. Uploading the `models/` folder (cached
   searches, ~40 MB) avoids several hours of retraining on Colab's two CPU cores.
2. Open a notebook from Drive with Google Colab. If the folder is elsewhere, change `PROJECT_DIR` in the first cell.
3. Run the first cell. It mounts Drive (approve access), installs the versions in `requirements-colab.txt` and adds
   `src/` to the import path. If it reports that numpy or pandas was upgraded, choose *Runtime > Restart session* and
   run it again, then run the remaining cells in order.

Results were produced on the local environment described above; library versions are pinned so that Colab reproduces
the same tables, but timings will differ.

## Main results

Eight algorithms from the module (lab and lecture topics) share one pipeline, the same five stratified folds and the
same metrics. Hyper-parameters below are the ones selected by cross-validated macro F1; the test set has 15,000 flows,
9 of them FTP-BruteForce.

| Model | CV macro F1 | Test errors | Test FTP recall | Inference (ms / 1,000 flows) | Macro F1 drop without host fingerprint |
|---|---|---|---|---|---|
| Decision Tree (recommended for deployment) | 0.99534 ± 0.00890 | 1 | 9 / 9 | 0.55 | 0.000 |
| Random Forest | 0.99549 ± 0.00892 | 1 | 9 / 9 | 3.43 | 0.057 |
| Gaussian Naive Bayes | 0.99952 ± 0.00029 | 3 | 9 / 9 | 1.24 | 0.171 |
| **k-Nearest Neighbours (selected by the pre-registered rule)** | 0.99998 ± 0.00004 | 1 | 9 / 9 | 39.01 | 0.103 |
| Logistic Regression | 0.98368 ± 0.01551 | 2 | 9 / 9 | 1.21 | 0.048 |
| Support Vector Machine | 0.99547 ± 0.00891 | 2 | 9 / 9 | 2.18 | 0.008 |
| Gradient Boosting | 0.99551 ± 0.00893 | 0 | 9 / 9 | 6.80 | 0.016 |
| Multi-layer Perceptron | 0.99534 ± 0.00884 | 3 | 9 / 9 | 0.97 | 0.008 |

Exact figures, with bootstrap confidence intervals, are in `reports/tables/5_7_final_comparison.csv`; timings vary
between runs. The rule fixed before the test set was opened (highest CV macro F1, ties within one standard deviation
broken by speed) selects k-Nearest Neighbours. The Decision Tree is recommended for deployment: it is within 0.01 macro
F1 of the leader, about 70 times faster, readable as rules, and its score does not depend on TCP values fixed by the
attacking machines. Ablations show that the destination port is a label shortcut, that SMOTE, class weights and Tomek
links bring no consistent benefit on this data, and that L1 regularisation keeps only 19 of 69 features for Logistic
Regression at almost the same macro F1.

## Layout

| Path | Content |
|---|---|
| `src/nids/` | Reusable code: configuration, data preparation, features, models, evaluation, plotting |
| `notebooks/` | Annotated notebooks whose headings follow the report template |
| `data/processed/` | Cleaned, de-duplicated 75,000-row sample (Parquet) and `manifest.json` (counts, seed, SHA-256) |
| `reports/` | Figures (PNG, 300 dpi), tables (CSV) and copy-ready text for the report |
| `requirements-colab.txt` | Pinned library versions for Google Colab |
| `tests/` | Checks for data integrity, train/test leakage, models, course techniques and executed notebooks (58 tests) |
