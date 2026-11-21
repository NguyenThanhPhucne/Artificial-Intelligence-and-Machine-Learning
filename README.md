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

The unseen-day check in notebook 04 (Step 6 d) uses a second file of the same dataset,
`Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv` (Kaggle: `02-15-2018.csv`), placed in the same folder. It is only
needed to rebuild `data/processed/unseen_day_02-15_sample.parquet`, which is included.

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
  `data/processed/unseen_day_02-15_sample.parquet` is derived from `02-15-2018.csv` with the same cleaning and
  de-duplication, after removing flows whose features already occur in the 75,000-row sample, and a sample of 20,000
  benign and 5,000 flows of each attack (seed 42); stage counts are in `reports/tables/5_6_unseen_day_stages.csv`.

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

## Predict new flows

The recommended model (the Decision Tree) labels any CSV with the CICFlowMeter columns of CSE-CIC-IDS2018:

```bash
uv run python -m nids.predict "IDS 2018 Intrusion CSVs (CSE-CIC-IDS2018)/02-15-2018.csv" --out predictions.csv
```

It applies the same row-wise cleaning as notebook 01 (rows with Infinity, NaN or impossible negative values are dropped
and counted), then the fitted pipeline. Notebook 04 saves the model to `models/final_decision_tree.joblib`; in a fresh
clone the command rebuilds it from the search in about 20 seconds. The model only knows Benign, FTP-BruteForce and
SSH-Bruteforce, so every other attack is given one of these three labels.

## Run on Google Colab

The notebooks also run on Google Colab; the first code cell of each notebook does the setup and does nothing locally.

1. Upload the project folder to Google Drive so that it sits at `MyDrive/CMP6202_NIDS/Project/` or
   `MyDrive/Project/`; the first cell of each notebook finds either. It needs `src/`,
   `notebooks/`, `data/`, `reports/`, `pyproject.toml` and `requirements-colab.txt`. Step 6 (d) of notebook 04 reads
   the stored 15 February sample in `data/processed/`, so the second raw file is not needed. Notebook 01 and Step 6 (a) of
   notebook 04 also need the raw CSV at `Project/IDS 2018 Intrusion CSVs (CSE-CIC-IDS2018)/02-14-2018.csv` (358 MB);
   notebooks 02–04 otherwise read only the processed sample in `data/processed/`. Uploading the `models/` folder (cached
   searches, ~40 MB) avoids several hours of retraining on Colab's two CPU cores.
2. Open a notebook from Drive with Google Colab. If the folder is elsewhere, add its path to `candidates` in the
   first cell.
3. Run the first cell. It mounts Drive (approve access), installs the versions in `requirements-colab.txt` and adds
   `src/` to the import path. If it reports that numpy or pandas was upgraded, choose *Runtime > Restart session* and
   run it again, then run the remaining cells in order.

Results were produced on the local environment described above; library versions are pinned so that Colab reproduces
the same tables, but timings will differ.

## Main results

Eight algorithms from the module (lab and lecture topics) share one pipeline, the same five stratified folds and the
same metrics. Hyper-parameters below are the ones selected by cross-validated macro F1; the test set has
15,000 flows, 9 of them FTP-BruteForce. Test errors range from 0 to 3 per
model, and every model finds all FTP test flows.

| Model | CV macro F1 | Test errors | Test FTP recall | Inference (ms / 1,000 flows) | Macro F1 drop without host fingerprint |
|---|---|---|---|---|---|
| Decision Tree (recommended for deployment) | 0.99534 ± 0.00890 | 1 | 9 / 9 | 0.46 | 0.0051 |
| Random Forest | 0.99549 ± 0.00892 | 1 | 9 / 9 | 2.74 | 0.0568 |
| Gaussian Naive Bayes | 0.99952 ± 0.00029 | 3 | 9 / 9 | 0.96 | 0.1709 |
| **k-Nearest Neighbours (selected by the pre-registered rule)** | 0.99998 ± 0.00004 | 1 | 9 / 9 | 37.64 | 0.1028 |
| Logistic Regression | 0.98368 ± 0.01551 | 2 | 9 / 9 | 1.17 | 0.0080 |
| Support Vector Machine | 0.99547 ± 0.00891 | 2 | 9 / 9 | 2.13 | 0.0079 |
| Gradient Boosting | 0.99551 ± 0.00893 | 0 | 9 / 9 | 7.30 | 0.0163 |
| Multi-layer Perceptron | 0.99534 ± 0.00884 | 3 | 9 / 9 | 0.84 | 0.0840 |

Exact figures, with bootstrap confidence intervals, are in `reports/tables/5_7_final_comparison.csv`; timings vary
between runs. The rule fixed before the test set was opened (highest CV macro F1, ties within one standard deviation
broken by speed) selects k-Nearest Neighbours. The Decision Tree is recommended for deployment: it is 0.00464 macro F1 below the leader,
about 82 times faster at inference, readable as rules, and it loses only 0.0051 macro F1 when the TCP values fixed by the attacking machines are withheld. In the ablations no model gains more than one standard deviation from the destination port, which stays excluded as a label proxy;
SMOTE helped beyond one standard deviation for no model, class weights and Tomek links brought no
consistent benefit, and L1 regularisation keeps 19 of 69 features for Logistic Regression at
macro F1 0.9830 (L2: 0.9671).

Correlation-based feature selection (Week 5 lab) keeps 16 of 66 numeric features at
|rho| ≥ 0.7, but 8 of 8 models then lose more than one standard deviation of macro F1; at 0.95
(39 features kept) all 8 hold. On a day the models never saw (15 February 2018,
`reports/tables/5_6_unseen_day.csv`) they raise 0–3 false alarms on
20,000 benign flows, but they were never trained on DoS: the Decision Tree flags only
46% of DoS-GoldenEye and 1% of DoS-Slowloris flows, and
98% of the flows it flags carry the attacking hosts' TCP fingerprint. Refitted
without those TCP columns, it flags 0% of GoldenEye and 0% of Slowloris
(`reports/tables/5_6_unseen_day_tree_without_fingerprint.csv`): its unseen-day alarms come from recognising the
attacking machines.

## Layout

| Path | Content |
|---|---|
| `src/nids/` | Reusable code: configuration, data preparation, features, models, evaluation, plotting |
| `notebooks/` | Four annotated notebooks (01 data preparation, 02 EDA, 03 modelling, 04 evaluation); headings follow the report template, so each report section's evidence is easy to find |
| `data/processed/` | Cleaned, de-duplicated 75,000-row sample (Parquet) and `manifest.json` (counts, seed, SHA-256) |
| `reports/` | Figures (PNG, 300 dpi), tables (CSV) and `numbers.json`, the values the report quotes |
| `requirements-colab.txt` | Pinned library versions for Google Colab |
| `tests/` | Checks for data integrity, train/test leakage, models, course techniques, the unseen day, the predictor, the results table and the executed notebooks (98 tests) |
