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
| `02_exploratory_analysis.ipynb` | 2.1–2.4 – stratified split, leakage check, EDA on the training set | ~20 s |
| `03_modelling.ipynb` | 3.1, 3.2, 3.4, 4.1, 4.2 – baselines, randomised search, cross-validation | ~4 min first run, ~30 s cached |
| `04_evaluation.ipynb` | 5, 6.1, 6.2 – test set, ablations, model choice, interpretation | ~1.5 min |

To reproduce everything from a clean kernel, including refitting every model instead of loading the cached searches
from `models/`:

```bash
for nb in 01_data_preparation 02_exploratory_analysis 03_modelling 04_evaluation; do
  NIDS_FORCE_RETRAIN=1 uv run jupyter nbconvert --to notebook --execute --inplace notebooks/$nb.ipynb
done
```

All randomness is seeded (seed 42); a full refit reproduces every reported table exactly (only timings vary).

## Main results

| Model (hyper-parameters selected by CV macro F1) | CV macro F1 | Test accuracy | Test FTP recall | Training (s) | Inference (ms / 1,000 flows) |
|---|---|---|---|---|---|
| Logistic Regression | 0.9837 ± 0.0155 | 0.99987 | 9 / 9 | 0.69 | 1.22 |
| **Decision Tree (selected)** | 0.9953 ± 0.0089 | 0.99993 | 9 / 9 | 0.45 | 0.58 |
| Random Forest | 0.9955 ± 0.0089 | 0.99993 | 9 / 9 | 3.80 | 3.12 |

The exact figures, with confidence intervals, are in `reports/tables/5_5_final_comparison.csv`; timings vary slightly
between runs. Ablations show that the destination port is a label shortcut, that SMOTE brings no benefit on this data,
and that Logistic Regression and the Random Forest partly rely on TCP values fixed by the attacking machines.

## Layout

| Path | Content |
|---|---|
| `src/nids/` | Reusable code: configuration, data preparation, features, models, evaluation, plotting |
| `notebooks/` | Annotated notebooks whose headings follow the report template |
| `data/processed/` | Cleaned, de-duplicated 75,000-row sample (Parquet) and `manifest.json` (counts, seed, SHA-256) |
| `reports/` | Figures (PNG, 300 dpi), tables (CSV) and copy-ready text for the report |
| `tests/` | Checks for data integrity and train/test leakage |
