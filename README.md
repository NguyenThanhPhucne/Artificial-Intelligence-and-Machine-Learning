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

Terms of use (Canadian Institute for Cybersecurity, <https://www.unb.ca/cic/datasets/ids-2018.html>, section 5 “License”):
redistribution is permitted in any form provided that any use cites the CSE-CIC-IDS2018 dataset and links to the AWS
registry page. Dataset paper: Sharafaldin, Lashkari and Ghorbani (2018), ICISSP.

## Setup (Apple Silicon, Python 3.12 arm64)

```bash
uv sync                      # creates .venv from uv.lock
uv run pytest                # data-integrity and leakage tests
uv run jupyter notebook      # notebooks/01_… → 03_…, run in order
```

To re-execute a notebook end to end from a clean kernel:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_preparation.ipynb
```

## Layout

| Path | Content |
|---|---|
| `src/nids/` | Reusable code: configuration, data preparation, features, models, evaluation, plotting |
| `notebooks/` | Annotated notebooks whose headings follow the report template |
| `data/processed/` | Cleaned, de-duplicated 75,000-row sample (Parquet) and `manifest.json` (counts, seed, SHA-256) |
| `reports/` | Figures (PNG, 300 dpi), tables (CSV) and copy-ready text for the report |
| `tests/` | Checks for data integrity and train/test leakage |
