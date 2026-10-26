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
