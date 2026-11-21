"""Single source of truth for paths, seeds and data-handling constants.

I keep every constant that changes the data or the experiment in this module, next to the reason
for its value, so that the notebooks and the report cite one place and a reader can audit every
decision without searching through the code.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Raw data ------------------------------------------------------------------------------------
# I use the Kaggle mirror, which renames the original AWS file
# "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv" to "02-14-2018.csv"; the content is the same day.
# The NIDS_RAW_DIR environment variable can point elsewhere, e.g. to a Google Drive folder on Colab,
# without editing the code; by default the CSVs sit next to the project.
RAW_DIR = Path(os.environ.get("NIDS_RAW_DIR", PROJECT_ROOT / "IDS 2018 Intrusion CSVs (CSE-CIC-IDS2018)"))
RAW_CSV = RAW_DIR / "02-14-2018.csv"
ORIGINAL_AWS_FILENAME = "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"
# I copied the column descriptions verbatim from CIC Table 3 ("List of extracted traffic features
# by CICFlowMeter-V3") and kept my own notes on its gaps and copy errors in a separate column, so
# the source wording and my interpretation are never mixed.
FEATURE_REFERENCE_CSV = PROJECT_ROOT / "data" / "reference" / "cicflowmeter_features.csv"

LABEL_COL = "Label"
TIMESTAMP_COL = "Timestamp"
DST_PORT_COL = "Dst Port"
PROTOCOL_COL = "Protocol"
FLOW_DURATION_COL = "Flow Duration"

# I read 200k rows per chunk: about 125 MB of float64 per chunk, which keeps peak memory low while
# the per-chunk overhead stays negligible.
CHUNK_SIZE = 200_000

# --- Cleaning and de-duplication -----------------------------------------------------------------
# I keep -1 in these two columns because it means "no TCP window observed", not an error. I inferred
# this from the data (CIC does not document it): -1 appears in 100% of non-TCP flows (protocols 0
# and 17), and in TCP flows only in the backward column and only when the flow has no backward
# packets. Negative values anywhere else (durations, inter-arrival times, rates) are physically
# impossible, so I treat them as broken records.
SENTINEL_NEGATIVE_COLS: tuple[str, ...] = ("Init Fwd Win Byts", "Init Bwd Win Byts")

# Columns I ignore when deciding whether two flows are duplicates.
#  * Timestamp records when a flow was seen, not what it was; identical flows recorded at different
#    seconds are the same observation for a model that never sees time.
#  * Dst Port is excluded from my main model because it almost encodes the label (FTP = port 21,
#    SSH = port 22). De-duplicating in this smaller feature space guarantees uniqueness in every
#    larger space too, so my "with Dst Port" ablation also stays leakage-free.
DEDUP_IGNORE_COLS: tuple[str, ...] = (TIMESTAMP_COL, DST_PORT_COL)

# --- Sampling ------------------------------------------------------------------------------------
SEED = 42
SAMPLE_SIZE = 75_000
# I give each class at least min(class size, floor) rows. After de-duplication FTP-BruteForce has
# only 46 unique flows; proportional allocation would keep about 6 and make the class unlearnable.
RARE_CLASS_FLOOR = 2_000

# --- Splitting and modelling ---------------------------------------------------------------------
# 80/20 leaves 60,000 flows for 5-fold cross-validation and 15,000 for a test set that is touched once.
TEST_SIZE = 0.20
CV_FOLDS = 5
# Columns my main model never sees: Timestamp is a recording time on a broken 12-hour clock, and
# Dst Port (like time) was one of the inputs CIC used to create the labels, so a model using them
# would learn the labelling procedure rather than the traffic.
MODEL_EXCLUDED_COLS: tuple[str, ...] = (TIMESTAMP_COL, DST_PORT_COL)
CATEGORICAL_COLS: tuple[str, ...] = (PROTOCOL_COL,)
# Feature-selection ablation (report Step 5): 0.70 is the "high correlation" cut-off used in Lab 5;
# 0.95 is a conservative setting that only merges near-duplicate columns. Testing both shows whether
# any loss comes from removing redundancy or from removing genuinely different information.
CORRELATION_THRESHOLDS: tuple[float, ...] = (0.70, 0.95)

# --- Outputs -------------------------------------------------------------------------------------
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SAMPLE_PARQUET = PROCESSED_DIR / "sample_75k_dedup.parquet"
MANIFEST_JSON = PROCESSED_DIR / "manifest.json"
SPLIT_PARQUET = PROCESSED_DIR / "split_assignment.parquet"
# Unseen-day check (Week 6 lab, "Predict on New Data"): 15 February 2018 from the same dataset. It
# holds benign traffic from another day and two DoS attacks my models never saw (GoldenEye,
# Slowloris). I cap the sample so that it stays small enough to copy to Google Drive with the code.
UNSEEN_DAY_CSV = RAW_DIR / "02-15-2018.csv"
UNSEEN_DAY_PARQUET = PROCESSED_DIR / "unseen_day_02-15_sample.parquet"
UNSEEN_BENIGN_CAP = 20_000
UNSEEN_ATTACK_CAP = 5_000

# Fitted searches are cached here (git-ignored); notebooks rebuild them when FORCE_RETRAIN is set.
MODELS_DIR = PROJECT_ROOT / "models"
N_SEARCH_ITER = 20

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURE_DPI = 300

# --- Licence and citation (checked 26/10/2026) ---------------------------------------------------
DATASET_LICENSE = "CC BY 4.0 (Attribution 4.0 International), as listed for the Kaggle mirror"
DATASET_CITATION_TERMS = (
    "CIC/UNB dataset page, section 5 'License': 'You may redistribute, republish, and mirror the "
    "CSE-CIC-IDS2018 dataset in any form. However, any use or redistribution of the data must include "
    "a citation to the CSE-CIC-IDS2018 dataset and a link to this page in AWS.'"
)
DATASET_URLS = {
    "license": "https://creativecommons.org/licenses/by/4.0/",
    "cic": "https://www.unb.ca/cic/datasets/ids-2018.html",
    "aws_registry": "https://registry.opendata.aws/cse-cic-ids2018",
    "paper_doi": "https://doi.org/10.5220/0006639801080116",
}
UNSEEN_DAY_STAGES_CSV = TABLES_DIR / "5_6_unseen_day_stages.csv"
