"""Single source of truth for paths, seeds and data-handling constants.

Every constant that changes the data or the experiment lives here, with the reason for its value,
so the notebooks and the report can cite one place.

AI assistance: drafted with Claude (Anthropic) and reviewed by the author (report section 1.3).
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Raw data ------------------------------------------------------------------------------------
# Kaggle mirror name of the original AWS file "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv".
RAW_DIR = PROJECT_ROOT / "IDS 2018 Intrusion CSVs (CSE-CIC-IDS2018)"
RAW_CSV = RAW_DIR / "02-14-2018.csv"
ORIGINAL_AWS_FILENAME = "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"
# Column descriptions copied verbatim from CIC Table 3 ("List of extracted traffic features by
# CICFlowMeter-V3"), with the author's notes on gaps and copy errors kept in a separate column.
FEATURE_REFERENCE_CSV = PROJECT_ROOT / "data" / "reference" / "cicflowmeter_features.csv"

LABEL_COL = "Label"
TIMESTAMP_COL = "Timestamp"
DST_PORT_COL = "Dst Port"
PROTOCOL_COL = "Protocol"
FLOW_DURATION_COL = "Flow Duration"

# 200k rows x 78 float64 columns is ~125 MB per chunk: small enough to keep peak memory low,
# large enough that per-chunk overhead is negligible.
CHUNK_SIZE = 200_000

# --- Cleaning and de-duplication -----------------------------------------------------------------
# -1 in these two columns means "no TCP window observed". Inferred from the data, not from CIC
# documentation: -1 appears in 100% of non-TCP flows (protocols 0 and 17), and in TCP flows only
# for the backward window and only when the flow has no backward packets. Negative values anywhere
# else (durations, inter-arrival times, rates) are physically impossible and mark a broken record.
SENTINEL_NEGATIVE_COLS: tuple[str, ...] = ("Init Fwd Win Byts", "Init Bwd Win Byts")

# Columns ignored when deciding whether two flows are duplicates.
#  * Timestamp is a recording time, not a property of the flow; identical flows recorded at
#    different seconds are the same observation for a model that never sees time.
#  * Dst Port is excluded from the main model because it almost encodes the label (FTP = port 21,
#    SSH = port 22). De-duplicating in this smaller feature space guarantees uniqueness in every
#    larger space too, so the "with Dst Port" ablation also stays leakage-free.
DEDUP_IGNORE_COLS: tuple[str, ...] = (TIMESTAMP_COL, DST_PORT_COL)

# --- Sampling ------------------------------------------------------------------------------------
SEED = 42
SAMPLE_SIZE = 75_000
# Each class receives at least min(class size, floor) rows. After de-duplication FTP-BruteForce has
# only 46 unique flows; proportional allocation would keep ~6 and make the class unlearnable.
RARE_CLASS_FLOOR = 2_000

# --- Outputs -------------------------------------------------------------------------------------
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SAMPLE_PARQUET = PROCESSED_DIR / "sample_75k_dedup.parquet"
MANIFEST_JSON = PROCESSED_DIR / "manifest.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
TEXT_DIR = REPORTS_DIR / "text"

# --- Citation (verified 26/10/2026 against the CIC and AWS pages) --------------------------------
DATASET_LICENSE_NOTE = (
    "No standard licence name is given. CIC/UNB terms (dataset page, section 5 'License'): "
    "'You may redistribute, republish, and mirror the CSE-CIC-IDS2018 dataset in any form. "
    "However, any use or redistribution of the data must include a citation to the CSE-CIC-IDS2018 "
    "dataset and a link to this page in AWS.'"
)
DATASET_URLS = {
    "cic": "https://www.unb.ca/cic/datasets/ids-2018.html",
    "aws_registry": "https://registry.opendata.aws/cse-cic-ids2018",
}
