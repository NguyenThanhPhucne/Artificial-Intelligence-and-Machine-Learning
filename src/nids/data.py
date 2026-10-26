"""Data preparation: chunked loading, row-wise cleaning, de-duplication and stratified sampling.

Every step here is row-wise or deterministic (nothing is *learned* from the data), so it can run
before the train/test split without leaking test information. Anything fitted on data (scaling,
SMOTE, feature selection) belongs in the model pipeline instead.

AI assistance: drafted with Claude (Anthropic) and reviewed by the author (report section 1.3).
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

LABEL = config.LABEL_COL
NON_FEATURE_COLS = (config.LABEL_COL, config.TIMESTAMP_COL)


# --- Loading -------------------------------------------------------------------------------------

def raw_columns(path: Path = config.RAW_CSV) -> list[str]:
    """Column names of the raw CSV, read from the header only."""
    return pd.read_csv(path, nrows=0).columns.tolist()


def read_raw_in_chunks(
    path: Path = config.RAW_CSV, chunksize: int = config.CHUNK_SIZE
) -> Iterator[pd.DataFrame]:
    """Yield the raw CSV in chunks with explicit dtypes.

    Explicit float64 avoids pandas' per-chunk type inference (which can disagree between chunks)
    and parses the literal strings 'Infinity' and 'NaN' into IEEE values. float64 rather than
    float32 because large integers such as Flow Duration (> 2**24 microseconds) lose precision in
    float32, which would merge distinct flows and distort the duplicate count.
    """
    dtypes: dict[str, str] = {c: "float64" for c in raw_columns(path) if c not in NON_FEATURE_COLS}
    dtypes.update({config.LABEL_COL: "string", config.TIMESTAMP_COL: "string"})
    yield from pd.read_csv(path, dtype=dtypes, chunksize=chunksize)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric flow features: every column except the label and the timestamp."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def attribute_table(
    frame: pd.DataFrame, reference_path: Path = config.FEATURE_REFERENCE_CSV
) -> pd.DataFrame:
    """One row per column for report section 1.1: CIC description, role and observed values.

    `frame` is the data whose observed statistics are reported (normally the sampling frame).
    Raises if the reference table and the data disagree on column names, so a renamed or missing
    column cannot slip through silently.
    """
    reference = pd.read_csv(reference_path, keep_default_na=False)
    missing = set(frame.columns) ^ set(reference["csv_column"])
    if missing:
        raise ValueError(f"Reference table and data disagree on columns: {sorted(missing)}")

    observed = pd.DataFrame({
        "csv_column": frame.columns,
        "dtype": [str(t) for t in frame.dtypes],
        "n_unique": [frame[c].nunique() for c in frame.columns],
    })
    table = reference.merge(observed, on="csv_column", how="left", validate="one_to_one")
    table["constant"] = table["n_unique"] == 1
    return table


# --- Row-wise cleaning ---------------------------------------------------------------------------

@dataclass
class ChunkCleaning:
    """Rows kept from one chunk plus what was removed and why."""

    kept: pd.DataFrame
    nonfinite_by_label: pd.Series
    negative_by_label: pd.Series
    inf_by_column: pd.Series
    nan_by_column: pd.Series
    negative_by_column: pd.Series


def clean_chunk(chunk: pd.DataFrame) -> ChunkCleaning:
    """Drop rows with non-finite values or physically impossible negative values.

    1. +/-Infinity -> NaN, then drop rows with any NaN. Infinity comes from CICFlowMeter dividing
       by a zero flow duration (bytes/s, packets/s); imputing a rate for a zero-length flow would
       invent a value, and the affected rows are a tiny share of the file.
    2. Drop rows with a negative value outside the "-1 = no TCP window" columns (see
       config.SENTINEL_NEGATIVE_COLS): a negative duration or inter-arrival time is a clock error.
    """
    features = feature_columns(chunk)
    values = chunk[features]
    inf_mask = np.isinf(values.to_numpy())
    inf_by_column = pd.Series(inf_mask.sum(axis=0), index=features)
    nan_by_column = values.isna().sum()

    chunk = chunk.copy()
    chunk[features] = values.replace([np.inf, -np.inf], np.nan)
    nonfinite = chunk[features].isna().any(axis=1)
    finite = chunk.loc[~nonfinite]

    checked = [c for c in features if c not in config.SENTINEL_NEGATIVE_COLS]
    negative_cells = finite[checked] < 0
    negative = negative_cells.any(axis=1)

    return ChunkCleaning(
        kept=finite.loc[~negative],
        nonfinite_by_label=chunk.loc[nonfinite, LABEL].value_counts(),
        negative_by_label=finite.loc[negative, LABEL].value_counts(),
        inf_by_column=inf_by_column,
        nan_by_column=nan_by_column,
        negative_by_column=negative_cells.sum(),
    )


# --- De-duplication ------------------------------------------------------------------------------

def flow_hash(df: pd.DataFrame, ignore: tuple[str, ...] = config.DEDUP_IGNORE_COLS) -> pd.Series:
    """64-bit hash of each row's feature vector, ignoring the label and `ignore` columns.

    Hashing lets us de-duplicate ~1M rows chunk by chunk without holding every row in memory.
    With ~1M rows the chance of any 64-bit collision is about 3e-8. Adding 0.0 maps -0.0 to 0.0
    so equal numbers always hash equally.
    """
    cols = [c for c in df.columns if c != LABEL and c not in ignore]
    return pd.util.hash_pandas_object(df[cols] + 0.0, index=False)


@dataclass
class PreparationReport:
    """Per-label row counts at each cleaning stage, plus column-level detail for section 3.3."""

    raw: pd.Series
    nonfinite_removed: pd.Series
    negative_removed: pd.Series
    after_cleaning: pd.Series
    duplicates_removed: pd.Series
    conflicts_removed: pd.Series
    unique: pd.Series
    inf_by_column: pd.Series
    nan_by_column: pd.Series
    negative_by_column: pd.Series
    n_conflicting_vectors: int

    def stage_table(self, sample: pd.Series | None = None) -> pd.DataFrame:
        """Rows = stages, columns = labels + Total; ready to print or save for the report."""
        stages = {
            "1. Raw file": self.raw,
            "2. Removed: non-finite values (Infinity/NaN)": self.nonfinite_removed,
            "3. Removed: impossible negative values": self.negative_removed,
            "4. After row-wise cleaning": self.after_cleaning,
            "5. Removed: duplicate flows (same features and label)": self.duplicates_removed,
            "6. Removed: same features with different labels": self.conflicts_removed,
            "7. Unique flows (sampling frame)": self.unique,
        }
        if sample is not None:
            stages["8. Stratified sample"] = sample
        table = pd.DataFrame(stages).T.fillna(0).astype(int)
        table = table.reindex(columns=sorted(table.columns))
        table["Total"] = table.sum(axis=1)
        return table


def _sum_series(parts: list[pd.Series]) -> pd.Series:
    if not parts:
        return pd.Series(dtype="int64")
    return pd.concat(parts, axis=1).fillna(0).sum(axis=1).astype("int64")


def build_unique_flows(
    path: Path = config.RAW_CSV, chunksize: int = config.CHUNK_SIZE
) -> tuple[pd.DataFrame, PreparationReport]:
    """Read, clean and de-duplicate the raw file; return the sampling frame and a stage report.

    A flow is a duplicate when its features (ignoring Timestamp and Dst Port) *and* label match an
    earlier row; the first occurrence is kept. Feature vectors that appear under more than one
    label are removed entirely: their ground truth is contradictory and no model can be right on
    both copies.
    """
    raw, nonfinite, negative, cleaned_counts = [], [], [], []
    inf_cols, nan_cols, neg_cols, parts = [], [], [], []

    for chunk in read_raw_in_chunks(path, chunksize):
        raw.append(chunk[LABEL].value_counts())
        result = clean_chunk(chunk)
        nonfinite.append(result.nonfinite_by_label)
        negative.append(result.negative_by_label)
        inf_cols.append(result.inf_by_column)
        nan_cols.append(result.nan_by_column)
        neg_cols.append(result.negative_by_column)
        cleaned_counts.append(result.kept[LABEL].value_counts())
        # De-duplicating inside each chunk first keeps memory small (FTP collapses to ~53 rows).
        kept = result.kept.assign(_flow_hash=flow_hash(result.kept))
        parts.append(kept.drop_duplicates(subset=["_flow_hash", LABEL]))

    flows = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["_flow_hash", LABEL])
    after_cleaning = _sum_series(cleaned_counts)
    deduplicated = flows[LABEL].value_counts()

    labels_per_vector = flows.groupby("_flow_hash")[LABEL].nunique()
    conflicting = labels_per_vector.index[labels_per_vector > 1]
    conflict_mask = flows["_flow_hash"].isin(conflicting)
    conflicts_removed = flows.loc[conflict_mask, LABEL].value_counts()

    flows = flows.loc[~conflict_mask].drop(columns="_flow_hash").reset_index(drop=True)
    flows[LABEL] = flows[LABEL].astype("category")

    report = PreparationReport(
        raw=_sum_series(raw),
        nonfinite_removed=_sum_series(nonfinite),
        negative_removed=_sum_series(negative),
        after_cleaning=after_cleaning,
        duplicates_removed=after_cleaning.sub(deduplicated, fill_value=0).astype("int64"),
        conflicts_removed=conflicts_removed.astype("int64"),
        unique=flows[LABEL].value_counts().astype("int64"),
        inf_by_column=_sum_series(inf_cols),
        nan_by_column=_sum_series(nan_cols),
        negative_by_column=_sum_series(neg_cols),
        n_conflicting_vectors=len(conflicting),
    )
    return flows, report


# --- Sampling ------------------------------------------------------------------------------------

def allocate_with_floor(class_counts: pd.Series, n_total: int, floor: int) -> pd.Series:
    """Rows to draw per class: proportional to class size, but at least min(class size, floor).

    Classes whose proportional share falls below their floor are fixed first; the remaining budget
    is shared proportionally among the other classes using largest-remainder rounding, so the
    allocation sums exactly to n_total.
    """
    counts = class_counts.astype("int64").sort_index()
    if n_total > counts.sum():
        raise ValueError(f"Requested {n_total} rows but only {counts.sum()} are available")

    fixed = pd.Series(dtype="int64")
    remaining, budget = counts.copy(), n_total
    while True:
        share = remaining / remaining.sum() * budget
        below_floor = share < np.minimum(remaining, floor)
        if not below_floor.any():
            break
        take = np.minimum(remaining[below_floor], floor)
        fixed = pd.concat([fixed, take])
        budget -= int(take.sum())
        remaining = remaining[~below_floor]

    share = remaining / remaining.sum() * budget
    base = np.floor(share).astype("int64")
    leftover = budget - int(base.sum())
    base[(share - base).sort_values(ascending=False).index[:leftover]] += 1

    allocation = pd.concat([fixed, base]).reindex(counts.index).astype("int64")
    if allocation.sum() != n_total or (allocation > counts).any():
        raise RuntimeError(f"Invalid allocation {allocation.to_dict()} for counts {counts.to_dict()}")
    return allocation


def stratified_sample(
    flows: pd.DataFrame,
    n_total: int = config.SAMPLE_SIZE,
    floor: int = config.RARE_CLASS_FLOOR,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Draw a reproducible stratified sample (see allocate_with_floor) and shuffle its rows."""
    allocation = allocate_with_floor(flows[LABEL].value_counts(), n_total, floor)
    parts = [
        flows[flows[LABEL] == label].sample(n=int(n), random_state=seed)
        for label, n in allocation.items()
    ]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


# --- Persistence ---------------------------------------------------------------------------------

def sha256_of(path: Path, block: int = 1 << 20) -> str:
    """SHA-256 of a file, read in 1 MB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def save_sample(
    sample: pd.DataFrame,
    report: PreparationReport,
    parquet_path: Path = config.SAMPLE_PARQUET,
    manifest_path: Path = config.MANIFEST_JSON,
    raw_path: Path = config.RAW_CSV,
) -> dict:
    """Write the sample as Parquet and a JSON manifest that lets anyone verify and rebuild it."""
    import imblearn
    import sklearn

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(parquet_path, index=False)

    stages = report.stage_table(sample[LABEL].value_counts())
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_file": raw_path.name,
        "raw_original_aws_name": config.ORIGINAL_AWS_FILENAME,
        "raw_sha256": sha256_of(raw_path),
        "raw_size_bytes": raw_path.stat().st_size,
        "seed": config.SEED,
        "sample_size": config.SAMPLE_SIZE,
        "rare_class_floor": config.RARE_CLASS_FLOOR,
        "dedup_ignore_columns": list(config.DEDUP_IGNORE_COLS),
        "sentinel_negative_columns": list(config.SENTINEL_NEGATIVE_COLS),
        "n_conflicting_vectors_removed": report.n_conflicting_vectors,
        "stage_counts": {stage: row.to_dict() for stage, row in stages.iterrows()},
        "sample_file": parquet_path.name,
        "sample_sha256": sha256_of(parquet_path),
        "sample_shape": list(sample.shape),
        "versions": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "imbalanced-learn": imblearn.__version__,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def load_sample(path: Path = config.SAMPLE_PARQUET) -> pd.DataFrame:
    """Load the processed sample with the label as a categorical column."""
    sample = pd.read_parquet(path)
    sample[LABEL] = sample[LABEL].astype("category")
    return sample
