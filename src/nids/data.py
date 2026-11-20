"""Data preparation: chunked loading, row-wise cleaning, de-duplication, sampling and splitting.

I keep every step before the train/test split row-wise or deterministic: nothing here is *learned*
from the data, so running it before the split cannot leak test information. Anything fitted to data
(scaling, SMOTE, feature selection, tuning) lives in the model pipeline instead (see features.py).
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
from sklearn.model_selection import train_test_split

from . import config

LABEL = config.LABEL_COL
NON_FEATURE_COLS = (config.LABEL_COL, config.TIMESTAMP_COL)


# --- Loading -------------------------------------------------------------------------------------

def raw_columns(path: Path = config.RAW_CSV) -> list[str]:
    """Return the column names of the raw CSV by reading its header only."""
    return pd.read_csv(path, nrows=0).columns.tolist()


def read_raw_in_chunks(
    path: Path = config.RAW_CSV, chunksize: int = config.CHUNK_SIZE
) -> Iterator[pd.DataFrame]:
    """Yield the raw CSV in chunks with explicit dtypes.

    I fix every feature to float64 so that pandas does not infer types per chunk (which can disagree
    between chunks), and so that the literal strings 'Infinity' and 'NaN' parse into IEEE values.
    I deliberately avoid float32: large integers such as Flow Duration (> 2**24 microseconds) lose
    precision in float32, which would merge distinct flows and distort the duplicate count.
    """
    dtypes: dict[str, str] = {c: "float64" for c in raw_columns(path) if c not in NON_FEATURE_COLS}
    dtypes.update({config.LABEL_COL: "string", config.TIMESTAMP_COL: "string"})
    yield from pd.read_csv(path, dtype=dtypes, chunksize=chunksize)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the flow attributes: every column except the label and the timestamp."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def attribute_table(
    frame: pd.DataFrame, reference_path: Path = config.FEATURE_REFERENCE_CSV
) -> pd.DataFrame:
    """Build one row per column for report section 1.1: CIC description, role and observed values.

    I raise an error when the reference table and the data disagree on column names, so that a
    renamed or missing column cannot slip into the report unnoticed.
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
    table["level_of_measurement"] = [
        level_of_measurement(row.csv_column, row.role, frame[row.csv_column])
        for row in table.itertuples()
    ]
    return table


def level_of_measurement(column: str, role: str, values: pd.Series) -> str:
    """Classify a column by Stevens' levels of measurement (weeks 2-3 of the module).

    * Label and Protocol are nominal: the numbers 0, 6, 17 name protocols and have no order.
    * Dst Port is nominal too: port 22 is not "more" than port 21.
    * Timestamp is interval: differences between times are meaningful, but there is no true zero.
    * Every other column is a count, a size in bytes, a duration or a rate, all with a true zero,
      so they are ratio scale; I add whether the observed values are whole numbers (discrete) or not
      (continuous). The -1 "no window observed" code in the window columns is a sentinel inside an
      otherwise ratio-scale count, which I note in the report rather than here.
    """
    if column in (config.LABEL_COL, config.PROTOCOL_COL, config.DST_PORT_COL):
        return "nominal"
    if column == config.TIMESTAMP_COL:
        return "interval"
    numeric = pd.to_numeric(values, errors="coerce")
    discrete = bool(np.all(np.mod(numeric.dropna(), 1) == 0))
    return f"ratio ({'discrete' if discrete else 'continuous'})"


# --- Row-wise cleaning ---------------------------------------------------------------------------

@dataclass
class ChunkCleaning:
    """Rows I keep from one chunk, plus what I removed and why."""

    kept: pd.DataFrame
    nonfinite_by_label: pd.Series
    negative_by_label: pd.Series
    inf_by_column: pd.Series
    nan_by_column: pd.Series
    negative_by_column: pd.Series


def clean_chunk(chunk: pd.DataFrame) -> ChunkCleaning:
    """Drop rows with non-finite values or physically impossible negative values.

    1. I convert +/-Infinity to NaN and drop every row containing NaN. Infinity comes from
       CICFlowMeter dividing by a zero flow duration (bytes/s, packets/s); imputing a rate for a
       zero-length flow would invent a value, and these rows are a tiny share of the file.
    2. I drop rows with a negative value outside the "-1 = no TCP window" columns
       (config.SENTINEL_NEGATIVE_COLS), because a negative duration or inter-arrival time can only
       be a clock error.
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
    """Return a 64-bit hash of each row's feature vector, ignoring the label and `ignore` columns.

    I hash rows so that I can de-duplicate ~1M rows chunk by chunk without holding every row in
    memory; with ~1M rows the probability of any 64-bit collision is about 3e-8. Adding 0.0 maps
    -0.0 to 0.0, so equal numbers always hash equally.
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
        """Return rows = stages, columns = labels + Total, ready to print or save for the report."""
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

    I treat a flow as a duplicate when its features (ignoring Timestamp and Dst Port) *and* its
    label match an earlier row, and keep the first occurrence. I remove feature vectors that appear
    under more than one label entirely: their ground truth is contradictory, so no model can be
    right on both copies, and keeping either copy would be an arbitrary choice.
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
        # I de-duplicate inside each chunk first to keep memory small (FTP collapses to a few rows).
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


def build_cleaned_rows(
    path: Path = config.RAW_CSV, chunksize: int = config.CHUNK_SIZE
) -> pd.DataFrame:
    """Return every row that survives row-wise cleaning, *without* de-duplication.

    I only use this for the ablation that shows what a naive pipeline would report: sampling and
    splitting these rows puts copies of the same flow in both the training and the test set.
    """
    parts = [clean_chunk(chunk).kept for chunk in read_raw_in_chunks(path, chunksize)]
    rows = pd.concat(parts, ignore_index=True)
    rows[LABEL] = rows[LABEL].astype("category")
    return rows


def train_test_overlap(split: Split) -> pd.Series:
    """Share of test flows (per class) whose feature vector also occurs in the training set."""
    train_hashes = set(flow_hash(split.X_train))
    in_train = flow_hash(split.X_test).isin(train_hashes)
    return in_train.groupby(split.y_test.to_numpy(), observed=True).mean()


# --- Sampling ------------------------------------------------------------------------------------

def allocate_with_floor(class_counts: pd.Series, n_total: int, floor: int) -> pd.Series:
    """Return rows to draw per class: proportional to class size, but at least min(size, floor).

    I fix the classes whose proportional share falls below their floor first, then share the
    remaining budget proportionally among the other classes with largest-remainder rounding, so the
    allocation always sums exactly to n_total.
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


# --- Train/test split ----------------------------------------------------------------------------

@dataclass
class Split:
    """Training and test partitions of the sample, with features and target separated."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split_sample(
    sample: pd.DataFrame, test_size: float = config.TEST_SIZE, seed: int = config.SEED
) -> Split:
    """Split the sample 80/20, stratified by label.

    I stratify so that the 46 FTP-BruteForce flows are shared between training and test in the same
    proportion as the other classes; a purely random split could leave the test set with almost no
    FTP flows by chance. The split is made once and the test set is not looked at again until the
    final evaluation: EDA, scaling, SMOTE and tuning all use the training set only.
    """
    X = sample.drop(columns=LABEL)
    y = sample[LABEL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    return Split(X_train, X_test, y_train, y_test)


def assert_no_leakage(split: Split) -> dict[str, int]:
    """Prove that no flow (in the de-duplication feature space) appears in both partitions.

    I compare the feature-vector hashes of the two partitions rather than row indices, because the
    leakage that matters is an identical flow being memorised in training and then "predicted" in
    testing. I also re-check uniqueness inside each partition. Returns the counts I checked.
    """
    train_hashes = set(flow_hash(split.X_train))
    test_hashes = set(flow_hash(split.X_test))
    shared = train_hashes & test_hashes
    if shared:
        raise AssertionError(f"{len(shared)} feature vectors occur in both train and test")
    if len(train_hashes) != len(split.X_train) or len(test_hashes) != len(split.X_test):
        raise AssertionError("Duplicate feature vectors found inside a partition")
    if set(split.X_train.index) & set(split.X_test.index):
        raise AssertionError("A sample row was assigned to both partitions")
    return {
        "train_rows": len(split.X_train),
        "test_rows": len(split.X_test),
        "unique_train_vectors": len(train_hashes),
        "unique_test_vectors": len(test_hashes),
        "shared_vectors": len(shared),
    }


def save_split(split: Split, path: Path = config.SPLIT_PARQUET) -> None:
    """Store which sample row went to which partition, so later notebooks reuse the exact split."""
    assignment = pd.concat([
        pd.DataFrame({"row": split.X_train.index, "partition": "train"}),
        pd.DataFrame({"row": split.X_test.index, "partition": "test"}),
    ]).sort_values("row")
    assignment.to_parquet(path, index=False)


def load_split(sample: pd.DataFrame, path: Path = config.SPLIT_PARQUET) -> Split:
    """Rebuild the saved split from the sample and the stored row assignment."""
    assignment = pd.read_parquet(path).set_index("row")["partition"]
    is_test = assignment.reindex(sample.index).eq("test").to_numpy()
    X = sample.drop(columns=LABEL)
    y = sample[LABEL]
    return Split(X[~is_test], X[is_test], y[~is_test], y[is_test])


# --- Persistence ---------------------------------------------------------------------------------

def sha256_of(path: Path, block: int = 1 << 20) -> str:
    """Return the SHA-256 of a file, read in 1 MB blocks."""
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
    """Write the sample as Parquet plus a JSON manifest that lets anyone verify and rebuild it."""
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


# --- Unseen day (Week 6 lab: predict on new data) --------------------------------------------------

def unseen_day_sample(
    known: pd.DataFrame,
    path: Path | None = None,
    benign_cap: int = config.UNSEEN_BENIGN_CAP,
    attack_cap: int = config.UNSEEN_ATTACK_CAP,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean and de-duplicate another day's file exactly as I did for 14 February, then sample it.

    Any flow whose feature vector (ignoring Timestamp and Dst Port) already appears in `known`
    (my 75,000-flow sample) is removed: otherwise a "new day" score would partly re-test flows the
    model was trained or tested on, and the false-alarm rate would look better than it is.
    Returns the sampled flows and a table of row counts per label at each stage.
    """
    path = config.UNSEEN_DAY_CSV if path is None else path
    flows, report = build_unique_flows(path)
    known_hashes = set(flow_hash(known[flows.columns]))
    is_known = flow_hash(flows).isin(known_hashes).to_numpy()
    removed = flows.loc[is_known, LABEL].value_counts()
    fresh = flows.loc[~is_known]
    parts = []
    for label, group in fresh.groupby(LABEL, observed=True):
        cap = benign_cap if label == "Benign" else attack_cap
        parts.append(group.sample(n=min(cap, len(group)), random_state=seed))
    sample = pd.concat(parts).sort_index().reset_index(drop=True)
    sample[LABEL] = sample[LABEL].astype(str)
    stages = pd.DataFrame(
        {
            "raw": report.raw,
            "unique": report.unique,
            "removed_as_known": removed,
            "sampled": sample[LABEL].value_counts(),
        }
    ).T.fillna(0).astype("int64")
    stages.columns = [str(c) for c in stages.columns]
    return sample, stages


def load_or_build_unseen_day(known: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the stored unseen-day sample if it exists, otherwise build and store it.

    Building needs the 350 MB CSV of 15 February; the stored parquet (a few MB) lets the notebooks
    run on Google Colab without uploading that file.
    """
    if config.UNSEEN_DAY_PARQUET.exists() and config.UNSEEN_DAY_STAGES_CSV.exists():
        return pd.read_parquet(config.UNSEEN_DAY_PARQUET), pd.read_csv(config.UNSEEN_DAY_STAGES_CSV, index_col=0)
    sample, stages = unseen_day_sample(known)
    config.UNSEEN_DAY_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(config.UNSEEN_DAY_PARQUET, index=False)
    config.UNSEEN_DAY_STAGES_CSV.parent.mkdir(parents=True, exist_ok=True)
    stages.to_csv(config.UNSEEN_DAY_STAGES_CSV)
    return sample, stages
