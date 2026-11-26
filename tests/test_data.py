"""Tests for the data-preparation step (cleaning, de-duplication, sampling, saved artefacts)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from nids import config, data


def _chunk(rows: list[dict]) -> pd.DataFrame:
    """Minimal raw-like chunk with the columns the cleaning rules depend on."""
    base = {
        "Dst Port": 22.0,
        "Protocol": 6.0,
        "Timestamp": "14/02/2018 08:31:01",
        "Flow Duration": 100.0,
        "Flow Byts/s": 10.0,
        "Flow Pkts/s": 1.0,
        "Init Fwd Win Byts": 512.0,
        "Label": "Benign",
    }
    frame = pd.DataFrame([{**base, **row} for row in rows])
    frame["Label"] = frame["Label"].astype("string")
    frame["Timestamp"] = frame["Timestamp"].astype("string")
    return frame


def test_clean_chunk_removes_non_finite_and_negative_rows_but_keeps_sentinels():
    chunk = _chunk(
        [
            {},  # valid
            {"Flow Byts/s": np.inf},  # +Infinity
            {"Flow Pkts/s": -np.inf},  # -Infinity
            {"Flow Byts/s": np.nan},  # NaN
            {"Flow Duration": -5.0},  # impossible negative duration
            {"Init Fwd Win Byts": -1.0},  # documented sentinel: must be kept
        ]
    )
    result = data.clean_chunk(chunk)

    assert len(result.kept) == 2
    assert result.kept["Init Fwd Win Byts"].tolist() == [512.0, -1.0]
    assert result.nonfinite_by_label["Benign"] == 3
    assert result.negative_by_label["Benign"] == 1
    assert result.inf_by_column["Flow Byts/s"] == 1
    assert result.inf_by_column["Flow Pkts/s"] == 1
    assert result.nan_by_column["Flow Byts/s"] == 1


def test_flow_hash_ignores_timestamp_and_port_but_not_features():
    chunk = _chunk(
        [
            {},
            {"Timestamp": "14/02/2018 09:00:00", "Dst Port": 21.0},  # same flow, other time/port
            {"Flow Duration": 101.0},  # different flow
            {"Flow Duration": 0.0},
            {"Flow Duration": -0.0},  # -0.0 must equal 0.0
        ]
    )
    hashes = data.flow_hash(chunk)
    assert hashes[0] == hashes[1]
    assert hashes[0] != hashes[2]
    assert hashes[3] == hashes[4]


def test_allocation_keeps_rare_class_whole_and_sums_exactly():
    counts = pd.Series({"Benign": 525_240, "FTP-BruteForce": 46, "SSH-Bruteforce": 94_041})
    allocation = data.allocate_with_floor(counts, 75_000, floor=2_000)
    assert allocation.sum() == 75_000
    assert allocation["FTP-BruteForce"] == 46
    # The two large classes keep their relative proportions (within rounding).
    ratio = allocation["SSH-Bruteforce"] / allocation["Benign"]
    assert ratio == pytest.approx(94_041 / 525_240, rel=1e-3)


def test_allocation_raises_small_class_to_floor():
    counts = pd.Series({"A": 100_000, "B": 5_000})
    allocation = data.allocate_with_floor(counts, 10_000, floor=2_000)
    assert allocation.to_dict() == {"A": 8_000, "B": 2_000}


def test_allocation_rejects_oversized_request():
    counts = pd.Series({"A": 10})
    with pytest.raises(ValueError):
        data.allocate_with_floor(counts, 11, floor=2)


def test_stratified_sample_is_deterministic():
    rng = np.random.default_rng(0)
    flows = pd.DataFrame(
        {
            "x": rng.normal(size=1_000),
            "Label": ["a"] * 900 + ["b"] * 95 + ["c"] * 5,
        }
    )
    first = data.stratified_sample(flows, n_total=100, floor=10, seed=42)
    second = data.stratified_sample(flows, n_total=100, floor=10, seed=42)
    assert first.equals(second)
    # Proportional shares would be a=90, b=9.5, c=0.5: b is lifted to the floor (10), c is kept
    # whole (5 rows < floor) and a receives the remaining 85.
    assert first["Label"].value_counts().to_dict() == {"a": 85, "b": 10, "c": 5}


def test_feature_reference_covers_exactly_the_raw_columns():
    if not config.RAW_CSV.exists():
        pytest.skip("raw CSV not present")
    reference = pd.read_csv(config.FEATURE_REFERENCE_CSV, keep_default_na=False)
    assert reference["csv_column"].tolist() == data.raw_columns()


# --- Saved artefacts (skipped until notebook 01 has been run) ------------------------------------

needs_sample = pytest.mark.skipif(
    not config.SAMPLE_PARQUET.exists(), reason="run notebooks/01_data_preparation.ipynb first"
)


@needs_sample
def test_saved_sample_has_no_duplicate_or_contradictory_flows():
    sample = data.load_sample()
    hashes = data.flow_hash(sample)
    assert len(sample) == config.SAMPLE_SIZE
    assert hashes.is_unique, "every flow in the sample must be unique in the main feature space"
    assert not sample.isna().any().any()
    assert np.isfinite(sample[data.feature_columns(sample)].to_numpy()).all()


@needs_sample
def test_manifest_matches_saved_sample():
    manifest = json.loads(config.MANIFEST_JSON.read_text())
    assert manifest["sample_sha256"] == data.sha256_of(config.SAMPLE_PARQUET)
    assert manifest["seed"] == config.SEED
    sample_counts = manifest["stage_counts"]["8. Stratified sample"]
    assert sample_counts["Total"] == config.SAMPLE_SIZE
