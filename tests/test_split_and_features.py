"""Tests for the train/test split, the leakage check and the base preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

from nids import config, data, features

needs_sample = pytest.mark.skipif(
    not config.SAMPLE_PARQUET.exists(), reason="run notebooks/01_data_preparation.ipynb first"
)


@pytest.fixture(scope="module")
def split() -> data.Split:
    return data.split_sample(data.load_sample())


def _toy_flows() -> pd.DataFrame:
    """A small frame with the columns my preprocessing expects, including one constant column."""
    rng = np.random.default_rng(0)
    n = 120
    return pd.DataFrame(
        {
            "Dst Port": rng.choice([21.0, 22.0, 80.0], n),
            "Protocol": rng.choice([6.0, 17.0], n),
            "Timestamp": ["14/02/2018 10:00:00"] * n,
            "Flow Duration": rng.exponential(1e5, n),
            "Tot Fwd Pkts": rng.integers(1, 20, n).astype(float),
            "Bwd PSH Flags": np.zeros(n),
        }
    )


@needs_sample
def test_split_is_stratified_80_20(split):
    assert len(split.X_train) == 60_000
    assert len(split.X_test) == 15_000
    train_share = split.y_train.value_counts(normalize=True)
    test_share = split.y_test.value_counts(normalize=True)
    assert (train_share - test_share).abs().max() < 0.002
    assert split.y_test.value_counts()["FTP-BruteForce"] == 9


@needs_sample
def test_no_feature_vector_is_shared_between_train_and_test(split):
    report = data.assert_no_leakage(split)
    assert report["shared_vectors"] == 0
    assert report["unique_train_vectors"] == report["train_rows"]


def test_leakage_check_catches_a_copied_flow():
    flows = _toy_flows()
    leaky = data.Split(
        X_train=flows.iloc[:100],
        X_test=pd.concat([flows.iloc[100:], flows.iloc[[0]].set_axis([999])]),
        y_train=pd.Series(["Benign"] * 100),
        y_test=pd.Series(["Benign"] * 21),
    )
    with pytest.raises(AssertionError):
        data.assert_no_leakage(leaky)


@needs_sample
def test_saved_split_round_trips(split, tmp_path):
    path = tmp_path / "split.parquet"
    data.save_split(split, path)
    reloaded = data.load_split(data.load_sample(), path)
    assert reloaded.X_test.index.equals(split.X_test.sort_index().index)
    assert set(reloaded.X_train.index) == set(split.X_train.index)


def test_main_model_never_sees_timestamp_or_port():
    categorical, numeric = features.model_columns(_toy_flows())
    assert categorical == ["Protocol"]
    assert "Dst Port" not in numeric
    assert "Timestamp" not in numeric
    _, with_port = features.model_columns(_toy_flows(), include_dst_port=True)
    assert "Dst Port" in with_port


def test_preprocessor_one_hot_encodes_protocol_and_drops_constants():
    flows = _toy_flows()
    transformed = features.build_preprocessor(flows).fit(flows).get_feature_names_out()
    assert "Bwd PSH Flags" not in transformed
    assert {"Protocol_6.0", "Protocol_17.0"} <= set(transformed)


def test_symlog_handles_the_minus_one_sentinel():
    values = np.array([-1.0, 0.0, 1.0, 1e6])
    out = features.symlog(values)
    assert np.isfinite(out).all()
    assert out[1] == 0.0
    assert out[0] == -out[2]


def test_smote_runs_only_inside_fit():
    flows = _toy_flows()
    y = np.array(["a"] * 100 + ["b"] * 20)
    pipe = features.build_pipeline(
        flows, LogisticRegression(max_iter=500), sampler=SMOTE(k_neighbors=3, random_state=0)
    )
    pipe.fit(flows, y)
    # imblearn skips samplers at predict time: one prediction per input row, no synthetic rows.
    assert len(pipe.predict(flows)) == len(flows)


def test_timing_columns_on_a_toy_frame_keeps_order_and_skips_shape_columns():
    cols = ["Protocol", "Flow Duration", "Tot Fwd Pkts", "Flow Byts/s", "Init Fwd Win Byts"]
    assert features.timing_columns(cols) == ["Flow Duration", "Flow Byts/s"]


@needs_sample
def test_timing_columns_on_the_sample_are_27_numeric_columns(split):
    timing = features.timing_columns(split.X_train.columns)
    assert len(timing) == 27
    assert all(pd.api.types.is_numeric_dtype(split.X_train[c]) for c in timing)
    assert "Init Fwd Win Byts" not in timing and "Timestamp" not in timing


@needs_sample
def test_shape_columns_are_the_model_columns_without_timing(split):
    shape = features.shape_columns(split.X_train)
    _, numeric = features.model_columns(split.X_train)
    assert "Protocol" in shape and "Dst Port" not in shape
    assert set(shape) == ({"Protocol"} | set(numeric)) - set(features.timing_columns(split.X_train.columns))


def test_shape_columns_never_include_the_label_or_text_columns():
    X = _toy_flows().assign(Label="Benign")
    shape = features.shape_columns(X)
    assert config.LABEL_COL not in shape and "Timestamp" not in shape
    assert shape == ["Protocol", "Tot Fwd Pkts", "Bwd PSH Flags"]
