"""Tests for the command-line predictor that labels a new CICFlowMeter CSV (Week 6 lab, new data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nids import data, predict

CLASSES = {"Benign", "FTP-BruteForce", "SSH-Bruteforce"}


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("models") / "final_decision_tree.joblib"
    mp = pytest.MonkeyPatch()
    mp.setattr(predict, "FINAL_MODEL_PATH", path)
    fitted = predict.load_or_fit_final_model()
    yield fitted
    mp.undo()


def _rows(n: int = 20) -> pd.DataFrame:
    rows = data.load_sample().sample(n=n, random_state=0).reset_index(drop=True)
    rows["Label"] = rows["Label"].astype(str)
    return rows


def test_predict_csv_returns_label_per_kept_row(tmp_path, model):
    path = tmp_path / "flows.csv"
    _rows().to_csv(path, index=False)
    predictions, dropped = predict.predict_csv(path, model=model)
    assert len(predictions) == 20 and dropped == 0
    assert set(predictions["predicted_label"]) <= CLASSES


def test_predict_works_without_label_column(tmp_path, model):
    path = tmp_path / "flows.csv"
    _rows(5).drop(columns="Label").to_csv(path, index=False)
    predictions, _ = predict.predict_csv(path, model=model)
    assert len(predictions) == 5


def test_predict_reports_dropped_rows(tmp_path, model):
    rows = _rows()
    rows.loc[0, "Flow Byts/s"] = np.inf
    rows.loc[1, "Flow Duration"] = -5.0
    path = tmp_path / "flows.csv"
    rows.to_csv(path, index=False)
    predictions, dropped = predict.predict_csv(path, model=model)
    assert dropped == 2
    assert len(predictions) == 18
    assert set(predictions["row"]) == set(range(2, 20))  # row numbers refer to the input file


def test_predict_missing_column_message(tmp_path, model):
    path = tmp_path / "flows.csv"
    _rows(5).drop(columns="Flow Duration").to_csv(path, index=False)
    with pytest.raises(ValueError, match="Flow Duration"):
        predict.predict_csv(path, model=model)
