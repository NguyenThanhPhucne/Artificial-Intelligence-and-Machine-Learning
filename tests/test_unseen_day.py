"""Tests for the unseen-day check (Week 6 lab, "Predict on New Data"): 15 February 2018 traffic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nids import config, data, evaluation


@pytest.fixture(scope="module")
def known() -> pd.DataFrame:
    return data.load_sample().head(200)


def _write_day(tmp_path, known: pd.DataFrame) -> tuple:
    """Five rows identical to known flows (labelled as another day would label them) + five new flows."""
    repeated = known.head(5).copy()
    repeated["Label"] = "Benign"
    repeated["Timestamp"] = "15/02/2018 10:00:00"  # a different time must not make a copy look new
    fresh = known.iloc[5:10].copy()
    fresh["Flow Duration"] = [9_000_001.0, 9_000_002.0, 9_000_003.0, 9_000_004.0, 9_000_005.0]
    fresh["Label"] = ["Benign", "Benign", "DoS attacks-GoldenEye", "DoS attacks-GoldenEye", "DoS attacks-Slowloris"]
    day = pd.concat([repeated, fresh], ignore_index=True)
    day["Label"] = day["Label"].astype(str)
    path = tmp_path / "day.csv"
    day.to_csv(path, index=False)
    return path, fresh


def test_unseen_day_excludes_known_vectors(tmp_path, known):
    path, fresh = _write_day(tmp_path, known)
    flows, stages = data.unseen_day_sample(known, path=path, benign_cap=100, attack_cap=100)
    assert len(flows) == 5
    assert sorted(flows["Flow Duration"]) == sorted(fresh["Flow Duration"])
    assert int(stages.loc["removed_as_known"].sum()) == 5


def test_unseen_day_caps_classes(tmp_path, known):
    path, _ = _write_day(tmp_path, known)
    flows, stages = data.unseen_day_sample(known, path=path, benign_cap=1, attack_cap=1)
    counts = flows["Label"].value_counts()
    assert counts["Benign"] == 1
    assert counts["DoS attacks-GoldenEye"] == 1  # labels from the new day are kept verbatim
    assert counts["DoS attacks-Slowloris"] == 1
    assert list(stages.index) == ["raw", "unique", "removed_as_known", "sampled"]


def test_unseen_day_loads_parquet_without_csv(tmp_path, monkeypatch, known):
    stored = known.head(3).assign(Label=["Benign", "Benign", "DoS attacks-Slowloris"])
    parquet = tmp_path / "unseen.parquet"
    stages_csv = tmp_path / "stages.csv"
    stored.to_parquet(parquet)
    pd.DataFrame({"Benign": [2]}, index=["sampled"]).to_csv(stages_csv)
    monkeypatch.setattr(config, "UNSEEN_DAY_CSV", tmp_path / "missing.csv")
    monkeypatch.setattr(config, "UNSEEN_DAY_PARQUET", parquet)
    monkeypatch.setattr(config, "UNSEEN_DAY_STAGES_CSV", stages_csv)
    flows, stages = data.load_or_build_unseen_day(known)
    assert len(flows) == 3
    assert "sampled" in stages.index


class _AlwaysBenign:
    def predict(self, X):
        return np.array(["Benign"] * len(X))


def test_unseen_day_report_shares_sum_to_one(known):
    flows = known.head(6).assign(Label=["Benign"] * 3 + ["DoS attacks-Slowloris"] * 3)
    report = evaluation.unseen_day_report(_AlwaysBenign(), flows)
    assert report.loc["DoS attacks-Slowloris", "predicted Benign"] == pytest.approx(1.0)
    shares = report.filter(like="predicted ")
    assert np.allclose(shares.sum(axis=1), 1.0)
    assert report.loc["Benign", "flows"] == 3


def test_false_alarm_rate_wilson():
    result = evaluation.wilson_interval(errors=2, n=1_000)
    assert result["rate"] == pytest.approx(0.002)
    assert 0 <= result["ci_low"] < 0.002 < result["ci_high"]
    zero = evaluation.wilson_interval(errors=0, n=1_000)
    assert zero["ci_low"] == 0 and zero["ci_high"] > 0  # zero errors still has an upper bound
