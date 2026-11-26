"""Tests for the configurable paths used when the notebooks run on Google Colab."""

from __future__ import annotations

import importlib

from nids import config


def test_raw_dir_can_be_redirected_with_an_environment_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("NIDS_RAW_DIR", str(tmp_path))
    reloaded = importlib.reload(config)
    try:
        assert tmp_path == reloaded.RAW_DIR
        assert tmp_path / "02-14-2018.csv" == reloaded.RAW_CSV
    finally:
        monkeypatch.delenv("NIDS_RAW_DIR")
        importlib.reload(config)


def test_raw_dir_defaults_to_the_folder_next_to_the_project(monkeypatch):
    monkeypatch.delenv("NIDS_RAW_DIR", raising=False)
    reloaded = importlib.reload(config)
    assert reloaded.RAW_DIR == reloaded.PROJECT_ROOT / "IDS 2018 Intrusion CSVs (CSE-CIC-IDS2018)"


def test_timing_pattern_matches_time_and_rate_columns_only():
    import re

    timing = ["Flow Duration", "Flow IAT Mean", "Fwd IAT Tot", "Active Max", "Idle Min", "Flow Byts/s", "Bwd Pkts/s"]
    shape = ["Tot Fwd Pkts", "TotLen Fwd Pkts", "Init Fwd Win Byts", "Fwd Header Len", "Down/Up Ratio", "Protocol"]
    assert all(re.search(config.TIMING_COL_PATTERN, c) for c in timing)
    assert not any(re.search(config.TIMING_COL_PATTERN, c) for c in shape)
