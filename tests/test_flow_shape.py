"""Tests for the flow "shape" (every packet count, size, header, flag and window, but no timing column).

Two flows with the same shape exchanged the same packets and differ only in when the packets arrived. I use the
shape to describe the FTP class (H4) and to group near-identical flows for grouped cross-validation (H2).
"""

from __future__ import annotations

import pandas as pd
import pytest

from nids import config, data, eda

needs_sample = pytest.mark.skipif(
    not config.SAMPLE_PARQUET.exists(), reason="run notebooks/01_data_preparation.ipynb first"
)


def _flows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Dst Port": [21.0, 22.0, 21.0, 21.0],
            "Protocol": [6.0, 6.0, 6.0, 6.0],
            "Timestamp": ["14/02/2018 10:00:00"] * 4,
            "Flow Duration": [5.0, 63.0, 5.0, 5.0],
            "Flow Byts/s": [0.0, 0.0, 0.0, 0.0],
            "Tot Fwd Pkts": [1.0, 1.0, 2.0, 1.0],
            "TotLen Fwd Pkts": [0.0, 0.0, 0.0, 0.0],
        }
    )


def test_flows_that_differ_only_in_timing_or_port_share_a_signature():
    signature = eda.shape_signature(_flows())
    # Rows 0 and 1 differ in Flow Duration and Dst Port only; row 3 is row 0 again.
    assert signature.iloc[0] == signature.iloc[1] == signature.iloc[3]
    assert list(signature.index) == list(_flows().index)


def test_a_different_packet_count_gives_a_different_signature():
    signature = eda.shape_signature(_flows())
    assert signature.iloc[2] != signature.iloc[0]


def test_class_shape_profile_counts_flows_shapes_and_the_largest_shape():
    y = pd.Series(["FTP-BruteForce", "FTP-BruteForce", "SSH-Bruteforce", "SSH-Bruteforce"])
    profile = eda.class_shape_profile(_flows(), y)
    assert profile.loc["FTP-BruteForce"].to_dict() == {
        "flows": 2,
        "shapes": 1,
        "largest shape": 2,
        "largest shape share": 1.0,
    }
    assert profile.loc["SSH-Bruteforce", "shapes"] == 2


def test_packet_pattern_reports_min_and_max_of_the_packet_columns_and_the_duration():
    y = pd.Series(["FTP-BruteForce", "FTP-BruteForce", "SSH-Bruteforce", "SSH-Bruteforce"])
    pattern = eda.packet_pattern(_flows(), y, "FTP-BruteForce")
    assert pattern.loc["Tot Fwd Pkts"].to_dict() == {"min": 1.0, "max": 1.0}
    assert pattern.loc["Flow Duration"].to_dict() == {"min": 5.0, "max": 63.0}
    assert "Bwd Header Len" not in pattern.index  # columns missing from the frame are skipped


@needs_sample
def test_on_the_training_set_every_ftp_flow_has_one_shape_and_ssh_has_few():
    split = data.load_split(data.load_sample())
    profile = eda.class_shape_profile(split.X_train, split.y_train)
    assert profile.loc["FTP-BruteForce", "shapes"] == 1
    assert profile.loc["SSH-Bruteforce", "shapes"] < profile.loc["SSH-Bruteforce", "flows"] / 20
    pattern = eda.packet_pattern(split.X_train, split.y_train, "FTP-BruteForce")
    assert (pattern.drop(index="Flow Duration")["min"] == pattern.drop(index="Flow Duration")["max"]).all()
    assert pattern.loc["TotLen Fwd Pkts", "max"] == 0 and pattern.loc["TotLen Bwd Pkts", "max"] == 0
