"""Tests for the nearest-training-flow distances (H1: are test flows near-copies of training flows?)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from nids import evaluation


def _train() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame({"a": [0.0, 3.0, 0.0], "b": [0.0, 0.0, 4.0]}, index=[10, 11, 12])
    y = pd.Series(["Benign", "SSH-Bruteforce", "FTP-BruteForce"], index=X.index)
    return X, y


class Identity:
    """A fitted 'preprocess' that changes nothing, so distances can be checked by hand."""

    def transform(self, X):
        return np.asarray(X, dtype=float)


def test_distance_and_label_of_the_nearest_training_flow_match_a_hand_calculation():
    X, y = _train()
    other = pd.DataFrame({"a": [0.0, 3.0, 1.0], "b": [0.0, 1.0, 4.0]}, index=["p", "q", "r"])
    result = evaluation.nearest_train_distance(Identity(), X, y, other)
    assert list(result.index) == ["p", "q", "r"]
    np.testing.assert_allclose(result["distance"], [0.0, 1.0, 1.0])
    assert list(result["neighbour_label"]) == ["Benign", "SSH-Bruteforce", "FTP-BruteForce"]


def test_train_to_train_distance_never_returns_the_flow_itself():
    X, y = _train()
    result = evaluation.train_self_distance(Identity(), X, y)
    # (0,0)-(3,0) = 3, (3,0)-(0,0) = 3, (0,4)-(0,0) = 4: no zero although each flow is in the index.
    np.testing.assert_allclose(result["distance"], [3.0, 3.0, 4.0])
    assert list(result["neighbour_label"]) == ["SSH-Bruteforce", "Benign", "Benign"]


def test_an_exact_copy_in_training_has_distance_zero_but_is_not_its_own_neighbour():
    X, y = _train()
    X = pd.concat([X, X.iloc[[1]].set_axis([13])])
    y = pd.concat([y, pd.Series(["SSH-Bruteforce"], index=[13])])
    result = evaluation.train_self_distance(Identity(), X, y)
    assert result.loc[11, "distance"] == 0.0 and result.loc[13, "distance"] == 0.0
    assert result.loc[10, "distance"] == pytest.approx(3.0)


def test_the_already_fitted_preprocess_is_used_and_not_refitted():
    X, y = _train()
    # Fitted on a range ten times wider than the training rows: a refit would rescale and change the distances.
    wide = MinMaxScaler().fit(pd.DataFrame({"a": [0.0, 30.0], "b": [0.0, 40.0]}))
    other = pd.DataFrame({"a": [3.0], "b": [1.0]})
    result = evaluation.nearest_train_distance(wide, X, y, other)
    assert result["distance"].iloc[0] == pytest.approx(1.0 / 40.0)


def test_distance_summary_reports_share_below_each_threshold_and_same_label_share_by_class():
    dist = pd.DataFrame(
        {
            "distance": [0.0, 0.0005, 0.005, 0.5, 0.0002],
            "neighbour_label": ["Benign", "Benign", "SSH-Bruteforce", "Benign", "SSH-Bruteforce"],
        }
    )
    labels = pd.Series(["Benign", "Benign", "Benign", "Benign", "SSH-Bruteforce"])
    table = evaluation.distance_summary(dist, labels, thresholds=(1e-3, 1e-2))
    benign = table.loc["Benign"]
    assert benign["flows"] == 4
    assert benign["median"] == pytest.approx(0.00275)
    assert benign["share < 0.001"] == pytest.approx(0.5)
    assert benign["share < 0.01"] == pytest.approx(0.75)
    assert benign["neighbour same class"] == pytest.approx(0.75)
    assert table.loc["SSH-Bruteforce", "share < 0.001"] == 1.0
    assert table.loc["All", "flows"] == 5


def test_the_distance_figure_renders_every_curve_and_the_threshold():
    from nids import plotting

    rng = np.random.default_rng(0)
    curves = {
        "train ↔ train": rng.exponential(1e-4, 50),
        "test Benign": rng.exponential(1e-4, 50),
        "test SSH-Bruteforce": np.r_[0.0, rng.exponential(1e-4, 49)],
        "15 Feb DoS": rng.exponential(0.5, 50),
    }
    fig = plotting.plot_neighbour_distance(curves, threshold=1e-3)
    labels = [line.get_label() for line in fig.axes[0].get_lines()]
    assert all(any(name in label for label in labels) for name in curves)
