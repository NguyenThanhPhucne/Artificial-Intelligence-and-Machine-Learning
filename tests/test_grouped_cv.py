"""Tests for cross-validation grouped by flow shape (H2: no shape on both sides of a fold)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from nids import evaluation


def _grouped_flows() -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """40 benign shapes, 10 SSH shapes and a single FTP shape, three flows each; x encodes the shape."""
    rows = [("Benign", g) for g in range(40) for _ in range(3)]
    rows += [("SSH-Bruteforce", 100 + g) for g in range(10) for _ in range(3)]
    rows += [("FTP-BruteForce", 200) for _ in range(5)]
    labels, groups = zip(*rows, strict=True)
    rng = np.random.default_rng(0)
    groups = np.array(groups)
    X = pd.DataFrame({"shape": groups.astype(float), "noise": rng.normal(size=len(groups))})
    return X, pd.Series(labels), groups


def test_grouped_splitter_never_puts_one_shape_on_both_sides_but_plain_kfold_does():
    X, y, groups = _grouped_flows()
    grouped = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    assert evaluation.group_overlap(grouped, X, y, groups) == 0
    assert evaluation.group_overlap(KFold(n_splits=5, shuffle=True, random_state=42), X, y, groups) > 0


def test_a_class_with_one_shape_is_validated_in_exactly_one_fold_without_training_examples():
    X, y, groups = _grouped_flows()
    folds = evaluation.grouped_cv(DecisionTreeClassifier(random_state=0), X, y, groups)
    assert len(folds) == 5
    ftp_folds = folds[folds["FTP valid"] > 0]
    assert len(ftp_folds) == 1 and ftp_folds["FTP train"].iloc[0] == 0
    # A model that never saw FTP cannot predict it; in the other folds there is no FTP to score.
    assert ftp_folds["recall FTP-BruteForce"].iloc[0] == 0
    assert folds.loc[folds["FTP valid"] == 0, "recall FTP-BruteForce"].isna().all()


def test_false_alarm_rate_is_the_share_of_benign_flows_not_called_benign():
    X, y, groups = _grouped_flows()
    folds = evaluation.grouped_cv(DecisionTreeClassifier(random_state=0), X, y, groups)
    np.testing.assert_allclose(folds["false alarm rate"], 1 - folds["recall Benign"])
    assert folds["macro F1 (Benign, SSH)"].between(0, 1).all()


def test_any_splitter_can_be_passed_for_the_ungrouped_comparison():
    X, y, groups = _grouped_flows()
    folds = evaluation.grouped_cv(
        DecisionTreeClassifier(random_state=0),
        X,
        y,
        groups,
        splitter=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    )
    # Stratified but ungrouped: every fold trains on FTP and sees shapes it was trained on.
    assert (folds["FTP train"] > 0).all()
    assert folds["recall SSH-Bruteforce"].min() == 1.0


def test_the_grouped_cv_figure_has_one_bar_pair_per_model_in_each_panel():
    from nids import plotting

    names = ["Decision Tree", "k-Nearest Neighbours"]
    summary = pd.DataFrame(
        {
            "splitter": ["stratified", "grouped"] * 2,
            "model": [names[0]] * 2 + [names[1]] * 2,
            "ssh_recall_mean": [1.0, 0.98, 1.0, 0.9],
            "ssh_recall_std": [0.0, 0.01, 0.0, 0.05],
            "false_alarm_mean": [0.0001, 0.002, 0.0, 0.01],
            "false_alarm_std": [0.0, 0.001, 0.0, 0.004],
        }
    )
    fig = plotting.plot_grouped_cv(summary)
    assert len(fig.axes) == 2
    assert all(len(ax.patches) == 4 for ax in fig.axes)


def test_out_of_fold_predictions_cover_every_flow_once_and_match_the_fold_recalls():
    X, y, groups = _grouped_flows()
    folds, predictions = evaluation.grouped_cv(
        DecisionTreeClassifier(random_state=0), X, y, groups, return_predictions=True
    )
    assert list(predictions.index) == list(X.index) and predictions.notna().all()
    ftp = (y == "FTP-BruteForce").to_numpy()
    assert (predictions[ftp] != "FTP-BruteForce").all()  # FTP was never in a training fold with itself
    assert folds["FTP valid"].sum() == ftp.sum()
