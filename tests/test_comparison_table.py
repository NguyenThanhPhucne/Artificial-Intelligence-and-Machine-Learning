"""The final comparison table (report Table 3, reports/tables/5_7_final_comparison.csv) built by one function."""

from __future__ import annotations

import pandas as pd

from nids import evaluation

COLUMNS = [
    "CV macro F1",
    "Test accuracy",
    "Test macro F1 [95% CI]",
    "Test errors (of 15,000)",
    "Test FTP recall",
    "Training time (s)",
    "Inference (ms / 1,000 flows)",
    "CV macro F1 re-tuned without host fingerprint",
]


NOFP_CV = {"k-Nearest Neighbours": 0.979276, "Decision Tree": 0.990254}


def _selection() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cv_f1_macro (macro-selected)": [0.999982, 0.995341],
            "cv_f1_macro_std": [0.0000415, 0.008904],
            "test_accuracy": [0.999933, 0.999933],
            "test_f1_macro": [0.99993, 0.99991],
            "test_ci_low": [0.99971, 0.99968],
            "test_ci_high": [1.0, 1.0],
            "test_errors": [1, 1],
            "test_recall_ftp": [1.0, 1.0],
            "fit_seconds": [0.2712, 0.4689],
            "inference_ms_per_1000": [34.119, 0.6013],
        },
        index=["k-Nearest Neighbours", "Decision Tree"],
    )


def test_final_comparison_table_has_the_report_columns_in_model_order():
    table = evaluation.final_comparison_table(_selection(), NOFP_CV, ["Decision Tree", "k-Nearest Neighbours"])
    assert list(table.columns) == COLUMNS
    assert list(table.index) == ["Decision Tree", "k-Nearest Neighbours"]


def test_final_comparison_table_formats_scores_for_the_reader():
    table = evaluation.final_comparison_table(_selection(), NOFP_CV, ["Decision Tree", "k-Nearest Neighbours"])
    knn = table.loc["k-Nearest Neighbours"]
    # Five decimals for CV, so k-NN's tiny spread does not print as "1.0 ± 0.0".
    assert knn["CV macro F1"] == "0.99998 ± 0.00004"
    assert knn["Test macro F1 [95% CI]"] == "0.9999 [0.9997, 1.0000]"
    assert knn["Inference (ms / 1,000 flows)"] == 34.12


def test_final_comparison_table_shows_the_re_tuned_fingerprint_free_score():
    # Step 5 (v) re-tunes every model without the fingerprint columns; the table shows that score, not the drop
    # measured with settings tuned on the fingerprint, because re-tuning recovers most of that drop.
    table = evaluation.final_comparison_table(_selection(), NOFP_CV, ["Decision Tree", "k-Nearest Neighbours"])
    assert table.loc["k-Nearest Neighbours", "CV macro F1 re-tuned without host fingerprint"] == "0.97928"
    assert table.loc["Decision Tree", "CV macro F1 re-tuned without host fingerprint"] == "0.99025"
