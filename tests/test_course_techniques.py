"""Tests for the techniques added from the module's lectures and labs (EDA statistics, fold
composition, regularisation, learning curves, plots and the result cache)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from nids import config, data, eda, evaluation, features, models, plotting


def _toy(n_benign: int = 160, n_ftp: int = 20, n_ssh: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = n_benign + n_ftp + n_ssh
    X = pd.DataFrame(
        {
            "Dst Port": rng.choice([21.0, 22.0, 443.0], n),
            "Protocol": rng.choice([6.0, 17.0], n),
            "Timestamp": ["14/02/2018 10:00:00"] * n,
            "Flow Duration": rng.exponential(1e5, n),
            "Init Fwd Win Byts": rng.choice([-1.0, 8192.0, 26883.0], n),
            "Fwd Seg Size Min": rng.choice([20.0, 32.0, 40.0], n),
            "Init Bwd Win Byts": rng.choice([-1.0, 0.0, 255.0], n),
        }
    )
    y = pd.Series(["Benign"] * n_benign + ["FTP-BruteForce"] * n_ftp + ["SSH-Bruteforce"] * n_ssh)
    return X, y


def test_descriptive_statistics_match_pandas():
    X = pd.DataFrame({"a": [1.0, 2.0, 2.0, 10.0], "b": [0.0, 0.0, 1.0, 1.0]})
    stats = eda.descriptive_statistics(X)
    assert stats.loc["a", "median"] == 2.0
    assert stats.loc["a", "mode"] == 2.0
    assert stats.loc["a", "range"] == 9.0
    assert stats.loc["a", "variance"] == pytest.approx(X["a"].var())
    assert stats.loc["a", "skewness"] > 0  # one large value pulls the tail to the right


def test_missing_values_reports_only_columns_with_gaps():
    X = pd.DataFrame({"a": [1.0, np.nan], "b": [1.0, 2.0]})
    assert eda.missing_values(X).to_dict() == {"a": 1}


@pytest.mark.parametrize(
    "column, values, expected",
    [
        ("Protocol", [0, 6, 17], "nominal"),
        ("Dst Port", [21, 22], "nominal"),
        ("Timestamp", ["14/02/2018"], "interval"),
        ("Tot Fwd Pkts", [1.0, 2.0], "ratio (discrete)"),
        ("Flow Byts/s", [1.5, 2.0], "ratio (continuous)"),
    ],
)
def test_level_of_measurement(column, values, expected):
    assert data.level_of_measurement(column, "numeric", pd.Series(values)) == expected


def test_stratified_folds_keep_the_rare_class_balanced():
    _, y = _toy()
    table = evaluation.fold_composition(y)
    stratified = table.loc["StratifiedKFold (shuffled)"]
    assert stratified["max"] - stratified["min"] <= 1
    assert table.loc["KFold (no shuffle)", "max"] >= stratified["max"]


def test_l1_removes_features_and_l2_keeps_them():
    X, y = _toy(300, 40, 120)
    pipeline = features.build_pipeline(X, models.default_estimators()[models.LOGREG], sampler=None)
    run = models.SearchRun(models.LOGREG, type("S", (), {"estimator": pipeline})(), 0, 0, 0)
    scores, _ = evaluation.regularisation_comparison(run, X, y, {"model__C": 0.05}, extra_c=())
    by_penalty = scores.set_index("penalty")
    assert by_penalty.loc["L2 (ridge)", "features_removed"] == 0
    assert by_penalty.loc["L1 (lasso)", "features_removed"] > 0


def test_learning_curve_has_one_row_per_size():
    X, y = _toy(300, 40, 120)
    pipeline = features.build_pipeline(X, models.default_estimators()[models.TREE])
    curve = evaluation.learning_curve_table(pipeline, X, y, sizes=(0.5, 1.0))
    assert list(curve.columns) == ["train_flows", "train_mean", "train_std", "valid_mean", "valid_std"]
    assert len(curve) == 2 and curve["train_flows"].is_monotonic_increasing


def test_cached_reuses_the_stored_result(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.delenv("NIDS_FORCE_RETRAIN", raising=False)
    X = pd.DataFrame({"a": [1, 2]})
    calls = []
    first = models.cached("toy", lambda: calls.append(1) or 41, X)
    second = models.cached("toy", lambda: calls.append(1) or 42, X)
    assert first == second == 41 and len(calls) == 1


def test_course_figures_render():
    X, y = _toy()
    stats = eda.descriptive_statistics(X[["Flow Duration", "Init Fwd Win Byts"]])
    plotting.plot_histograms(X, ["Flow Duration", "Init Fwd Win Byts"], stats)
    plotting.plot_scatter(X, y, "Flow Duration", "Init Fwd Win Byts")
    plotting.plot_scaling_comparison(X["Flow Duration"], np.zeros(len(X)), np.zeros(len(X)))
    table = pd.DataFrame({m: {"accuracy": 0.99, "f1_macro": 0.98} for m in plotting.MODEL_ORDER}).T
    plotting.plot_metric_bars(table, {"accuracy": "Accuracy", "f1_macro": "Macro F1"})
    curve = pd.DataFrame(
        {
            "train_flows": [10, 20],
            "train_mean": [1, 1],
            "train_std": [0, 0],
            "valid_mean": [0.9, 0.95],
            "valid_std": [0.01, 0.01],
        }
    )
    plotting.plot_learning_curves({m: curve for m in plotting.MODEL_ORDER})


# --- Week 5-6 labs: Pearson vs Spearman, mosaic plot, pair plot, importance -----------------------


def _curved_pair(n: int = 400) -> pd.DataFrame:
    a = np.linspace(0, 10, n)
    return pd.DataFrame({"a": a, "b": np.exp(a), "c": np.random.default_rng(0).normal(size=n)})


def test_pairwise_correlations_detects_monotonic_curve():
    pairs = eda.pairwise_correlations(_curved_pair())
    row = pairs.set_index(["feature_a", "feature_b"]).loc[("a", "b")]
    assert row["spearman"] == pytest.approx(1.0)
    assert row["pearson_raw"] < 0.9
    assert row["pearson_symlog"] > row["pearson_raw"]
    assert len(pairs) == 3  # every unordered pair once


def test_pairwise_correlations_skips_constant_columns():
    frame = _curved_pair().assign(constant=1.0)
    pairs = eda.pairwise_correlations(frame)
    assert "constant" not in set(pairs["feature_a"]) | set(pairs["feature_b"])


def test_correlation_method_counts_shape():
    split = data.load_split(data.load_sample())
    pairs = eda.pairwise_correlations(eda.numeric_model_features(split.X_train))
    counts = eda.correlation_method_counts(pairs)
    assert list(counts.index) == [0.7, 0.9, 0.95]
    assert list(counts.columns) == ["pearson_raw", "pearson_symlog", "spearman"]
    assert (counts.diff().dropna() <= 0).all().all()  # stricter thresholds never count more pairs


def test_stratified_subsample_keeps_all_ftp():
    split = data.load_split(data.load_sample())
    X1, y1 = eda.stratified_subsample(split.X_train, split.y_train, n=2_000)
    X2, _ = eda.stratified_subsample(split.X_train, split.y_train, n=2_000)
    assert len(X1) == 2_000 and len(y1) == 2_000
    assert (y1 == models.MINORITY_CLASS).sum() == (split.y_train == models.MINORITY_CLASS).sum()
    assert X1.index.equals(X2.index)


def test_lab5_lab6_figures_render():
    split = data.load_split(data.load_sample())
    pairs = eda.pairwise_correlations(_curved_pair())
    assert isinstance(plotting.plot_pearson_vs_spearman(pairs), matplotlib.figure.Figure)
    counts = pd.crosstab(split.X_train["Protocol"], split.y_train)
    assert isinstance(plotting.plot_mosaic(counts), matplotlib.figure.Figure)
    columns = ["Flow Duration", "Tot Fwd Pkts", "Init Fwd Win Byts"]
    X, y = eda.stratified_subsample(split.X_train, split.y_train, n=500)
    fig = plotting.plot_pair_matrix(X, y, columns)
    assert len(fig.axes) == len(columns) ** 2
    matplotlib.pyplot.close("all")


def test_importance_pair_has_ten_bars_each():
    rng = np.random.default_rng(0)
    left = pd.Series(rng.random(15), index=[f"f{i}" for i in range(15)])
    right = pd.Series(rng.random(15), index=[f"g{i}" for i in range(15)])
    fig = plotting.plot_importance_pair(left, right, ("Decision Tree", "Random Forest"), top=10)
    assert len(fig.axes) == 2
    for ax, series in zip(fig.axes, (left, right), strict=True):
        assert len(ax.patches) == 10
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert set(labels) == set(series.nlargest(10).index)  # the ten largest, not the first ten
    matplotlib.pyplot.close("all")


def test_unseen_day_figure_has_one_group_per_model():
    table = pd.DataFrame(
        {
            "benign false alarms": [0.01, 0.2],
            "benign ci_low": [0.005, 0.15],
            "benign ci_high": [0.02, 0.25],
            "DoS attacks-GoldenEye flagged": [0.7, 0.1],
            "DoS attacks-Slowloris flagged": [0.0, 0.9],
        },
        index=["Decision Tree", "k-Nearest Neighbours"],
    )
    fig = plotting.plot_unseen_day(table)
    assert len(fig.axes) == 2  # false alarms | unseen attacks flagged
    assert len(fig.axes[1].patches) == 2 * 2  # two DoS types x two models
    matplotlib.pyplot.close("all")
