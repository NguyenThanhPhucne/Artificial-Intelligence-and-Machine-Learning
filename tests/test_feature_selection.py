"""Tests for correlation-based feature selection (Lab 5) and for the scale invariance of trees (Lab 6)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from nids import data, evaluation, features, models


def _linear_pair(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    a = rng.normal(size=n)
    return pd.DataFrame({"a": a, "b": 2 * a + 1, "c": rng.normal(size=n)})


def test_corr_filter_keeps_first_of_correlated_pair():
    selector = features.CorrelationFilter(0.7).fit(_linear_pair())
    assert selector.kept_columns_ == ["a", "c"]
    assert selector.dropped_ == {"b": "a"}


def test_corr_filter_uses_rank_correlation():
    a = np.linspace(0, 10, 400)
    df = pd.DataFrame({"a": a, "b": np.exp(a)})
    # The relationship is monotonic but strongly curved, so Pearson alone would keep both columns.
    assert abs(df["a"].corr(df["b"])) < 0.95
    selector = features.CorrelationFilter(0.95).fit(df)
    assert selector.kept_columns_ == ["a"]


def test_corr_filter_learns_only_from_fit_data():
    fit_part = _linear_pair()
    rng = np.random.default_rng(1)
    unrelated = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50), "c": rng.normal(size=50)})
    selector = features.CorrelationFilter(0.7).fit(fit_part)
    out = selector.transform(unrelated)
    assert out.shape == (50, 2)
    assert np.allclose(np.asarray(out), unrelated[["a", "c"]].to_numpy())


def test_corr_filter_feature_names():
    selector = features.CorrelationFilter(0.7).fit(_linear_pair())
    assert list(selector.get_feature_names_out()) == ["a", "c"]


def test_corr_filter_works_on_arrays_inside_column_transformer():
    frame = _linear_pair()
    selector = features.CorrelationFilter(0.7).fit(frame.to_numpy())
    assert selector.transform(frame.to_numpy()).shape == (len(frame), 2)


def test_pipeline_default_has_passthrough_filter():
    flows = _linear_pair().rename(columns={"a": "Flow Duration", "b": "Tot Fwd Pkts", "c": "Fwd Seg Size Min"})
    flows["Protocol"] = 6.0
    pipe = features.build_pipeline(flows, DecisionTreeClassifier())
    assert pipe.get_params()["preprocess__numeric__corr_filter"] == "passthrough"


@pytest.mark.parametrize("variant, threshold", [("corr_filter_0.70", 0.70), ("corr_filter_0.95", 0.95)])
def test_variant_corr_filter_sets_threshold(variant, threshold):
    flows = _linear_pair().rename(columns={"a": "Flow Duration", "b": "Tot Fwd Pkts", "c": "Fwd Seg Size Min"})
    flows["Protocol"] = 6.0
    run = SimpleNamespace(
        name=models.TREE,
        search=SimpleNamespace(best_params_={"model__max_depth": 3, "resample__k_neighbors": 3}),
    )
    pipe, X_variant = evaluation.variant_pipeline(run, flows, variant)
    step = pipe.get_params()["preprocess__numeric__corr_filter"]
    assert isinstance(step, features.CorrelationFilter)
    assert step.threshold == threshold
    assert "resample" in pipe.named_steps
    assert pipe.get_params()["resample__k_neighbors"] == 3
    assert pipe.get_params()["model__max_depth"] == 3
    assert X_variant is flows


def test_tree_predictions_invariant_to_scaling():
    split = data.load_split(data.load_sample())
    _, numeric = features.model_columns(split.X_train)
    X_train = split.X_train[numeric].to_numpy(dtype=float)
    X_test = split.X_test[numeric].to_numpy(dtype=float)
    raw = DecisionTreeClassifier(random_state=42).fit(X_train, split.y_train)
    scaler = StandardScaler().fit(X_train)
    scaled = DecisionTreeClassifier(random_state=42).fit(scaler.transform(X_train), split.y_train)
    assert np.array_equal(raw.predict(X_test), scaled.predict(scaler.transform(X_test)))


def test_variant_uses_given_params_instead_of_search_best():
    flows = _linear_pair().rename(columns={"a": "Flow Duration", "b": "Tot Fwd Pkts", "c": "Fwd Seg Size Min"})
    flows["Protocol"] = 6.0
    run = SimpleNamespace(
        name=models.TREE,
        search=SimpleNamespace(best_params_={"model__max_depth": 3, "resample__k_neighbors": 3}),
    )
    macro_params = {"model__max_depth": 7, "resample__k_neighbors": 5}
    pipe, _ = evaluation.variant_pipeline(run, flows, "main", params=macro_params)
    assert pipe.get_params()["model__max_depth"] == 7
    assert pipe.get_params()["resample__k_neighbors"] == 5
