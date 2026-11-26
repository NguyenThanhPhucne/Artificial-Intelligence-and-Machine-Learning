"""Tests for the model definitions: search spaces, SMOTE constraints and scorers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rv_continuous, rv_discrete
from sklearn.base import BaseEstimator, ClassifierMixin

from nids import config, data, features, models


def _toy_flows(n: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "Dst Port": rng.choice([21.0, 22.0, 80.0], n),
            "Protocol": rng.choice([6.0, 17.0], n),
            "Timestamp": ["14/02/2018 10:00:00"] * n,
            "Flow Duration": rng.exponential(1e5, n),
            "Init Fwd Win Byts": rng.choice([-1.0, 8192.0, 26883.0], n),
        }
    )


def _sample_value(dist):
    if isinstance(dist, list):
        return dist[-1]
    return dist.rvs(random_state=0)


@pytest.mark.parametrize("name", models.ALL_MODEL_NAMES)
def test_every_search_parameter_exists_on_the_pipeline(name):
    flows = _toy_flows()
    pipeline = features.build_pipeline(flows, models.default_estimators()[name], sampler=models.make_smote())
    params = {key: _sample_value(dist) for key, dist in models.search_spaces()[name].items()}
    pipeline.set_params(**params)  # raises ValueError if any key is misspelled


def test_smote_neighbours_fit_inside_the_smallest_fold():
    if not config.SPLIT_PARQUET.exists():
        pytest.skip("run notebook 02 first")
    split = data.load_split(data.load_sample())
    ftp_per_training_fold = [
        int((split.y_train.iloc[train] == models.MINORITY_CLASS).sum())
        for train, _ in models.cv_splitter().split(split.X_train, split.y_train)
    ]
    largest_k = max(models.search_spaces()[models.FOREST]["resample__k_neighbors"])
    assert largest_k < min(ftp_per_training_fold)
    smallest_target = min(
        next(iter(s.values())) for s in models.search_spaces()[models.FOREST]["resample__sampling_strategy"]
    )
    assert smallest_target > max(ftp_per_training_fold), "SMOTE target must exceed the real FTP count"


def test_ftp_recall_scorer_ignores_other_classes():
    scorer = models.scorers()["recall_ftp"]

    class Echo(ClassifierMixin, BaseEstimator):
        """A stand-in classifier that returns its input, so I control the predictions exactly."""

        classes_ = np.array(["Benign", "FTP-BruteForce", "SSH-Bruteforce"])

        def predict(self, X):
            return np.asarray(X)

    y_true = np.array(["FTP-BruteForce", "FTP-BruteForce", "Benign", "Benign"])
    y_pred = np.array(["FTP-BruteForce", "Benign", "SSH-Bruteforce", "SSH-Bruteforce"])
    assert scorer(Echo(), y_pred, y_true) == pytest.approx(0.5)


def test_search_distributions_are_valid():
    for space in models.search_spaces().values():
        for dist in space.values():
            assert isinstance(dist, list) or hasattr(dist, "rvs")
            if not isinstance(dist, list):
                assert isinstance(dist.dist, (rv_continuous, rv_discrete))


@pytest.mark.parametrize("metric", ["accuracy", "f1_macro", "f1_weighted"])
def test_fast_bootstrap_score_matches_scikit_learn(metric):
    from nids import evaluation

    rng = np.random.default_rng(3)
    labels = np.array(["Benign", "FTP-BruteForce", "SSH-Bruteforce"])
    y_true = rng.choice(labels, 500, p=[0.8, 0.05, 0.15])
    y_pred = np.where(rng.random(500) < 0.9, y_true, rng.choice(labels, 500))
    classes = np.unique(np.concatenate([y_true, y_pred]))
    fast = evaluation._fast_score(
        np.searchsorted(classes, y_true), np.searchsorted(classes, y_pred), len(classes), metric
    )
    assert fast == pytest.approx(evaluation.score_predictions(y_true, y_pred)[metric])


def test_bootstrap_interval_brackets_the_point_estimate():
    from nids import evaluation

    y_true = np.array(["Benign"] * 90 + ["FTP-BruteForce"] * 10)
    y_pred = y_true.copy()
    y_pred[0] = "FTP-BruteForce"
    point = evaluation.score_predictions(y_true, y_pred)["f1_macro"]
    low, high = evaluation.bootstrap_interval(y_true, y_pred, n_boot=300)
    assert low <= point <= high


def test_knn_neighbourhood_stays_below_the_real_ftp_count_of_a_fold():
    # An FTP flow can only win a k-NN vote if most of its k neighbours are FTP; without SMOTE (the
    # "no resampling" ablation) a training fold has only ~30 real FTP flows to supply them.
    if not config.SPLIT_PARQUET.exists():
        pytest.skip("run notebook 02 first")
    split = data.load_split(data.load_sample())
    ftp_per_training_fold = [
        int((split.y_train.iloc[train] == models.MINORITY_CLASS).sum())
        for train, _ in models.cv_splitter().split(split.X_train, split.y_train)
    ]
    largest_k = max(models.search_spaces()[models.KNN]["model__n_neighbors"])
    assert largest_k < min(ftp_per_training_fold)


def test_class_weight_variant_only_for_models_that_support_it():
    assert models.supports_class_weight(models.TREE)
    assert models.supports_class_weight(models.SVM)
    assert models.supports_class_weight(models.BOOSTING)
    assert not models.supports_class_weight(models.KNN)
    assert not models.supports_class_weight(models.NAIVE_BAYES)
    assert not models.supports_class_weight(models.MLP)


def test_every_algorithm_has_a_baseline_and_a_search_space():
    assert set(models.default_estimators()) == set(models.ALL_MODEL_NAMES)
    assert set(models.search_spaces()) == set(models.ALL_MODEL_NAMES)
    assert len(models.ALL_MODEL_NAMES) == len(set(models.ALL_MODEL_NAMES)) == 8


@pytest.mark.parametrize("name", [models.SVM, models.BOOSTING, models.MLP])
def test_new_algorithms_fit_and_predict_inside_the_smote_pipeline(name):
    # A small end-to-end fit catches parameter combinations the estimator rejects only at fit time.
    flows = _toy_flows(120)
    labels = np.array(["Benign"] * 80 + ["SSH-Bruteforce"] * 30 + ["FTP-BruteForce"] * 10)
    pipe = features.build_pipeline(flows, models.default_estimators()[name], sampler=models.make_smote())
    pipe.set_params(resample__sampling_strategy={models.MINORITY_CLASS: 40}, resample__k_neighbors=3)
    pipe.fit(flows, labels)
    assert set(pipe.predict(flows)) <= set(labels)


def test_tomek_links_remove_only_majority_rows_and_only_during_fit():
    from imblearn.under_sampling import TomekLinks

    rng = np.random.default_rng(0)
    # Two overlapping classes, so some cross-class nearest-neighbour pairs (Tomek links) must exist.
    X = np.vstack([rng.normal(0, 1, (200, 2)), rng.normal(1, 1, (20, 2))])
    y = np.array(["Benign"] * 200 + ["FTP-BruteForce"] * 20)
    X_res, y_res = TomekLinks(sampling_strategy="auto").fit_resample(X, y)
    assert (y_res == "FTP-BruteForce").sum() == 20  # minority rows are never removed
    assert (y_res == "Benign").sum() < 200  # some majority rows on the boundary are
    flows = _toy_flows(120)
    labels = np.array(["a"] * 100 + ["b"] * 20)
    pipe = features.build_pipeline(flows, models.default_estimators()[models.NAIVE_BAYES], sampler=TomekLinks())
    pipe.fit(flows, labels)
    assert len(pipe.predict(flows)) == len(flows)
