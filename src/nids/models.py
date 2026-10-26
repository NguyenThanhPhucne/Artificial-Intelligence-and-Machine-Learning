"""Model definitions, hyper-parameter search spaces, scoring and timing.

I compare three algorithms that sit at different points of the bias/variance and interpretability
spectrum: a linear model (Logistic Regression), a single tree (Decision Tree) and a bagged ensemble
of trees (Random Forest). All three share the same preprocessing pipeline, the same cross-validation
folds and the same scorers, so differences in their scores come from the algorithms, not the setup.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.stats import loguniform, randint
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from . import config
from .features import build_pipeline, symlog_transformer

LOGREG, TREE, FOREST = "Logistic Regression", "Decision Tree", "Random Forest"
MODEL_NAMES = (LOGREG, TREE, FOREST)
MINORITY_CLASS = "FTP-BruteForce"
REFIT_METRIC = "f1_weighted"


def cv_splitter(seed: int = config.SEED) -> StratifiedKFold:
    """Return the 5-fold stratified splitter that every model and every experiment uses.

    I shuffle with a fixed seed so the folds are random but identical across models; comparing
    models on different folds would add fold-to-fold noise to the comparison.
    """
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed)


def scorers() -> dict[str, object]:
    """Return the metrics I record in every cross-validation.

    Weighted F1 is the selection metric required by the project brief. Because Benign is 85% of the
    data, weighted F1 can stay near 1.0 while the rare class is missed, so I also record macro F1
    (every class counts equally), macro recall, and the recall of FTP-BruteForce on its own.
    """
    return {
        "accuracy": "accuracy",
        "f1_weighted": "f1_weighted",
        "f1_macro": "f1_macro",
        "recall_macro": "recall_macro",
        "precision_macro": "precision_macro",
        "recall_ftp": make_scorer(
            recall_score, labels=[MINORITY_CLASS], average="macro", zero_division=0
        ),
    }


def make_smote(seed: int = config.SEED) -> SMOTE:
    """SMOTE that oversamples only FTP-BruteForce, to a capped target.

    I do not balance every class to the majority size: FTP has about 30 flows per training fold, and
    interpolating ~50,000 synthetic points from 30 real ones would teach the model a region of
    feature space that no real flow occupies. SSH-Bruteforce (15%) is left as it is. The target size
    and k_neighbors are tuned in the search (k must stay below the ~30 FTP flows in a fold).
    """
    return SMOTE(sampling_strategy={MINORITY_CLASS: 1_000}, k_neighbors=5, random_state=seed)


def default_estimators(seed: int = config.SEED) -> dict[str, BaseEstimator]:
    """Return each algorithm with scikit-learn defaults (seeded), used as the untuned baseline.

    I raise Logistic Regression's max_iter from 100 to 2,000 only because lbfgs does not converge in
    100 iterations on this data; a non-converged baseline would not be a fair comparison. Random
    Forest uses one thread inside each fit because the search already runs folds in parallel.
    """
    return {
        LOGREG: LogisticRegression(max_iter=2_000, random_state=seed),
        TREE: DecisionTreeClassifier(random_state=seed),
        FOREST: RandomForestClassifier(random_state=seed, n_jobs=1),
    }


def _smote_space() -> dict[str, list]:
    targets = [{MINORITY_CLASS: n} for n in (500, 1_000, 2_000)]
    return {"resample__sampling_strategy": targets, "resample__k_neighbors": [3, 5]}


def search_spaces() -> dict[str, dict[str, object]]:
    """Return the hyper-parameter distributions I sample from for each algorithm.

    * Logistic Regression: the inverse regularisation strength C on a log scale (redundant,
      correlated features make coefficients unstable without regularisation), and whether to apply
      symlog before scaling.
    * Decision Tree: depth and minimum leaf size control over-fitting; criterion changes the
      impurity measure.
    * Random Forest: number of trees, depth, leaf size and the share of features tried at each split,
      which controls how decorrelated the trees are.
    """
    return {
        LOGREG: {
            "model__C": loguniform(1e-3, 1e2),
            "preprocess__numeric__log": ["passthrough", symlog_transformer()],
            **_smote_space(),
        },
        TREE: {
            "model__max_depth": [None, 5, 10, 15, 20, 30],
            "model__min_samples_leaf": randint(1, 30),
            "model__criterion": ["gini", "entropy"],
            **_smote_space(),
        },
        FOREST: {
            "model__n_estimators": randint(100, 401),
            "model__max_depth": [None, 10, 20, 30],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.3],
            **_smote_space(),
        },
    }


def baseline_cv(
    name: str, X: pd.DataFrame, y: pd.Series, seed: int = config.SEED
) -> pd.DataFrame:
    """Cross-validate the default model with my preprocessing but without SMOTE.

    This is the reference point that tuning and resampling must beat; it uses the same folds and
    scorers as the search.
    """
    pipeline = build_pipeline(X, default_estimators(seed)[name])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        result = cross_validate(
            pipeline, X, y, cv=cv_splitter(seed), scoring=scorers(),
            return_train_score=True, n_jobs=-1,
        )
    return pd.DataFrame(result)


@dataclass
class SearchRun:
    """A finished hyper-parameter search plus the timings I report in the comparative analysis."""

    name: str
    search: RandomizedSearchCV
    search_seconds: float
    fit_seconds: float
    predict_ms_per_1000: float


def run_search(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 20,
    seed: int = config.SEED,
) -> SearchRun:
    """Tune one algorithm with RandomizedSearchCV inside the SMOTE pipeline, then time it.

    I use a randomized rather than an exhaustive grid search: with 20 sampled settings per model it
    explores continuous ranges such as C and the number of trees at a fraction of the cost, which
    matters because every setting is fitted five times. The search refits the best setting on the
    whole training set; scikit-learn records that single refit as `refit_time_`, which I report as
    the training time. Inference time is the median of three passes over the training set, scaled
    to 1,000 flows (the test set is not touched until the final evaluation).
    """
    search = RandomizedSearchCV(
        estimator=build_pipeline(X, default_estimators(seed)[name], sampler=make_smote(seed)),
        param_distributions=search_spaces()[name],
        n_iter=n_iter,
        scoring=scorers(),
        refit=REFIT_METRIC,
        cv=cv_splitter(seed),
        n_jobs=-1,
        random_state=seed,
        return_train_score=True,
        error_score="raise",
    )
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        search.fit(X, y)
    search_seconds = time.perf_counter() - start

    timings = []
    for _ in range(3):
        start = time.perf_counter()
        search.best_estimator_.predict(X)
        timings.append(time.perf_counter() - start)
    predict_ms = float(np.median(timings)) / len(X) * 1_000 * 1_000

    return SearchRun(name, search, search_seconds, float(search.refit_time_), predict_ms)


def _cache_path(name: str, n_iter: int, seed: int) -> Path:
    slug = name.lower().replace(" ", "_")
    return config.MODELS_DIR / f"{slug}_search_n{n_iter}_seed{seed}.joblib"


def load_or_run_search(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = config.N_SEARCH_ITER,
    seed: int = config.SEED,
    force: bool = False,
) -> SearchRun:
    """Return a cached search if one exists for these settings, otherwise run and cache it.

    Searching takes minutes, while rewriting the explanation around the results does not need a
    refit. I key the cache on the model, n_iter and seed, and store the training-data shape with
    it so a cache built on different data is never reused silently.
    """
    path = _cache_path(name, n_iter, seed)
    if path.exists() and not force:
        # joblib unpickles arbitrary objects, so I only ever load files this function wrote itself
        # into the local, git-ignored models/ folder; never point this at a downloaded file.
        cached = joblib.load(path)
        if cached["shape"] == X.shape:
            return cached["run"]
    run = run_search(name, X, y, n_iter=n_iter, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"shape": X.shape, "run": run}, path)
    return run


def best_cv_summary(run: SearchRun) -> pd.DataFrame:
    """Return mean and standard deviation of every metric for the selected setting.

    Rows are metrics; columns give the validation-fold scores (unseen within CV) and the
    training-fold scores, whose gap shows how much each model over-fits.
    """
    results, best = run.search.cv_results_, run.search.best_index_
    rows = {}
    for metric in scorers():
        rows[metric] = {
            "validation_mean": results[f"mean_test_{metric}"][best],
            "validation_std": results[f"std_test_{metric}"][best],
            "train_mean": results[f"mean_train_{metric}"][best],
            "train_std": results[f"std_train_{metric}"][best],
        }
    return pd.DataFrame(rows).T


def fold_scores(run: SearchRun, metric: str) -> np.ndarray:
    """Return the per-fold validation scores of the selected setting for one metric."""
    results, best = run.search.cv_results_, run.search.best_index_
    return np.array([results[f"split{k}_test_{metric}"][best] for k in range(config.CV_FOLDS)])


def out_of_fold_predictions(run: SearchRun, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Predict every training flow with a model that did not see it (same 5 folds).

    These out-of-fold predictions give a confusion matrix on "seen" data that is not inflated by
    the model memorising the rows it was fitted on.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return cross_val_predict(clone(run.search.best_estimator_), X, y, cv=cv_splitter(), n_jobs=-1)
