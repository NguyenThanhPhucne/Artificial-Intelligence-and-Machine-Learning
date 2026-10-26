"""Test-set evaluation, ablation experiments and model interpretation.

I keep the test set out of every decision: the models, their hyper-parameters and the choice between
selection metrics are all fixed on the training set by cross-validation before anything here looks
at the test set. Ablations are scored with the same cross-validation, so they explain the models
without turning the test set into a second validation set.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_validate

from . import config, models
from .features import build_pipeline

SENTINEL_WINDOW_COLS = list(config.SENTINEL_NEGATIVE_COLS)
# Columns that are constant within each attack and plausibly describe the attacking host's TCP stack
# rather than brute-force behaviour: FTP flows always have Fwd Seg Size Min = 40 and SSH flows 32,
# and both attacks share Init Fwd Win Byts = 26,883.
HOST_FINGERPRINT_COLS = [*SENTINEL_WINDOW_COLS, "Fwd Seg Size Min"]


def score_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Return the same metrics I used in cross-validation, computed on one set of predictions."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_ftp": recall_score(
            y_true, y_pred, labels=[models.MINORITY_CLASS], average="macro", zero_division=0
        ),
    }


def per_class_report(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """Return precision, recall, F1 and support for each class as a table."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    classes = [c for c in report if c not in ("accuracy", "macro avg", "weighted avg")]
    return pd.DataFrame({c: report[c] for c in classes}).T


def inference_ms_per_1000(model: BaseEstimator, X: pd.DataFrame, repeats: int = 5) -> float:
    """Median wall-clock prediction time over `repeats` passes, scaled to 1,000 flows."""
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(X)
        timings.append(time.perf_counter() - start)
    return float(np.median(timings)) / len(X) * 1_000_000


def selected_pipeline(run: models.SearchRun, metric: str) -> tuple[BaseEstimator, dict, int]:
    """Return an unfitted pipeline set to the search candidate that is best on `metric`.

    The search already scored every candidate on every metric, so choosing by macro instead of
    weighted F1 needs no new search: I take the arg-max of the stored cross-validation means.
    """
    results = run.search.cv_results_
    index = int(np.argmax(results[f"mean_test_{metric}"]))
    params = results["params"][index]
    return clone(run.search.estimator).set_params(**params), params, index


def fit_timed(pipeline: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> tuple[BaseEstimator, float]:
    """Fit a pipeline and return it with the fit time in seconds."""
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pipeline.fit(X, y)
    return pipeline, time.perf_counter() - start


def variant_pipeline(
    run: models.SearchRun,
    X: pd.DataFrame,
    variant: str,
) -> tuple[BaseEstimator, pd.DataFrame]:
    """Rebuild the selected model of `run` for one ablation variant; return (pipeline, X to use).

    Variants keep the tuned model hyper-parameters and change exactly one thing:
      * "main"            – the tuned pipeline as selected (no Dst Port, SMOTE).
      * "with_dst_port"   – Dst Port added back as a numeric feature.
      * "no_window_sizes" – Init Fwd/Bwd Win Byts removed (suspected attacker fingerprint).
      * "no_host_fingerprint" – the window sizes and Fwd Seg Size Min removed together.
      * "class_weight"    – no SMOTE; the classifier weights classes inversely to their frequency.
      * "no_resampling"   – no SMOTE and no class weights.
    """
    params = run.search.best_params_
    model_params = {k.removeprefix("model__"): v for k, v in params.items() if k.startswith("model__")}
    other_params = {k: v for k, v in params.items() if not k.startswith(("model__", "resample__"))}
    resample_params = {k: v for k, v in params.items() if k.startswith("resample__")}
    estimator = clone(models.default_estimators()[run.name]).set_params(**model_params)

    X_variant = X
    include_port = variant == "with_dst_port"
    sampler = models.make_smote()
    if variant == "no_window_sizes":
        X_variant = X.drop(columns=SENTINEL_WINDOW_COLS)
    elif variant == "no_host_fingerprint":
        X_variant = X.drop(columns=HOST_FINGERPRINT_COLS)
    elif variant == "class_weight":
        estimator.set_params(class_weight="balanced")
        sampler = None
    elif variant == "no_resampling":
        sampler = None
    elif variant not in ("main", "with_dst_port"):
        raise ValueError(f"Unknown variant {variant!r}")

    pipeline = build_pipeline(X_variant, estimator, sampler=sampler, include_dst_port=include_port)
    pipeline.set_params(**other_params, **(resample_params if sampler is not None else {}))
    return pipeline, X_variant


def cross_validated(pipeline: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Mean and standard deviation of every scorer over my 5 standard folds."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        result = cross_validate(pipeline, X, y, cv=models.cv_splitter(), scoring=models.scorers(), n_jobs=-1)
    summary = {}
    for metric in models.scorers():
        summary[f"{metric}_mean"] = float(np.mean(result[f"test_{metric}"]))
        summary[f"{metric}_std"] = float(np.std(result[f"test_{metric}"]))
    return summary


def importance_table(model: BaseEstimator, X: pd.DataFrame, y: pd.Series, top: int = 15) -> pd.DataFrame:
    """Impurity importance and permutation importance (macro F1) for a fitted tree-based pipeline.

    Impurity importance comes from training and favours features with many split points;
    permutation importance measures how much macro F1 drops when one input column is shuffled.
    Both under-rate features that have a near-copy (shuffling one leaves its twin intact), so I
    read them as "what the model uses", not "what matters in the traffic".
    """
    fitted_names = model.named_steps["preprocess"].get_feature_names_out()
    impurity = pd.Series(model.named_steps["model"].feature_importances_, index=fitted_names)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perm = permutation_importance(
            model, X, y, scoring="f1_macro", n_repeats=5, random_state=config.SEED, n_jobs=-1
        )
    permutation = pd.Series(perm.importances_mean, index=X.columns)
    permutation_std = pd.Series(perm.importances_std, index=X.columns)
    table = pd.DataFrame({
        "impurity_importance": impurity,
        "permutation_drop_f1_macro": permutation.reindex(impurity.index),
        "permutation_std": permutation_std.reindex(impurity.index),
    })
    return table.sort_values("impurity_importance", ascending=False).head(top)


def bootstrap_interval(
    y_true: pd.Series,
    y_pred: np.ndarray,
    metric: str = "f1_macro",
    n_boot: int = 2_000,
    seed: int = config.SEED,
) -> tuple[float, float]:
    """95% percentile bootstrap interval for a test-set metric.

    With only 9 FTP-BruteForce flows in the test set, one prediction moves macro F1 by about 0.02.
    Resampling the test flows with replacement shows how much of a difference between two models
    could arise from which flows happen to be in the test set. I stratify the resampling by class so
    that every replicate keeps the 9 FTP flows' share instead of sometimes dropping the class.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    true_codes = np.searchsorted(classes, np.asarray(y_true))
    pred_codes = np.searchsorted(classes, np.asarray(y_pred))
    by_class = [np.flatnonzero(true_codes == k) for k in range(len(classes))]
    by_class = [rows for rows in by_class if len(rows)]
    scores = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(rows, size=len(rows), replace=True) for rows in by_class])
        scores[b] = _fast_score(true_codes[idx], pred_codes[idx], len(classes), metric)
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def _fast_score(true_codes: np.ndarray, pred_codes: np.ndarray, k: int, metric: str) -> float:
    """Accuracy, macro F1 or weighted F1 from a confusion matrix built with one bincount.

    Calling scikit-learn's metric functions 2,000 times per model takes minutes because each call
    re-validates its inputs; a k x k confusion matrix gives the same numbers in microseconds. The
    test suite checks this function against scikit-learn.
    """
    cm = np.bincount(true_codes * k + pred_codes, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    if metric == "accuracy":
        return tp.sum() / cm.sum()
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    denominator = support + predicted
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    if metric == "f1_macro":
        present = (support + predicted) > 0
        return float(f1[present].mean())
    if metric == "f1_weighted":
        return float((f1 * support).sum() / support.sum())
    raise ValueError(f"Unsupported metric {metric!r}")


def tree_rules_in_original_units(pipeline: BaseEstimator, max_depth: int = 3) -> str:
    """Print the fitted decision tree's top rules with thresholds converted back to raw units.

    The tree is trained on standardised inputs, so its thresholds are z-scores ("<= 1.43"), which no
    network engineer can read. I invert the StandardScaler (x = z * scale + mean) for every numeric
    split; the symlog step is a pass-through for the Decision Tree, which I check before converting.
    """
    preprocess = pipeline.named_steps["preprocess"]
    numeric = preprocess.named_transformers_["numeric"]
    if numeric.named_steps["log"] != "passthrough":
        raise ValueError("Thresholds can only be inverted when the symlog step is disabled")
    names = list(preprocess.get_feature_names_out())
    n_protocol = len(preprocess.named_transformers_["protocol"].get_feature_names_out())
    scaler = numeric.named_steps["scale"]
    tree = pipeline.named_steps["model"].tree_
    classes = pipeline.named_steps["model"].classes_

    def raw_threshold(feature: int, z: float) -> str:
        if feature < n_protocol:                       # one-hot Protocol column: 0/1 indicator
            return f"{names[feature]} is {'not set' if z < 0.5 else 'set'}"
        j = feature - n_protocol
        return f"{z * scale[j] + mean[j]:,.2f}"

    scale, mean = scaler.scale_, scaler.mean_
    lines: list[str] = []

    def walk(node: int, depth: int) -> None:
        indent = "|   " * depth
        if tree.children_left[node] == -1 or depth == max_depth:
            counts = tree.value[node][0]
            label = classes[int(np.argmax(counts))]
            suffix = "" if tree.children_left[node] == -1 else " (subtree truncated)"
            lines.append(f"{indent}→ {label}{suffix}")
            return
        feature, z = tree.feature[node], tree.threshold[node]
        value = raw_threshold(feature, z)
        lines.append(f"{indent}if {names[feature]} <= {value}:")
        walk(tree.children_left[node], depth + 1)
        lines.append(f"{indent}else ({names[feature]} > {value}):")
        walk(tree.children_right[node], depth + 1)

    walk(0, 0)
    return "\n".join(lines)
