"""Test-set evaluation, ablation experiments and model interpretation.

I keep the test set out of every decision: the models, their hyper-parameters and the choice between
selection metrics are all fixed on the training set by cross-validation before anything here looks
at the test set. Ablations are scored with the same cross-validation, so they explain the models
without turning the test set into a second validation set.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd
from imblearn.under_sampling import TomekLinks
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedGroupKFold,
    cross_val_score,
    cross_validate,
    learning_curve,
)
from sklearn.neighbors import NearestNeighbors

from . import config, models
from .features import CorrelationFilter, build_pipeline

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
        "recall_ftp": recall_score(y_true, y_pred, labels=[models.MINORITY_CLASS], average="macro", zero_division=0),
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
    params: dict | None = None,
) -> tuple[BaseEstimator, pd.DataFrame]:
    """Rebuild the selected model of `run` for one ablation variant; return (pipeline, X to use).

    Variants keep the given (by default the refit) hyper-parameters and change exactly one thing:
      * "main"            – the tuned pipeline as selected (no Dst Port, SMOTE).
      * "with_dst_port"   – Dst Port added back as a numeric feature.
      * "no_window_sizes" – Init Fwd/Bwd Win Byts removed (suspected attacker fingerprint).
      * "no_host_fingerprint" – the window sizes and Fwd Seg Size Min removed together.
      * "class_weight"    – no SMOTE; the classifier weights classes inversely to their frequency.
      * "no_resampling"   – no SMOTE and no class weights.
      * "tomek_links"     – SMOTE replaced by Tomek-link under-sampling.
      * "corr_filter_0.70", "corr_filter_0.95" – numeric columns whose Spearman |rho| with an earlier
        kept column reaches the threshold are dropped inside each training fold (Lab 5 feature selection).

    Tomek links are pairs of nearest neighbours with different labels. Removing the majority-class
    member of each pair cleans the class boundary instead of inventing minority flows, the opposite
    strategy to SMOTE. Like SMOTE it runs after scaling (it is distance-based) and only during fit.
    """
    # `params` lets a caller ablate the setting it actually reports (for example the one chosen by macro F1);
    # without it I use the search's refit setting (best weighted F1).
    params = run.search.best_params_ if params is None else params
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
        if not models.supports_class_weight(run.name):
            raise ValueError(f"{run.name} has no class_weight parameter")
        estimator.set_params(class_weight="balanced")
        sampler = None
    elif variant == "no_resampling":
        sampler = None
    elif variant == "tomek_links":
        sampler = TomekLinks(sampling_strategy="auto", n_jobs=1)
    elif variant.startswith("corr_filter_"):
        threshold = float(variant.removeprefix("corr_filter_"))
        if threshold not in config.CORRELATION_THRESHOLDS:
            raise ValueError(f"Unknown variant {variant!r}")
    elif variant not in ("main", "with_dst_port"):
        raise ValueError(f"Unknown variant {variant!r}")

    pipeline = build_pipeline(X_variant, estimator, sampler=sampler, include_dst_port=include_port)
    if variant.startswith("corr_filter_"):
        pipeline.set_params(preprocess__numeric__corr_filter=CorrelationFilter(threshold))
    keep_smote_params = variant in ("main", "with_dst_port", "no_window_sizes", "no_host_fingerprint") or (
        variant.startswith("corr_filter_")
    )
    pipeline.set_params(**other_params, **(resample_params if keep_smote_params else {}))
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
        perm = permutation_importance(model, X, y, scoring="f1_macro", n_repeats=5, random_state=config.SEED, n_jobs=-1)
    permutation = pd.Series(perm.importances_mean, index=X.columns)
    permutation_std = pd.Series(perm.importances_std, index=X.columns)
    table = pd.DataFrame(
        {
            "impurity_importance": impurity,
            "permutation_drop_f1_macro": permutation.reindex(impurity.index),
            "permutation_std": permutation_std.reindex(impurity.index),
        }
    )
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
        if feature < n_protocol:  # one-hot Protocol column: 0/1 indicator
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


# --- Techniques from the module's lectures, applied to this data -----------------------------------


def fold_composition(y: pd.Series, minority: str = models.MINORITY_CLASS) -> pd.DataFrame:
    """Count the minority-class flows in each validation fold under three 5-fold splitters.

    The Week 7 lecture introduces KFold and StratifiedKFold. With only 37 FTP-BruteForce flows in the
    training set, the difference is not academic: plain KFold can leave a fold with far fewer FTP
    flows than another, so per-fold FTP scores would vary for reasons that have nothing to do with
    the model. Stratification gives every fold the same share of each class.
    """
    splitters = {
        "KFold (no shuffle)": KFold(n_splits=config.CV_FOLDS),
        "KFold (shuffled)": KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED),
        "StratifiedKFold (shuffled)": models.cv_splitter(),
    }
    dummy = np.zeros(len(y))
    rows = {}
    for name, splitter in splitters.items():
        rows[name] = [int((y.iloc[val] == minority).sum()) for _, val in splitter.split(dummy, y)]
    table = pd.DataFrame(rows, index=[f"fold {k + 1}" for k in range(config.CV_FOLDS)]).T
    table["min"], table["max"] = table.min(axis=1), table.max(axis=1)
    return table


def splitter_scores(pipeline: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Score one pipeline with `cross_val_score` under KFold and StratifiedKFold (Week 7 style).

    `cross_val_score` returns one score per fold; I report the mean and standard deviation of
    accuracy and macro F1 so the effect of the splitter on the spread of the scores is visible.
    """
    splitters = {
        "KFold (shuffled)": KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED),
        "StratifiedKFold (shuffled)": models.cv_splitter(),
    }
    rows = []
    for name, splitter in splitters.items():
        for metric in ("accuracy", "f1_macro"):
            scores = cross_val_score(clone(pipeline), X, y, cv=splitter, scoring=metric, n_jobs=-1)
            rows.append(
                {
                    "splitter": name,
                    "metric": metric,
                    "mean": scores.mean(),
                    "std": scores.std(),
                    "per_fold": ", ".join(f"{v:.4f}" for v in scores),
                }
            )
    return pd.DataFrame(rows)


def regularisation_comparison(
    run: models.SearchRun,
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    extra_c: tuple[float, ...] = (1.0, 0.1, 0.01),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare L2 (ridge) and L1 (lasso) penalties for the selected Logistic Regression (Week 9).

    I keep every selected setting (transform, SMOTE) and change only the penalty and its strength,
    using the saga solver because it supports both. scikit-learn 1.8 replaced the `penalty` argument
    by `l1_ratio`: 0 gives the L2 penalty and 1 the L1 penalty. C is the inverse of the lecture's
    alpha (lambda), so a smaller C means stronger shrinkage; besides the selected C I try stronger
    penalties to show how L1 removes features while L2 only shrinks them. With saga's default
    tolerance (1e-4) one fit took over ten minutes; at 1e-3 it converges in a few hundred passes.

    For each setting I count the features whose coefficient is exactly zero for every class (L1's
    embedded feature selection) and report training- and validation-fold macro F1, so a change in
    over-fitting is visible. Returns (scores per setting, coefficients at the selected C).
    """
    rows, coefficients = [], {}
    c_values = (float(params["model__C"]), *extra_c)
    for c in c_values:
        for label, ratio in (("L2 (ridge)", 0.0), ("L1 (lasso)", 1.0)):
            pipeline = clone(run.search.estimator).set_params(**params)
            pipeline.set_params(
                model=LogisticRegression(
                    C=c,
                    l1_ratio=ratio,
                    solver="saga",
                    tol=1e-3,
                    max_iter=5_000,
                    random_state=config.SEED,
                )
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                cv = cross_validate(
                    pipeline, X, y, cv=models.cv_splitter(), scoring="f1_macro", return_train_score=True, n_jobs=-1
                )
                fitted = clone(pipeline).fit(X, y)
            coef = pd.DataFrame(
                fitted.named_steps["model"].coef_.T,
                index=fitted.named_steps["preprocess"].get_feature_names_out(),
                columns=fitted.named_steps["model"].classes_,
            )
            if c == c_values[0]:
                coefficients[label] = coef
            zero = (coef.abs() < 1e-10).all(axis=1)
            rows.append(
                {
                    "C": c,
                    "penalty": label,
                    "train_f1_macro": cv["train_score"].mean(),
                    "cv_f1_macro_mean": cv["test_score"].mean(),
                    "cv_f1_macro_std": cv["test_score"].std(),
                    "features_removed": int(zero.sum()),
                    "features_used": int((~zero).sum()),
                }
            )
    return pd.DataFrame(rows), pd.concat(coefficients, axis=1)


def learning_curve_table(
    pipeline: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    sizes: tuple[float, ...] = (0.3, 0.45, 0.6, 0.8, 1.0),
) -> pd.DataFrame:
    """Training- and validation-fold macro F1 as the training set grows (Week 2: under/over-fitting).

    I use the same 5 stratified folds and shuffle each training fold before taking the first n flows,
    so small subsets still contain FTP-BruteForce. I start at 30% because below that a fold would hold
    fewer FTP flows than SMOTE's neighbour count, and the pipeline could not be fitted.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        n, train, valid = learning_curve(
            clone(pipeline),
            X,
            y,
            train_sizes=list(sizes),
            cv=models.cv_splitter(),
            scoring="f1_macro",
            shuffle=True,
            random_state=config.SEED,
            n_jobs=-1,
            error_score=np.nan,
        )
    return pd.DataFrame(
        {
            "train_flows": n,
            "train_mean": np.nanmean(train, axis=1),
            "train_std": np.nanstd(train, axis=1),
            "valid_mean": np.nanmean(valid, axis=1),
            "valid_std": np.nanstd(valid, axis=1),
        }
    )


# --- Unseen day (Week 6 lab: predict on new data) --------------------------------------------------


def wilson_interval(errors: int, n: int, z: float = 1.96) -> dict[str, float]:
    """Error rate with a 95% Wilson score interval.

    I use the Wilson interval rather than rate +/- 1.96 * SE because false alarms are rare: the
    normal approximation gives a negative lower bound, or a zero-width interval when no error is seen.
    """
    rate = errors / n
    centre = (rate + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return {"rate": rate, "ci_low": max(0.0, centre - half), "ci_high": min(1.0, centre + half), "n": n}


def apply_selection_rule(cv_mean: pd.Series, cv_std: pd.Series, inference_ms: pd.Series) -> tuple[str, list[str]]:
    """My pre-registered rule: highest CV macro F1; models within one SD of the leader tie; the fastest tied wins.

    Returns (chosen model, tied models in the order given). I fixed this rule before looking at any test score.
    """
    leader = cv_mean.idxmax()
    tied = cv_mean[cv_mean >= cv_mean[leader] - cv_std[leader]].index.tolist()
    return inference_ms[tied].idxmin(), tied


def nearest_train_distance(
    preprocess, X_train: pd.DataFrame, y_train: pd.Series, X_other: pd.DataFrame, p: int = 2
) -> pd.DataFrame:
    """For each row of X_other: distance to the nearest training flow and that flow's label.

    I measure in the space the k-NN model works in: `preprocess` is the already fitted first step of its
    pipeline (one-hot Protocol, constant columns dropped, MinMax scaling), and `p` its Minkowski power. I never
    refit it here; refitting on another set would change the scale and so the distances.
    """
    train = preprocess.transform(X_train)
    finder = NearestNeighbors(n_neighbors=1, p=p).fit(train)
    distance, index = finder.kneighbors(preprocess.transform(X_other))
    return pd.DataFrame(
        {"distance": distance[:, 0], "neighbour_label": y_train.to_numpy()[index[:, 0]]}, index=X_other.index
    )


def train_self_distance(preprocess, X_train: pd.DataFrame, y_train: pd.Series, p: int = 2) -> pd.DataFrame:
    """Like nearest_train_distance for the training flows themselves, leaving each flow out of its own search.

    This is the yardstick: how close two different training flows normally are. Calling kneighbors without
    a query makes scikit-learn skip the point itself, so an exact copy elsewhere still counts as distance 0.
    """
    finder = NearestNeighbors(n_neighbors=1, p=p).fit(preprocess.transform(X_train))
    distance, index = finder.kneighbors()
    return pd.DataFrame(
        {"distance": distance[:, 0], "neighbour_label": y_train.to_numpy()[index[:, 0]]}, index=X_train.index
    )


def distance_summary(
    dist: pd.DataFrame, labels: pd.Series, thresholds: tuple[float, ...] = (1e-3, 1e-2)
) -> pd.DataFrame:
    """Per true class (and "All"): flows, median and 90th percentile distance, share closer than each
    threshold, and share whose nearest training flow has the same class."""
    frame = dist.assign(label=np.asarray(labels))
    rows = {}
    for name, part in [*frame.groupby("label", sort=False), ("All", frame)]:
        row = {"flows": len(part), "median": part["distance"].median(), "p90": part["distance"].quantile(0.9)}
        row.update({f"share < {t:g}": (part["distance"] < t).mean() for t in thresholds})
        row["neighbour same class"] = (part["neighbour_label"] == part["label"]).mean()
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index")


CLASS_LABELS = ("Benign", "FTP-BruteForce", "SSH-Bruteforce")


def shape_group_splitter() -> StratifiedGroupKFold:
    """Week 7's StratifiedKFold extended with groups: no flow shape is in both the training and validation fold."""
    return StratifiedGroupKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)


def _folds(splitter, X: pd.DataFrame, labels: np.ndarray, groups: np.ndarray):
    # Only group-aware splitters take `groups`; scikit-learn warns when the others are given it.
    return splitter.split(X, labels, groups) if "Group" in type(splitter).__name__ else splitter.split(X, labels)


def group_overlap(splitter, X: pd.DataFrame, y: pd.Series, groups) -> int:
    """Number of (fold, group) pairs where a group sits on both sides of the split; 0 for a grouped splitter."""
    groups = np.asarray(groups)
    return sum(
        len(set(groups[train]) & set(groups[valid])) for train, valid in _folds(splitter, X, np.asarray(y), groups)
    )


def grouped_cv(
    pipeline: BaseEstimator, X: pd.DataFrame, y: pd.Series, groups, splitter=None, return_predictions: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
    """Per fold: recall of each class, benign false-alarm rate, macro F1 on Benign and SSH flows, and counts.

    By default the folds are grouped by `groups` (shape_group_splitter); pass another splitter for the ungrouped
    comparison. A class absent from a validation fold gets NaN recall. A class absent from a training fold
    cannot be predicted there, so its recall in that validation fold is 0: I report this, I do not hide it.
    I score macro F1 on the Benign and SSH flows only, so the comparison does not depend on where FTP lands.
    With `return_predictions`, I also return each flow's out-of-fold prediction (to see *which* flows are missed).
    """
    splitter = shape_group_splitter() if splitter is None else splitter
    labels, groups = np.asarray(y), np.asarray(groups)
    benign, ftp, ssh = CLASS_LABELS
    rows, predictions = [], pd.Series(index=X.index, dtype=object)
    for fold, (train, valid) in enumerate(_folds(splitter, X, labels, groups), start=1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = clone(pipeline).fit(X.iloc[train], labels[train])
        true, pred = labels[valid], model.predict(X.iloc[valid])
        predictions.iloc[valid] = pred
        row = {
            "fold": fold,
            "valid groups": len(set(groups[valid])),
            "FTP train": int((labels[train] == ftp).sum()),
            "FTP valid": int((true == ftp).sum()),
        }
        for name in CLASS_LABELS:
            mask = true == name
            row[f"recall {name}"] = float((pred[mask] == name).mean()) if mask.any() else np.nan
        row["false alarm rate"] = 1 - row[f"recall {benign}"]
        two = np.isin(true, [benign, ssh])
        row["macro F1 (Benign, SSH)"] = f1_score(
            true[two], pred[two], labels=[benign, ssh], average="macro", zero_division=0
        )
        rows.append(row)
    table = pd.DataFrame(rows).set_index("fold")
    return (table, predictions) if return_predictions else table


def unseen_day_report(model, flows: pd.DataFrame) -> pd.DataFrame:
    """How a fitted model labels each true class of a new day's traffic.

    Rows are the true labels of the new day (including attack types the model never saw); columns
    give the number of flows and the share predicted as each of the model's three classes. For an
    unseen attack, the "predicted Benign" share is the share of that attack the detector would miss.
    """
    X = flows.drop(columns=[config.LABEL_COL])
    predicted = pd.Series(model.predict(X), index=flows.index)
    table = pd.crosstab(flows[config.LABEL_COL], predicted, normalize="index")
    for cls in ("Benign", "FTP-BruteForce", "SSH-Bruteforce"):
        if cls not in table.columns:
            table[cls] = 0.0
    table = table[["Benign", "FTP-BruteForce", "SSH-Bruteforce"]].add_prefix("predicted ")
    table.insert(0, "flows", flows[config.LABEL_COL].value_counts())
    table.index.name = "true label"
    return table


def final_comparison_table(selection: pd.DataFrame, nofp_cv: Mapping[str, float], order: list[str]) -> pd.DataFrame:
    """Build the comparison of all algorithms that the report shows as its main results table.

    `selection` holds one row per model with the settings re-selected by macro F1 (notebook 04, Step 5),
    and `nofp_cv` each model's cross-validated macro F1 after its search is rerun without the attacking
    hosts' TCP settings (Step 5 (v)). I show that re-tuned score rather than the drop with the original
    settings, because re-tuning recovers most of the drop and the reader should see what a model achieves
    when it is built without the fingerprint. I format the scores as text here because the table is read by
    people: five decimals for the cross-validated score, so k-NN's tiny spread does not print as
    "1.0 ± 0.0", and the test score with its 95% bootstrap interval in one cell.
    """

    def fixed(values: pd.Series, digits: int) -> pd.Series:
        return values.map(f"{{:.{digits}f}}".format)

    table = pd.DataFrame(
        {
            "CV macro F1": fixed(selection["cv_f1_macro (macro-selected)"], 5)
            + " ± "
            + fixed(selection["cv_f1_macro_std"], 5),
            "Test accuracy": fixed(selection["test_accuracy"], 5),
            "Test macro F1 [95% CI]": fixed(selection["test_f1_macro"], 4)
            + " ["
            + fixed(selection["test_ci_low"], 4)
            + ", "
            + fixed(selection["test_ci_high"], 4)
            + "]",
            "Test errors (of 15,000)": selection["test_errors"],
            "Test FTP recall": selection["test_recall_ftp"],
            "Training time (s)": selection["fit_seconds"].round(2),
            "Inference (ms / 1,000 flows)": selection["inference_ms_per_1000"].round(2),
            "CV macro F1 re-tuned without host fingerprint": fixed(pd.Series(nofp_cv), 5),
        }
    )
    return table.reindex(order)
