"""Exploratory analysis on the training set only.

I never pass the test set to anything in this module. Every ranking here (variance, correlation,
mutual information) is a statistic of the data; computing it on the full sample would let test-set
information shape my feature and model choices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_classif

from . import config
from .features import model_columns, symlog


def numeric_model_features(X: pd.DataFrame) -> pd.DataFrame:
    """Return the non-constant numeric columns my main model can see (no Timestamp, Dst Port, Protocol)."""
    _, numeric = model_columns(X)
    numeric_frame = X[numeric]
    return numeric_frame.loc[:, numeric_frame.nunique() > 1]


def class_counts(y: pd.Series) -> pd.DataFrame:
    """Return the count and percentage of each class, sorted by class name."""
    counts = y.value_counts().sort_index()
    return pd.DataFrame({"flows": counts, "percent": counts / counts.sum() * 100})


def missing_values(X: pd.DataFrame) -> pd.Series:
    """Count missing values per column (`isnull().sum()`), keeping only columns that have any.

    Missing and infinite values were removed during cleaning in notebook 01 (3,824 rows), so on the
    sample this should be empty; I compute it anyway as a check that cleaning really happened.
    """
    counts = X.isnull().sum()
    return counts[counts > 0]


def descriptive_statistics(X_numeric: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics for every numeric feature on the raw scale, as taught in weeks 3 and 5.

    Measures of location (mean, median, mode), of spread (standard deviation, variance, range,
    inter-quartile range) and of shape (skewness, kurtosis). For these features the mean is far above
    the median wherever a few very long or very large flows exist, which skewness quantifies; I
    report kurtosis as excess kurtosis (0 for a normal distribution), pandas' convention.
    """
    q1, q3 = X_numeric.quantile(0.25), X_numeric.quantile(0.75)
    return pd.DataFrame({
        "mean": X_numeric.mean(),
        "median": X_numeric.median(),
        "mode": X_numeric.mode().iloc[0],
        "std": X_numeric.std(),
        "variance": X_numeric.var(),
        "min": X_numeric.min(),
        "max": X_numeric.max(),
        "range": X_numeric.max() - X_numeric.min(),
        "IQR": q3 - q1,
        "skewness": X_numeric.skew(),
        "kurtosis": X_numeric.kurt(),
    })


def variance_ranking(X_numeric: pd.DataFrame) -> pd.Series:
    """Rank features by variance after the symlog transform, highest first.

    On the raw scale the ranking would simply reproduce the units: rates in bytes per second and
    durations in microseconds have variances many orders of magnitude larger than flag counts. After
    symlog the variance reflects how much a feature actually spreads relative to its own magnitude.
    """
    return symlog(X_numeric).var().sort_values(ascending=False)


def correlation_matrix(X_numeric: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return the Pearson correlation of the symlog-transformed columns, ordered by clustering.

    I correlate after symlog because Pearson's r is sensitive to extreme values; on the raw scale a
    handful of very long flows would dominate. I reorder rows and columns with average-linkage
    clustering on 1 - |r| so that groups of redundant features appear as blocks in the heatmap.
    """
    corr = symlog(X_numeric[columns]).corr()
    distance = 1 - corr.abs().to_numpy()
    np.fill_diagonal(distance, 0)
    order = leaves_list(linkage(squareform(distance, checks=False), method="average"))
    ordered = corr.columns[order]
    return corr.loc[ordered, ordered]


def highly_correlated_pairs(corr: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """List each feature pair with |r| >= threshold once (upper triangle only)."""
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().rename("r").reset_index()
    pairs.columns = ["feature_a", "feature_b", "r"]
    return pairs[pairs["r"].abs() >= threshold].sort_values("r", key=np.abs, ascending=False)


def mutual_information_ranking(
    X_numeric: pd.DataFrame, y: pd.Series, seed: int = config.SEED
) -> pd.Series:
    """Rank features by mutual information with the class label, highest first.

    I use mutual information because it captures any dependency, including the non-linear,
    threshold-like separations that brute-force flows show, whereas an ANOVA F-test assumes
    differences in means of roughly normal data, which these heavy-tailed features are not.
    """
    scores = mutual_info_classif(X_numeric, y, random_state=seed)
    return pd.Series(scores, index=X_numeric.columns).sort_values(ascending=False)


def label_entropy(y: pd.Series) -> float:
    """Return the Shannon entropy H(Y) of the labels in nats.

    Mutual information between a feature and the label can never exceed H(Y), so I report MI as a
    share of H(Y): a share near 100% means the feature alone almost determines the class.
    """
    p = y.value_counts(normalize=True).to_numpy()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def select_diverse(
    ranking: pd.Series, X_numeric: pd.DataFrame, k: int = 6, max_abs_r: float = 0.9
) -> list[str]:
    """Pick the k best-ranked features, skipping any that is redundant with one already picked.

    Many top features carry almost the same information (e.g. packet-length maximum and mean), so
    plotting the raw top-k would show the same pattern several times. I walk down the ranking and
    keep a feature only if its |r| (after symlog) with every kept feature is below `max_abs_r`.
    """
    transformed = symlog(X_numeric[ranking.index])
    chosen: list[str] = []
    for feature in ranking.index:
        if all(abs(transformed[feature].corr(transformed[c])) < max_abs_r for c in chosen):
            chosen.append(feature)
        if len(chosen) == k:
            break
    return chosen


def outlier_share(X_numeric: pd.DataFrame, y: pd.Series, columns: list[str]) -> pd.DataFrame:
    """Return the % of flows outside Tukey's fences (Q1 - 1.5 IQR, Q3 + 1.5 IQR), per class.

    I compute fences within each class because a value that is extreme for benign traffic can be
    perfectly typical for an attack; fences over the pooled data would mix the two.
    """
    rows = {}
    for label, group in X_numeric[columns].groupby(y, observed=True):
        q1, q3 = group.quantile(0.25), group.quantile(0.75)
        iqr = q3 - q1
        outside = (group < q1 - 1.5 * iqr) | (group > q3 + 1.5 * iqr)
        rows[label] = outside.mean() * 100
    return pd.DataFrame(rows)


def port_profile(X: pd.DataFrame, y: pd.Series, top: int = 3) -> pd.DataFrame:
    """Return each class's most frequent destination ports with their share of the class."""
    frames = []
    for label, ports in X[config.DST_PORT_COL].groupby(y, observed=True):
        share = ports.value_counts(normalize=True).head(top) * 100
        frames.append(pd.DataFrame({
            "class": label,
            "dst_port": share.index.astype(int),
            "percent_of_class": share.to_numpy(),
            "distinct_ports_in_class": ports.nunique(),
        }))
    return pd.concat(frames, ignore_index=True)


# --- Week 5 lab: Pearson versus Spearman -----------------------------------------------------------

CORRELATION_METHODS = ("pearson_raw", "pearson_symlog", "spearman")


def pairwise_correlations(X_numeric: pd.DataFrame) -> pd.DataFrame:
    """Return one row per unordered pair of non-constant columns with three correlation measures.

    Lab 5 (Exercise 3) contrasts Pearson's r, which measures *linear* association, with Spearman's
    rank correlation, which measures any *monotonic* association; a Spearman value well above the
    Pearson value suggests a curved (for example logarithmic) relationship that a log transform
    would straighten. I therefore also report Pearson after my symlog transform: if symlog is the
    right transform, it should bring Pearson close to Spearman.
    """
    frame = X_numeric.loc[:, X_numeric.nunique() > 1].astype(float)
    matrices = {
        "pearson_raw": frame.corr(method="pearson"),
        "pearson_symlog": symlog(frame).corr(method="pearson"),
        "spearman": frame.corr(method="spearman"),
    }
    upper_i, upper_j = np.triu_indices(frame.shape[1], k=1)
    names = np.asarray(frame.columns, dtype=object)
    pairs = pd.DataFrame({"feature_a": names[upper_i], "feature_b": names[upper_j]})
    for method, matrix in matrices.items():
        pairs[method] = matrix.to_numpy()[upper_i, upper_j]
    return pairs


def correlation_method_counts(
    pairs: pd.DataFrame, thresholds: tuple[float, ...] = (0.7, 0.9, 0.95)
) -> pd.DataFrame:
    """Count the pairs whose |correlation| reaches each threshold, for each measure.

    0.7 is the "high correlation" cut-off used in the Week 5 lab; 0.9 and 0.95 show how the picture
    changes for near-duplicate columns.
    """
    counts = {
        method: [int((pairs[method].abs() >= t).sum()) for t in thresholds] for method in CORRELATION_METHODS
    }
    return pd.DataFrame(counts, index=pd.Index(list(thresholds), name="threshold"))


# --- Week 6 lab: subsample for a readable pair plot -----------------------------------------------

def stratified_subsample(
    X: pd.DataFrame, y: pd.Series, n: int, keep_all: str = "FTP-BruteForce", seed: int = config.SEED
) -> tuple[pd.DataFrame, pd.Series]:
    """Draw `n` training rows for plotting, keeping every row of the class `keep_all`.

    The Week 6 lab subsamples 2,000 rows so that a scatter matrix stays readable. A plain random
    draw would keep almost no FTP flows (37 of 60,000), so I keep all of them and fill the rest of
    the sample in proportion to the other classes.
    """
    rare = y.index[y == keep_all]
    others = y.drop(index=rare)
    remaining = n - len(rare)
    shares = others.value_counts(normalize=True)
    quotas = (shares * remaining).round().astype(int)
    quotas.iloc[0] += remaining - quotas.sum()  # make the total exactly n after rounding
    rng = np.random.default_rng(seed)
    chosen = list(rare)
    for label, quota in quotas.items():
        pool = others.index[others == label].to_numpy()
        chosen.extend(rng.choice(pool, size=min(quota, len(pool)), replace=False))
    index = pd.Index(sorted(chosen))
    return X.loc[index], y.loc[index]
