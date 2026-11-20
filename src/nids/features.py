"""Feature selection and the base preprocessing pipeline shared by every model.

I put every transformation that learns from data (dropping constant columns, scaling, later SMOTE)
inside an imbalanced-learn Pipeline. Cross-validation then refits these steps on each training fold
only, so no statistic from a validation fold or from the test set can leak into the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config


def model_columns(
    X: pd.DataFrame, include_dst_port: bool = False
) -> tuple[list[str], list[str]]:
    """Return (categorical, numeric) column lists that my models are allowed to see.

    I always drop Timestamp. I drop Dst Port in the main model because CIC created the labels from
    the attack schedule, IPs, ports and protocol, so the port encodes the labelling procedure rather
    than traffic behaviour; `include_dst_port=True` exists only for the ablation that measures this.
    """
    excluded = set(config.MODEL_EXCLUDED_COLS)
    if include_dst_port:
        excluded.discard(config.DST_PORT_COL)
    categorical = [c for c in config.CATEGORICAL_COLS if c in X.columns]
    numeric = [c for c in X.columns if c not in excluded and c not in categorical]
    return categorical, numeric


def symlog(values: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
    """Apply sign(x) * log1p(|x|).

    I use this instead of plain log1p because the window columns contain -1 ("no TCP window"), for
    which log1p is -infinity. The transform is monotonic, keeps 0 at 0, and compresses the heavy
    right tails of byte counts and microsecond durations so that variances and correlations are not
    dominated by a few extreme flows.
    """
    return np.sign(values) * np.log1p(np.abs(values))


def symlog_transformer() -> FunctionTransformer:
    """Wrap symlog as a pipeline step that keeps feature names (for importance plots later)."""
    return FunctionTransformer(symlog, feature_names_out="one-to-one")


class CorrelationFilter(TransformerMixin, BaseEstimator):
    """Drop numeric columns that are strongly rank-correlated with a column I already kept.

    Lab 5 treats |r| >= 0.7 as a high correlation; two such columns carry largely the same
    information, which makes linear coefficients unstable and costs fitting time. I use Spearman's
    rank correlation rather than Pearson's because my EDA (section 2.3.3) showed that most strong
    relationships between flow features are monotonic but curved, which Pearson under-states; ranks
    are also unchanged by the optional symlog step, so the same columns are chosen with or without it.

    The scan is greedy in input order: a column is kept when its |rho| with every column kept so far
    is below `threshold`. The rule is learned in `fit` only, so inside a cross-validated pipeline each
    training fold chooses its own columns and the validation fold never influences the choice.
    """

    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold

    def fit(self, X, y=None) -> "CorrelationFilter":
        frame = self._as_frame(X)
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        # Pearson correlation of the ranks is Spearman's rho; NaN (a constant column) counts as 0.
        rho = frame.rank().corr().abs().fillna(0.0).to_numpy()
        names = list(frame.columns)
        kept: list[int] = []
        dropped: dict[str, str] = {}
        for j in range(len(names)):
            partner = next((k for k in kept if rho[j, k] >= self.threshold), None)
            if partner is None:
                kept.append(j)
            else:
                dropped[names[j]] = names[partner]
        self.n_features_in_ = len(names)
        self.kept_indices_ = np.array(kept, dtype=int)
        self.kept_columns_ = [names[k] for k in kept]
        self.dropped_ = dropped
        return self

    def transform(self, X):
        if hasattr(X, "iloc"):
            return X.iloc[:, self.kept_indices_]
        return np.asarray(X)[:, self.kept_indices_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is not None:
            return np.asarray(input_features, dtype=object)[self.kept_indices_]
        return np.asarray(self.kept_columns_, dtype=object)

    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        array = np.asarray(X, dtype=float)
        return pd.DataFrame(array, columns=[f"x{i}" for i in range(array.shape[1])])


def build_preprocessor(X: pd.DataFrame, include_dst_port: bool = False) -> ColumnTransformer:
    """Build the unfitted column transformer: one-hot Protocol, drop constants and scale numerics.

    * Protocol (0, 6, 17) is a category, not a quantity, so I one-hot encode it; `handle_unknown`
      is set to ignore so an unseen protocol at prediction time does not crash the model.
    * VarianceThreshold(0) removes the columns that are constant in the training fold. Ten columns
      are constant in the whole file, but I let the pipeline find them on each fold rather than
      hard-coding the list, which keeps the rule correct if the data changes.
    * StandardScaler puts features on comparable scales. Logistic Regression and SMOTE (which
      interpolates between nearest neighbours using Euclidean distance) both need this; the tree
      models are scale-invariant, so the step is harmless for them and keeps one pipeline for all.
    """
    categorical, numeric = model_columns(X, include_dst_port)
    # A plain scikit-learn Pipeline is enough here; the imblearn Pipeline is only needed at the
    # outer level, where the resampler sits. The "log" step is a pass-through by default; I let the
    # hyper-parameter search switch it to symlog, because my EDA showed heavy right tails that can
    # dominate a linear model even after standardisation (trees are unaffected by monotonic maps).
    numeric_steps = SkPipeline([
        ("log", "passthrough"),
        ("drop_constant", VarianceThreshold(threshold=0.0)),
        # Off by default; the feature-selection ablation sets it to CorrelationFilter(threshold).
        ("corr_filter", "passthrough"),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(
        [
            ("protocol", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("numeric", numeric_steps, numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(
    X: pd.DataFrame,
    estimator: BaseEstimator,
    sampler: BaseEstimator | None = None,
    include_dst_port: bool = False,
) -> Pipeline:
    """Assemble preprocessing -> optional resampler (SMOTE) -> estimator.

    I place SMOTE after scaling and inside the imblearn Pipeline rather than resampling the data
    before cross-validation: imblearn applies samplers only during `fit`, so synthetic flows are
    created from each training fold alone and never appear in a validation fold or the test set.
    Resampling before the split would let synthetic copies of test-like flows into training and
    inflate every score.
    """
    steps: list[tuple[str, BaseEstimator]] = [("preprocess", build_preprocessor(X, include_dst_port))]
    if sampler is not None:
        steps.append(("resample", sampler))
    steps.append(("model", estimator))
    return Pipeline(steps)
