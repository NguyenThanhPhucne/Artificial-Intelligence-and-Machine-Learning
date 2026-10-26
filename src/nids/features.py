"""Feature selection and the base preprocessing pipeline shared by every model.

I put every transformation that learns from data (dropping constant columns, scaling, later SMOTE)
inside an imbalanced-learn Pipeline. Cross-validation then refits these steps on each training fold
only, so no statistic from a validation fold or from the test set can leak into the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    # outer level, where the resampler sits.
    numeric_steps = SkPipeline([
        ("drop_constant", VarianceThreshold(threshold=0.0)),
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
