"""Model-family-appropriate preprocessing pipelines.

Two different model families need two different representations of the same
features, and getting this wrong is a common source of unfair model comparisons:

* **Gradient-boosted trees** (XGBoost, LightGBM, Random Forest) split on ordered
  values and handle missingness natively. They get ordinal-coded categoricals
  and untouched numerics, with NaN preserved as a real signal (a missing
  procalcitonin means the test was not ordered, which is informative).

* **Linear / margin models** (Logistic Regression, SVM) need one-hot categorical
  input, imputed numerics and standardised scales, or they will silently rank
  `organism_clean = 173` above `organism_clean = 12`.

Both are wrapped in `Pipeline`s so that every fitted transformation is learned on
the training fold only and applied to the validation fold — the same guarantee
the feature-selection stage relies on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class OrdinalCategoryEncoder(BaseEstimator, TransformerMixin):
    """Map categoricals to integer codes learned on the training fold only.

    Unseen categories at transform time become -1, which trees treat as their
    own branch rather than silently colliding with a real category.
    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        cols = self.columns if self.columns is not None else list(X.columns)
        self.columns_ = [c for c in cols if c in X.columns]
        self.mapping_ = {}
        for c in self.columns_:
            cats = pd.Index(pd.unique(X[c].astype("object").dropna()))
            self.mapping_[c] = {v: i for i, v in enumerate(sorted(cats, key=str))}
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.columns_:
            m = self.mapping_[c]
            X[c] = X[c].astype("object").map(m).astype("float32")
            X[c] = X[c].fillna(-1.0)
        for c in X.columns:
            if c not in self.columns_:
                X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)


def tree_preprocessor(categorical_columns) -> Pipeline:
    """Ordinal-code categoricals, leave numerics (and their NaNs) alone."""
    return Pipeline([("encode", OrdinalCategoryEncoder(columns=categorical_columns))])


def linear_preprocessor(categorical_columns, numeric_columns,
                        min_frequency: int = 200) -> ColumnTransformer:
    """One-hot categoricals (rare levels pooled), impute + scale numerics.

    `min_frequency` pools categories seen fewer than N times in the training fold
    into a single `infrequent` level. Without it, `organism_clean` alone would
    add 245 columns, most of them near-empty, and the linear models would be
    fitting noise.
    """
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=min_frequency,
                                 sparse_output=True)),
    ])
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(
        [("cat", cat, list(categorical_columns)),
         ("num", num, list(numeric_columns))],
        remainder="drop", sparse_threshold=0.3)


def as_object_frame(df: pd.DataFrame, categorical_columns) -> pd.DataFrame:
    """Ensure categorical columns are plain object dtype for the encoders."""
    out = df.copy()
    for c in categorical_columns:
        if c in out.columns:
            out[c] = out[c].astype("object")
    return out
