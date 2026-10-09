"""Reusable model components that must keep a stable import path.

Objects from this module can safely be referenced by joblib/pickle artifacts and
loaded by the web application in a different Python process.
"""

from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class SelectiveStandardScaler(BaseEstimator, TransformerMixin):
    """Scale continuous columns while preserving categorical codes and names."""

    def __init__(self, categorical_columns: Optional[Iterable[str]] = None):
        self.categorical_columns = categorical_columns

    def fit(self, X, y=None):
        X_df = self._as_dataframe(X)
        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        categorical = set(self.categorical_columns or [])
        self.scale_columns_ = [
            column for column in X_df.columns if column not in categorical
        ]
        self.scaler_ = StandardScaler()
        if self.scale_columns_:
            self.scaler_.fit(X_df[self.scale_columns_])
        return self

    def transform(self, X):
        X_df = self._as_dataframe(X).copy()
        missing = [
            column
            for column in self.feature_names_in_
            if column not in X_df.columns
        ]
        if missing:
            raise ValueError(f"Scaler input is missing columns: {missing[:20]}")
        X_df = X_df.loc[:, self.feature_names_in_]
        if self.scale_columns_:
            X_df.loc[:, self.scale_columns_] = self.scaler_.transform(
                X_df[self.scale_columns_]
            )
        return X_df

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)

    @staticmethod
    def _as_dataframe(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "ML9 preprocessing expects a pandas DataFrame with named columns."
            )
        return X
