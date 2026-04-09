"""
Custom imputer for the PLD forecast model.
Lives in its own module so joblib can deserialise the pipeline from any entry point.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class MixedImputer(BaseEstimator, TransformerMixin):
    """
    Imputes NaN values with different strategies per column:
      - zero_fill_cols: fill with 0 (features where NaN means "absent / zero activity")
      - all other columns: fill with median (fit on training data)

    Used for thermal dispatch features (NaN = no data collected pre-2022, which
    semantically means no emergency dispatch occurred, so 0 is the correct fill).
    """

    def __init__(self, zero_fill_cols: list[str]):
        self.zero_fill_cols = zero_fill_cols

    def fit(self, X: pd.DataFrame, y=None):
        self._median_cols = [c for c in X.columns if c not in self.zero_fill_cols]
        self._all_cols = list(X.columns)
        self._median_imputer = SimpleImputer(strategy="median")
        self._median_imputer.set_output(transform="pandas")
        self._median_imputer.fit(X[self._median_cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self._median_cols] = self._median_imputer.transform(X[self._median_cols])
        for col in self.zero_fill_cols:
            if col in X.columns:
                X[col] = X[col].fillna(0.0)
        return X[self._all_cols]

    def set_output(self, *, transform=None):
        return self
