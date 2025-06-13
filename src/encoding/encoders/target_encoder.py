import pandas as pd
from typing import Iterable

class TargetEncoder:
    """Simple target mean encoder."""

    def __init__(self, columns: Iterable[str]):
        self.columns = list(columns)
        self.maps = {}
        self.global_mean = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.global_mean = y.mean()
        df = pd.concat([X[self.columns], y], axis=1)
        for col in self.columns:
            self.maps[col] = df.groupby(col)[y.name].mean()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            mapping = self.maps.get(col, {})
            X[col] = X[col].map(mapping).fillna(self.global_mean)
        return X
