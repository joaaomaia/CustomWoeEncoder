import pandas as pd
from typing import Iterable

class LeaveOneOutEncoder:
    """Naive leave-one-out target encoder."""

    def __init__(self, columns: Iterable[str]):
        self.columns = list(columns)
        self.sum_ = {}
        self.count_ = {}
        self.global_mean = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.global_mean = y.mean()
        df = pd.concat([X[self.columns], y], axis=1)
        for col in self.columns:
            grouped = df.groupby(col)[y.name]
            self.sum_[col] = grouped.sum()
            self.count_[col] = grouped.count()
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            s = X[col]
            sums = self.sum_[col].reindex(s).values
            counts = self.count_[col].reindex(s).values
            if y is not None:
                sums -= y.values
                counts = counts - 1
            enc = sums / counts.clip(min=1)
            X[col] = pd.Series(enc, index=s.index).fillna(self.global_mean)
        return X
