from sklearn.preprocessing import OneHotEncoder as _OneHotEncoder
from typing import Iterable

class OneHotEncoder:
    """Thin wrapper around scikit-learn's OneHotEncoder using sparse output."""

    def __init__(self, columns: Iterable[str], **kwargs) -> None:
        self.columns = list(columns)
        self.encoder = _OneHotEncoder(handle_unknown="ignore", sparse_output=True, **kwargs)

    def fit(self, X, y=None):
        self.encoder.fit(X[self.columns])
        return self

    def transform(self, X):
        return self.encoder.transform(X[self.columns])
