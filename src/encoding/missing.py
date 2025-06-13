from __future__ import annotations

import numpy as np
import pandas as pd

class MissingHandler:
    """Central utility for NaN management inspired by XGBoost."""

    def __init__(self, sentinel: str | int | float | None = np.nan) -> None:
        self.sentinel = sentinel
        self.mapping: dict[str, str] = {}

    def fit(self, df: pd.DataFrame) -> "MissingHandler":
        self.dtypes_ = df.dtypes.to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, dtype in self.dtypes_.items():
            if out[col].isna().any():
                if self.sentinel is np.nan:
                    self.mapping[col] = "kept_as_nan"
                else:
                    self.mapping[col] = f"filled_with_{self.sentinel}"
                    out[col] = out[col].fillna(self.sentinel).astype(dtype)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def to_dict(self) -> dict[str, object]:
        return {"sentinel": self.sentinel, "mapping": self.mapping}
