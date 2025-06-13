from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer

from encoding import EncodingManager


def build_preprocessor(categorical: list[str]):
    manager = EncodingManager("onehot", columns=categorical)
    encoder = manager.encoder  # underlying encoder
    return ColumnTransformer(
        [
            ("cat", encoder.encoder, categorical),
        ],
        remainder="passthrough",
        sparse_output=True,
    )
