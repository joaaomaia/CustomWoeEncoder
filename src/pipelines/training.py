from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor


def build_pipeline(categorical: list[str]):
    pre = build_preprocessor(categorical)
    clf = LogisticRegression(max_iter=100)
    return Pipeline([
        ("pre", pre),
        ("clf", clf),
    ])
