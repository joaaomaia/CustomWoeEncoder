from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt

from ..comparison import ComparisonResult


class ReportBuilder:
    """Generate simple HTML and text reports comparing raw vs encoded data."""

    def __init__(
        self,
        X_raw: pd.DataFrame,
        X_enc: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
        comparison: Optional[ComparisonResult] = None,
    ) -> None:
        self.X_raw = X_raw
        self.X_enc = X_enc
        self.metadata = metadata or {}
        self.comparison = comparison

    def _summary(self) -> pd.DataFrame:
        mem_raw = self.X_raw.memory_usage(deep=True).sum()
        mem_enc = self.X_enc.memory_usage(deep=True).sum()
        return pd.DataFrame(
            {
                "n_features": [self.X_raw.shape[1], self.X_enc.shape[1]],
                "memory": [mem_raw, mem_enc],
            },
            index=["raw", "encoded"],
        )

    def to_html(self, path: str | Path) -> None:
        html = self._summary().to_html()
        Path(path).write_text(html, encoding="utf-8")

    def to_text(self) -> str:
        return self._summary().to_string()

    def save_plot(self, path: str | Path) -> None:
        counts = [self.X_raw.shape[1], self.X_enc.shape[1]]
        plt.bar(["raw", "encoded"], counts)
        plt.ylabel("n_features")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
