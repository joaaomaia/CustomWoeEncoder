from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ColumnNameManager:
    """Maintain mapping between original and encoded column names."""

    mapping: Dict[str, List[str]] = field(default_factory=dict)

    def add(self, original: str, encoded: List[str]) -> None:
        self.mapping[original] = encoded

    def original_to_encoded(self, col: str) -> List[str]:
        return self.mapping.get(col, [])

    def encoded_to_original(self, col: str) -> str | None:
        for orig, enc_list in self.mapping.items():
            if col in enc_list:
                return orig
        return None

    def explain(self, feature: str) -> str:
        origin = self.encoded_to_original(feature)
        if origin:
            return f"{feature} derives from {origin}"
        return feature

    def flatten_names(self, df):
        df.columns = [
            c.replace(" ", "_") if isinstance(c, str) else "__".join(map(str, c))
            for c in df.columns
        ]
        return df
