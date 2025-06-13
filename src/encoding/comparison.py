from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class ComparisonResult:
    shape_change: Tuple[int, int] | None = None
    densification_factor: float | None = None
    dtype_changes: Dict[str, str] | None = None
    time_fit: float | None = None
    time_transform: float | None = None
