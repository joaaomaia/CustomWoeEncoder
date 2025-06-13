from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib


class ArtefactRegistry:
    """Interface for saving and loading encoder artefacts."""

    def save(self, name: str, encoder: Any, metadata: Dict[str, Any]) -> Path:
        raise NotImplementedError

    def load(self, name: str):
        raise NotImplementedError


class FilesystemRegistry(ArtefactRegistry):
    """Simple filesystem-based registry."""

    def __init__(self, base_path: str | Path = "artefacts") -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, encoder: Any, metadata: Dict[str, Any]) -> Path:
        artefact_dir = self.base_path / name
        artefact_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(encoder, artefact_dir / "encoder.pkl")
        with open(artefact_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return artefact_dir

    def load(self, name: str):
        artefact_dir = self.base_path / name
        return joblib.load(artefact_dir / "encoder.pkl")
