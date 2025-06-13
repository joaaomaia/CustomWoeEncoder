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

    def save(
        self,
        name: str,
        encoder: Any,
        metadata: Dict[str, Any],
        missing_handler: Any | None = None,
    ) -> Path:
        artefact_dir = self.base_path / name
        artefact_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(encoder, artefact_dir / "encoder.pkl")
        if missing_handler is not None:
            joblib.dump(missing_handler, artefact_dir / "missing_handler.pkl")
        with open(artefact_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return artefact_dir

    def load(self, name: str):
        artefact_dir = self.base_path / name
        encoder = joblib.load(artefact_dir / "encoder.pkl")
        mh_path = artefact_dir / "missing_handler.pkl"
        missing_handler = joblib.load(mh_path) if mh_path.exists() else None
        with open(artefact_dir / "meta.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return encoder, missing_handler, metadata
