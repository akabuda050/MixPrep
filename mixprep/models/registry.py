from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

_MANIFEST_PATH = Path(__file__).parent / "manifest.yaml"


class ModelEntry(BaseModel):
    name: str
    url: str
    framework: str
    description: str


class ModelRegistry:
    def __init__(self, manifest_path: Path = _MANIFEST_PATH) -> None:
        if not manifest_path.exists():
            self._models: dict[str, ModelEntry] = {}
            return
        with open(manifest_path) as f:
            raw = yaml.safe_load(f) or {}
        self._models = {
            name: ModelEntry(name=name, **entry) for name, entry in raw.get("models", {}).items()
        }

    def get(self, name: str) -> ModelEntry | None:
        return self._models.get(name)

    def all(self) -> list[ModelEntry]:
        return list(self._models.values())

    def names(self) -> list[str]:
        return list(self._models.keys())
