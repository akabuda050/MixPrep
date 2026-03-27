from __future__ import annotations

import os
from pathlib import Path


def _xdg_data_home() -> Path:
    xdg = os.environ.get("XDG_DATA_HOME")
    return Path(xdg) if xdg else Path.home() / ".local" / "share"


def models_dir() -> Path:
    """Resolve model cache directory: MIXPREP_MODELS_DIR → XDG default."""
    env = os.environ.get("MIXPREP_MODELS_DIR")
    if env:
        return Path(env)
    return _xdg_data_home() / "mixprep" / "models"


def data_dir() -> Path:
    """Resolve data directory: MIXPREP_DATA_DIR → XDG default."""
    env = os.environ.get("MIXPREP_DATA_DIR")
    if env:
        return Path(env)
    return _xdg_data_home() / "mixprep" / "data"


def model_path(filename: str) -> Path:
    return models_dir() / filename


def ensure_models_dir() -> Path:
    d = models_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_data_dir() -> Path:
    d = data_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d
