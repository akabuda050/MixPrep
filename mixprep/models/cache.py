from __future__ import annotations

import os
from pathlib import Path


def _resolve_base(env_var: str, default: Path) -> Path:
    """Resolve a base directory from an env var, resolving to absolute path."""
    env = os.environ.get(env_var)
    return Path(env).resolve() if env else default


def _xdg_data_home() -> Path:
    xdg = os.environ.get("XDG_DATA_HOME")
    return Path(xdg).resolve() if xdg else Path.home() / ".local" / "share"


def models_dir() -> Path:
    """Resolve model cache directory: MIXPREP_MODELS_DIR → XDG default."""
    return _resolve_base("MIXPREP_MODELS_DIR", _xdg_data_home() / "mixprep" / "models")


def data_dir(library: str) -> Path:
    """Resolve data directory for *library*.

    MIXPREP_DATA_DIR / libraries/<name>  (base from env or XDG default).

    Raises ValueError if *library* contains path separators or relative
    path components (e.g. '..') to prevent directory traversal.
    """
    if (
        not library
        or "/" in library
        or "\\" in library
        or library == ".."
        or library.startswith("..")
    ):
        raise ValueError(f"Invalid library name: {library!r}")
    base = _resolve_base("MIXPREP_DATA_DIR", _xdg_data_home() / "mixprep" / "data")
    return base / "libraries" / library


def model_path(filename: str) -> Path:
    """Resolve path for a model file by filename.

    Raises ValueError if *filename* contains path separators to prevent
    directory traversal.
    """
    if not filename or "/" in filename or "\\" in filename or ".." in filename:
        raise ValueError(f"Invalid model filename: {filename!r}")
    return models_dir() / filename


def ensure_models_dir() -> Path:
    d = models_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_data_dir(library: str) -> Path:
    d = data_dir(library)
    d.mkdir(parents=True, exist_ok=True)
    return d
