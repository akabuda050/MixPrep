from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path

from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn

from mixprep.models.cache import ensure_models_dir, model_path
from mixprep.models.registry import ModelEntry, ModelRegistry

_MAX_RETRIES = 3
_RETRY_DELAY = 10  # seconds


def _fetch(url: str, dest: Path, reporthook=None) -> None:
    """Download url → dest atomically with retries."""
    tmp = dest.with_suffix(".tmp")
    last_err: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            urllib.request.urlretrieve(url, tmp, reporthook=reporthook)
            tmp.replace(dest)
            return
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            tmp.unlink(missing_ok=True)
            last_err = e
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY)
    raise RuntimeError(f"Download failed after {_MAX_RETRIES} attempts: {last_err}")


def download_model(entry: ModelEntry, force: bool = False, show_progress: bool = True) -> Path:
    ensure_models_dir()
    filename = entry.url.split("/")[-1]
    dest = model_path(filename)

    if not force and dest.exists():
        return dest

    if show_progress:
        with Progress(
            TextColumn(f"[bold]{entry.name}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
        ) as progress:
            task = progress.add_task("Downloading...", total=None)

            def reporthook(block: int, block_size: int, total: int) -> None:
                if total > 0:
                    progress.update(task, total=total, completed=block * block_size)

            _fetch(entry.url, dest, reporthook=reporthook)
    else:
        from rich.console import Console

        Console().print(f"  Downloading [bold]{entry.name}[/bold]...")
        _fetch(entry.url, dest)

    return dest


def download_all(
    registry: ModelRegistry, force: bool = False
) -> tuple[dict[str, Path], dict[str, Exception]]:
    """Download all models. Returns (succeeded, failed) dicts."""
    from rich.console import Console

    succeeded: dict[str, Path] = {}
    failed: dict[str, Exception] = {}
    console = Console()
    for entry in registry.all():
        try:
            succeeded[entry.name] = download_model(entry, force=force)
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Failed [bold]{entry.name}[/bold]: {e}")
            failed[entry.name] = e
    return succeeded, failed


def model_status(registry: ModelRegistry) -> list[dict]:
    rows = []
    for entry in registry.all():
        filename = entry.url.split("/")[-1]
        path = model_path(filename)
        present = path.exists()
        size_mb = round(path.stat().st_size / 1_048_576, 1) if present else None
        rows.append(
            {
                "name": entry.name,
                "filename": filename,
                "present": present,
                "size_mb": size_mb,
                "description": entry.description,
            }
        )
    return rows
