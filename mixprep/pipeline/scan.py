"""
F1 — Scan stage.

Walks a directory, identifies audio files, assigns stable KSUIDs keyed by
file content hash (not path), and reads duration/sample_rate via mutagen.

Identity rule:
    track_id is tied to file_hash. If an existing scan already contains a
    matching file_hash, the same track_id is reused regardless of path.

Atomic write:
    scan.json is written via .tmp + replace() to avoid partial files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import mutagen
from cyksuid.v2 import ksuid

from mixprep.pipeline.schemas import TrackIndex

log = logging.getLogger(__name__)

AUDIO_EXTENSIONS = frozenset({".mp3", ".flac", ".wav", ".aiff", ".aif", ".m4a", ".ogg", ".mp4"})


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_audio_info(
    path: Path,
) -> tuple[Optional[float], Optional[str], Optional[int]]:
    """
    Return (duration, format_name, sample_rate) for the given audio file.

    Returns (None, None, None) if mutagen fails to open the file — the caller
    is responsible for deciding whether to skip the file or keep it with nulls.

    A successful open with a missing individual field returns None for that
    field only; the file is not skipped.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = mutagen.File(path)
    except Exception as exc:
        raise _MutagenParseError(str(exc)) from exc

    if info is None:
        raise _MutagenParseError("mutagen returned None (unsupported format)")

    audio_info = getattr(info, "info", None)

    duration: Optional[float] = None
    sample_rate: Optional[int] = None

    if audio_info is not None:
        raw_dur = getattr(audio_info, "length", None)
        if raw_dur is not None:
            duration = float(raw_dur)

        raw_sr = getattr(audio_info, "sample_rate", None)
        if raw_sr is not None:
            sample_rate = int(raw_sr)

    # Derive format from the mutagen class name, e.g. "MP3" from "mutagen.mp3.MP3"
    fmt: Optional[str] = None
    cls_name = type(info).__name__
    if cls_name:
        fmt = cls_name.lower()

    return duration, fmt, sample_rate


class _MutagenParseError(Exception):
    """Raised when mutagen cannot open or parse a file."""


def collect_audio_files(directory: Path) -> list[Path]:
    """Return sorted list of audio files under *directory*. Symlinks are not followed."""
    return sorted(
        p
        for p in directory.rglob("*")
        if not p.is_symlink() and p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def scan_library(
    directory: Path,
    existing: list[TrackIndex],
    files: list[Path] | None = None,
    progress_callback: Callable[[Path], None] | None = None,
) -> list[TrackIndex]:
    """
    Scan *directory* recursively for audio files.

    existing — previously scanned entries (from scan.json). Files whose
    file_hash matches an existing entry reuse the same track_id.

    files — optional pre-collected file list (from collect_audio_files).
    progress_callback — optional callable(path) called after each file is
        processed (used by CLI to advance a progress bar).

    Files that mutagen cannot open at all are skipped with a warning.
    Files mutagen opens but with missing metadata fields retain null values.
    """
    hash_to_id: dict[str, str] = {e.file_hash: e.track_id for e in existing}
    _raw_paths = files if files is not None else collect_audio_files(directory)
    # Deduplicate: keep first occurrence of each resolved path
    seen_resolved: set[Path] = set()
    paths: list[Path] = []
    for p in _raw_paths:
        rp = p.resolve()
        if rp not in seen_resolved:
            seen_resolved.add(rp)
            paths.append(p)
    results: list[TrackIndex] = []

    for path in paths:
        file_hash = _md5(path)

        try:
            duration, fmt, sample_rate = _read_audio_info(path)
        except _MutagenParseError as exc:
            log.warning("Skipping %s: %s", path, exc)
            if progress_callback is not None:
                progress_callback(path)
            continue

        track_id = hash_to_id.get(file_hash) or str(ksuid())
        hash_to_id[file_hash] = track_id

        entry = TrackIndex(
            track_id=track_id,
            file_path=str(path),
            file_hash=file_hash,
            duration=duration,
            format=fmt,
            sample_rate=sample_rate,
        )
        results.append(entry)

        if progress_callback is not None:
            progress_callback(path)

    return results


def write_scan(entries: list[TrackIndex], dest: Path) -> None:
    """Write scan.json atomically."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    data = [e.model_dump() for e in entries]
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(dest)


def load_scan(path: Path) -> list[TrackIndex]:
    """Load and validate an existing scan.json."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [TrackIndex.model_validate(r) for r in raw]
