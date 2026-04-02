"""
F1 — Scan stage.

Walks a directory, identifies audio files, assigns stable KSUIDs keyed by
absolute file path, and reads both audio metadata (duration, format,
sample_rate) and DJ tags (BPM, key, title, artist, album) via a single
mutagen pass.

Per-file storage:
    Each track is stored as tracks/<track_id>.json. Existing entries are
    loaded by path, so track_id is stable across rescans as long as the
    file path does not change.

Identity rule:
    track_id is tied to the resolved absolute file_path. Moving a file
    outside of mixprep creates a new track_id on next scan (same as
    rekordbox/Lightroom behaviour — the user must relocate).

Atomic write:
    Each track file is written via .tmp + replace() to avoid partial files.

Orphan pruning:
    Tracks whose file_path no longer exists on disk can be removed by
    calling prune_orphans(tracks_dir). This is opt-in (--prune flag in CLI).
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import mutagen
from cyksuid.v2 import ksuid

from mixprep.pipeline.schemas import TaggedValue, TrackIndex

log = logging.getLogger(__name__)

AUDIO_EXTENSIONS = frozenset({".mp3", ".flac", ".wav", ".aiff", ".aif", ".m4a", ".ogg", ".mp4"})

_FILE_TAG_SOURCE = "file_tag"
_FILE_TAG_CONFIDENCE = 1.0


def _get_tag(tags: object, *keys: str) -> str | None:
    """Return the first non-empty string value found among *keys* in *tags*."""
    for key in keys:
        try:
            val = tags[key]  # type: ignore[index]
        except (KeyError, TypeError):
            continue
        if isinstance(val, list):
            val = val[0] if val else None
        if val is not None:
            if isinstance(val, bytes):
                val = val.decode("utf-8", errors="replace")
            s = str(val).strip()
            if s:
                return s
    return None


def _parse_bpm(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


def _tagged(value: float | str | None) -> TaggedValue:
    return TaggedValue(value=value, source=_FILE_TAG_SOURCE, confidence=_FILE_TAG_CONFIDENCE)


def _read_track_info(
    path: Path,
) -> tuple[
    Optional[float],
    Optional[str],
    Optional[int],
    Optional[TaggedValue],
    Optional[TaggedValue],
    Optional[str],
    Optional[str],
    Optional[str],
]:
    """
    Open *path* once via mutagen and return:
        (duration, format, sample_rate, bpm, key, title, artist, album)

    Raises _MutagenParseError if mutagen cannot open the file at all.
    Individual missing fields are returned as None.
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

    fmt: Optional[str] = None
    cls_name = type(info).__name__
    if cls_name:
        fmt = cls_name.lower()

    tags = getattr(info, "tags", None)

    bpm_raw = _get_tag(
        tags,
        "TBPM",
        "tbpm",
        "BPM",
        "bpm",
        "fBPM",
        "tmpo",
        "----:com.apple.iTunes:BPM",
    )
    bpm_val = _parse_bpm(bpm_raw)
    bpm = _tagged(bpm_val) if bpm_val is not None else None

    key_raw = _get_tag(
        tags,
        "TKEY",
        "tkey",
        "KEY",
        "key",
        "initialkey",
        "INITIALKEY",
        "----:com.apple.iTunes:initialkey",
        "----:com.apple.iTunes:INITIALKEY",
    )
    key = _tagged(key_raw) if key_raw is not None else None

    title = _get_tag(tags, "TIT2", "tit2", "\xa9nam", "TITLE", "title")
    artist = _get_tag(tags, "TPE1", "tpe1", "\xa9ART", "ARTIST", "artist")
    album = _get_tag(tags, "TALB", "talb", "\xa9alb", "ALBUM", "album")

    return duration, fmt, sample_rate, bpm, key, title, artist, album


class _MutagenParseError(Exception):
    """Raised when mutagen cannot open or parse a file."""


def collect_audio_files(directory: Path) -> list[Path]:
    """Return sorted list of audio files under *directory*. Symlinks are not followed."""
    return sorted(
        p
        for p in directory.rglob("*")
        if not p.is_symlink() and p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def load_tracks_dir(tracks_dir: Path) -> list[TrackIndex]:
    """Load all track entries from *tracks_dir*/<track_id>.json files."""
    entries: list[TrackIndex] = []
    if not tracks_dir.is_dir():
        return entries
    for f in tracks_dir.glob("*.json"):
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
            entries.append(TrackIndex.model_validate(raw))
        except Exception as exc:
            log.warning("Skipping corrupt track file %s: %s", f, exc)
    return entries


def write_track(entry: TrackIndex, tracks_dir: Path) -> None:
    """Write a single track entry atomically to tracks_dir/<track_id>.json."""
    tracks_dir.mkdir(parents=True, exist_ok=True)
    dest = tracks_dir / f"{entry.track_id}.json"
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(entry.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(dest)


def prune_orphans(tracks_dir: Path) -> int:
    """
    Remove track files whose file_path no longer exists on disk.
    Returns the number of files removed.
    """
    removed = 0
    if not tracks_dir.is_dir():
        return removed
    for f in tracks_dir.glob("*.json"):
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
            file_path = raw.get("file_path", "")
            if file_path and not Path(file_path).exists():
                f.unlink()
                removed += 1
                log.info("Pruned orphan: %s (file gone: %s)", f.name, file_path)
        except Exception as exc:
            log.warning("Could not check %s: %s", f, exc)
    return removed


def scan_library(
    directory: Path,
    tracks_dir: Path,
    files: list[Path] | None = None,
    progress_callback: Callable[[Path], None] | None = None,
) -> list[TrackIndex]:
    """
    Scan *directory* recursively for audio files and write per-file track entries.

    tracks_dir — destination directory (libraries/<name>/tracks/).
    Existing entries are loaded first; files whose resolved path already has a
    track_id reuse that ID.

    files — optional pre-collected file list (from collect_audio_files).
    progress_callback — optional callable(path) called after each file is processed.

    Files that mutagen cannot open at all are skipped with a warning.
    Files mutagen opens but with missing metadata fields retain null values.
    """
    existing = load_tracks_dir(tracks_dir)
    path_to_id: dict[str, str] = {e.file_path: e.track_id for e in existing}

    _raw_paths = files if files is not None else collect_audio_files(directory)
    seen: set[Path] = set()
    paths: list[Path] = []
    for p in _raw_paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            paths.append(rp)

    results: list[TrackIndex] = []

    for path in paths:
        try:
            duration, fmt, sample_rate, bpm, key, title, artist, album = _read_track_info(path)
        except _MutagenParseError as exc:
            log.warning("Skipping %s: %s", path, exc)
            if progress_callback is not None:
                progress_callback(path)
            continue

        path_str = str(path)
        track_id = path_to_id.get(path_str) or str(ksuid())

        entry = TrackIndex(
            track_id=track_id,
            file_path=path_str,
            duration=duration,
            format=fmt,
            sample_rate=sample_rate,
            bpm=bpm,
            key=key,
            title=title,
            artist=artist,
            album=album,
        )
        write_track(entry, tracks_dir)
        results.append(entry)

        if progress_callback is not None:
            progress_callback(path)

    return results
