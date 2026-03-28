"""
F2 — Ingest stage.

Extracts DJ tags (BPM, key, title, artist, album) from file tags via mutagen.

Rules:
- Every tag is stored with source="file_tag" and confidence=1.0.
- A missing tag is stored as null (None), never guessed or defaulted.
- Each track produces one artifact: metadata/<track_id>.json
- Artifacts are written atomically via .tmp + replace().
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mutagen

from mixprep.pipeline.schemas import TaggedValue, TrackIndex, TrackMetadata

log = logging.getLogger(__name__)

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
            s = str(val).strip()
            if s:
                return s
    return None


def _parse_bpm(raw: str | None) -> float | None:
    """Parse BPM string to float; return None if unparseable."""
    if raw is None:
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


def ingest_track(track: TrackIndex) -> TrackMetadata:
    """
    Extract DJ tags from file tags for *track*.

    Returns TrackMetadata with null fields for any absent tag.
    Never raises — if mutagen fails entirely, all tag fields are null.
    """
    tags: object = None
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mf = mutagen.File(track.file_path)
        if mf is not None:
            tags = mf.tags
    except Exception as exc:
        log.warning("mutagen failed for %s: %s", track.file_path, exc)

    def tagged(value: float | str | None) -> TaggedValue | None:
        return TaggedValue(value=value, source=_FILE_TAG_SOURCE, confidence=_FILE_TAG_CONFIDENCE)

    bpm_raw = _get_tag(tags, "TBPM", "tbpm", "BPM", "bpm", "fBPM")
    bpm_val = _parse_bpm(bpm_raw)
    bpm = tagged(bpm_val) if bpm_val is not None else None

    key_raw = _get_tag(tags, "TKEY", "tkey", "KEY", "key", "initialkey", "INITIALKEY")
    key = tagged(key_raw) if key_raw is not None else None

    title = _get_tag(tags, "TIT2", "tit2", "\xa9nam", "TITLE", "title")
    artist = _get_tag(tags, "TPE1", "tpe1", "\xa9ART", "ARTIST", "artist")
    album = _get_tag(tags, "TALB", "talb", "\xa9alb", "ALBUM", "album")

    return TrackMetadata(
        track_id=track.track_id,
        bpm=bpm,
        key=key,
        title=title,
        artist=artist,
        album=album,
    )


def write_metadata(meta: TrackMetadata, dest_dir: Path) -> None:
    """Write metadata/<track_id>.json atomically."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{meta.track_id}.json"
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(dest)


def load_metadata(path: Path) -> TrackMetadata:
    """Load and validate a metadata/<track_id>.json file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return TrackMetadata.model_validate(raw)
