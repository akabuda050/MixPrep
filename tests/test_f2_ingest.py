"""
F2 Ingest — behavior tests.

Covers:
- BPM and key mapped correctly from file tags
- Missing tags produce null fields (not defaulted)
- source is always "file_tag", confidence always 1.0
- mutagen failure → all tag fields null, no crash
- write_metadata / load_metadata round-trip
- Idempotency: writing same metadata twice produces identical files
- Unicode characters in title/artist preserved
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from mixprep.pipeline.ingest import ingest_track, load_metadata, write_metadata
from mixprep.pipeline.schemas import TrackIndex, TrackMetadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_track(**kwargs) -> TrackIndex:
    defaults = dict(
        track_id="tid1",
        file_path="/music/track.mp3",
        file_hash="deadbeef",
        duration=300.0,
        format="mp3",
        sample_rate=44100,
    )
    defaults.update(kwargs)
    return TrackIndex(**defaults)


def _mock_mutagen_file(tags: dict) -> MagicMock:
    mf = MagicMock()
    mf.tags = tags
    return mf


# ---------------------------------------------------------------------------
# ingest_track unit tests
# ---------------------------------------------------------------------------


def test_ingest_bpm_and_key_extracted():
    track = _make_track()
    tags = {"TBPM": ["138.0"], "TKEY": ["Am"]}
    with patch("mutagen.File", return_value=_mock_mutagen_file(tags)):
        meta = ingest_track(track)

    assert meta.bpm is not None
    assert meta.bpm.value == pytest.approx(138.0)
    assert meta.bpm.source == "file_tag"
    assert meta.bpm.confidence == pytest.approx(1.0)

    assert meta.key is not None
    assert meta.key.value == "Am"
    assert meta.key.source == "file_tag"
    assert meta.key.confidence == pytest.approx(1.0)


def test_ingest_missing_bpm_is_null():
    track = _make_track()
    tags = {"TKEY": ["8A"]}
    with patch("mutagen.File", return_value=_mock_mutagen_file(tags)):
        meta = ingest_track(track)

    assert meta.bpm is None


def test_ingest_missing_key_is_null():
    track = _make_track()
    tags = {"TBPM": ["140"]}
    with patch("mutagen.File", return_value=_mock_mutagen_file(tags)):
        meta = ingest_track(track)

    assert meta.key is None


def test_ingest_all_tags_missing():
    track = _make_track()
    with patch("mutagen.File", return_value=_mock_mutagen_file({})):
        meta = ingest_track(track)

    assert meta.bpm is None
    assert meta.key is None
    assert meta.title is None
    assert meta.artist is None
    assert meta.album is None


def test_ingest_title_artist_album():
    track = _make_track()
    tags = {
        "TIT2": ["My Track"],
        "TPE1": ["DJ Artist"],
        "TALB": ["The Album"],
    }
    with patch("mutagen.File", return_value=_mock_mutagen_file(tags)):
        meta = ingest_track(track)

    assert meta.title == "My Track"
    assert meta.artist == "DJ Artist"
    assert meta.album == "The Album"


def test_ingest_mutagen_returns_none_all_null():
    """mutagen.File returns None (unsupported format) → all tags null."""
    track = _make_track()
    with patch("mutagen.File", return_value=None):
        meta = ingest_track(track)

    assert meta.bpm is None
    assert meta.key is None
    assert meta.title is None


def test_ingest_mutagen_raises_all_null():
    """mutagen.File raises exception → all tags null, no crash."""
    track = _make_track()
    with patch("mutagen.File", side_effect=Exception("corrupt")):
        meta = ingest_track(track)

    assert meta.bpm is None
    assert meta.key is None


def test_ingest_mutagen_tags_none_all_null():
    """mutagen.File opens but tags is None → all tags null."""
    track = _make_track()
    mf = MagicMock()
    mf.tags = None
    with patch("mutagen.File", return_value=mf):
        meta = ingest_track(track)

    assert meta.bpm is None
    assert meta.key is None


def test_ingest_bpm_unparseable_stored_as_null():
    """BPM tag present but not a valid number → bpm is null."""
    track = _make_track()
    tags = {"TBPM": ["not-a-number"]}
    with patch("mutagen.File", return_value=_mock_mutagen_file(tags)):
        meta = ingest_track(track)

    assert meta.bpm is None


def test_ingest_bpm_integer_string():
    track = _make_track()
    tags = {"TBPM": ["145"]}
    with patch("mutagen.File", return_value=_mock_mutagen_file(tags)):
        meta = ingest_track(track)

    assert meta.bpm is not None
    assert meta.bpm.value == pytest.approx(145.0)


def test_ingest_unicode_title_artist():
    track = _make_track()
    tags = {"TIT2": ["рандом"], "TPE1": ["Я подарю"]}
    with patch("mutagen.File", return_value=_mock_mutagen_file(tags)):
        meta = ingest_track(track)

    assert meta.title == "рандом"
    assert meta.artist == "Я подарю"


def test_ingest_track_id_preserved():
    track = _make_track(track_id="custom-id-abc")
    with patch("mutagen.File", return_value=_mock_mutagen_file({})):
        meta = ingest_track(track)

    assert meta.track_id == "custom-id-abc"


# ---------------------------------------------------------------------------
# write_metadata / load_metadata
# ---------------------------------------------------------------------------


def test_write_metadata_creates_file(tmp_path):
    from mixprep.pipeline.schemas import TaggedValue

    meta = TrackMetadata(
        track_id="tid1",
        bpm=TaggedValue(value=138.0, source="file_tag", confidence=1.0),
        key=TaggedValue(value="Am", source="file_tag", confidence=1.0),
        title="Track",
        artist="Artist",
        album=None,
    )
    write_metadata(meta, tmp_path)

    dest = tmp_path / "tid1.json"
    assert dest.exists()
    data = json.loads(dest.read_text())
    assert data["track_id"] == "tid1"
    assert data["bpm"]["value"] == pytest.approx(138.0)
    assert data["key"]["value"] == "Am"
    assert data["album"] is None


def test_write_metadata_no_tmp_left_behind(tmp_path):
    meta = TrackMetadata(
        track_id="tid2",
        bpm=None,
        key=None,
        title=None,
        artist=None,
        album=None,
    )
    write_metadata(meta, tmp_path)
    assert not (tmp_path / "tid2.tmp").exists()


def test_write_load_roundtrip(tmp_path):
    from mixprep.pipeline.schemas import TaggedValue

    meta = TrackMetadata(
        track_id="rt1",
        bpm=TaggedValue(value=132.0, source="file_tag", confidence=1.0),
        key=None,
        title="Hello",
        artist=None,
        album=None,
    )
    write_metadata(meta, tmp_path)
    loaded = load_metadata(tmp_path / "rt1.json")

    assert loaded.track_id == "rt1"
    assert loaded.bpm is not None
    assert loaded.bpm.value == pytest.approx(132.0)
    assert loaded.title == "Hello"
    assert loaded.key is None


def test_write_metadata_idempotency(tmp_path):
    meta = TrackMetadata(
        track_id="idem1",
        bpm=None,
        key=None,
        title="Stable",
        artist=None,
        album=None,
    )
    write_metadata(meta, tmp_path)
    first = (tmp_path / "idem1.json").read_text()

    write_metadata(meta, tmp_path)
    second = (tmp_path / "idem1.json").read_text()

    assert first == second


def test_write_metadata_unicode_preserved(tmp_path):
    meta = TrackMetadata(
        track_id="uni1",
        bpm=None,
        key=None,
        title="рандом",
        artist="Я подарю",
        album=None,
    )
    write_metadata(meta, tmp_path)
    text = (tmp_path / "uni1.json").read_text(encoding="utf-8")
    assert "рандом" in text
    assert "Я подарю" in text
    assert "\\u" not in text
