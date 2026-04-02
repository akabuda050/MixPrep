"""
F2 Tag extraction — behavior tests.

Tag extraction is now part of the scan stage (_read_track_info in scan.py).

Covers:
- BPM and key extracted correctly from file tags
- Missing tags produce null fields
- source is always "file_tag", confidence always 1.0
- mutagen failure → all tag fields null, no crash (handled by scan_library)
- Unicode characters in title/artist preserved
- write_track / load_tracks_dir round-trip with tag fields
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from mixprep.pipeline.scan import _read_track_info, load_tracks_dir, write_track
from mixprep.pipeline.schemas import TaggedValue, TrackIndex


def _mock_mutagen_file(tags: dict, duration: float = 300.0, sample_rate: int = 44100) -> MagicMock:
    mf = MagicMock()
    mf.info.length = duration
    mf.info.sample_rate = sample_rate
    mf.tags = tags
    return mf


def _make_track(**kwargs) -> TrackIndex:
    defaults = dict(track_id="tid1", file_path="/music/track.mp3")
    defaults.update(kwargs)
    return TrackIndex(**defaults)


# ---------------------------------------------------------------------------
# _read_track_info tag extraction
# ---------------------------------------------------------------------------


def test_bpm_and_key_extracted(tmp_path):
    path = tmp_path / "track.mp3"
    path.write_bytes(b"data")

    tags = {"TBPM": ["138.0"], "TKEY": ["Am"]}
    with patch("mutagen.File", return_value=_mock_mutagen_file(tags)):
        _, _, _, bpm, key, title, artist, album = _read_track_info(path)

    assert bpm is not None
    assert bpm.value == pytest.approx(138.0)
    assert bpm.source == "file_tag"
    assert bpm.confidence == pytest.approx(1.0)
    assert key is not None
    assert key.value == "Am"
    assert key.source == "file_tag"
    assert key.confidence == pytest.approx(1.0)


def test_missing_bpm_is_null(tmp_path):
    path = tmp_path / "track.mp3"
    path.write_bytes(b"data")

    with patch("mutagen.File", return_value=_mock_mutagen_file({"TKEY": ["8A"]})):
        _, _, _, bpm, key, *_ = _read_track_info(path)

    assert bpm is None
    assert key is not None


def test_missing_key_is_null(tmp_path):
    path = tmp_path / "track.mp3"
    path.write_bytes(b"data")

    with patch("mutagen.File", return_value=_mock_mutagen_file({"TBPM": ["140"]})):
        _, _, _, bpm, key, *_ = _read_track_info(path)

    assert key is None
    assert bpm is not None


def test_all_tags_missing(tmp_path):
    path = tmp_path / "track.mp3"
    path.write_bytes(b"data")

    with patch("mutagen.File", return_value=_mock_mutagen_file({})):
        _, _, _, bpm, key, title, artist, album = _read_track_info(path)

    assert bpm is None
    assert key is None
    assert title is None
    assert artist is None
    assert album is None


def test_title_artist_album_extracted(tmp_path):
    path = tmp_path / "track.mp3"
    path.write_bytes(b"data")

    tags = {"TIT2": ["My Track"], "TPE1": ["DJ Artist"], "TALB": ["The Album"]}
    with patch("mutagen.File", return_value=_mock_mutagen_file(tags)):
        _, _, _, bpm, key, title, artist, album = _read_track_info(path)

    assert title == "My Track"
    assert artist == "DJ Artist"
    assert album == "The Album"


def test_bpm_unparseable_stored_as_null(tmp_path):
    path = tmp_path / "track.mp3"
    path.write_bytes(b"data")

    with patch("mutagen.File", return_value=_mock_mutagen_file({"TBPM": ["not-a-number"]})):
        _, _, _, bpm, *_ = _read_track_info(path)

    assert bpm is None


def test_bpm_integer_string(tmp_path):
    path = tmp_path / "track.mp3"
    path.write_bytes(b"data")

    with patch("mutagen.File", return_value=_mock_mutagen_file({"TBPM": ["145"]})):
        _, _, _, bpm, *_ = _read_track_info(path)

    assert bpm is not None
    assert bpm.value == pytest.approx(145.0)


def test_unicode_title_artist(tmp_path):
    path = tmp_path / "track.mp3"
    path.write_bytes(b"data")

    tags = {"TIT2": ["рандом"], "TPE1": ["Я подарю"]}
    with patch("mutagen.File", return_value=_mock_mutagen_file(tags)):
        _, _, _, bpm, key, title, artist, album = _read_track_info(path)

    assert title == "рандом"
    assert artist == "Я подарю"


# ---------------------------------------------------------------------------
# write_track / load_tracks_dir with tag fields
# ---------------------------------------------------------------------------


def test_write_track_preserves_tags(tmp_path):
    tracks_dir = tmp_path / "tracks"
    entry = TrackIndex(
        track_id="tid1",
        file_path="/music/track.mp3",
        duration=300.0,
        format="mp3",
        sample_rate=44100,
        bpm=TaggedValue(value=138.0, source="file_tag", confidence=1.0),
        key=TaggedValue(value="Am", source="file_tag", confidence=1.0),
        title="Track",
        artist="Artist",
        album=None,
    )
    write_track(entry, tracks_dir)

    dest = tracks_dir / "tid1.json"
    data = json.loads(dest.read_text())
    assert data["bpm"]["value"] == pytest.approx(138.0)
    assert data["key"]["value"] == "Am"
    assert data["title"] == "Track"
    assert data["album"] is None


def test_load_tracks_dir_restores_tags(tmp_path):
    tracks_dir = tmp_path / "tracks"
    entry = TrackIndex(
        track_id="rt1",
        file_path="/music/track.mp3",
        bpm=TaggedValue(value=132.0, source="file_tag", confidence=1.0),
        key=None,
        title="Hello",
    )
    write_track(entry, tracks_dir)
    loaded = load_tracks_dir(tracks_dir)

    assert len(loaded) == 1
    assert loaded[0].bpm is not None
    assert loaded[0].bpm.value == pytest.approx(132.0)
    assert loaded[0].title == "Hello"
    assert loaded[0].key is None


def test_write_track_unicode_preserved(tmp_path):
    tracks_dir = tmp_path / "tracks"
    entry = TrackIndex(
        track_id="uni1",
        file_path="/music/track.mp3",
        title="рандом",
        artist="Я подарю",
    )
    write_track(entry, tracks_dir)
    text = (tracks_dir / "uni1.json").read_text(encoding="utf-8")
    assert "рандом" in text
    assert "Я подарю" in text
    assert "\\u" not in text


def test_write_track_idempotency(tmp_path):
    tracks_dir = tmp_path / "tracks"
    entry = TrackIndex(track_id="idem1", file_path="/music/track.mp3", title="Stable")
    write_track(entry, tracks_dir)
    first = (tracks_dir / "idem1.json").read_text()
    write_track(entry, tracks_dir)
    second = (tracks_dir / "idem1.json").read_text()
    assert first == second
