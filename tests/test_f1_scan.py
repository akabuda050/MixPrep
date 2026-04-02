"""
F1 Test — scan behavior tests.

Covers:
- Happy path: correct entries for valid audio files
- Stable IDs: same file_path reuses same track_id on rescan
- Unreadable file: skipped with warning, not in output
- Non-audio files: ignored
- Null fields: duration/sample_rate null when mutagen can't provide them
- Per-file atomic write: tracks/<track_id>.json written via tmp
- Idempotency: two identical scans produce identical output
- write_track / load_tracks_dir round-trip
- prune_orphans removes entries whose file is gone
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mixprep.pipeline.scan import (
    AUDIO_EXTENSIONS,
    _MutagenParseError,
    _read_track_info,
    load_tracks_dir,
    prune_orphans,
    scan_library,
    write_track,
)
from mixprep.pipeline.schemas import TrackIndex

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_fake_mp3(path: Path, content: bytes = b"fake-mp3-content") -> Path:
    path.write_bytes(content)
    return path


def _make_track_index(**kwargs) -> TrackIndex:
    defaults = dict(
        track_id="abc",
        file_path="/music/t.mp3",
        duration=300.0,
        format="mp3",
        sample_rate=44100,
        bpm=None,
        key=None,
        title=None,
        artist=None,
        album=None,
    )
    defaults.update(kwargs)
    return TrackIndex(**defaults)


def _mock_audio_info(duration=300.0, sample_rate=44100):
    mock = MagicMock()
    mock.info.length = duration
    mock.info.sample_rate = sample_rate
    mock.tags = {}
    return mock


# ---------------------------------------------------------------------------
# _read_track_info unit tests
# ---------------------------------------------------------------------------


def test_read_track_info_returns_values_from_mutagen(tmp_path):
    path = tmp_path / "track.mp3"
    path.write_bytes(b"data")

    mock_info = MagicMock()
    mock_info.info.length = 300.5
    mock_info.info.sample_rate = 44100
    mock_info.tags = {"TBPM": ["138"], "TKEY": ["Am"], "TIT2": ["Title"], "TPE1": ["Artist"]}

    with patch("mutagen.File", return_value=mock_info):
        duration, fmt, sample_rate, bpm, key, title, artist, album = _read_track_info(path)

    assert duration == pytest.approx(300.5)
    assert sample_rate == 44100
    assert fmt is not None
    assert bpm is not None and bpm.value == pytest.approx(138.0)
    assert key is not None and key.value == "Am"
    assert title == "Title"
    assert artist == "Artist"


def test_read_track_info_returns_null_when_info_missing(tmp_path):
    path = tmp_path / "track.mp3"
    path.write_bytes(b"data")

    mock_file = MagicMock()
    del mock_file.info
    mock_file.tags = {}

    with patch("mutagen.File", return_value=mock_file):
        duration, fmt, sample_rate, bpm, key, title, artist, album = _read_track_info(path)

    assert duration is None
    assert sample_rate is None


def test_read_track_info_raises_on_none_result(tmp_path):
    path = tmp_path / "track.mp3"
    path.write_bytes(b"data")

    with patch("mutagen.File", return_value=None):
        with pytest.raises(_MutagenParseError, match="unsupported"):
            _read_track_info(path)


def test_read_track_info_raises_on_mutagen_exception(tmp_path):
    path = tmp_path / "track.mp3"
    path.write_bytes(b"data")

    with patch("mutagen.File", side_effect=Exception("corrupt")):
        with pytest.raises(_MutagenParseError, match="corrupt"):
            _read_track_info(path)


# ---------------------------------------------------------------------------
# scan_library behavior tests
# ---------------------------------------------------------------------------


def test_scan_library_finds_audio_files(tmp_path):
    mp3 = _make_fake_mp3(tmp_path / "a.mp3")
    (tmp_path / "b.txt").write_text("not audio")
    tracks_dir = tmp_path / "tracks"

    with patch("mutagen.File", return_value=_mock_audio_info()):
        results = scan_library(tmp_path, tracks_dir)

    assert len(results) == 1
    assert results[0].file_path == str(mp3.resolve())


def test_scan_library_ignores_non_audio_extensions(tmp_path):
    for name in ["a.pdf", "b.doc", "c.jpg", "d.avi"]:
        (tmp_path / name).write_bytes(b"data")
    tracks_dir = tmp_path / "tracks"

    with patch("mutagen.File", return_value=_mock_audio_info()):
        results = scan_library(tmp_path, tracks_dir)

    assert results == []


def test_scan_library_all_supported_extensions(tmp_path):
    for ext in AUDIO_EXTENSIONS:
        (tmp_path / f"track{ext}").write_bytes(b"x")
    tracks_dir = tmp_path / "tracks"

    with patch("mutagen.File", return_value=_mock_audio_info()):
        results = scan_library(tmp_path, tracks_dir)

    assert len(results) == len(AUDIO_EXTENSIONS)


def test_scan_library_stable_id_on_rescan(tmp_path):
    _make_fake_mp3(tmp_path / "track.mp3", content=b"same-content")
    tracks_dir = tmp_path / "tracks"

    with patch("mutagen.File", return_value=_mock_audio_info()):
        first = scan_library(tmp_path, tracks_dir)

    first_id = first[0].track_id

    with patch("mutagen.File", return_value=_mock_audio_info()):
        second = scan_library(tmp_path, tracks_dir)

    assert second[0].track_id == first_id


def test_scan_library_skips_unreadable_file(tmp_path, caplog):
    import logging

    _make_fake_mp3(tmp_path / "bad.mp3")
    _make_fake_mp3(tmp_path / "good.mp3")
    tracks_dir = tmp_path / "tracks"

    def side_effect(path, *args, **kwargs):
        if "bad" in str(path):
            raise Exception("corrupt")
        return _mock_audio_info()

    with patch("mutagen.File", side_effect=side_effect):
        with caplog.at_level(logging.WARNING):
            results = scan_library(tmp_path, tracks_dir)

    assert len(results) == 1
    assert "good" in results[0].file_path
    assert any("bad.mp3" in r.message for r in caplog.records)


def test_scan_library_null_duration_when_info_absent(tmp_path):
    _make_fake_mp3(tmp_path / "track.mp3")
    tracks_dir = tmp_path / "tracks"

    mock = MagicMock()
    mock.info.length = None
    mock.info.sample_rate = None
    mock.tags = {}

    with patch("mutagen.File", return_value=mock):
        results = scan_library(tmp_path, tracks_dir)

    assert results[0].duration is None
    assert results[0].sample_rate is None


def test_scan_library_recurses_subdirectories(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    _make_fake_mp3(sub / "deep.mp3")
    tracks_dir = tmp_path / "tracks"

    with patch("mutagen.File", return_value=_mock_audio_info()):
        results = scan_library(tmp_path, tracks_dir)

    assert len(results) == 1
    assert "deep.mp3" in results[0].file_path


def test_scan_idempotency(tmp_path):
    _make_fake_mp3(tmp_path / "t.mp3", content=b"stable")
    tracks_dir = tmp_path / "tracks"

    with patch("mutagen.File", return_value=_mock_audio_info()):
        first = scan_library(tmp_path, tracks_dir)

    with patch("mutagen.File", return_value=_mock_audio_info()):
        second = scan_library(tmp_path, tracks_dir)

    assert first[0].track_id == second[0].track_id
    assert first[0].file_path == second[0].file_path
    assert first[0].duration == second[0].duration


def test_scan_library_progress_callback_called_for_each_file(tmp_path):
    for i in range(3):
        _make_fake_mp3(tmp_path / f"t{i}.mp3", content=f"data{i}".encode())
    tracks_dir = tmp_path / "tracks"
    called = []

    with patch("mutagen.File", return_value=_mock_audio_info()):
        scan_library(tmp_path, tracks_dir, progress_callback=called.append)

    assert len(called) == 3


def test_scan_library_progress_callback_called_on_skipped_file(tmp_path):
    _make_fake_mp3(tmp_path / "bad.mp3")
    tracks_dir = tmp_path / "tracks"
    called = []

    with patch("mutagen.File", side_effect=Exception("corrupt")):
        scan_library(tmp_path, tracks_dir, progress_callback=called.append)

    assert len(called) == 1


# ---------------------------------------------------------------------------
# write_track / load_tracks_dir / round-trip
# ---------------------------------------------------------------------------


def test_write_track_creates_file(tmp_path):
    tracks_dir = tmp_path / "tracks"
    entry = _make_track_index(track_id="tid1")

    write_track(entry, tracks_dir)

    dest = tracks_dir / "tid1.json"
    assert dest.exists()
    data = json.loads(dest.read_text())
    assert data["track_id"] == "tid1"


def test_write_track_no_tmp_left_behind(tmp_path):
    tracks_dir = tmp_path / "tracks"
    write_track(_make_track_index(track_id="tid2"), tracks_dir)
    assert not (tracks_dir / "tid2.tmp").exists()


def test_write_track_load_tracks_dir_roundtrip(tmp_path):
    tracks_dir = tmp_path / "tracks"
    e1 = _make_track_index(track_id="x1", file_path="/music/a.mp3")
    e2 = _make_track_index(track_id="x2", file_path="/music/b.mp3")
    write_track(e1, tracks_dir)
    write_track(e2, tracks_dir)

    loaded = load_tracks_dir(tracks_dir)
    ids = {e.track_id for e in loaded}
    assert ids == {"x1", "x2"}


def test_load_tracks_dir_empty_when_dir_missing(tmp_path):
    assert load_tracks_dir(tmp_path / "nonexistent") == []


def test_load_tracks_dir_skips_corrupt_files(tmp_path):
    tracks_dir = tmp_path / "tracks"
    tracks_dir.mkdir()
    (tracks_dir / "corrupt.json").write_text("not valid json")
    write_track(_make_track_index(track_id="good1"), tracks_dir)

    loaded = load_tracks_dir(tracks_dir)
    assert len(loaded) == 1
    assert loaded[0].track_id == "good1"


def test_write_track_idempotency(tmp_path):
    tracks_dir = tmp_path / "tracks"
    entry = _make_track_index(track_id="idem1")

    write_track(entry, tracks_dir)
    first = (tracks_dir / "idem1.json").read_text()

    write_track(entry, tracks_dir)
    second = (tracks_dir / "idem1.json").read_text()

    assert first == second


# ---------------------------------------------------------------------------
# prune_orphans
# ---------------------------------------------------------------------------


def test_prune_orphans_removes_missing_file_entries(tmp_path):
    tracks_dir = tmp_path / "tracks"
    # Write entry pointing to a file that doesn't exist
    entry = _make_track_index(track_id="gone1", file_path=str(tmp_path / "missing.mp3"))
    write_track(entry, tracks_dir)

    removed = prune_orphans(tracks_dir)

    assert removed == 1
    assert not (tracks_dir / "gone1.json").exists()


def test_prune_orphans_keeps_existing_file_entries(tmp_path):
    tracks_dir = tmp_path / "tracks"
    real_file = tmp_path / "real.mp3"
    real_file.write_bytes(b"audio")
    entry = _make_track_index(track_id="keep1", file_path=str(real_file))
    write_track(entry, tracks_dir)

    removed = prune_orphans(tracks_dir)

    assert removed == 0
    assert (tracks_dir / "keep1.json").exists()


def test_prune_orphans_returns_zero_on_missing_dir(tmp_path):
    assert prune_orphans(tmp_path / "nonexistent") == 0
