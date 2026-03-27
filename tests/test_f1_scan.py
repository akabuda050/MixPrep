"""
F1 Test — scan behavior tests.

Covers:
- Happy path: correct entries for valid audio files
- Stable IDs: same file_hash reuses same track_id on rescan
- New ID on changed content
- Unreadable file: skipped with warning, not in output
- Non-audio files: ignored
- Null fields: duration/sample_rate null when mutagen can't provide them
- Atomic write: scan.json written via tmp
- Idempotency: two identical scans produce identical output
- write_scan / load_scan round-trip
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mixprep.pipeline.scan import (
    AUDIO_EXTENSIONS,
    _MutagenParseError,
    _read_audio_info,
    load_scan,
    scan_library,
    write_scan,
)
from mixprep.pipeline.schemas import TrackIndex

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_fake_mp3(path: Path, content: bytes = b"fake-mp3-content") -> Path:
    """Write a file with .mp3 extension and given content."""
    path.write_bytes(content)
    return path


def _make_track_index(**kwargs) -> TrackIndex:
    defaults = dict(
        track_id="abc",
        file_path="/music/t.mp3",
        file_hash="deadbeef",
        duration=300.0,
        format="mp3",
        sample_rate=44100,
    )
    defaults.update(kwargs)
    return TrackIndex(**defaults)


# ---------------------------------------------------------------------------
# _read_audio_info unit tests
# ---------------------------------------------------------------------------


def test_read_audio_info_returns_values_from_mutagen(tmp_path):
    path = tmp_path / "track.mp3"
    path.write_bytes(b"data")

    mock_info = MagicMock()
    mock_info.info.length = 300.5
    mock_info.info.sample_rate = 44100

    with patch("mutagen.File", return_value=mock_info):
        duration, fmt, sample_rate = _read_audio_info(path)

    assert duration == pytest.approx(300.5)
    assert sample_rate == 44100
    assert fmt is not None  # derived from class name


def test_read_audio_info_returns_null_when_info_missing(tmp_path):
    path = tmp_path / "track.mp3"
    path.write_bytes(b"data")

    mock_file = MagicMock()
    del mock_file.info  # simulate no .info attribute

    with patch("mutagen.File", return_value=mock_file):
        duration, fmt, sample_rate = _read_audio_info(path)

    assert duration is None
    assert sample_rate is None


def test_read_audio_info_raises_on_none_result(tmp_path):
    path = tmp_path / "track.mp3"
    path.write_bytes(b"data")

    with patch("mutagen.File", return_value=None):
        with pytest.raises(_MutagenParseError, match="unsupported"):
            _read_audio_info(path)


def test_read_audio_info_raises_on_mutagen_exception(tmp_path):
    path = tmp_path / "track.mp3"
    path.write_bytes(b"data")

    with patch("mutagen.File", side_effect=Exception("corrupt")):
        with pytest.raises(_MutagenParseError, match="corrupt"):
            _read_audio_info(path)


# ---------------------------------------------------------------------------
# scan_library behavior tests
# ---------------------------------------------------------------------------


def _mock_audio_info(duration=300.0, sample_rate=44100):
    """Return a patch target and mock that simulates a valid mutagen file."""
    mock = MagicMock()
    mock.info.length = duration
    mock.info.sample_rate = sample_rate
    return mock


def test_scan_library_finds_audio_files(tmp_path):
    mp3 = _make_fake_mp3(tmp_path / "a.mp3")
    (tmp_path / "b.txt").write_text("not audio")

    mock = _mock_audio_info()
    with patch("mutagen.File", return_value=mock):
        results = scan_library(tmp_path, [])

    assert len(results) == 1
    assert results[0].file_path == str(mp3)


def test_scan_library_ignores_non_audio_extensions(tmp_path):
    for name in ["a.pdf", "b.doc", "c.jpg", "d.avi"]:
        (tmp_path / name).write_bytes(b"data")

    with patch("mutagen.File", return_value=_mock_audio_info()):
        results = scan_library(tmp_path, [])

    assert results == []


def test_scan_library_all_supported_extensions(tmp_path):
    for ext in AUDIO_EXTENSIONS:
        (tmp_path / f"track{ext}").write_bytes(b"x")

    with patch("mutagen.File", return_value=_mock_audio_info()):
        results = scan_library(tmp_path, [])

    assert len(results) == len(AUDIO_EXTENSIONS)


def test_scan_library_stable_id_on_rescan(tmp_path):
    _make_fake_mp3(tmp_path / "track.mp3", content=b"same-content")

    with patch("mutagen.File", return_value=_mock_audio_info()):
        first = scan_library(tmp_path, [])

    assert len(first) == 1
    first_id = first[0].track_id

    with patch("mutagen.File", return_value=_mock_audio_info()):
        second = scan_library(tmp_path, first)

    assert len(second) == 1
    assert second[0].track_id == first_id


def test_scan_library_new_id_when_content_changes(tmp_path):
    f = tmp_path / "track.mp3"
    f.write_bytes(b"original")

    with patch("mutagen.File", return_value=_mock_audio_info()):
        first = scan_library(tmp_path, [])

    first_id = first[0].track_id

    f.write_bytes(b"changed-content")

    with patch("mutagen.File", return_value=_mock_audio_info()):
        second = scan_library(tmp_path, first)

    assert len(second) == 1
    assert second[0].track_id != first_id


def test_scan_library_skips_unreadable_file(tmp_path, caplog):
    import logging

    _make_fake_mp3(tmp_path / "bad.mp3")
    _make_fake_mp3(tmp_path / "good.mp3")

    def side_effect(path, *args, **kwargs):
        if "bad" in str(path):
            raise Exception("corrupt")
        return _mock_audio_info()

    with patch("mutagen.File", side_effect=side_effect):
        with caplog.at_level(logging.WARNING):
            results = scan_library(tmp_path, [])

    assert len(results) == 1
    assert "good" in results[0].file_path
    assert any("bad.mp3" in r.message for r in caplog.records)


def test_scan_library_null_duration_when_info_absent(tmp_path):
    _make_fake_mp3(tmp_path / "track.mp3")

    mock = MagicMock()
    mock.info.length = None
    mock.info.sample_rate = None

    with patch("mutagen.File", return_value=mock):
        results = scan_library(tmp_path, [])

    assert results[0].duration is None
    assert results[0].sample_rate is None


def test_scan_library_recurses_subdirectories(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    _make_fake_mp3(sub / "deep.mp3")

    with patch("mutagen.File", return_value=_mock_audio_info()):
        results = scan_library(tmp_path, [])

    assert len(results) == 1
    assert "deep.mp3" in results[0].file_path


# ---------------------------------------------------------------------------
# write_scan / load_scan / idempotency
# ---------------------------------------------------------------------------


def test_write_scan_creates_valid_json(tmp_path):
    dest = tmp_path / "scan.json"
    entries = [_make_track_index()]

    write_scan(entries, dest)

    assert dest.exists()
    data = json.loads(dest.read_text())
    assert isinstance(data, list)
    assert data[0]["track_id"] == "abc"


def test_write_scan_no_tmp_left_behind(tmp_path):
    dest = tmp_path / "scan.json"
    write_scan([_make_track_index()], dest)
    assert not (tmp_path / "scan.tmp").exists()


def test_write_scan_load_scan_roundtrip(tmp_path):
    dest = tmp_path / "scan.json"
    original = [_make_track_index(track_id="x1"), _make_track_index(track_id="x2")]

    write_scan(original, dest)
    loaded = load_scan(dest)

    assert len(loaded) == 2
    assert loaded[0].track_id == "x1"
    assert loaded[1].track_id == "x2"


def test_scan_idempotency(tmp_path):
    """Two identical scans of unchanged files produce identical output."""
    _make_fake_mp3(tmp_path / "t.mp3", content=b"stable")

    with patch("mutagen.File", return_value=_mock_audio_info()):
        first = scan_library(tmp_path, [])

    with patch("mutagen.File", return_value=_mock_audio_info()):
        second = scan_library(tmp_path, first)

    assert len(first) == len(second) == 1
    assert first[0].track_id == second[0].track_id
    assert first[0].file_hash == second[0].file_hash
    assert first[0].duration == second[0].duration


def test_scan_library_progress_callback_called_for_each_file(tmp_path):
    for i in range(3):
        _make_fake_mp3(tmp_path / f"t{i}.mp3", content=f"data{i}".encode())

    called = []

    with patch("mutagen.File", return_value=_mock_audio_info()):
        scan_library(tmp_path, [], progress_callback=called.append)

    assert len(called) == 3


def test_scan_library_progress_callback_called_on_skipped_file(tmp_path):
    _make_fake_mp3(tmp_path / "bad.mp3")

    called = []

    with patch("mutagen.File", side_effect=Exception("corrupt")):
        scan_library(tmp_path, [], progress_callback=called.append)

    assert len(called) == 1


def test_write_scan_idempotency(tmp_path):
    """Writing the same entries twice produces byte-identical files."""
    dest = tmp_path / "scan.json"
    entries = [_make_track_index()]

    write_scan(entries, dest)
    content_first = dest.read_text()

    write_scan(entries, dest)
    content_second = dest.read_text()

    assert content_first == content_second
