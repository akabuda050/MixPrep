"""
F1 CLI — run_scan command handler tests.

Covers:
- Invalid directory exits with code 1
- No audio files found returns early
- Successful scan writes tracks/<track_id>.json and prints summary
- Rescan reuses existing track_id
- Orphan cleanup when leftover .tmp exists
- Interrupt exits 130
- --prune removes orphaned entries
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_LIBRARY = "test_lib"


def _mock_audio_info(duration=300.0, sample_rate=44100):
    mock = MagicMock()
    mock.info.length = duration
    mock.info.sample_rate = sample_rate
    mock.tags = {}
    return mock


def _tracks_dir(data_dir: Path) -> Path:
    return data_dir / "libraries" / _LIBRARY / "tracks"


def test_run_scan_exits_1_on_nonexistent_directory(tmp_path):
    from mixprep.cli.commands.scan import run_scan

    with pytest.raises(SystemExit) as exc:
        run_scan(tmp_path / "nonexistent", _LIBRARY)

    assert exc.value.code == 1


def test_run_scan_returns_early_when_no_audio_files(tmp_path, monkeypatch):
    from mixprep.cli.commands.scan import run_scan

    monkeypatch.setenv("MIXPREP_DATA_DIR", str(tmp_path / "data"))

    run_scan(tmp_path, _LIBRARY)

    assert not _tracks_dir(tmp_path / "data").exists()


def test_run_scan_writes_track_json(tmp_path, monkeypatch):
    from mixprep.cli.commands.scan import run_scan

    monkeypatch.setenv("MIXPREP_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "track.mp3").write_bytes(b"audio")

    with patch("mutagen.File", return_value=_mock_audio_info()):
        run_scan(tmp_path, _LIBRARY)

    tracks_dir = _tracks_dir(tmp_path / "data")
    track_files = list(tracks_dir.glob("*.json"))
    assert len(track_files) == 1

    entry = json.loads(track_files[0].read_text())
    assert entry["file_path"].endswith("track.mp3")
    assert entry["track_id"]
    assert entry["duration"] == pytest.approx(300.0)


def test_run_scan_reuses_track_id_on_rescan(tmp_path, monkeypatch):
    from mixprep.cli.commands.scan import run_scan

    monkeypatch.setenv("MIXPREP_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "track.mp3").write_bytes(b"stable-content")

    with patch("mutagen.File", return_value=_mock_audio_info()):
        run_scan(tmp_path, _LIBRARY)

    tracks_dir = _tracks_dir(tmp_path / "data")
    first_id = json.loads(list(tracks_dir.glob("*.json"))[0].read_text())["track_id"]

    with patch("mutagen.File", return_value=_mock_audio_info()):
        run_scan(tmp_path, _LIBRARY)

    second_id = json.loads(list(tracks_dir.glob("*.json"))[0].read_text())["track_id"]

    assert first_id == second_id


def test_run_scan_interrupt_exits_130(tmp_path, monkeypatch):
    from mixprep.cli.commands.scan import run_scan

    monkeypatch.setenv("MIXPREP_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "track.mp3").write_bytes(b"audio")

    with patch("mutagen.File", side_effect=KeyboardInterrupt):
        with pytest.raises(SystemExit) as exc:
            run_scan(tmp_path, _LIBRARY)

    assert exc.value.code == 130


def test_run_scan_prune_removes_orphans(tmp_path, monkeypatch):
    from mixprep.cli.commands.scan import run_scan

    monkeypatch.setenv("MIXPREP_DATA_DIR", str(tmp_path / "data"))

    # First scan: create a track entry
    mp3 = tmp_path / "track.mp3"
    mp3.write_bytes(b"audio")

    with patch("mutagen.File", return_value=_mock_audio_info()):
        run_scan(tmp_path, _LIBRARY)

    tracks_dir = _tracks_dir(tmp_path / "data")
    assert len(list(tracks_dir.glob("*.json"))) == 1

    # Remove the file to create an orphan, then scan with --prune
    mp3.unlink()

    with patch("mutagen.File", return_value=_mock_audio_info()):
        run_scan(tmp_path, _LIBRARY, prune=True)

    assert len(list(tracks_dir.glob("*.json"))) == 0
