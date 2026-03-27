"""
F1 CLI — run_scan command handler tests.

Covers:
- Invalid directory exits with code 1
- No audio files found returns early
- Successful scan writes scan.json and prints summary
- Corrupt existing scan.json prints warning and continues
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


def _mock_audio_info(duration=300.0, sample_rate=44100):
    mock = MagicMock()
    mock.info.length = duration
    mock.info.sample_rate = sample_rate
    return mock


def test_run_scan_exits_1_on_nonexistent_directory(tmp_path):
    from mixprep.cli.commands.scan import run_scan

    with pytest.raises(SystemExit) as exc:
        run_scan(tmp_path / "nonexistent")

    assert exc.value.code == 1


def test_run_scan_returns_early_when_no_audio_files(tmp_path, monkeypatch):
    from mixprep.cli.commands.scan import run_scan

    monkeypatch.setenv("MIXPREP_DATA_DIR", str(tmp_path / "data"))

    run_scan(tmp_path)  # empty dir — should not raise or write anything

    assert not (tmp_path / "data" / "scan.json").exists()


def test_run_scan_writes_scan_json(tmp_path, monkeypatch):
    from mixprep.cli.commands.scan import run_scan

    data_dir = tmp_path / "data"
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))

    (tmp_path / "track.mp3").write_bytes(b"audio")

    with patch("mutagen.File", return_value=_mock_audio_info()):
        run_scan(tmp_path)

    dest = data_dir / "scan.json"
    assert dest.exists()
    entries = json.loads(dest.read_text())
    assert len(entries) == 1
    assert entries[0]["file_path"].endswith("track.mp3")
    assert entries[0]["track_id"]  # non-empty KSUID
    assert entries[0]["file_hash"]  # MD5 was computed
    assert entries[0]["duration"] == pytest.approx(300.0)


def test_run_scan_loads_existing_scan_and_reuses_id(tmp_path, monkeypatch):
    from mixprep.cli.commands.scan import run_scan

    data_dir = tmp_path / "data"
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))

    (tmp_path / "track.mp3").write_bytes(b"stable-content")

    with patch("mutagen.File", return_value=_mock_audio_info()):
        run_scan(tmp_path)

    first_id = json.loads((data_dir / "scan.json").read_text())[0]["track_id"]

    with patch("mutagen.File", return_value=_mock_audio_info()):
        run_scan(tmp_path)

    second_id = json.loads((data_dir / "scan.json").read_text())[0]["track_id"]

    assert first_id == second_id


def test_run_scan_cleans_up_orphaned_tmp(tmp_path, monkeypatch):
    from mixprep.cli.commands.scan import run_scan

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))

    # Simulate a leftover .tmp from a previous interrupted scan
    orphan = data_dir / "scan.tmp"
    orphan.write_text("leftover")

    (tmp_path / "track.mp3").write_bytes(b"audio")

    with patch("mutagen.File", return_value=_mock_audio_info()):
        run_scan(tmp_path)

    assert not orphan.exists()


def test_run_scan_interrupt_leaves_no_tmp_and_exits_130(tmp_path, monkeypatch):
    from mixprep.cli.commands.scan import run_scan

    data_dir = tmp_path / "data"
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))

    (tmp_path / "track.mp3").write_bytes(b"audio")

    with patch("mutagen.File", side_effect=KeyboardInterrupt):
        with pytest.raises(SystemExit) as exc:
            run_scan(tmp_path)

    assert exc.value.code == 130
    assert not (data_dir / "scan.json").exists()
    assert not (data_dir / "scan.tmp").exists()


def test_run_scan_warns_on_corrupt_existing_scan(tmp_path, monkeypatch):
    from mixprep.cli.commands.scan import console, run_scan

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))

    (data_dir / "scan.json").write_text("not valid json")
    (tmp_path / "track.mp3").write_bytes(b"audio")

    printed = []

    def capture(*args, **kwargs):
        printed.append(str(args))

    with patch.object(console, "print", side_effect=capture):
        with patch("mutagen.File", return_value=_mock_audio_info()):
            run_scan(tmp_path)

    # Warning was printed
    assert any("Warning" in m for m in printed)

    # scan.json still written with fresh result despite corrupt existing file
    entries = json.loads((data_dir / "scan.json").read_text())
    assert len(entries) == 1
