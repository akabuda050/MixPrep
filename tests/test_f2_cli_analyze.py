"""
F2 CLI — run_analyze ingest command tests.

Covers:
- Missing scan.json exits with code 1
- Unknown stage exits with code 1
- Empty scan.json returns early
- Successful ingest writes metadata/<track_id>.json
- Interrupt exits 130
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_scan_json(data_dir: Path, tracks: list[dict]) -> None:
    scan = data_dir / "scan.json"
    scan.write_text(json.dumps(tracks, ensure_ascii=False), encoding="utf-8")


def _mock_mutagen_file(tags: dict) -> MagicMock:
    mf = MagicMock()
    mf.tags = tags
    return mf


def test_run_analyze_exits_1_on_missing_scan(tmp_path, monkeypatch):
    from mixprep.cli.commands.analyze import run_analyze

    monkeypatch.setenv("MIXPREP_DATA_DIR", str(tmp_path / "data"))

    with pytest.raises(SystemExit) as exc:
        run_analyze("ingest")

    assert exc.value.code == 1


def test_run_analyze_exits_1_on_unknown_stage(tmp_path, monkeypatch):
    from mixprep.cli.commands.analyze import run_analyze

    monkeypatch.setenv("MIXPREP_DATA_DIR", str(tmp_path / "data"))

    with pytest.raises(SystemExit) as exc:
        run_analyze("bogus")

    assert exc.value.code == 1


def test_run_analyze_returns_early_on_empty_scan(tmp_path, monkeypatch):
    from mixprep.cli.commands.analyze import run_analyze

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))
    _make_scan_json(data_dir, [])

    run_analyze("ingest")  # should not raise

    assert not (data_dir / "metadata").exists()


def test_run_analyze_ingest_writes_metadata(tmp_path, monkeypatch):
    from mixprep.cli.commands.analyze import run_analyze

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))

    tracks = [
        {
            "track_id": "tid1",
            "file_path": str(tmp_path / "track.mp3"),
            "file_hash": "abc123",
            "duration": 300.0,
            "format": "mp3",
            "sample_rate": 44100,
        }
    ]
    _make_scan_json(data_dir, tracks)

    tags = {"TBPM": ["138"], "TKEY": ["Am"], "TIT2": ["Test Track"], "TPE1": ["Artist"]}
    with patch("mutagen.File", return_value=_mock_mutagen_file(tags)):
        run_analyze("ingest")

    dest = data_dir / "metadata" / "tid1.json"
    assert dest.exists()
    data = json.loads(dest.read_text())
    assert data["track_id"] == "tid1"
    assert data["bpm"]["value"] == pytest.approx(138.0)
    assert data["key"]["value"] == "Am"
    assert data["title"] == "Test Track"
    assert data["artist"] == "Artist"


def test_run_analyze_ingest_interrupt_exits_130(tmp_path, monkeypatch):
    from mixprep.cli.commands.analyze import run_analyze

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))

    tracks = [
        {
            "track_id": "tid1",
            "file_path": str(tmp_path / "track.mp3"),
            "file_hash": "abc123",
            "duration": 300.0,
            "format": "mp3",
            "sample_rate": 44100,
        }
    ]
    _make_scan_json(data_dir, tracks)

    with patch("mutagen.File", side_effect=KeyboardInterrupt):
        with pytest.raises(SystemExit) as exc:
            run_analyze("ingest")

    assert exc.value.code == 130


def test_run_analyze_ingest_idempotent(tmp_path, monkeypatch):
    """Running ingest twice on same data produces identical output."""
    from mixprep.cli.commands.analyze import run_analyze

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))

    tracks = [
        {
            "track_id": "tid1",
            "file_path": str(tmp_path / "track.mp3"),
            "file_hash": "abc123",
            "duration": 300.0,
            "format": "mp3",
            "sample_rate": 44100,
        }
    ]
    _make_scan_json(data_dir, tracks)

    tags = {"TBPM": ["138"], "TKEY": ["Am"]}
    with patch("mutagen.File", return_value=_mock_mutagen_file(tags)):
        run_analyze("ingest")
    first = (data_dir / "metadata" / "tid1.json").read_text()

    with patch("mutagen.File", return_value=_mock_mutagen_file(tags)):
        run_analyze("ingest")
    second = (data_dir / "metadata" / "tid1.json").read_text()

    assert first == second
