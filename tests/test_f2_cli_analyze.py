"""
F2 CLI — run_analyze ingest stage removed; essentia now reads from tracks/.

Kept tests that still apply after scan+ingest merge:
- Missing tracks dir exits with code 1
- Unknown stage exits with code 1
- Empty tracks dir returns early
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_LIBRARY = "test_lib"


def _lib_dir(data_dir: Path) -> Path:
    return data_dir / "libraries" / _LIBRARY


def _write_track(data_dir: Path, track_id: str, file_path: str) -> None:
    tracks_dir = _lib_dir(data_dir) / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "track_id": track_id,
        "file_path": file_path,
        "duration": 300.0,
        "format": "mp3",
        "sample_rate": 44100,
        "bpm": None,
        "key": None,
        "title": None,
        "artist": None,
        "album": None,
    }
    (tracks_dir / f"{track_id}.json").write_text(json.dumps(entry), encoding="utf-8")


def test_run_analyze_exits_1_on_missing_tracks_dir(tmp_path, monkeypatch):
    from mixprep.cli.commands.analyze import run_analyze

    monkeypatch.setenv("MIXPREP_DATA_DIR", str(tmp_path / "data"))

    with pytest.raises(SystemExit) as exc:
        run_analyze("essentia", _LIBRARY)

    assert exc.value.code == 1


def test_run_analyze_exits_1_on_unknown_stage(tmp_path, monkeypatch):
    from mixprep.cli.commands.analyze import run_analyze

    monkeypatch.setenv("MIXPREP_DATA_DIR", str(tmp_path / "data"))

    with pytest.raises(SystemExit) as exc:
        run_analyze("bogus", _LIBRARY)

    assert exc.value.code == 1


def test_run_analyze_returns_early_on_empty_tracks_dir(tmp_path, monkeypatch):
    from mixprep.cli.commands.analyze import run_analyze

    data_dir = tmp_path / "data"
    tracks_dir = _lib_dir(data_dir) / "tracks"
    tracks_dir.mkdir(parents=True)
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))

    run_analyze("essentia", _LIBRARY)  # should not raise

    assert not (_lib_dir(data_dir) / "essentia").exists()
