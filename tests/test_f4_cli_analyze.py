"""
F4 CLI — run_analyze profile command tests.

Covers:
- Missing tracks dir exits with code 1
- Profile stage writes profiles/<track_id>.json
- Profile stage skips tracks without essentia artifact (logs warning)
- Profile stage handles corrupt essentia artifact gracefully
- Profile stage idempotent (running twice produces same output)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

_LIBRARY = "test_lib"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lib_dir(data_dir: Path) -> Path:
    return data_dir / "libraries" / _LIBRARY


def _write_track(data_dir: Path, track_id: str, file_path: str = "/music/track.mp3") -> None:
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


def _write_essentia(data_dir: Path, track_id: str, failed: bool = False) -> None:
    from mixprep.pipeline.schemas import (
        EssentiaFlags,
        EssentiaOutput,
        EssentiaRaw,
        EssentiaScores,
    )

    output = EssentiaOutput(
        track_id=track_id,
        raw=EssentiaRaw(
            discogs_effnet_embedding=None,
            msd_musicnn_embedding=None,
            discogs_effnet_activations=None,
            maest_activations=None,
            jamendo_genre_activations=None,
        ),
        scores=EssentiaScores(
            danceability=None if failed else 0.8,
            arousal=None if failed else 5.0,
            tonal=None if failed else 0.9,
            timbre_bright=None if failed else 0.5,
            approachability=None if failed else 0.6,
            engagement=None if failed else 0.7,
            vocal_probability=None if failed else 0.2,
        ),
        time_curves=None,
        flags=EssentiaFlags(essentia_failed=failed),
    )
    essentia_dir = _lib_dir(data_dir) / "essentia"
    essentia_dir.mkdir(parents=True, exist_ok=True)
    dest = essentia_dir / f"{track_id}.json"
    dest.write_text(json.dumps(output.model_dump(), indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_analyze_profile_exits_1_on_missing_tracks_dir(tmp_path, monkeypatch):
    from mixprep.cli.commands.analyze import run_analyze

    data_dir = tmp_path / "data"
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))

    with pytest.raises(SystemExit) as exc:
        run_analyze("profile", _LIBRARY)

    assert exc.value.code == 1


def test_run_analyze_profile_writes_artifact(tmp_path, monkeypatch):
    from mixprep.cli.commands.analyze import run_analyze

    data_dir = tmp_path / "data"
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))
    _write_track(data_dir, "tid1")
    _write_essentia(data_dir, "tid1")

    run_analyze("profile", _LIBRARY)

    dest = _lib_dir(data_dir) / "profiles" / "tid1.json"
    assert dest.exists()
    profile = json.loads(dest.read_text())
    assert profile["track_id"] == "tid1"


def test_run_analyze_profile_skips_without_essentia(tmp_path, monkeypatch):
    from mixprep.cli.commands.analyze import console, run_analyze

    data_dir = tmp_path / "data"
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))
    _write_track(data_dir, "tid1")
    # No essentia artifact written

    printed = []
    with patch.object(console, "print", side_effect=lambda *a, **k: printed.append(str(a))):
        run_analyze("profile", _LIBRARY)

    profiles_dir = _lib_dir(data_dir) / "profiles"
    assert not profiles_dir.exists() or not list(profiles_dir.glob("*.json"))
    assert any("skipped" in m for m in printed)


def test_run_analyze_profile_handles_corrupt_essentia(tmp_path, monkeypatch):
    from mixprep.cli.commands.analyze import run_analyze

    data_dir = tmp_path / "data"
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))
    _write_track(data_dir, "tid1")

    # Write a corrupt essentia file
    essentia_dir = _lib_dir(data_dir) / "essentia"
    essentia_dir.mkdir(parents=True, exist_ok=True)
    (essentia_dir / "tid1.json").write_text("{invalid}", encoding="utf-8")

    # Should not raise
    run_analyze("profile", _LIBRARY)

    dest = _lib_dir(data_dir) / "profiles" / "tid1.json"
    assert not dest.exists()


def test_run_analyze_profile_idempotent(tmp_path, monkeypatch):
    from mixprep.cli.commands.analyze import run_analyze

    data_dir = tmp_path / "data"
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))
    _write_track(data_dir, "tid1")
    _write_essentia(data_dir, "tid1")

    run_analyze("profile", _LIBRARY)
    first = (_lib_dir(data_dir) / "profiles" / "tid1.json").read_text()

    run_analyze("profile", _LIBRARY)
    second = (_lib_dir(data_dir) / "profiles" / "tid1.json").read_text()

    assert first == second


def test_run_analyze_profile_multiple_tracks(tmp_path, monkeypatch):
    from mixprep.cli.commands.analyze import run_analyze

    data_dir = tmp_path / "data"
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))

    for i in range(3):
        _write_track(data_dir, f"tid{i}")
        _write_essentia(data_dir, f"tid{i}")

    run_analyze("profile", _LIBRARY)

    profiles_dir = _lib_dir(data_dir) / "profiles"
    profiles = list(profiles_dir.glob("*.json"))
    assert len(profiles) == 3
