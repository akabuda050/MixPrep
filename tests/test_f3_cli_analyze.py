"""
F3 CLI — run_analyze essentia command tests.

Covers:
- Missing tracks dir exits with code 1
- Essentia stage: writes essentia/<track_id>.json
- Essentia stage: essentia_failed tracks counted in failed total
- Essentia stage: model load failure exits with code 1
- Essentia stage: interrupt exits 130
- Essentia stage: idempotent (running twice produces same output)
- classify stage not yet implemented prints warning
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
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


def _make_fake_essentia_output(track_id: str, failed: bool = False):
    from mixprep.pipeline.schemas import (
        EssentiaFlags,
        EssentiaOutput,
        EssentiaRaw,
        EssentiaScores,
    )

    return EssentiaOutput(
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
            arousal=None,
            tonal=None,
            timbre_bright=None,
            approachability=None,
            engagement=None,
            vocal_probability=None,
        ),
        time_curves=None,
        flags=EssentiaFlags(essentia_failed=failed),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_analyze_exits_1_on_missing_tracks_dir(tmp_path, monkeypatch):
    from mixprep.cli.commands.analyze import run_analyze

    data_dir = tmp_path / "data"
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))

    with pytest.raises(SystemExit) as exc:
        run_analyze("essentia", _LIBRARY)

    assert exc.value.code == 1


def test_run_analyze_essentia_writes_artifact(tmp_path, monkeypatch):
    import mixprep.pipeline.essentia_runner as er
    from mixprep.cli.commands.analyze import run_analyze

    data_dir = tmp_path / "data"
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))
    _write_track(data_dir, "tid1")

    fake_output = _make_fake_essentia_output("tid1")

    with (
        patch.object(er, "load_models"),
        patch.object(er, "run_essentia", return_value=fake_output),
    ):
        run_analyze("essentia", _LIBRARY)

    dest = _lib_dir(data_dir) / "essentia" / "tid1.json"
    assert dest.exists()
    data = json.loads(dest.read_text())
    assert data["track_id"] == "tid1"
    assert not data["flags"]["essentia_failed"]


def test_run_analyze_essentia_counts_failed_tracks(tmp_path, monkeypatch):
    import mixprep.pipeline.essentia_runner as er
    from mixprep.cli.commands.analyze import console, run_analyze

    data_dir = tmp_path / "data"
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))
    _write_track(data_dir, "tid1")

    failed_output = _make_fake_essentia_output("tid1", failed=True)
    printed = []

    with (
        patch.object(er, "load_models"),
        patch.object(er, "run_essentia", return_value=failed_output),
        patch.object(console, "print", side_effect=lambda *a, **k: printed.append(str(a))),
    ):
        run_analyze("essentia", _LIBRARY)

    assert any("1" in m and "failed" in m for m in printed)


def test_run_analyze_essentia_model_load_failure_exits_1(tmp_path, monkeypatch):
    import mixprep.pipeline.essentia_runner as er
    from mixprep.cli.commands.analyze import run_analyze

    data_dir = tmp_path / "data"
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))
    _write_track(data_dir, "tid1")

    with (
        patch.object(er, "load_models", side_effect=Exception("model files missing")),
        pytest.raises(SystemExit) as exc,
    ):
        run_analyze("essentia", _LIBRARY)

    assert exc.value.code == 1


def test_run_analyze_essentia_interrupt_exits_130(tmp_path, monkeypatch):
    import mixprep.pipeline.essentia_runner as er
    from mixprep.cli.commands.analyze import run_analyze

    data_dir = tmp_path / "data"
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))
    _write_track(data_dir, "tid1")

    with (
        patch.object(er, "load_models"),
        patch.object(er, "run_essentia", side_effect=KeyboardInterrupt),
        pytest.raises(SystemExit) as exc,
    ):
        run_analyze("essentia", _LIBRARY)

    assert exc.value.code == 130


def test_run_analyze_essentia_idempotent(tmp_path, monkeypatch):
    import mixprep.pipeline.essentia_runner as er
    from mixprep.cli.commands.analyze import run_analyze

    data_dir = tmp_path / "data"
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))
    _write_track(data_dir, "tid1")

    fake_output = _make_fake_essentia_output("tid1")

    with (
        patch.object(er, "load_models"),
        patch.object(er, "run_essentia", return_value=fake_output),
    ):
        run_analyze("essentia", _LIBRARY)
    first = (_lib_dir(data_dir) / "essentia" / "tid1.json").read_text()

    with (
        patch.object(er, "load_models"),
        patch.object(er, "run_essentia", return_value=fake_output),
    ):
        run_analyze("essentia", _LIBRARY)
    second = (_lib_dir(data_dir) / "essentia" / "tid1.json").read_text()

    assert first == second


def test_run_analyze_classify_not_implemented(tmp_path, monkeypatch):
    from mixprep.cli.commands.analyze import console, run_analyze

    data_dir = tmp_path / "data"
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))
    _write_track(data_dir, "tid1")

    printed = []
    with patch.object(console, "print", side_effect=lambda *a, **k: printed.append(str(a))):
        run_analyze("classify", _LIBRARY)

    assert any("not yet implemented" in m for m in printed)


def test_run_analyze_essentia_time_curves_warning(tmp_path, monkeypatch, caplog):
    """time_curves computation failure logs a warning but does not crash."""
    import logging

    import mixprep.pipeline.essentia_runner as er
    from mixprep.cli.commands.analyze import run_analyze

    data_dir = tmp_path / "data"
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))
    _write_track(data_dir, "tid1", file_path=str(tmp_path / "track.mp3"))

    fake_audio = np.zeros(16000, dtype=np.float32)
    loader_instance = MagicMock(return_value=fake_audio)
    loader_cls = MagicMock(return_value=loader_instance)

    fake_models = {
        k: MagicMock(return_value=np.zeros((1, 2), dtype=np.float32))
        for k in [
            "effnet_emb",
            "effnet_act",
            "musicnn",
            "maest",
            "genre_jamendo",
            "tonal_atonal",
            "timbre",
            "approachability",
            "engagement",
            "voice_instrumental",
            "danceability",
            "arousal_valence",
        ]
    }
    fake_models["arousal_valence"] = MagicMock(return_value=np.array([[5.0, 4.0]]))
    fake_models["approachability"] = MagicMock(return_value=np.array([[0.6]]))
    fake_models["engagement"] = MagicMock(return_value=np.array([[0.7]]))

    with (
        patch.object(er, "load_models"),
        patch.object(er, "_models_loaded", True),
        patch.object(er, "_models", fake_models),
        patch.object(er, "_labels", {}),
        patch("essentia.standard.MonoLoader", loader_cls),
        patch(
            "mixprep.pipeline.essentia_runner.compute_time_curves",
            side_effect=Exception("librosa error"),
        ),
        caplog.at_level(logging.WARNING, logger="mixprep.pipeline.essentia_runner"),
    ):
        run_analyze("essentia", _LIBRARY)

    dest = _lib_dir(data_dir) / "essentia" / "tid1.json"
    assert dest.exists()
    data = json.loads(dest.read_text())
    assert data["time_curves"] is None
    assert any("librosa error" in r.message for r in caplog.records)
