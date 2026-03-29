"""
F3 CLI — run_analyze essentia command tests.

Covers:
- Corrupt scan.json exits with code 1
- Essentia stage: tracks missing metadata are skipped
- Essentia stage: all tracks missing metadata → no eligible tracks, returns early
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _track_dict(track_id: str, file_path: str = "/music/track.mp3") -> dict:
    return {
        "track_id": track_id,
        "file_path": file_path,
        "file_hash": "abc123",
        "duration": 300.0,
        "format": "mp3",
        "sample_rate": 44100,
    }


def _write_scan(data_dir: Path, tracks: list[dict]) -> None:
    (data_dir / "scan.json").write_text(json.dumps(tracks), encoding="utf-8")


def _write_metadata(data_dir: Path, track_id: str) -> None:
    meta_dir = data_dir / "metadata"
    meta_dir.mkdir(exist_ok=True)
    (meta_dir / f"{track_id}.json").write_text(
        json.dumps(
            {
                "track_id": track_id,
                "bpm": None,
                "key": None,
                "title": None,
                "artist": None,
                "album": None,
            }
        ),
        encoding="utf-8",
    )


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
            mood_theme_activations=None,
            moods_mirex_activations=None,
            instrument_activations=None,
        ),
        scores=EssentiaScores(
            danceability=None if failed else 0.8,
            arousal=None,
            valence=None,
            aggressive=None,
            happy=None,
            party=None,
            relaxed=None,
            sad=None,
            acoustic=None,
            tonal=None,
            timbre_bright=None,
            approachability=None,
            engagement=None,
            vocal_probability=None,
            vocalist_gender=None,
        ),
        time_curves=None,
        flags=EssentiaFlags(essentia_failed=failed),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_analyze_exits_1_on_corrupt_scan(tmp_path, monkeypatch):
    from mixprep.cli.commands.analyze import run_analyze

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))
    (data_dir / "scan.json").write_text("not valid json", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        run_analyze("essentia")

    assert exc.value.code == 1


def test_run_analyze_essentia_skips_missing_metadata(tmp_path, monkeypatch):
    from mixprep.cli.commands.analyze import run_analyze

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))

    # Track with no metadata artifact
    _write_scan(data_dir, [_track_dict("tid1")])
    # Do NOT write metadata for tid1

    run_analyze("essentia")  # should not raise — skips and returns early

    assert not (data_dir / "essentia" / "tid1.json").exists()


def test_run_analyze_essentia_writes_artifact(tmp_path, monkeypatch):
    import mixprep.pipeline.essentia_runner as er
    from mixprep.cli.commands.analyze import run_analyze

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))

    _write_scan(data_dir, [_track_dict("tid1")])
    _write_metadata(data_dir, "tid1")

    fake_output = _make_fake_essentia_output("tid1")

    with (
        patch.object(er, "load_models"),
        patch.object(er, "run_essentia", return_value=fake_output),
    ):
        run_analyze("essentia")

    dest = data_dir / "essentia" / "tid1.json"
    assert dest.exists()
    data = json.loads(dest.read_text())
    assert data["track_id"] == "tid1"
    assert not data["flags"]["essentia_failed"]


def test_run_analyze_essentia_counts_failed_tracks(tmp_path, monkeypatch):
    import mixprep.pipeline.essentia_runner as er
    from mixprep.cli.commands.analyze import console, run_analyze

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))

    _write_scan(data_dir, [_track_dict("tid1")])
    _write_metadata(data_dir, "tid1")

    failed_output = _make_fake_essentia_output("tid1", failed=True)
    printed = []

    with (
        patch.object(er, "load_models"),
        patch.object(er, "run_essentia", return_value=failed_output),
        patch.object(console, "print", side_effect=lambda *a, **k: printed.append(str(a))),
    ):
        run_analyze("essentia")

    # Failed count was reported
    assert any("1" in m and "failed" in m for m in printed)


def test_run_analyze_essentia_model_load_failure_exits_1(tmp_path, monkeypatch):
    import mixprep.pipeline.essentia_runner as er
    from mixprep.cli.commands.analyze import run_analyze

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))

    _write_scan(data_dir, [_track_dict("tid1")])
    _write_metadata(data_dir, "tid1")

    with (
        patch.object(er, "load_models", side_effect=Exception("model files missing")),
        pytest.raises(SystemExit) as exc,
    ):
        run_analyze("essentia")

    assert exc.value.code == 1


def test_run_analyze_essentia_interrupt_exits_130(tmp_path, monkeypatch):
    import mixprep.pipeline.essentia_runner as er
    from mixprep.cli.commands.analyze import run_analyze

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))

    _write_scan(data_dir, [_track_dict("tid1")])
    _write_metadata(data_dir, "tid1")

    with (
        patch.object(er, "load_models"),
        patch.object(er, "run_essentia", side_effect=KeyboardInterrupt),
        pytest.raises(SystemExit) as exc,
    ):
        run_analyze("essentia")

    assert exc.value.code == 130


def test_run_analyze_essentia_idempotent(tmp_path, monkeypatch):
    import mixprep.pipeline.essentia_runner as er
    from mixprep.cli.commands.analyze import run_analyze

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))

    _write_scan(data_dir, [_track_dict("tid1")])
    _write_metadata(data_dir, "tid1")

    fake_output = _make_fake_essentia_output("tid1")

    with (
        patch.object(er, "load_models"),
        patch.object(er, "run_essentia", return_value=fake_output),
    ):
        run_analyze("essentia")
    first = (data_dir / "essentia" / "tid1.json").read_text()

    with (
        patch.object(er, "load_models"),
        patch.object(er, "run_essentia", return_value=fake_output),
    ):
        run_analyze("essentia")
    second = (data_dir / "essentia" / "tid1.json").read_text()

    assert first == second


def test_run_analyze_classify_not_implemented(tmp_path, monkeypatch):
    from mixprep.cli.commands.analyze import console, run_analyze

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))
    _write_scan(data_dir, [_track_dict("tid1")])

    printed = []
    with patch.object(console, "print", side_effect=lambda *a, **k: printed.append(str(a))):
        run_analyze("classify")

    assert any("not yet implemented" in m for m in printed)


def test_run_analyze_essentia_time_curves_warning(tmp_path, monkeypatch, caplog):
    """time_curves computation failure logs a warning but does not crash."""
    import logging

    import mixprep.pipeline.essentia_runner as er
    from mixprep.cli.commands.analyze import run_analyze

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("MIXPREP_DATA_DIR", str(data_dir))

    _write_scan(data_dir, [_track_dict("tid1", file_path=str(tmp_path / "track.mp3"))])
    _write_metadata(data_dir, "tid1")

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
            "mood_theme",
            "instrument",
            "mood_aggressive",
            "mood_happy",
            "mood_party",
            "mood_relaxed",
            "mood_sad",
            "mood_acoustic",
            "tonal_atonal",
            "timbre",
            "approachability",
            "engagement",
            "voice_instrumental",
            "gender",
            "danceability",
            "arousal_valence",
            "moods_mirex",
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
        run_analyze("essentia")

    dest = data_dir / "essentia" / "tid1.json"
    assert dest.exists()
    data = json.loads(dest.read_text())
    assert data["time_curves"] is None
    assert any("librosa error" in r.message for r in caplog.records)
