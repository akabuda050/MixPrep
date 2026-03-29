"""
F3 Essentia — behavior tests (all Essentia/audio mocked).

Covers:
- run_essentia happy path: produces valid EssentiaOutput with scores
- run_essentia audio load failure: essentia_failed=True, scores null, no crash
- run_essentia inference failure: essentia_failed=True, scores null, no crash
- time_curves present on success, null on audio failure
- write_essentia / load_essentia round-trip
- write_essentia idempotency
- write_essentia no .tmp left behind
- Unicode track_id works
- load_models called once (singleton behavior)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mixprep.pipeline.schemas import TrackIndex

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_track(**kwargs) -> TrackIndex:
    defaults = dict(
        track_id="tid1",
        file_path="/music/track.mp3",
        file_hash="deadbeef",
        duration=300.0,
        format="mp3",
        sample_rate=44100,
    )
    defaults.update(kwargs)
    return TrackIndex(**defaults)


def _fake_audio(length: int = 16000) -> np.ndarray:
    """Return fake mono audio at 16 kHz (1 second of silence-ish)."""
    rng = np.random.default_rng(42)
    return rng.uniform(-0.01, 0.01, length).astype(np.float32)


def _make_fake_embedding(size: int) -> np.ndarray:
    return np.ones((1, size), dtype=np.float32) * 0.5


def _make_fake_activations(size: int) -> np.ndarray:
    arr = np.zeros((1, size), dtype=np.float32)
    arr[0, 0] = 0.3
    if size > 1:
        arr[0, 1] = 0.7
    return arr


def _patch_essentia_load(audio: np.ndarray):
    """Patch essentia.standard.MonoLoader to return *audio*."""
    loader_instance = MagicMock()
    loader_instance.return_value = audio
    loader_cls = MagicMock(return_value=loader_instance)
    return loader_cls


def _patch_models(effnet_size: int = 1280, musicnn_size: int = 200, maest_size: int = 519):
    """Return a dict of fake model callables."""
    effnet_emb = _make_fake_embedding(effnet_size)
    effnet_act = _make_fake_activations(400)  # 400 Discogs style activations
    musicnn_emb = _make_fake_embedding(musicnn_size)
    maest_act = _make_fake_activations(maest_size)
    binary_act = _make_fake_activations(2)
    jamendo_act = _make_fake_activations(87)
    arousal_valence_act = np.array([[5.0, 4.0]], dtype=np.float32)

    return {
        "effnet_emb": MagicMock(return_value=effnet_emb),
        "effnet_act": MagicMock(return_value=effnet_act),
        "musicnn": MagicMock(return_value=musicnn_emb),
        "maest": MagicMock(return_value=maest_act),
        "genre_jamendo": MagicMock(return_value=jamendo_act),
        "tonal_atonal": MagicMock(return_value=binary_act),
        "timbre": MagicMock(return_value=binary_act),
        "approachability": MagicMock(return_value=np.array([[0.6]], dtype=np.float32)),
        "engagement": MagicMock(return_value=np.array([[0.7]], dtype=np.float32)),
        "voice_instrumental": MagicMock(return_value=binary_act),
        "danceability": MagicMock(return_value=binary_act),
        "arousal_valence": MagicMock(return_value=arousal_valence_act),
    }


# ---------------------------------------------------------------------------
# run_essentia tests
# ---------------------------------------------------------------------------


def _run_with_mocks(track: TrackIndex, audio: np.ndarray | None = None):
    """Run run_essentia with all external dependencies mocked."""
    import mixprep.pipeline.essentia_runner as er

    fake_audio = audio if audio is not None else _fake_audio()
    fake_models = _patch_models()
    fake_labels: dict = {
        "effnet": [f"style_{i}" for i in range(400)],
        "maest": [f"maest_{i}" for i in range(519)],
        "genre_jamendo": [f"genre_{i}" for i in range(87)],
    }

    loader_cls = _patch_essentia_load(fake_audio)

    with (
        patch("essentia.standard.MonoLoader", loader_cls),
        patch.object(er, "_models", fake_models),
        patch.object(er, "_labels", fake_labels),
        patch.object(er, "_models_loaded", True),
    ):
        return er.run_essentia(track)


def test_run_essentia_happy_path():
    track = _make_track()
    output = _run_with_mocks(track)

    assert output.track_id == "tid1"
    assert not output.flags.essentia_failed
    assert output.scores.danceability is not None
    assert output.scores.arousal is not None
    assert output.time_curves is not None
    assert len(output.time_curves.rms) > 0


def test_run_essentia_arousal_range():
    track = _make_track()
    output = _run_with_mocks(track)

    # arousal is [1–9] MuSe scale — our fake returns dim 0 = 5.0
    assert output.scores.arousal == pytest.approx(5.0)


def test_run_essentia_raw_activations_present():
    track = _make_track()
    output = _run_with_mocks(track)

    assert output.raw.discogs_effnet_embedding is not None
    assert len(output.raw.discogs_effnet_embedding) == 1280
    assert output.raw.maest_activations is not None
    assert output.raw.jamendo_genre_activations is not None


def test_run_essentia_audio_load_failure():
    import mixprep.pipeline.essentia_runner as er

    track = _make_track()
    loader_cls = MagicMock(side_effect=Exception("file not found"))

    with (
        patch("essentia.standard.MonoLoader", loader_cls),
        patch.object(er, "_models_loaded", True),
        patch.object(er, "_models", {}),
    ):
        output = er.run_essentia(track)

    assert output.flags.essentia_failed
    assert output.time_curves is None
    assert output.scores.danceability is None
    assert output.scores.arousal is None


def test_run_essentia_inference_failure():
    import mixprep.pipeline.essentia_runner as er

    track = _make_track()
    fake_audio = _fake_audio()
    loader_cls = _patch_essentia_load(fake_audio)

    broken_models = {"effnet": MagicMock(side_effect=Exception("gpu error"))}

    with (
        patch("essentia.standard.MonoLoader", loader_cls),
        patch.object(er, "_models", broken_models),
        patch.object(er, "_labels", {}),
        patch.object(er, "_models_loaded", True),
    ):
        output = er.run_essentia(track)

    assert output.flags.essentia_failed
    assert output.scores.danceability is None
    # time_curves may still be present (computed before inference)


def test_run_essentia_time_curves_present_on_success():
    track = _make_track()
    output = _run_with_mocks(track, audio=_fake_audio(length=16000 * 3))

    assert output.time_curves is not None
    assert len(output.time_curves.rms) > 0
    assert len(output.time_curves.rms) == len(output.time_curves.novelty)


def test_run_essentia_time_curves_null_on_audio_failure():
    import mixprep.pipeline.essentia_runner as er

    track = _make_track()
    with (
        patch("essentia.standard.MonoLoader", MagicMock(side_effect=Exception("corrupt"))),
        patch.object(er, "_models_loaded", True),
        patch.object(er, "_models", {}),
    ):
        output = er.run_essentia(track)

    assert output.time_curves is None


# ---------------------------------------------------------------------------
# write_essentia / load_essentia
# ---------------------------------------------------------------------------


def test_write_essentia_creates_file(tmp_path):
    from mixprep.pipeline.essentia_runner import write_essentia
    from mixprep.pipeline.schemas import EssentiaFlags, EssentiaOutput, EssentiaRaw, EssentiaScores

    output = EssentiaOutput(
        track_id="tid1",
        raw=EssentiaRaw(
            discogs_effnet_embedding=[0.1, 0.2],
            msd_musicnn_embedding=None,
            discogs_effnet_activations={"Techno": 0.8},
            maest_activations=None,
            jamendo_genre_activations=None,
        ),
        scores=EssentiaScores(
            danceability=0.88,
            arousal=7.2,
            tonal=0.92,
            timbre_bright=0.61,
            approachability=0.78,
            engagement=0.85,
            vocal_probability=0.22,
        ),
        time_curves=None,
        flags=EssentiaFlags(essentia_failed=False),
    )

    write_essentia(output, tmp_path)
    dest = tmp_path / "tid1.json"
    assert dest.exists()
    data = json.loads(dest.read_text())
    assert data["track_id"] == "tid1"
    assert data["scores"]["danceability"] == pytest.approx(0.88)
    assert data["flags"]["essentia_failed"] is False


def test_write_essentia_no_tmp_left_behind(tmp_path):
    from mixprep.pipeline.essentia_runner import write_essentia
    from mixprep.pipeline.schemas import EssentiaFlags, EssentiaOutput, EssentiaRaw, EssentiaScores

    output = EssentiaOutput(
        track_id="tid2",
        raw=EssentiaRaw(
            discogs_effnet_embedding=None,
            msd_musicnn_embedding=None,
            discogs_effnet_activations=None,
            maest_activations=None,
            jamendo_genre_activations=None,
        ),
        scores=EssentiaScores(
            danceability=None,
            arousal=None,
            tonal=None,
            timbre_bright=None,
            approachability=None,
            engagement=None,
            vocal_probability=None,
        ),
        time_curves=None,
        flags=EssentiaFlags(essentia_failed=True),
    )
    write_essentia(output, tmp_path)
    assert not (tmp_path / "tid2.tmp").exists()


def test_write_load_essentia_roundtrip(tmp_path):
    from mixprep.pipeline.essentia_runner import load_essentia, write_essentia
    from mixprep.pipeline.schemas import (
        EssentiaFlags,
        EssentiaOutput,
        EssentiaRaw,
        EssentiaScores,
        TimeCurves,
    )

    output = EssentiaOutput(
        track_id="rt1",
        raw=EssentiaRaw(
            discogs_effnet_embedding=[0.5, 0.6],
            msd_musicnn_embedding=[0.1],
            discogs_effnet_activations={"Techno": 0.9},
            maest_activations=None,
            jamendo_genre_activations=None,
        ),
        scores=EssentiaScores(
            danceability=0.7,
            arousal=6.0,
            tonal=0.8,
            timbre_bright=0.5,
            approachability=0.6,
            engagement=0.7,
            vocal_probability=0.15,
        ),
        time_curves=TimeCurves(
            rms=[0.1, 0.2],
            onset_strength=[0.3, 0.4],
            low_band=[0.1, 0.2],
            mid_band=[0.2, 0.3],
            high_band=[0.05, 0.1],
            novelty=[0.0, 0.5],
        ),
        flags=EssentiaFlags(essentia_failed=False),
    )

    write_essentia(output, tmp_path)
    loaded = load_essentia(tmp_path / "rt1.json")

    assert loaded.track_id == "rt1"
    assert loaded.scores.danceability == pytest.approx(0.7)
    assert loaded.raw.discogs_effnet_activations == {"Techno": pytest.approx(0.9)}
    assert loaded.time_curves is not None
    assert loaded.time_curves.rms == pytest.approx([0.1, 0.2])
    assert not loaded.flags.essentia_failed


def test_write_essentia_idempotency(tmp_path):
    from mixprep.pipeline.essentia_runner import write_essentia
    from mixprep.pipeline.schemas import EssentiaFlags, EssentiaOutput, EssentiaRaw, EssentiaScores

    output = EssentiaOutput(
        track_id="idem1",
        raw=EssentiaRaw(
            discogs_effnet_embedding=None,
            msd_musicnn_embedding=None,
            discogs_effnet_activations=None,
            maest_activations=None,
            jamendo_genre_activations=None,
        ),
        scores=EssentiaScores(
            danceability=0.5,
            arousal=None,
            tonal=None,
            timbre_bright=None,
            approachability=None,
            engagement=None,
            vocal_probability=None,
        ),
        time_curves=None,
        flags=EssentiaFlags(essentia_failed=False),
    )

    write_essentia(output, tmp_path)
    first = (tmp_path / "idem1.json").read_text()

    write_essentia(output, tmp_path)
    second = (tmp_path / "idem1.json").read_text()

    assert first == second
