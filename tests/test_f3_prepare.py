"""
F3 Prepare — contract tests for Essentia stage.

Covers:
- librosa importable
- essentia importable
- EssentiaRaw: valid data, all-null accepted
- EssentiaScores: valid data, all-null accepted
- TimeCurves: valid data, curve lengths match
- EssentiaOutput: valid, rejects empty track_id
- essentia_failed flag propagates
"""

from __future__ import annotations

import pytest


def test_librosa_importable():
    import librosa  # noqa: F401


def test_essentia_importable():
    import essentia  # noqa: F401


def test_essentia_raw_all_null():
    from mixprep.pipeline.schemas import EssentiaRaw

    raw = EssentiaRaw(
        discogs_effnet_embedding=None,
        msd_musicnn_embedding=None,
        discogs_effnet_activations=None,
        maest_activations=None,
        jamendo_genre_activations=None,
    )
    assert raw.discogs_effnet_embedding is None
    assert raw.maest_activations is None


def test_essentia_raw_with_data():
    from mixprep.pipeline.schemas import EssentiaRaw

    raw = EssentiaRaw(
        discogs_effnet_embedding=[0.1, 0.2, 0.3],
        msd_musicnn_embedding=[0.4, 0.5],
        discogs_effnet_activations={"Techno": 0.82, "House": 0.45},
        maest_activations={"Techno": 0.91},
        jamendo_genre_activations={"techno": 0.77},
    )
    assert raw.discogs_effnet_embedding == pytest.approx([0.1, 0.2, 0.3])
    assert raw.maest_activations == {"Techno": pytest.approx(0.91)}


def test_essentia_scores_all_null():
    from mixprep.pipeline.schemas import EssentiaScores

    scores = EssentiaScores(
        danceability=None,
        arousal=None,
        tonal=None,
        timbre_bright=None,
        approachability=None,
        engagement=None,
        vocal_probability=None,
    )
    assert scores.danceability is None
    assert scores.arousal is None


def test_essentia_scores_with_data():
    from mixprep.pipeline.schemas import EssentiaScores

    scores = EssentiaScores(
        danceability=0.88,
        arousal=7.2,
        tonal=0.92,
        timbre_bright=0.61,
        approachability=0.78,
        engagement=0.85,
        vocal_probability=0.22,
    )
    assert scores.danceability == pytest.approx(0.88)
    assert scores.arousal == pytest.approx(7.2)
    assert scores.tonal == pytest.approx(0.92)


def test_time_curves_valid():
    from mixprep.pipeline.schemas import TimeCurves

    curves = TimeCurves(
        rms=[0.1, 0.2, 0.3],
        onset_strength=[0.0, 1.0, 0.5],
        low_band=[0.1, 0.2, 0.3],
        mid_band=[0.2, 0.3, 0.4],
        high_band=[0.05, 0.1, 0.15],
        novelty=[0.0, 0.8, 0.3],
    )
    assert len(curves.rms) == 3
    assert len(curves.novelty) == 3


def test_essentia_output_valid():
    from mixprep.pipeline.schemas import (
        EssentiaFlags,
        EssentiaOutput,
        EssentiaRaw,
        EssentiaScores,
        TimeCurves,
    )

    output = EssentiaOutput(
        track_id="tid1",
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
            danceability=None,
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
        time_curves=TimeCurves(
            rms=[0.1],
            onset_strength=[0.2],
            low_band=[0.1],
            mid_band=[0.2],
            high_band=[0.05],
            novelty=[0.0],
        ),
        flags=EssentiaFlags(essentia_failed=False),
    )
    assert output.track_id == "tid1"
    assert not output.flags.essentia_failed


def test_essentia_output_failed_flag():
    from mixprep.pipeline.schemas import (
        EssentiaFlags,
        EssentiaOutput,
        EssentiaRaw,
        EssentiaScores,
    )

    output = EssentiaOutput(
        track_id="tid1",
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
            danceability=None,
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
        flags=EssentiaFlags(essentia_failed=True),
    )
    assert output.flags.essentia_failed
    assert output.time_curves is None


def test_essentia_output_rejects_empty_track_id():
    from pydantic import ValidationError

    from mixprep.pipeline.schemas import (
        EssentiaFlags,
        EssentiaOutput,
        EssentiaRaw,
        EssentiaScores,
    )

    with pytest.raises(ValidationError):
        EssentiaOutput(
            track_id="",
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
                danceability=None,
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
            flags=EssentiaFlags(essentia_failed=False),
        )
