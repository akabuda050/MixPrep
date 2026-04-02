"""
F4 Prepare — contract tests for profile stage.

Verifies:
- TrackProfile schema: valid data, required fields, None optionals
- GenreLabel schema: valid data
- parse_camelot: all supported formats
- camelot_from_essentia: major/minor
- merge_genres: dedup, averaging, sort, top_n
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Schema contracts
# ---------------------------------------------------------------------------


def test_track_profile_minimal():
    from mixprep.pipeline.schemas import TrackProfile

    p = TrackProfile(track_id="t1")
    assert p.track_id == "t1"
    assert p.camelot is None
    assert p.bpm is None
    assert p.energy is None
    assert p.genres == []


def test_track_profile_all_fields():
    from mixprep.pipeline.schemas import GenreLabel, TrackProfile

    p = TrackProfile(
        track_id="t1",
        camelot="8A",
        bpm=128.0,
        energy=0.7,
        groove=0.6,
        danceability=0.8,
        arousal=0.5,
        tonal=0.9,
        timbre_bright=0.4,
        approachability=0.6,
        engagement=0.7,
        vocal_probability=0.2,
        warmup_score=0.3,
        build_score=0.6,
        peak_score=0.9,
        reset_score=0.4,
        winddown_score=0.2,
        genres=[GenreLabel(label="Techno", score=0.8, sources=["maest"])],
    )
    assert p.camelot == "8A"
    assert p.bpm == 128.0
    assert p.energy == 0.7
    assert len(p.genres) == 1
    assert p.genres[0].label == "Techno"


def test_track_profile_rejects_empty_track_id():
    from pydantic import ValidationError

    from mixprep.pipeline.schemas import TrackProfile

    with pytest.raises(ValidationError):
        TrackProfile(track_id="")


def test_genre_label_valid():
    from mixprep.pipeline.schemas import GenreLabel

    g = GenreLabel(label="House", score=0.75, sources=["maest", "effnet"])
    assert g.label == "House"
    assert g.score == 0.75
    assert g.sources == ["maest", "effnet"]


# ---------------------------------------------------------------------------
# parse_camelot
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,expected",
    [
        # Camelot passthrough
        ("8A", "8A"),
        ("12B", "12B"),
        ("1A", "1A"),
        ("10b", "10B"),
        # Open Key
        ("1m", "6A"),
        ("6d", "11B"),
        ("12m", "5A"),
        # Classical short form (minor)
        ("Am", "8A"),
        ("C#m", "12A"),
        ("Bbm", "3A"),
        ("Gm", "6A"),
        # Classical short form (major)
        ("C", "8B"),
        ("F#", "2B"),
        ("Bb", "6B"),
        # Words
        ("A minor", "8A"),
        ("F# Major", "2B"),
        ("A min", "8A"),
        ("C maj", "8B"),
        # Unicode
        ("C♯m", "12A"),
        ("D♭", "3B"),
        # Enharmonics
        ("Dbm", "12A"),
        ("Ebm", "2A"),
    ],
)
def test_parse_camelot(key, expected):
    from mixprep.pipeline.profile import parse_camelot

    assert parse_camelot(key) == expected


def test_parse_camelot_none_on_garbage():
    from mixprep.pipeline.profile import parse_camelot

    assert parse_camelot("xyz123") is None
    assert parse_camelot("") is None
    assert parse_camelot(None) is None


def test_camelot_from_essentia_minor():
    from mixprep.pipeline.profile import camelot_from_essentia

    assert camelot_from_essentia("A", "minor") == "8A"
    assert camelot_from_essentia("C", "minor") == "5A"


def test_camelot_from_essentia_major():
    from mixprep.pipeline.profile import camelot_from_essentia

    assert camelot_from_essentia("C", "major") == "8B"
    assert camelot_from_essentia("F#", "major") == "2B"


# ---------------------------------------------------------------------------
# merge_genres
# ---------------------------------------------------------------------------


def test_merge_genres_dedup_and_average():
    from mixprep.pipeline.profile import merge_genres
    from mixprep.pipeline.schemas import EssentiaRaw

    raw = EssentiaRaw(
        discogs_effnet_embedding=None,
        msd_musicnn_embedding=None,
        discogs_effnet_activations={"Techno": 0.8, "House": 0.5},
        maest_activations={"Techno": 0.6, "Ambient": 0.3},
        jamendo_genre_activations=None,
    )
    result = merge_genres(raw)
    labels = [g.label for g in result]
    assert "Techno" in labels

    techno = next(g for g in result if g.label.lower() == "techno")
    # averaged: (0.8 + 0.6) / 2 = 0.7
    assert abs(techno.score - 0.7) < 1e-4
    assert set(techno.sources) == {"effnet", "maest"}


def test_merge_genres_sorted_descending():
    from mixprep.pipeline.profile import merge_genres
    from mixprep.pipeline.schemas import EssentiaRaw

    raw = EssentiaRaw(
        discogs_effnet_embedding=None,
        msd_musicnn_embedding=None,
        discogs_effnet_activations={"House": 0.3, "Techno": 0.9},
        maest_activations=None,
        jamendo_genre_activations=None,
    )
    result = merge_genres(raw)
    scores = [g.score for g in result]
    assert scores == sorted(scores, reverse=True)


def test_merge_genres_top_n():
    from mixprep.pipeline.profile import merge_genres
    from mixprep.pipeline.schemas import EssentiaRaw

    activations = {f"Genre{i}": float(i) / 100 for i in range(30)}
    raw = EssentiaRaw(
        discogs_effnet_embedding=None,
        msd_musicnn_embedding=None,
        discogs_effnet_activations=activations,
        maest_activations=None,
        jamendo_genre_activations=None,
    )
    result = merge_genres(raw, top_n=10)
    assert len(result) == 10


def test_merge_genres_label_normalization():
    """dash-separated labels should deduplicate with space-separated."""
    from mixprep.pipeline.profile import merge_genres
    from mixprep.pipeline.schemas import EssentiaRaw

    raw = EssentiaRaw(
        discogs_effnet_embedding=None,
        msd_musicnn_embedding=None,
        discogs_effnet_activations={"deep-house": 0.7},
        maest_activations={"deep house": 0.5},
        jamendo_genre_activations=None,
    )
    result = merge_genres(raw)
    # Should be deduplicated into one entry
    assert len(result) == 1
    assert abs(result[0].score - 0.6) < 1e-4


def test_merge_genres_empty_sources():
    from mixprep.pipeline.profile import merge_genres
    from mixprep.pipeline.schemas import EssentiaRaw

    raw = EssentiaRaw(
        discogs_effnet_embedding=None,
        msd_musicnn_embedding=None,
        discogs_effnet_activations=None,
        maest_activations=None,
        jamendo_genre_activations=None,
    )
    result = merge_genres(raw)
    assert result == []
