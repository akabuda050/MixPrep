"""
F1 Prepare — contract tests.

These tests validate that all prerequisites for the Scan feature are in place:
- Required dependencies are importable
- TrackIndex schema accepts valid data and rejects invalid data
"""

import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Dependency contracts
# ---------------------------------------------------------------------------


def test_mutagen_importable():
    import mutagen  # noqa: F401


def test_cyksuid_importable():
    import cyksuid  # noqa: F401


def test_cyksuid_can_generate_id():
    from cyksuid.v2 import ksuid

    kid = ksuid()
    assert isinstance(str(kid), str)
    assert len(str(kid)) > 0


# ---------------------------------------------------------------------------
# TrackIndex schema — valid data
# ---------------------------------------------------------------------------


def test_track_index_valid_full():
    from mixprep.pipeline.schemas import TrackIndex

    t = TrackIndex(
        track_id="abc123",
        file_path="/music/track.mp3",
        duration=420.5,
        format="mp3",
        sample_rate=44100,
    )
    assert t.track_id == "abc123"
    assert t.duration == 420.5
    assert t.sample_rate == 44100


def test_track_index_valid_null_optional_fields():
    from mixprep.pipeline.schemas import TrackIndex

    t = TrackIndex(
        track_id="abc123",
        file_path="/music/track.mp3",
        duration=None,
        format=None,
        sample_rate=None,
    )
    assert t.duration is None
    assert t.format is None
    assert t.sample_rate is None


# ---------------------------------------------------------------------------
# TrackIndex schema — invalid data
# ---------------------------------------------------------------------------


def test_track_index_rejects_empty_track_id():
    from mixprep.pipeline.schemas import TrackIndex

    with pytest.raises(ValidationError, match="track_id"):
        TrackIndex(
            track_id="   ",
            file_path="/music/track.mp3",
            duration=None,
            format=None,
            sample_rate=None,
        )


def test_track_index_rejects_empty_file_path():
    from mixprep.pipeline.schemas import TrackIndex

    with pytest.raises(ValidationError, match="file_path"):
        TrackIndex(
            track_id="abc",
            file_path="",
            duration=None,
            format=None,
            sample_rate=None,
        )


def test_track_index_rejects_zero_duration():
    from mixprep.pipeline.schemas import TrackIndex

    with pytest.raises(ValidationError, match="duration"):
        TrackIndex(
            track_id="abc",
            file_path="/music/track.mp3",
            duration=0.0,
            format=None,
            sample_rate=None,
        )


def test_track_index_rejects_negative_duration():
    from mixprep.pipeline.schemas import TrackIndex

    with pytest.raises(ValidationError, match="duration"):
        TrackIndex(
            track_id="abc",
            file_path="/music/track.mp3",
            duration=-1.0,
            format=None,
            sample_rate=None,
        )


def test_track_index_rejects_zero_sample_rate():
    from mixprep.pipeline.schemas import TrackIndex

    with pytest.raises(ValidationError, match="sample_rate"):
        TrackIndex(
            track_id="abc",
            file_path="/music/track.mp3",
            duration=None,
            format=None,
            sample_rate=0,
        )


def test_track_index_rejects_missing_required_fields():
    from mixprep.pipeline.schemas import TrackIndex

    with pytest.raises(ValidationError):
        TrackIndex(track_id="abc")  # type: ignore[call-arg]


def test_track_index_serializes_to_dict():
    from mixprep.pipeline.schemas import TrackIndex

    t = TrackIndex(
        track_id="abc",
        file_path="/music/track.mp3",
        duration=300.0,
        format="flac",
        sample_rate=48000,
    )
    d = t.model_dump()
    assert d["track_id"] == "abc"
    assert d["duration"] == 300.0
    assert d["format"] == "flac"
