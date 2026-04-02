"""
F2 Prepare — contract tests for tag extraction.

Tag extraction is now part of scan stage. These tests verify:
- mutagen importable
- TaggedValue schema: valid data, rejects bad source/confidence
- TrackIndex accepts tag fields, allows null tags
"""

from __future__ import annotations

import pytest


def test_mutagen_importable():
    import mutagen  # noqa: F401


def test_tagged_value_valid():
    from mixprep.pipeline.schemas import TaggedValue

    tv = TaggedValue(value=138.0, source="file_tag", confidence=1.0)
    assert tv.value == 138.0
    assert tv.source == "file_tag"
    assert tv.confidence == 1.0


def test_tagged_value_string_value():
    from mixprep.pipeline.schemas import TaggedValue

    tv = TaggedValue(value="Am", source="file_tag", confidence=1.0)
    assert tv.value == "Am"


def test_tagged_value_null_value():
    from mixprep.pipeline.schemas import TaggedValue

    tv = TaggedValue(value=None, source="file_tag", confidence=1.0)
    assert tv.value is None


def test_tagged_value_rejects_empty_source():
    from pydantic import ValidationError

    from mixprep.pipeline.schemas import TaggedValue

    with pytest.raises(ValidationError):
        TaggedValue(value=138.0, source="", confidence=1.0)


def test_tagged_value_rejects_confidence_above_1():
    from pydantic import ValidationError

    from mixprep.pipeline.schemas import TaggedValue

    with pytest.raises(ValidationError):
        TaggedValue(value=138.0, source="file_tag", confidence=1.1)


def test_tagged_value_rejects_confidence_below_0():
    from pydantic import ValidationError

    from mixprep.pipeline.schemas import TaggedValue

    with pytest.raises(ValidationError):
        TaggedValue(value=138.0, source="file_tag", confidence=-0.1)


def test_track_index_with_tag_fields():
    from mixprep.pipeline.schemas import TaggedValue, TrackIndex

    t = TrackIndex(
        track_id="abc123",
        file_path="/music/track.mp3",
        bpm=TaggedValue(value=138.0, source="file_tag", confidence=1.0),
        key=TaggedValue(value="Am", source="file_tag", confidence=1.0),
        title="Track Title",
        artist="Artist Name",
        album="Album",
    )
    assert t.bpm is not None and t.bpm.value == 138.0
    assert t.key is not None and t.key.value == "Am"
    assert t.title == "Track Title"


def test_track_index_null_tag_fields():
    from mixprep.pipeline.schemas import TrackIndex

    t = TrackIndex(track_id="abc123", file_path="/music/track.mp3")
    assert t.bpm is None
    assert t.key is None
    assert t.title is None
    assert t.artist is None
    assert t.album is None


def test_track_index_serializes_tag_fields():
    from mixprep.pipeline.schemas import TaggedValue, TrackIndex

    t = TrackIndex(
        track_id="x1",
        file_path="/music/track.mp3",
        bpm=TaggedValue(value=140.0, source="file_tag", confidence=1.0),
    )
    d = t.model_dump()
    assert d["bpm"]["value"] == 140.0
    assert d["key"] is None
