"""
F2 Prepare — contract tests for ingest stage.

Covers:
- mutagen importable
- TaggedValue schema: valid data, rejects bad source/confidence
- TrackMetadata schema: valid data, null fields allowed, rejects empty track_id
- source is always "file_tag", confidence always 1.0
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


def test_track_metadata_valid():
    from mixprep.pipeline.schemas import TaggedValue, TrackMetadata

    meta = TrackMetadata(
        track_id="abc123",
        bpm=TaggedValue(value=138.0, source="file_tag", confidence=1.0),
        key=TaggedValue(value="Am", source="file_tag", confidence=1.0),
        title="Track Title",
        artist="Artist Name",
        album="Album",
    )
    assert meta.track_id == "abc123"
    assert meta.bpm is not None
    assert meta.bpm.value == 138.0
    assert meta.key is not None
    assert meta.key.value == "Am"


def test_track_metadata_all_null_tags():
    from mixprep.pipeline.schemas import TrackMetadata

    meta = TrackMetadata(
        track_id="abc123",
        bpm=None,
        key=None,
        title=None,
        artist=None,
        album=None,
    )
    assert meta.bpm is None
    assert meta.key is None
    assert meta.title is None
    assert meta.artist is None
    assert meta.album is None


def test_track_metadata_rejects_empty_track_id():
    from pydantic import ValidationError

    from mixprep.pipeline.schemas import TrackMetadata

    with pytest.raises(ValidationError):
        TrackMetadata(
            track_id="",
            bpm=None,
            key=None,
            title=None,
            artist=None,
            album=None,
        )


def test_track_metadata_serializes_to_dict():
    from mixprep.pipeline.schemas import TaggedValue, TrackMetadata

    meta = TrackMetadata(
        track_id="x1",
        bpm=TaggedValue(value=140.0, source="file_tag", confidence=1.0),
        key=None,
        title="T",
        artist=None,
        album=None,
    )
    d = meta.model_dump()
    assert d["track_id"] == "x1"
    assert d["bpm"]["value"] == 140.0
    assert d["key"] is None
