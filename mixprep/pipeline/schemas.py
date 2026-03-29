"""
Pydantic schemas for all pipeline stages.

Naming convention:
- Raw Essentia entries carry `source: str` (single model name).
- Merged/classified outputs carry `sources: list[str]` (all contributing models).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# F1 — Scan
# ---------------------------------------------------------------------------


class TrackIndex(BaseModel):
    """Stable file index entry produced by the scan stage."""

    track_id: str
    file_path: str
    file_hash: str  # MD5 hex digest of file contents
    duration: Optional[float]  # seconds; null if mutagen cannot read it
    format: Optional[str]  # e.g. "mp3", "flac"; null if indeterminate
    sample_rate: Optional[int]  # Hz; null if mutagen cannot read it

    @field_validator("track_id")
    @classmethod
    def track_id_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("track_id must not be empty")
        return v

    @field_validator("file_hash")
    @classmethod
    def file_hash_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("file_hash must not be empty")
        return v

    @field_validator("file_path")
    @classmethod
    def file_path_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("file_path must not be empty")
        return v

    @field_validator("duration")
    @classmethod
    def duration_positive(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError("duration must be positive")
        return v

    @field_validator("sample_rate")
    @classmethod
    def sample_rate_positive(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("sample_rate must be positive")
        return v


# ---------------------------------------------------------------------------
# F2 — Ingest
# ---------------------------------------------------------------------------


class TaggedValue(BaseModel):
    """A single DJ tag value with provenance."""

    value: Optional[float | str]  # null if the tag is absent in the file
    source: str  # always "file_tag" for ingest stage
    confidence: float  # always 1.0 for file tags

    @field_validator("source")
    @classmethod
    def source_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("source must not be empty")
        return v

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be in [0, 1]")
        return v


class TrackMetadata(BaseModel):
    """DJ tag metadata extracted from file tags."""

    track_id: str
    bpm: Optional[TaggedValue]  # null if tag absent
    key: Optional[TaggedValue]  # null if tag absent
    title: Optional[str]
    artist: Optional[str]
    album: Optional[str]

    @field_validator("track_id")
    @classmethod
    def track_id_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("track_id must not be empty")
        return v


# ---------------------------------------------------------------------------
# F3 — Essentia
# ---------------------------------------------------------------------------


class EssentiaRaw(BaseModel):
    """Raw model outputs — exact values before any normalization."""

    discogs_effnet_embedding: Optional[list[float]]
    msd_musicnn_embedding: Optional[list[float]]
    discogs_effnet_activations: Optional[dict[str, float]]  # 400 Discogs styles
    maest_activations: Optional[dict[str, float]]  # 519 Discogs styles
    jamendo_genre_activations: Optional[dict[str, float]]  # 87 Jamendo genres


class EssentiaScores(BaseModel):
    """Normalized scalar scores ready for downstream use.

    Only scores with reliable signal for DJ use-cases are included.
    Last.fm-derived mood tags (aggressive, happy, party, sad, acoustic, etc.)
    are excluded — they reflect subjective tagging semantics, not DJ-relevant
    audio properties.
    """

    danceability: Optional[float]  # [0–1] danceable vs not
    arousal: Optional[float]  # [1–9] MuSe energy scale
    tonal: Optional[float]  # [0–1] tonal vs atonal
    timbre_bright: Optional[float]  # [0–1] bright vs dark timbre
    approachability: Optional[float]  # [0–1] mainstream vs niche
    engagement: Optional[float]  # [0–1] active vs background listening
    vocal_probability: Optional[float]  # [0–1] vocal vs instrumental


class TimeCurves(BaseModel):
    """Per-frame time curves computed via librosa."""

    rms: list[float]
    onset_strength: list[float]
    low_band: list[float]  # 0–500 Hz mean energy per frame
    mid_band: list[float]  # 500–4000 Hz mean energy per frame
    high_band: list[float]  # 4000+ Hz mean energy per frame
    novelty: list[float]  # spectral novelty (onset envelope)


class EssentiaFlags(BaseModel):
    essentia_failed: bool


class EssentiaOutput(BaseModel):
    """Full Essentia stage artifact for one track."""

    track_id: str
    raw: EssentiaRaw
    scores: EssentiaScores
    time_curves: Optional[TimeCurves]  # null if audio load failed
    flags: EssentiaFlags

    @field_validator("track_id")
    @classmethod
    def track_id_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("track_id must not be empty")
        return v
