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


class TaggedValue(BaseModel):
    """A single DJ tag value with provenance."""

    value: Optional[float | str]  # null if the tag is absent in the file
    source: str  # always "file_tag" for scan stage
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


class TrackIndex(BaseModel):
    """Stable file index entry produced by the scan stage.

    Contains both file metadata (duration, format, sample_rate) and DJ tags
    (bpm, key, title, artist, album) — extracted in a single mutagen pass.

    Identity rule: track_id is assigned once per absolute file_path and reused
    on every subsequent scan as long as the path exists.  No content hashing —
    tag edits, format conversions and metadata updates do not change identity.
    """

    track_id: str
    file_path: str  # absolute, resolved path
    duration: Optional[float] = None  # seconds; null if mutagen cannot read it
    format: Optional[str] = None  # e.g. "mp3", "flac"; null if indeterminate
    sample_rate: Optional[int] = None  # Hz; null if mutagen cannot read it
    bpm: Optional[TaggedValue] = None  # null if tag absent
    key: Optional[TaggedValue] = None  # null if tag absent
    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None

    @field_validator("track_id")
    @classmethod
    def track_id_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("track_id must not be empty")
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
# F3 — Essentia
# ---------------------------------------------------------------------------


class EssentiaRaw(BaseModel):
    """Raw model outputs — exact values before any normalization."""

    discogs_effnet_embedding: Optional[list[float]]  # 1280-dim EfficientNet embedding
    msd_musicnn_embedding: Optional[list[float]]  # 200-dim MusiCNN embedding
    discogs_effnet_activations: Optional[dict[str, float]]  # 400 Discogs styles, sigmoid [0–1]
    maest_activations: Optional[dict[str, float]]  # 519 Discogs styles, softmax (sum=1)
    jamendo_genre_activations: Optional[dict[str, float]]  # 87 Jamendo genres, sigmoid [0–1]


class EssentiaScores(BaseModel):
    """Normalized scalar scores ready for downstream use."""

    danceability: Optional[float]  # [0–1]  danceable probability (model idx 0)
    arousal: Optional[float]  # [1–9]  DEAM scale (model output dim 1)
    tonal: Optional[float]  # [0–1]  tonal probability (model idx 0)
    timbre_bright: Optional[float]  # [0–1]  bright timbre probability (model idx 1)
    approachability: Optional[float]  # [0–1]  regression output
    engagement: Optional[float]  # [0–1]  regression output
    vocal_probability: Optional[float]  # [0–1]  vocal probability (model idx 1)


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
