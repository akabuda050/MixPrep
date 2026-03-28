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
