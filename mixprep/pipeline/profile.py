"""
F4 — Profile stage.

Reads essentia/<track_id>.json + tracks/<track_id>.json, merges genres,
computes derived metrics and phase scores, writes profiles/<track_id>.json.

BPM priority:  file tag → essentia detected_bpm
Key priority:  file tag → essentia detected_key
All scores [0–1]. arousal normalized from DEAM [1–9]: (raw - 1) / 8.0

If any required input score is None, the dependent derived metric or phase
score is also None (never substituted with a fallback). A warning is logged.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from mixprep.pipeline.schemas import (
    EssentiaOutput,
    GenreLabel,
    TrackIndex,
    TrackProfile,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Camelot key parsing
# ---------------------------------------------------------------------------

_CAMELOT: dict[tuple[str, bool], str] = {
    ("c", True): "5A",
    ("c", False): "8B",
    ("db", True): "12A",
    ("db", False): "3B",
    ("d", True): "7A",
    ("d", False): "10B",
    ("eb", True): "2A",
    ("eb", False): "5B",
    ("e", True): "9A",
    ("e", False): "12B",
    ("f", True): "4A",
    ("f", False): "7B",
    ("gb", True): "11A",
    ("gb", False): "2B",
    ("g", True): "6A",
    ("g", False): "9B",
    ("ab", True): "1A",
    ("ab", False): "4B",
    ("a", True): "8A",
    ("a", False): "11B",
    ("bb", True): "3A",
    ("bb", False): "6B",
    ("b", True): "10A",
    ("b", False): "1B",
}

_ENHARMONIC: dict[str, str] = {
    "c#": "db",
    "d#": "eb",
    "e#": "f",
    "f#": "gb",
    "g#": "ab",
    "a#": "bb",
    "b#": "c",
}

_OPEN_KEY: dict[str, str] = {
    "1m": "6A",
    "2m": "7A",
    "3m": "8A",
    "4m": "9A",
    "5m": "10A",
    "6m": "11A",
    "7m": "12A",
    "8m": "1A",
    "9m": "2A",
    "10m": "3A",
    "11m": "4A",
    "12m": "5A",
    "1d": "6B",
    "2d": "7B",
    "3d": "8B",
    "4d": "9B",
    "5d": "10B",
    "6d": "11B",
    "7d": "12B",
    "8d": "1B",
    "9d": "2B",
    "10d": "3B",
    "11d": "4B",
    "12d": "5B",
}

_CAMELOT_RE = re.compile(r"^(1[0-2]|[1-9])[AB]$", re.IGNORECASE)
_OPEN_KEY_RE = re.compile(r"^(1[0-2]|[1-9])[md]$", re.IGNORECASE)
_MINOR_RE = re.compile(r"\s*(m|min|minor)\s*$", re.IGNORECASE)
_MAJOR_RE = re.compile(r"\s*(maj|major)\s*$", re.IGNORECASE)


def _normalize_note(raw: str) -> str:
    s = raw.strip().lower().replace("♯", "#").replace("♭", "b")
    return _ENHARMONIC.get(s, s)


def parse_camelot(key: str | None) -> str | None:
    """Parse any common key format to Camelot notation (e.g. '8A').

    Supported: Camelot (8A), Open Key (1m/6d), classical (Am, C#m, Bb),
    words (A minor, F# Major), unicode (C♯m, D♭).
    Returns None if unparseable.
    """
    if not key:
        return None

    s = key.strip()

    if _CAMELOT_RE.match(s):
        return s.upper()

    ok = s.lower()
    if _OPEN_KEY_RE.match(ok):
        return _OPEN_KEY.get(ok)

    s_norm = s.replace("♯", "#").replace("♭", "b")

    is_minor: bool
    if _MINOR_RE.search(s_norm):
        is_minor = True
        note_raw = _MINOR_RE.sub("", s_norm).strip()
    elif _MAJOR_RE.search(s_norm):
        is_minor = False
        note_raw = _MAJOR_RE.sub("", s_norm).strip()
    elif s_norm.endswith("m") and len(s_norm) > 1:
        is_minor = True
        note_raw = s_norm[:-1]
    else:
        is_minor = False
        note_raw = s_norm

    note = _normalize_note(note_raw)
    return _CAMELOT.get((note, is_minor))


def camelot_from_essentia(key: str, scale: str) -> str | None:
    """Convert Essentia KeyExtractor (key, scale) to Camelot."""
    note = _normalize_note(key)
    is_minor = scale.lower().strip() == "minor"
    return _CAMELOT.get((note, is_minor))


# ---------------------------------------------------------------------------
# Genre merging
# ---------------------------------------------------------------------------

_GENRE_SOURCES = (
    ("maest", "maest_activations"),
    ("effnet", "discogs_effnet_activations"),
    ("jamendo", "jamendo_genre_activations"),
)


def merge_genres(raw: object, top_n: int = 15) -> list[GenreLabel]:
    """Merge genre activations from maest, effnet, jamendo.

    Deduplicates by normalized label (lowercase, strip, - → space),
    averages scores across sources, sorts descending. Tie-break: alpha.
    """
    merged: dict[str, dict] = {}

    for source_name, field in _GENRE_SOURCES:
        activations: dict[str, float] | None = getattr(raw, field, None)
        if not activations:
            continue
        for label, score in activations.items():
            norm = label.lower().strip().replace("-", " ")
            if norm not in merged:
                merged[norm] = {
                    "original": label,
                    "best_score": score,
                    "total": 0.0,
                    "count": 0,
                    "sources": set(),
                }
            entry = merged[norm]
            entry["total"] += score
            entry["count"] += 1
            entry["sources"].add(source_name)
            if score > entry["best_score"]:
                entry["best_score"] = score
                entry["original"] = label

    results = sorted(
        merged.values(),
        key=lambda e: (-e["total"] / e["count"], e["original"].lower()),
    )

    return [
        GenreLabel(
            label=e["original"],
            score=round(e["total"] / e["count"], 4),
            sources=sorted(e["sources"]),
        )
        for e in results[:top_n]
    ]


# ---------------------------------------------------------------------------
# Profile computation
# ---------------------------------------------------------------------------


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _phase_scores(
    track_id: str,
    energy: Optional[float],
    groove: Optional[float],
    dance: Optional[float],
    tonal: Optional[float],
    timbre: Optional[float],
    appr: Optional[float],
    eng: Optional[float],
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Compute phase scores. Returns None for any score whose inputs are missing."""

    def _warn(score_name: str, missing: list[str]) -> None:
        log.warning(
            "Track %s: %s is None — missing inputs: %s",
            track_id,
            score_name,
            ", ".join(missing),
        )

    warmup: Optional[float] = None
    if None not in (appr, energy, timbre, dance, tonal):
        warmup = _clamp(
            0.30 * appr + 0.25 * (1 - energy) + 0.20 * (1 - timbre) + 0.15 * dance + 0.10 * tonal
        )
    else:
        _warn(
            "warmup_score",
            [
                k
                for k, v in [
                    ("approachability", appr),
                    ("energy", energy),
                    ("timbre_bright", timbre),
                    ("danceability", dance),
                    ("tonal", tonal),
                ]
                if v is None
            ],
        )

    build: Optional[float] = None
    if None not in (energy, dance, eng, timbre, tonal):
        build = _clamp(0.30 * energy + 0.25 * dance + 0.20 * eng + 0.15 * timbre + 0.10 * tonal)
    else:
        _warn(
            "build_score",
            [
                k
                for k, v in [
                    ("energy", energy),
                    ("danceability", dance),
                    ("engagement", eng),
                    ("timbre_bright", timbre),
                    ("tonal", tonal),
                ]
                if v is None
            ],
        )

    peak: Optional[float] = None
    if None not in (energy, dance, eng, timbre):
        peak = _clamp(0.35 * energy + 0.30 * dance + 0.20 * eng + 0.15 * timbre)
    else:
        _warn(
            "peak_score",
            [
                k
                for k, v in [
                    ("energy", energy),
                    ("danceability", dance),
                    ("engagement", eng),
                    ("timbre_bright", timbre),
                ]
                if v is None
            ],
        )

    reset: Optional[float] = None
    if None not in (groove, appr, timbre, tonal, energy):
        reset = _clamp(
            0.30 * groove + 0.25 * appr + 0.20 * (1 - timbre) + 0.15 * tonal + 0.10 * (1 - energy)
        )
    else:
        _warn(
            "reset_score",
            [
                k
                for k, v in [
                    ("groove", groove),
                    ("approachability", appr),
                    ("timbre_bright", timbre),
                    ("tonal", tonal),
                    ("energy", energy),
                ]
                if v is None
            ],
        )

    winddown: Optional[float] = None
    if None not in (energy, appr, tonal, timbre, eng):
        winddown = _clamp(
            0.30 * (1 - energy)
            + 0.25 * appr
            + 0.20 * tonal
            + 0.15 * (1 - timbre)
            + 0.10 * (1 - eng)
        )
    else:
        _warn(
            "winddown_score",
            [
                k
                for k, v in [
                    ("energy", energy),
                    ("approachability", appr),
                    ("tonal", tonal),
                    ("timbre_bright", timbre),
                    ("engagement", eng),
                ]
                if v is None
            ],
        )

    return warmup, build, peak, reset, winddown


def compute_profile(track: TrackIndex, essentia: EssentiaOutput) -> TrackProfile:
    """Compute TrackProfile from TrackIndex (scan) and EssentiaOutput."""
    s = essentia.scores

    arousal_raw = s.arousal
    arousal: Optional[float] = _clamp((arousal_raw - 1) / 8.0) if arousal_raw is not None else None

    dance = s.danceability
    tonal = s.tonal
    timbre = s.timbre_bright
    appr = s.approachability
    eng = s.engagement
    vocal = s.vocal_probability

    energy: Optional[float] = None
    if arousal is not None and dance is not None:
        energy = _clamp(0.6 * arousal + 0.4 * dance)
    else:
        log.warning(
            "Track %s: energy is None — missing: %s",
            track.track_id,
            ", ".join(k for k, v in [("arousal", arousal), ("danceability", dance)] if v is None),
        )

    groove: Optional[float] = None
    if dance is not None and appr is not None and tonal is not None:
        groove = _clamp(0.5 * dance + 0.3 * appr + 0.2 * tonal)
    else:
        log.warning(
            "Track %s: groove is None — missing: %s",
            track.track_id,
            ", ".join(
                k
                for k, v in [("danceability", dance), ("approachability", appr), ("tonal", tonal)]
                if v is None
            ),
        )

    warmup, build, peak, reset, winddown = _phase_scores(
        track.track_id, energy, groove, dance, tonal, timbre, appr, eng
    )

    # BPM: file tag → detected
    bpm: Optional[float] = None
    if track.bpm is not None and track.bpm.value is not None:
        try:
            bpm = float(track.bpm.value)
        except (ValueError, TypeError):
            log.warning("Track %s: could not parse BPM tag %r", track.track_id, track.bpm.value)
    if bpm is None and essentia.detected_bpm is not None:
        bpm = essentia.detected_bpm
    if bpm is None:
        log.warning("Track %s: bpm is None — no tag and no detected BPM", track.track_id)

    # Camelot key: file tag → detected
    camelot: Optional[str] = None
    if track.key is not None and track.key.value is not None:
        camelot = parse_camelot(str(track.key.value))
    if camelot is None and essentia.detected_key is not None:
        dk = essentia.detected_key
        camelot = camelot_from_essentia(dk.key, dk.scale)
    if camelot is None:
        log.warning("Track %s: camelot is None — no tag and no detected key", track.track_id)

    return TrackProfile(
        track_id=track.track_id,
        camelot=camelot,
        bpm=bpm,
        energy=energy,
        groove=groove,
        danceability=dance,
        arousal=arousal,
        tonal=tonal,
        timbre_bright=timbre,
        approachability=appr,
        engagement=eng,
        vocal_probability=vocal,
        warmup_score=warmup,
        build_score=build,
        peak_score=peak,
        reset_score=reset,
        winddown_score=winddown,
        genres=merge_genres(essentia.raw),
    )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_profile(profile: TrackProfile, profiles_dir: Path) -> None:
    """Write profiles/<track_id>.json atomically."""
    profiles_dir.mkdir(parents=True, exist_ok=True)
    dest = profiles_dir / f"{profile.track_id}.json"
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(profile.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(dest)


def load_profile(path: Path) -> TrackProfile:
    """Load and validate a profiles/<track_id>.json file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return TrackProfile.model_validate(raw)


def load_profiles_dir(profiles_dir: Path) -> list[TrackProfile]:
    """Load all profiles from profiles_dir/*.json."""
    profiles: list[TrackProfile] = []
    if not profiles_dir.is_dir():
        return profiles
    for f in profiles_dir.glob("*.json"):
        try:
            profiles.append(load_profile(f))
        except Exception as exc:
            log.warning("Skipping corrupt profile %s: %s", f, exc)
    return profiles
