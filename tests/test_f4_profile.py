"""
F4 Profile — behavior tests for compute_profile.

Covers:
- Full profile computation with all scores present
- Energy/groove derived correctly
- Phase scores computed correctly (warmup, build, peak, reset, winddown)
- None propagation: missing input → dependent scores None, warning logged
- BPM priority: file tag > detected
- Key priority: file tag > detected
- arousal normalized from [1–9] to [0–1]
- write_profile / load_profile roundtrip
- load_profiles_dir
"""

from __future__ import annotations

import logging
from typing import Optional

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_essentia(
    danceability: Optional[float] = 0.8,
    arousal: Optional[float] = 7.0,
    tonal: Optional[float] = 0.9,
    timbre_bright: Optional[float] = 0.5,
    approachability: Optional[float] = 0.6,
    engagement: Optional[float] = 0.7,
    vocal_probability: Optional[float] = 0.2,
    detected_bpm: Optional[float] = None,
    detected_key=None,
):
    from mixprep.pipeline.schemas import (
        EssentiaFlags,
        EssentiaOutput,
        EssentiaRaw,
        EssentiaScores,
    )

    return EssentiaOutput(
        track_id="t1",
        raw=EssentiaRaw(
            discogs_effnet_embedding=None,
            msd_musicnn_embedding=None,
            discogs_effnet_activations=None,
            maest_activations=None,
            jamendo_genre_activations=None,
        ),
        scores=EssentiaScores(
            danceability=danceability,
            arousal=arousal,
            tonal=tonal,
            timbre_bright=timbre_bright,
            approachability=approachability,
            engagement=engagement,
            vocal_probability=vocal_probability,
        ),
        time_curves=None,
        detected_bpm=detected_bpm,
        detected_key=detected_key,
        flags=EssentiaFlags(essentia_failed=False),
    )


def _make_track(
    track_id: str = "t1",
    bpm_value=None,
    key_value=None,
):
    from mixprep.pipeline.schemas import TaggedValue, TrackIndex

    bpm = (
        TaggedValue(value=bpm_value, source="file_tag", confidence=1.0)
        if bpm_value is not None
        else None
    )
    key = (
        TaggedValue(value=key_value, source="file_tag", confidence=1.0)
        if key_value is not None
        else None
    )
    return TrackIndex(
        track_id=track_id,
        file_path=f"/music/{track_id}.mp3",
        bpm=bpm,
        key=key,
    )


# ---------------------------------------------------------------------------
# compute_profile — happy path
# ---------------------------------------------------------------------------


def test_compute_profile_energy():
    from mixprep.pipeline.profile import compute_profile

    track = _make_track()
    essentia = _make_essentia(danceability=0.8, arousal=5.0)
    # arousal_norm = (5.0 - 1) / 8.0 = 0.5
    # energy = 0.6 * 0.5 + 0.4 * 0.8 = 0.3 + 0.32 = 0.62
    profile = compute_profile(track, essentia)
    assert profile.arousal is not None
    assert abs(profile.arousal - 0.5) < 1e-6
    assert profile.energy is not None
    assert abs(profile.energy - 0.62) < 1e-6


def test_compute_profile_groove():
    from mixprep.pipeline.profile import compute_profile

    track = _make_track()
    essentia = _make_essentia(danceability=0.8, approachability=0.6, tonal=0.9)
    # groove = 0.5 * 0.8 + 0.3 * 0.6 + 0.2 * 0.9 = 0.4 + 0.18 + 0.18 = 0.76
    profile = compute_profile(track, essentia)
    assert profile.groove is not None
    assert abs(profile.groove - 0.76) < 1e-6


def test_compute_profile_warmup():
    from mixprep.pipeline.profile import compute_profile

    track = _make_track()
    essentia = _make_essentia(
        danceability=0.8,
        arousal=5.0,
        tonal=0.9,
        timbre_bright=0.5,
        approachability=0.6,
    )
    # arousal_norm=0.5, energy=0.62
    # warmup = 0.30*appr + 0.25*(1-energy) + 0.20*(1-timbre) + 0.15*dance + 0.10*tonal
    #        = 0.30*0.6 + 0.25*(1-0.62) + 0.20*(1-0.5) + 0.15*0.8 + 0.10*0.9
    #        = 0.18 + 0.095 + 0.10 + 0.12 + 0.09 = 0.585
    profile = compute_profile(track, essentia)
    assert profile.warmup_score is not None
    assert abs(profile.warmup_score - 0.585) < 1e-5


def test_compute_profile_peak():
    from mixprep.pipeline.profile import compute_profile

    track = _make_track()
    essentia = _make_essentia(
        danceability=0.8,
        arousal=5.0,
        tonal=0.9,
        timbre_bright=0.5,
        engagement=0.7,
    )
    # energy=0.62, dance=0.8, eng=0.7, timbre=0.5
    # peak = 0.35*0.62 + 0.30*0.8 + 0.20*0.7 + 0.15*0.5
    #      = 0.217 + 0.24 + 0.14 + 0.075 = 0.672
    profile = compute_profile(track, essentia)
    assert profile.peak_score is not None
    assert abs(profile.peak_score - 0.672) < 1e-5


def test_compute_profile_arousal_normalization():
    from mixprep.pipeline.profile import compute_profile

    track = _make_track()
    essentia = _make_essentia(arousal=1.0)
    profile = compute_profile(track, essentia)
    assert abs(profile.arousal - 0.0) < 1e-6

    essentia9 = _make_essentia(arousal=9.0)
    profile9 = compute_profile(track, essentia9)
    assert abs(profile9.arousal - 1.0) < 1e-6


def test_compute_profile_all_scores_in_range():
    from mixprep.pipeline.profile import compute_profile

    track = _make_track()
    essentia = _make_essentia()
    profile = compute_profile(track, essentia)

    for field in (
        "energy",
        "groove",
        "danceability",
        "arousal",
        "tonal",
        "timbre_bright",
        "approachability",
        "engagement",
        "vocal_probability",
        "warmup_score",
        "build_score",
        "peak_score",
        "reset_score",
        "winddown_score",
    ):
        val = getattr(profile, field)
        assert val is not None, f"{field} should not be None"
        assert 0.0 <= val <= 1.0, f"{field}={val} not in [0,1]"


# ---------------------------------------------------------------------------
# None propagation
# ---------------------------------------------------------------------------


def test_energy_none_when_arousal_missing(caplog):
    from mixprep.pipeline.profile import compute_profile

    track = _make_track()
    essentia = _make_essentia(arousal=None)
    with caplog.at_level(logging.WARNING, logger="mixprep.pipeline.profile"):
        profile = compute_profile(track, essentia)
    assert profile.energy is None
    assert profile.arousal is None
    assert any("energy" in r.message for r in caplog.records)


def test_groove_none_when_dance_missing(caplog):
    from mixprep.pipeline.profile import compute_profile

    track = _make_track()
    essentia = _make_essentia(danceability=None)
    with caplog.at_level(logging.WARNING, logger="mixprep.pipeline.profile"):
        profile = compute_profile(track, essentia)
    assert profile.groove is None
    assert any("groove" in r.message for r in caplog.records)


def test_phase_scores_none_on_missing_input(caplog):
    from mixprep.pipeline.profile import compute_profile

    track = _make_track()
    # timbre_bright=None → warmup, build, peak, reset, winddown all require timbre
    essentia = _make_essentia(timbre_bright=None)
    with caplog.at_level(logging.WARNING, logger="mixprep.pipeline.profile"):
        profile = compute_profile(track, essentia)

    assert profile.warmup_score is None
    assert profile.build_score is None
    assert profile.peak_score is None
    assert profile.reset_score is None
    assert profile.winddown_score is None


# ---------------------------------------------------------------------------
# BPM / Key priority
# ---------------------------------------------------------------------------


def test_bpm_uses_file_tag():
    from mixprep.pipeline.profile import compute_profile

    track = _make_track(bpm_value=128.0)
    essentia = _make_essentia(detected_bpm=135.0)
    profile = compute_profile(track, essentia)
    assert profile.bpm == 128.0


def test_bpm_falls_back_to_detected():
    from mixprep.pipeline.profile import compute_profile

    track = _make_track()  # no tag
    essentia = _make_essentia(detected_bpm=132.5)
    profile = compute_profile(track, essentia)
    assert profile.bpm == 132.5


def test_bpm_none_when_both_absent(caplog):
    from mixprep.pipeline.profile import compute_profile

    track = _make_track()
    essentia = _make_essentia()
    with caplog.at_level(logging.WARNING, logger="mixprep.pipeline.profile"):
        profile = compute_profile(track, essentia)
    assert profile.bpm is None
    assert any("bpm" in r.message.lower() for r in caplog.records)


def test_key_uses_file_tag():
    from mixprep.pipeline.profile import compute_profile

    track = _make_track(key_value="Am")
    from mixprep.pipeline.schemas import DetectedKey

    dk = DetectedKey(key="C", scale="major", strength=0.9)
    essentia = _make_essentia(detected_key=dk)
    profile = compute_profile(track, essentia)
    assert profile.camelot == "8A"  # Am → 8A


def test_key_falls_back_to_detected():
    from mixprep.pipeline.profile import compute_profile
    from mixprep.pipeline.schemas import DetectedKey

    track = _make_track()  # no tag
    dk = DetectedKey(key="A", scale="minor", strength=0.85)
    essentia = _make_essentia(detected_key=dk)
    profile = compute_profile(track, essentia)
    assert profile.camelot == "8A"


def test_camelot_none_when_both_absent(caplog):
    from mixprep.pipeline.profile import compute_profile

    track = _make_track()
    essentia = _make_essentia()
    with caplog.at_level(logging.WARNING, logger="mixprep.pipeline.profile"):
        profile = compute_profile(track, essentia)
    assert profile.camelot is None
    assert any("camelot" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def test_write_and_load_profile(tmp_path):
    from mixprep.pipeline.profile import compute_profile, load_profile, write_profile

    track = _make_track(bpm_value=128.0, key_value="Am")
    essentia = _make_essentia()
    profile = compute_profile(track, essentia)

    write_profile(profile, tmp_path)
    dest = tmp_path / "t1.json"
    assert dest.exists()

    loaded = load_profile(dest)
    assert loaded.track_id == profile.track_id
    assert loaded.camelot == profile.camelot
    assert loaded.bpm == profile.bpm


def test_write_profile_atomic(tmp_path):
    """No .tmp file should remain after write."""
    from mixprep.pipeline.profile import compute_profile, write_profile

    track = _make_track()
    essentia = _make_essentia()
    profile = compute_profile(track, essentia)
    write_profile(profile, tmp_path)

    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == []


def test_load_profiles_dir(tmp_path):
    from mixprep.pipeline.profile import compute_profile, load_profiles_dir, write_profile

    for i in range(3):
        track = _make_track(track_id=f"t{i}")
        essentia = _make_essentia()
        essentia.track_id = f"t{i}"
        profile = compute_profile(track, essentia)
        write_profile(profile, tmp_path)

    profiles = load_profiles_dir(tmp_path)
    assert len(profiles) == 3


def test_load_profiles_dir_skips_corrupt(tmp_path):
    from mixprep.pipeline.profile import compute_profile, load_profiles_dir, write_profile

    track = _make_track()
    essentia = _make_essentia()
    profile = compute_profile(track, essentia)
    write_profile(profile, tmp_path)

    # Write a corrupt file
    (tmp_path / "corrupt.json").write_text("{bad json}", encoding="utf-8")

    profiles = load_profiles_dir(tmp_path)
    assert len(profiles) == 1


def test_write_profile_idempotent(tmp_path):
    from mixprep.pipeline.profile import compute_profile, write_profile

    track = _make_track(bpm_value=128.0, key_value="Am")
    essentia = _make_essentia()
    profile = compute_profile(track, essentia)

    write_profile(profile, tmp_path)
    first = (tmp_path / "t1.json").read_text()
    write_profile(profile, tmp_path)
    second = (tmp_path / "t1.json").read_text()

    assert first == second
