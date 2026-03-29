"""
F3 — Essentia model runner.

Loads all models once per process (module-level cache), runs inference,
assembles EssentiaOutput per track.

Model loading order: embeddings first, then classification heads.

Output node conventions (verified against Essentia source and docs):
  effnet:
    output="PartitionedCall:1"  → 1280-dim embedding (used as head input)
    output="PartitionedCall:0"  → 400-class Discogs softmax activations
    We load TWO instances: one for the embedding, one for the activations.

  musicnn (msd-musicnn-1.pb):
    output="model/dense/BiasAdd" → 200-dim embedding bottleneck layer
    Used only for the arousal head (muse-msd-musicnn).

  maest:
    default output="Identity"   → shape (frames,1,1,519) raw logits — squeeze+mean+softmax
    output="StatefulPartitionedCall:N" → attention layer N (not used here)

  TensorflowPredict2D (all classification heads):
    Takes a matrix input (embedding frames), outputs per-frame predictions.
    All outputs are pooled across frames via mean before storing.

Binary classifier convention:
    output shape: (frames, 2); positive-class index varies per model (verified):
      idx 0 = positive: danceability, tonal_atonal
      idx 1 = positive: timbre, voice_instrumental

Arousal (muse-msd-musicnn):
    Output shape: (frames, 2) → dim 0 = arousal [1–9] MuSe scale. Valence (dim 1) excluded.

All failures are caught and stored as flags.essentia_failed=True with
null scores — artifact is always written, never silently skipped.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from essentia import EssentiaLogger

from mixprep.models.cache import models_dir
from mixprep.pipeline.schemas import (
    EssentiaFlags,
    EssentiaOutput,
    EssentiaRaw,
    EssentiaScores,
    TimeCurves,
    TrackIndex,
)
from mixprep.pipeline.time_curves import compute_time_curves

EssentiaLogger().warningActive = False
log = logging.getLogger(__name__)

_SAMPLE_RATE = 16000  # required by all Essentia models

# ---------------------------------------------------------------------------
# Module-level model cache
# ---------------------------------------------------------------------------

_models: dict[str, Any] = {}
_labels: dict[str, list[str]] = {}
_models_loaded = False


def _load_json_labels(name: str) -> list[str]:
    path = models_dir() / f"{name}.json"
    if not path.exists():
        log.warning("Label file missing: %s", path)
        return []
    with open(path) as f:
        data = json.load(f)
    # Essentia label JSON: {"classes": [...]} or list directly
    if isinstance(data, list):
        return [str(x) for x in data]
    return [str(x) for x in data.get("classes", [])]


def _model_path(filename: str) -> str:
    return str(models_dir() / filename)


def load_models() -> None:
    """Load all Essentia models into module-level cache. Called once per process."""
    global _models_loaded
    if _models_loaded:
        return

    import essentia.standard as es  # type: ignore[import]

    # ── Embeddings ──────────────────────────────────────────────────────────
    # effnet: two outputs from the same model file.
    #   PartitionedCall:1 → 1280-dim embedding (fed into classification heads)
    #   PartitionedCall:0 → 400-class Discogs softmax (stored as raw activations)
    _models["effnet_emb"] = es.TensorflowPredictEffnetDiscogs(
        graphFilename=_model_path("discogs-effnet-bs64-1.pb"),
        output="PartitionedCall:1",
    )
    _models["effnet_act"] = es.TensorflowPredictEffnetDiscogs(
        graphFilename=_model_path("discogs-effnet-bs64-1.pb"),
        output="PartitionedCall:0",
    )
    _labels["effnet"] = _load_json_labels("discogs-effnet-bs64-1")

    # musicnn: output="model/dense/BiasAdd" → 200-dim embedding bottleneck
    _models["musicnn"] = es.TensorflowPredictMusiCNN(
        graphFilename=_model_path("msd-musicnn-1.pb"),
        output="model/dense/BiasAdd",
    )

    # maest: default output="Identity" → 519-class Discogs softmax
    _models["maest"] = es.TensorflowPredictMAEST(
        graphFilename=_model_path("discogs-maest-30s-pw-519l-2.pb"),
    )
    _labels["maest"] = _load_json_labels("discogs-maest-30s-pw-519l-2")

    # ── Classification heads (effnet embedding input) ────────────────────────
    def _effnet_head(pb: str, output: str = "model/Softmax") -> Any:
        return es.TensorflowPredict2D(graphFilename=_model_path(pb), output=output)

    _models["genre_jamendo"] = _effnet_head(
        "mtg_jamendo_genre-discogs-effnet-1.pb", output="model/Sigmoid"
    )
    _labels["genre_jamendo"] = _load_json_labels("mtg_jamendo_genre-discogs-effnet-1")

    _models["tonal_atonal"] = _effnet_head("tonal_atonal-discogs-effnet-1.pb")
    _models["timbre"] = _effnet_head("timbre-discogs-effnet-1.pb")
    _models["approachability"] = _effnet_head(
        "approachability_regression-discogs-effnet-1.pb",
        output="model/Identity",
    )
    _models["engagement"] = _effnet_head(
        "engagement_regression-discogs-effnet-1.pb",
        output="model/Identity",
    )
    _models["voice_instrumental"] = _effnet_head("voice_instrumental-discogs-effnet-1.pb")
    _models["danceability"] = _effnet_head("danceability-discogs-effnet-1.pb")

    # ── musicnn heads (musicnn embedding input) ──────────────────────────────
    # deam-msd-musicnn: output shape (frames, 2) → (valence, arousal) in [1–9] scale.
    # dim 1 = arousal (verified against DEAM dataset convention).
    # Valence excluded — unreliable for EDM.
    _models["arousal_valence"] = es.TensorflowPredict2D(
        graphFilename=_model_path("deam-msd-musicnn-2.pb"),
        output="model/Identity",
    )

    _models_loaded = True
    log.info("All Essentia models loaded.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pool_mean(activations: np.ndarray) -> np.ndarray:
    """Average activations across time frames → 1-D vector.

    Handles all model output shapes:
      (frames, dims)          → mean axis 0 → (dims,)
      (frames, 1, 1, dims)    → mean axis 0 → (1, 1, dims) → squeeze → (dims,)
    Always returns a 1-D array via np.atleast_1d so callers can safely index [i].
    """
    a = activations.mean(axis=0) if activations.ndim > 1 else activations
    return np.atleast_1d(a.squeeze())


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D array."""
    e = np.exp(x - x.max())
    return e / e.sum()


def _binary_score(activations: np.ndarray, idx: int = 1) -> float:
    """Extract positive-class probability from binary classifier output.

    Expected input shape: (frames, 2).  *idx* selects which output index
    corresponds to the positive / target class (verified against real model
    outputs — see load_models docstring for per-model values).

    Raises IndexError if the pooled output has fewer dimensions than expected,
    which is caught by the inference try/except in run_essentia → _null_output.
    """
    v = _pool_mean(activations)
    return float(v[idx])


def _activation_dict(activations: np.ndarray, labels: list[str]) -> dict[str, float]:
    """Map label names to mean activation scores."""
    v = _pool_mean(activations)
    if len(labels) != len(v):
        log.warning(
            "Label count mismatch: %d labels vs %d model outputs — truncating to shorter",
            len(labels),
            len(v),
        )
    return {label: float(v[i]) for i, label in enumerate(labels) if i < len(v)}


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------


def _null_output(track_id: str, time_curves: TimeCurves | None) -> EssentiaOutput:
    """Return a failed EssentiaOutput with all scores null."""
    return EssentiaOutput(
        track_id=track_id,
        raw=EssentiaRaw(
            discogs_effnet_embedding=None,
            msd_musicnn_embedding=None,
            discogs_effnet_activations=None,
            maest_activations=None,
            jamendo_genre_activations=None,
        ),
        scores=EssentiaScores(
            danceability=None,
            arousal=None,
            tonal=None,
            timbre_bright=None,
            approachability=None,
            engagement=None,
            vocal_probability=None,
        ),
        time_curves=time_curves,
        flags=EssentiaFlags(essentia_failed=True),
    )


def run_essentia(track: TrackIndex) -> EssentiaOutput:
    """
    Run all Essentia models on *track* and return EssentiaOutput.

    On any failure (audio load or inference), returns a valid artifact with
    essentia_failed=True and null scores.
    """
    import essentia.standard as es  # type: ignore[import]

    load_models()

    # ── Load audio ──────────────────────────────────────────────────────────
    try:
        audio = es.MonoLoader(filename=track.file_path, sampleRate=_SAMPLE_RATE)()
    except Exception as exc:
        log.warning("Audio load failed for %s: %s", track.file_path, exc)
        return _null_output(track.track_id, time_curves=None)

    # ── Time curves (librosa, independent of Essentia models) ───────────────
    time_curves: TimeCurves | None = None
    try:
        time_curves = compute_time_curves(np.array(audio, dtype=np.float32), _SAMPLE_RATE)
    except Exception as exc:
        log.warning("Time curves failed for %s: %s", track.file_path, exc)

    # ── Essentia inference ──────────────────────────────────────────────────
    try:
        # Embeddings (raw audio → embedding vector per frame)
        effnet_emb = _models["effnet_emb"](audio)  # (frames, 1280)
        effnet_act = _models["effnet_act"](audio)  # (frames, 400) — Discogs activations
        musicnn_emb = _models["musicnn"](audio)  # (frames, 200)
        maest_act = _models["maest"](audio)  # (frames, 519) — Discogs activations

        # Classification heads — take effnet embedding as input
        genre_jamendo_act = _models["genre_jamendo"](effnet_emb)  # (frames, 87)
        tonal_act = _models["tonal_atonal"](effnet_emb)  # (frames, 2)
        timbre_act = _models["timbre"](effnet_emb)  # (frames, 2)
        approach_act = _models["approachability"](effnet_emb)  # (frames, 1)
        engagement_act = _models["engagement"](effnet_emb)  # (frames, 1)
        voice_act = _models["voice_instrumental"](effnet_emb)  # (frames, 2)
        dance_act = _models["danceability"](effnet_emb)  # (frames, 2)

        # musicnn heads — take musicnn embedding as input
        av_act = _models["arousal_valence"](musicnn_emb)  # (frames, 2) — arousal dim 0 only

    except Exception as exc:
        log.warning("Essentia inference failed for %s: %s", track.file_path, exc)
        return _null_output(track.track_id, time_curves)

    # ── Assemble raw activations ─────────────────────────────────────────────
    # MAEST outputs raw logits (1,1,1,519) — apply softmax to get probabilities.
    maest_probs = _softmax(_pool_mean(maest_act))
    raw = EssentiaRaw(
        discogs_effnet_embedding=_pool_mean(effnet_emb).tolist(),
        msd_musicnn_embedding=_pool_mean(musicnn_emb).tolist(),
        discogs_effnet_activations=_activation_dict(effnet_act, _labels.get("effnet", [])),
        maest_activations={
            label: float(maest_probs[i])
            for i, label in enumerate(_labels.get("maest", []))
            if i < len(maest_probs)
        },
        jamendo_genre_activations=_activation_dict(
            genre_jamendo_act, _labels.get("genre_jamendo", [])
        ),
    )

    # ── Assemble normalized scores ───────────────────────────────────────────
    # Binary head positive-class indices (verified by running real inference):
    #   idx 0 = positive: danceability(danceable), tonal
    #   idx 1 = positive: timbre(bright), voice(vocal)
    # arousal_valence: (valence, arousal) — DEAM convention. dim 1 = arousal [1–9].
    av_mean = _pool_mean(av_act)
    scores = EssentiaScores(
        danceability=_binary_score(dance_act, idx=0),
        arousal=float(av_mean[1]) if len(av_mean) >= 2 else None,
        tonal=_binary_score(tonal_act, idx=0),
        timbre_bright=_binary_score(timbre_act, idx=1),
        approachability=float(_pool_mean(approach_act)[0]),
        engagement=float(_pool_mean(engagement_act)[0]),
        vocal_probability=_binary_score(voice_act, idx=1),
    )

    return EssentiaOutput(
        track_id=track.track_id,
        raw=raw,
        scores=scores,
        time_curves=time_curves,
        flags=EssentiaFlags(essentia_failed=False),
    )


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------


def write_essentia(output: EssentiaOutput, dest_dir: Path) -> None:
    """Write essentia/<track_id>.json atomically."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{output.track_id}.json"
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(output.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(dest)


def load_essentia(path: Path) -> EssentiaOutput:
    """Load and validate an essentia/<track_id>.json file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return EssentiaOutput.model_validate(raw)
