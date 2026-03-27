# MixPrep

DJ library intelligence and mix-set generation.

Analyzes your music library using machine learning to extract genre, mood, energy, instrumentation, and other musical properties. Uses that data to automatically build mix-ready track sets tailored to different DJ set profiles (warm-up, peak-time, cool-down, etc.).

## What it does

- **Scan** — indexes your music library, assigns stable track IDs
- **Analyze** — reads BPM and key from file metadata (set by your DJ software); analyzes audio directly if missing
- **Classify** — runs ML models to detect genre, mood, energy, danceability, vocals, instruments
- **Build** — generates candidate mix sets for a chosen profile using beam search
- **Export** — outputs playlists as M3U, HTML, or CSV

## Requirements

- Python 3.11–3.13
- [uv](https://docs.astral.sh/uv/) — package manager (replaces pip + venv)

Install uv once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation

```bash
git clone git@github.com:akabuda050/MixPrep.git
cd MixPrep
```

The project requires Python 3.11–3.13 (`essentia-tensorflow` has no 3.14+ wheels yet). `.python-version` pins 3.12 — uv picks it up automatically.

Pick the TensorFlow variant matching your hardware:

```bash
# CPU only (most machines)
uv sync --extra cpu

# NVIDIA GPU (CUDA)
uv sync --extra cuda
```

For development (adds pytest + ruff):

```bash
uv sync --extra cpu --extra dev
```

uv creates `.venv` automatically and installs everything from `uv.lock`. No manual `venv` or `pip install` needed.

> **Note:** Do not install bare `essentia` alongside `essentia-tensorflow` — they conflict. Do not use both `cpu` and `cuda` extras in the same environment.

## Keeping dependencies up to date

```bash
uv lock --upgrade            # re-resolve all deps to latest compatible versions
uv lock --upgrade-package rich   # upgrade one package only
uv sync --extra cpu --extra dev  # apply updated lockfile
```

Commit `uv.lock` to git — it pins every transitive dependency exactly, so every install is reproducible.

## Development

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Tests with coverage
uv run pytest
```

## Model management

MixPrep uses pre-trained Essentia models for classification. All commands are under `mixprep models`.

Models are stored in `~/.local/share/mixprep/models` by default. Override with `MIXPREP_MODELS_DIR`:

```bash
export MIXPREP_MODELS_DIR=/data/mixprep/models
```

### Download

```bash
mixprep models download                          # download all models
mixprep models download --model essentia-maest   # download one model
mixprep models download --force                  # re-download even if already present
```

The Essentia server can be slow — failed downloads retry automatically (3 attempts). If any fail, re-run the same command to pick up where it left off.

### Status

```bash
mixprep models status
```

Shows every model with:
- **File** — `✓` present on disk, `✗` missing
- **Size (MB)** — actual file size read from disk
- **Description** — what the model does

### Clear

```bash
mixprep models clear essentia-maest  # remove one model
mixprep models clear --all           # remove all models
```

Removes the cached model file. Use `download` to re-fetch.

## Models used

| Model name | What it does |
|---|---|
| `essentia-discogs-effnet` | Primary embedding — EfficientNet, powers most classification heads |
| `essentia-discogs-effnet-labels` | Label metadata for discogs-effnet (400 Discogs styles) |
| `essentia-maest` | Best genre accuracy — MAEST transformer, 519 Discogs styles |
| `essentia-maest-labels` | Label metadata for MAEST |
| `essentia-msd-musicnn` | Secondary embedding — required for arousal/valence and MIREX mood |
| `essentia-genre-jamendo` | 87 genre tags (house, techno, ambient, etc.) |
| `essentia-genre-jamendo-labels` | Label metadata for Jamendo genre |
| `essentia-mood-theme` | 56 mood/theme tags (energetic, dark, groovy, etc.) |
| `essentia-mood-theme-labels` | Label metadata for mood/theme |
| `essentia-mood-aggressive/happy/party/relaxed/sad/acoustic` | Binary mood scores |
| `essentia-tonal-atonal` | Tonal vs atonal — harmonic mixing compatibility |
| `essentia-timbre` | Bright vs dark timbre |
| `essentia-approachability` | Mainstream vs niche [0–1] |
| `essentia-engagement` | Active vs background listening [0–1] |
| `essentia-voice-instrumental` | Vocals vs instrumental |
| `essentia-gender` | Vocalist gender |
| `essentia-danceability` | Danceability score |
| `essentia-instrument` | 40 instrument tags |
| `essentia-instrument-labels` | Label metadata for instrument |
| `essentia-arousal-valence-muse` | Arousal + valence [1–9] — MuSe dataset |
| `essentia-moods-mirex` | 5 MIREX mood clusters |

## Output schema

Each classified track produces:

```json
{
  "track_id": "2abc...",
  "file_path": "/music/techno/artist - title.mp3",
  "title": "Title",
  "artist": "Artist",
  "bpm": 138.0,
  "key": "Am",
  "duration": 420.5,
  "genres": [{"label": "Techno", "score": 0.82}],
  "mood_tags": [{"label": "energetic", "score": 0.91}],
  "instruments": [{"label": "synthesizer", "score": 0.74}],
  "mood": {
    "arousal": 7.2,
    "valence": 4.1,
    "danceability": 0.88,
    "aggressive": 0.71,
    "happy": 0.12,
    "party": 0.83,
    "relaxed": 0.15,
    "sad": 0.08,
    "acoustic": 0.05,
    "tonal": 0.92,
    "timbre_bright": 0.61,
    "approachability": 0.78,
    "engagement": 0.85,
    "vocal_probability": 0.22,
    "vocalist_gender": 0.40
  },
  "flags": {
    "low_confidence": false,
    "essentia_failed": false,
    "bpm_from_metadata": true,
    "key_from_metadata": true
  }
}
```

`file_path` is the stable reference across all pipeline stages. `track_id` is a KSUID used internally. BPM and key come from file metadata when available (`bpm_from_metadata: true`); the flags indicate the source so you know whether to trust DJ software analysis or Essentia's fallback.

