# MixPrep

DJ library intelligence and mix-set generation.

Analyzes your music library using machine learning to extract genre, mood, energy, instrumentation, and other musical properties. Uses that data to automatically build mix-ready track sets tailored to different DJ set profiles (warm-up, peak-time, cool-down, etc.).

## What it does

- **Scan** — indexes your music library, assigns stable track IDs, and reads BPM, key, title, artist from file tags — all in one pass
- **Analyze** — runs ML models to detect genre, mood, energy, danceability, vocals, instruments
- **Build** — generates candidate mix sets for a chosen profile using beam search

## Workflow

```
1. Download models        mixprep models download
2. Scan your library      mixprep scan /path/to/music --library my_library
3. Analyze tracks         mixprep analyze --stage essentia --library my_library
                          mixprep analyze --stage profile --library my_library
4. Build a set            mixprep build --profile peak --duration 60 --library my_library
```

`--library` is required on every command — it namespaces all output so you can maintain multiple isolated libraries (e.g. `house_sets`, `techno_main`). `--duration` is in minutes. Each step produces reviewable JSON files you can inspect before running the next one.

To remove entries for files that no longer exist on disk:
```
mixprep scan /path/to/music --library my_library --prune
```

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

Models are stored in `~/.local/share/mixprep/models` by default (shared across all libraries). Override with `MIXPREP_MODELS_DIR`:

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
| `essentia-msd-musicnn` | Secondary embedding — required for arousal head |
| `essentia-genre-jamendo` | 87 genre tags (house, techno, ambient, etc.) |
| `essentia-genre-jamendo-labels` | Label metadata for Jamendo genre |
| `essentia-tonal-atonal` | Tonal vs atonal — harmonic mixing compatibility |
| `essentia-timbre` | Bright vs dark timbre |
| `essentia-approachability` | Mainstream vs niche [0–1] |
| `essentia-engagement` | Active vs background listening [0–1] |
| `essentia-voice-instrumental` | Vocal probability [0–1] |
| `essentia-danceability` | Danceability score [0–1] |
| `essentia-arousal-valence-deam` | Arousal [1–9] — DEAM dataset (dim 1 = arousal, valence excluded) |
| `essentia-arousal-valence-deam-labels` | Label metadata for arousal-valence-deam |

## Output

Analysis results are stored in `~/.local/share/mixprep/data` (override with `MIXPREP_DATA_DIR`):

```
~/.local/share/mixprep/data/
└── libraries/
    └── <library>/
        ├── tracks/<track_id>.json      # file index + BPM, key, title, artist
        ├── essentia/<track_id>.json    # raw ML scores + time curves
        ├── profiles/<track_id>.json    # unified per-track profile (energy, phase scores, genres)
        └── sets/<hash>_set.json        # generated DJ set plan
```

Each library is fully isolated. Models are shared across all libraries (`~/.local/share/mixprep/models`, override with `MIXPREP_MODELS_DIR`).

Each file is plain JSON — open and inspect at any stage.

### Score reference

| Score | Range | Meaning |
|---|---|---|
| `danceability` | 0–1 | How suitable the track is for dancing. High = danceable, low = not. |
| `arousal` | 0–1 | Energy and intensity level (normalized from DEAM [1–9]). ~0.25 = calm/ambient, ~0.5 = mid-energy house, ~0.9 = peak techno. |
| `energy` | 0–1 | Derived: `0.6 × arousal + 0.4 × danceability`. |
| `groove` | 0–1 | Derived: `0.5 × danceability + 0.3 × approachability + 0.2 × tonal`. |
| `tonal` | 0–1 | How harmonic/melodic the track is. High = tonal (chords, melody), low = atonal (noise, pure rhythm). |
| `timbre_bright` | 0–1 | Brightness of the sound. High = bright/sharp (lots of highs), low = dark/warm (more bass/sub). |
| `approachability` | 0–1 | How mainstream the sound is. High = accessible/commercial, low = niche/experimental. |
| `engagement` | 0–1 | Whether the track demands active listening. High = foreground/engaging, low = background/chill. |
| `vocal_probability` | 0–1 | Likelihood of a human voice. > 0.5 = likely vocal, < 0.3 = likely instrumental. |
| `camelot` | e.g. `8A` | Camelot wheel key. Parsed from file tag or detected by audio analysis. |
| `warmup_score` | 0–1 | Suitability for set warm-up phase. |
| `build_score` | 0–1 | Suitability for build phase. |
| `peak_score` | 0–1 | Suitability for peak time. |
| `reset_score` | 0–1 | Suitability for reset/breakdown. |
| `winddown_score` | 0–1 | Suitability for closing/wind-down. |

### Track profile (`profiles/<track_id>.json`)

```json
{
  "track_id": "2abc...",
  "duration": 412.5,
  "camelot": "11A",
  "bpm": 132.0,
  "energy": 0.74,
  "groove": 0.68,
  "danceability": 0.88,
  "arousal": 0.77,
  "tonal": 0.92,
  "timbre_bright": 0.61,
  "approachability": 0.54,
  "engagement": 0.73,
  "vocal_probability": 0.22,
  "warmup_score": 0.34,
  "build_score": 0.62,
  "peak_score": 0.87,
  "reset_score": 0.41,
  "winddown_score": 0.28,
  "genres": [{"label": "techno", "score": 0.82}, {"label": "electronic disco", "score": 0.39}]
}
```

