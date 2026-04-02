from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from mixprep.models.cache import (
    data_dir,
    ensure_data_dir,
    ensure_models_dir,
    model_path,
    models_dir,
)


class TestModelsDir:
    def test_env_var_takes_priority(self, tmp_path: Path):
        with patch.dict(os.environ, {"MIXPREP_MODELS_DIR": str(tmp_path)}):
            assert models_dir() == tmp_path

    def test_xdg_data_home_used_when_no_env(self, tmp_path: Path):
        env = {k: v for k, v in os.environ.items() if k != "MIXPREP_MODELS_DIR"}
        env["XDG_DATA_HOME"] = str(tmp_path)
        with patch.dict(os.environ, env, clear=True):
            result = models_dir()
        assert result == tmp_path / "mixprep" / "models"

    def test_xdg_fallback_to_home(self):
        env = {
            k: v for k, v in os.environ.items() if k not in ("MIXPREP_MODELS_DIR", "XDG_DATA_HOME")
        }
        with patch.dict(os.environ, env, clear=True):
            result = models_dir()
        assert result == Path.home() / ".local" / "share" / "mixprep" / "models"


class TestModelPath:
    def test_returns_models_dir_joined(self, tmp_path: Path):
        with patch.dict(os.environ, {"MIXPREP_MODELS_DIR": str(tmp_path)}):
            result = model_path("foo.pb")
        assert result == tmp_path / "foo.pb"


class TestDataDir:
    def test_env_var_takes_priority(self, tmp_path: Path):
        with patch.dict(os.environ, {"MIXPREP_DATA_DIR": str(tmp_path)}):
            assert data_dir("mylib") == tmp_path / "libraries" / "mylib"

    def test_xdg_data_home_used_when_no_env(self, tmp_path: Path):
        env = {k: v for k, v in os.environ.items() if k != "MIXPREP_DATA_DIR"}
        env["XDG_DATA_HOME"] = str(tmp_path)
        with patch.dict(os.environ, env, clear=True):
            result = data_dir("mylib")
        assert result == tmp_path / "mixprep" / "data" / "libraries" / "mylib"

    def test_xdg_fallback_to_home(self):
        env = {
            k: v for k, v in os.environ.items() if k not in ("MIXPREP_DATA_DIR", "XDG_DATA_HOME")
        }
        with patch.dict(os.environ, env, clear=True):
            result = data_dir("mylib")
        assert (
            result == Path.home() / ".local" / "share" / "mixprep" / "data" / "libraries" / "mylib"
        )


class TestEnsureDataDir:
    def test_creates_directory(self, tmp_path: Path):
        target = tmp_path / "nested" / "data"
        with patch.dict(os.environ, {"MIXPREP_DATA_DIR": str(target)}):
            result = ensure_data_dir("mylib")
        assert (target / "libraries" / "mylib").exists()
        assert result == target / "libraries" / "mylib"


class TestEnsureModelsDir:
    def test_creates_directory(self, tmp_path: Path):
        target = tmp_path / "nested" / "models"
        with patch.dict(os.environ, {"MIXPREP_MODELS_DIR": str(target)}):
            result = ensure_models_dir()
        assert target.exists()
        assert result == target

    def test_idempotent(self, tmp_path: Path):
        with patch.dict(os.environ, {"MIXPREP_MODELS_DIR": str(tmp_path)}):
            ensure_models_dir()
            ensure_models_dir()  # second call should not raise
        assert tmp_path.exists()
