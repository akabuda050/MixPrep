from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from mixprep.models.downloader import (
    download_model,
    model_status,
)
from mixprep.models.registry import ModelEntry


def _make_entry(**kwargs) -> ModelEntry:
    defaults = dict(
        name="test-model",
        url="https://example.com/model.pb",
        framework="tensorflow",
        description="test",
    )
    defaults.update(kwargs)
    return ModelEntry(**defaults)


class TestDownloadModel:
    def _make_fake_fetch(self, content: bytes = b"model bytes"):
        def fake_fetch(url, dest, reporthook=None):
            dest.write_bytes(content)

        return fake_fetch

    def test_skips_if_already_present(self, tmp_path: Path):
        entry = _make_entry()
        with patch.dict(os.environ, {"MIXPREP_MODELS_DIR": str(tmp_path)}):
            dest = tmp_path / "model.pb"
            dest.write_bytes(b"model bytes")
            with patch("mixprep.models.downloader._fetch") as mock_fetch:
                result = download_model(entry, force=False, show_progress=False)
        mock_fetch.assert_not_called()
        assert result == dest

    def test_downloads_when_missing(self, tmp_path: Path):
        entry = _make_entry()
        with patch.dict(os.environ, {"MIXPREP_MODELS_DIR": str(tmp_path)}):
            with patch("mixprep.models.downloader._fetch", side_effect=self._make_fake_fetch()):
                result = download_model(entry, force=False, show_progress=False)
        assert result.exists()

    def test_force_redownloads(self, tmp_path: Path):
        content = b"model bytes"
        entry = _make_entry()
        with patch.dict(os.environ, {"MIXPREP_MODELS_DIR": str(tmp_path)}):
            dest = tmp_path / "model.pb"
            dest.write_bytes(content)
            with patch(
                "mixprep.models.downloader._fetch",
                side_effect=self._make_fake_fetch(content),
            ) as mock_fetch:
                download_model(entry, force=True, show_progress=False)
        mock_fetch.assert_called_once()


class TestModelStatus:
    def test_missing_model(self, tmp_path: Path):
        entry = _make_entry(url="https://example.com/model.pb")
        registry = MagicMock()
        registry.all.return_value = [entry]
        with patch.dict(os.environ, {"MIXPREP_MODELS_DIR": str(tmp_path)}):
            rows = model_status(registry)
        assert rows[0]["present"] is False
        assert rows[0]["size_mb"] is None

    def test_present_model(self, tmp_path: Path):
        entry = _make_entry(url="https://example.com/model.pb")
        p = tmp_path / "model.pb"
        p.write_bytes(b"data")
        registry = MagicMock()
        registry.all.return_value = [entry]
        with patch.dict(os.environ, {"MIXPREP_MODELS_DIR": str(tmp_path)}):
            rows = model_status(registry)
        assert rows[0]["present"] is True

    def test_size_from_disk(self, tmp_path: Path):
        entry = _make_entry(url="https://example.com/model.pb")
        content = b"x" * 1_048_576  # exactly 1 MB
        (tmp_path / "model.pb").write_bytes(content)
        registry = MagicMock()
        registry.all.return_value = [entry]
        with patch.dict(os.environ, {"MIXPREP_MODELS_DIR": str(tmp_path)}):
            rows = model_status(registry)
        assert rows[0]["size_mb"] == 1.0
