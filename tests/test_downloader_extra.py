from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from mixprep.models.downloader import download_all, download_model
from mixprep.models.registry import ModelEntry, ModelRegistry


def _make_entry(**kwargs) -> ModelEntry:
    defaults = dict(
        name="test-model",
        url="https://example.com/model.pb",
        framework="tensorflow",
        description="test",
    )
    defaults.update(kwargs)
    return ModelEntry(**defaults)


class TestDownloadModelProgressBar:
    """Covers the show_progress=True branch."""

    def test_progress_bar_branch(self, tmp_path: Path):
        content = b"model bytes"
        entry = _make_entry()

        def write_fetch(url, dest, reporthook=None):
            dest.write_bytes(content)
            if reporthook:
                reporthook(1, len(content), len(content))

        with patch.dict(os.environ, {"MIXPREP_MODELS_DIR": str(tmp_path)}):
            with patch("mixprep.models.downloader._fetch", side_effect=write_fetch):
                result = download_model(entry, force=True, show_progress=True)

        assert result.exists()
        assert result.read_bytes() == content


class TestDownloadAll:
    def test_returns_all_successful(self, tmp_path: Path):
        entries = [
            _make_entry(name="model-a", url="https://example.com/model-a.pb"),
            _make_entry(name="model-b", url="https://example.com/model-b.pb"),
        ]
        registry = MagicMock(spec=ModelRegistry)
        registry.all.return_value = entries

        def fake_download(entry, force=False, show_progress=True):
            p = tmp_path / f"{entry.name}.pb"
            p.write_bytes(b"data")
            return p

        with patch.dict(os.environ, {"MIXPREP_MODELS_DIR": str(tmp_path)}):
            with patch("mixprep.models.downloader.download_model", side_effect=fake_download):
                succeeded, failed = download_all(registry, force=False)

        assert set(succeeded.keys()) == {"model-a", "model-b"}
        assert failed == {}

    def test_skips_failed_model_continues(self, tmp_path: Path):
        entries = [
            _make_entry(name="model-a", url="https://example.com/model-a.pb"),
            _make_entry(name="model-b", url="https://example.com/model-b.pb"),
        ]
        registry = MagicMock(spec=ModelRegistry)
        registry.all.return_value = entries

        call_count = 0

        def flaky_download(entry, force=False, show_progress=True):
            nonlocal call_count
            call_count += 1
            if entry.name == "model-a":
                raise RuntimeError("server error")
            p = tmp_path / f"{entry.name}.pb"
            p.write_bytes(b"data")
            return p

        with patch.dict(os.environ, {"MIXPREP_MODELS_DIR": str(tmp_path)}):
            with patch("mixprep.models.downloader.download_model", side_effect=flaky_download):
                succeeded, failed = download_all(registry, force=False)

        assert "model-a" not in succeeded
        assert "model-a" in failed
        assert "model-b" in succeeded
        assert call_count == 2
