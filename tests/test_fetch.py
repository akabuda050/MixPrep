from __future__ import annotations

import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

from mixprep.models.downloader import _fetch


class TestFetch:
    def test_success_on_first_attempt(self, tmp_path: Path):
        dest = tmp_path / "model.pb"

        def fake_retrieve(url, tmp, reporthook=None):
            Path(tmp).write_bytes(b"data")

        with patch("urllib.request.urlretrieve", side_effect=fake_retrieve):
            _fetch("https://example.com/model.pb", dest)

        assert dest.exists()
        assert dest.read_bytes() == b"data"

    def test_tmp_file_replaced_atomically(self, tmp_path: Path):
        dest = tmp_path / "model.pb"
        tmp = dest.with_suffix(".tmp")

        def fake_retrieve(url, tmp_path, reporthook=None):
            Path(tmp_path).write_bytes(b"data")

        with patch("urllib.request.urlretrieve", side_effect=fake_retrieve):
            _fetch("https://example.com/model.pb", dest)

        assert dest.exists()
        assert not tmp.exists()  # .tmp cleaned up

    def test_retries_on_failure_then_succeeds(self, tmp_path: Path):
        dest = tmp_path / "model.pb"
        attempts = []

        def fake_retrieve(url, tmp_path, reporthook=None):
            attempts.append(1)
            if len(attempts) < 2:
                raise urllib.error.URLError("timeout")
            Path(tmp_path).write_bytes(b"data")

        with patch("urllib.request.urlretrieve", side_effect=fake_retrieve):
            with patch("time.sleep"):  # don't actually wait
                _fetch("https://example.com/model.pb", dest)

        assert len(attempts) == 2
        assert dest.exists()

    def test_raises_after_max_retries(self, tmp_path: Path):
        dest = tmp_path / "model.pb"

        with patch("urllib.request.urlretrieve", side_effect=urllib.error.URLError("timeout")):
            with patch("time.sleep"):
                with pytest.raises(RuntimeError, match="Download failed after"):
                    _fetch("https://example.com/model.pb", dest)

    def test_tmp_cleaned_up_on_failure(self, tmp_path: Path):
        dest = tmp_path / "model.pb"
        tmp = dest.with_suffix(".tmp")

        def fake_retrieve(url, tmp_path, reporthook=None):
            Path(tmp_path).write_bytes(b"partial")
            raise urllib.error.URLError("connection reset")

        with patch("urllib.request.urlretrieve", side_effect=fake_retrieve):
            with patch("time.sleep"):
                with pytest.raises(RuntimeError):
                    _fetch("https://example.com/model.pb", dest)

        assert not tmp.exists()
        assert not dest.exists()

    def test_sleep_between_retries(self, tmp_path: Path):
        dest = tmp_path / "model.pb"
        calls = []

        def fake_retrieve(url, tmp_path, reporthook=None):
            calls.append(1)
            if len(calls) < 3:
                raise urllib.error.URLError("timeout")
            Path(tmp_path).write_bytes(b"data")

        with patch("urllib.request.urlretrieve", side_effect=fake_retrieve):
            with patch("time.sleep") as mock_sleep:
                _fetch("https://example.com/model.pb", dest)

        assert mock_sleep.call_count == 2  # sleep between attempt 1→2 and 2→3
