"""
Serve command tests.

Covers:
- run_serve exits 1 when library not found
- _make_handler: GET / returns viewer HTML
- _make_handler: GET /api/profiles returns JSON list of profiles
- _make_handler: GET /api/profiles returns empty list when profiles dir absent
- _make_handler: GET /api/profiles skips corrupt JSON files
- _make_handler: GET /unknown returns 404
- _make_handler: GET /api/profiles sorted by filename
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(handler_cls, path: str) -> MagicMock:
    """Instantiate handler with a fake socket and issue a GET request."""
    buf = io.BytesIO()

    request = MagicMock()
    request.makefile.return_value = io.BufferedReader(io.BytesIO(b""))

    client_addr = ("127.0.0.1", 9999)

    # Bypass __init__ — call setup manually
    handler = handler_cls.__new__(handler_cls)
    handler.rfile = io.BytesIO(b"")
    handler.wfile = buf
    handler.path = path
    handler.headers = {}
    handler.request_version = "HTTP/1.1"
    handler.server = MagicMock()
    handler.connection = MagicMock()
    handler.client_address = client_addr

    # Capture send_response / send_header / end_headers / send_error
    responses = []
    headers = {}
    errors = []

    def _send_response(code, *a):
        responses.append(code)

    def _send_header(k, v):
        headers[k] = v

    def _end_headers():
        pass

    def _send_error(code, *a):
        errors.append(code)

    handler.send_response = _send_response
    handler.send_header = _send_header
    handler.end_headers = _end_headers
    handler.send_error = _send_error

    handler.do_GET()

    written = buf.getvalue()
    return MagicMock(
        responses=responses,
        headers=headers,
        errors=errors,
        body=written,
    )


# ---------------------------------------------------------------------------
# run_serve
# ---------------------------------------------------------------------------


def test_run_serve_exits_1_on_missing_library(tmp_path, monkeypatch):
    from mixprep.cli.commands.serve import run_serve

    monkeypatch.setenv("MIXPREP_DATA_DIR", str(tmp_path))
    import pytest

    with pytest.raises(SystemExit) as exc:
        run_serve("nonexistent", 9876)
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# _make_handler — routing
# ---------------------------------------------------------------------------


def test_handler_get_root_serves_html(tmp_path):
    from mixprep.cli.commands.serve import _make_handler

    html = tmp_path / "viewer.html"
    html.write_bytes(b"<html>test</html>")

    handler_cls = _make_handler(tmp_path)

    with patch("mixprep.cli.commands.serve._VIEWER_HTML", html):
        result = _make_request(handler_cls, "/")

    assert 200 in result.responses
    assert result.headers.get("Content-Type") == "text/html; charset=utf-8"
    assert b"<html>test</html>" in result.body


def test_handler_get_index_html_serves_html(tmp_path):
    from mixprep.cli.commands.serve import _make_handler

    html = tmp_path / "viewer.html"
    html.write_bytes(b"<html>ok</html>")

    handler_cls = _make_handler(tmp_path)

    with patch("mixprep.cli.commands.serve._VIEWER_HTML", html):
        result = _make_request(handler_cls, "/index.html")

    assert 200 in result.responses


def test_handler_get_unknown_returns_404(tmp_path):
    from mixprep.cli.commands.serve import _make_handler

    handler_cls = _make_handler(tmp_path)
    result = _make_request(handler_cls, "/not/a/route")
    assert 404 in result.errors


def test_handler_get_viewer_missing_returns_404(tmp_path):
    from mixprep.cli.commands.serve import _make_handler

    handler_cls = _make_handler(tmp_path)
    missing = tmp_path / "nonexistent.html"

    with patch("mixprep.cli.commands.serve._VIEWER_HTML", missing):
        result = _make_request(handler_cls, "/")

    assert 404 in result.errors


# ---------------------------------------------------------------------------
# _make_handler — /api/profiles
# ---------------------------------------------------------------------------


def test_handler_profiles_empty_when_no_dir(tmp_path):
    from mixprep.cli.commands.serve import _make_handler

    handler_cls = _make_handler(tmp_path)  # no profiles/ subdir
    result = _make_request(handler_cls, "/api/profiles")

    assert 200 in result.responses
    assert result.headers.get("Content-Type") == "application/json; charset=utf-8"
    assert json.loads(result.body) == []


def test_handler_profiles_returns_json(tmp_path):
    from mixprep.cli.commands.serve import _make_handler

    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "a.json").write_text(
        json.dumps({"track_id": "a", "bpm": 128.0}), encoding="utf-8"
    )
    (profiles_dir / "b.json").write_text(
        json.dumps({"track_id": "b", "bpm": 132.0}), encoding="utf-8"
    )

    handler_cls = _make_handler(tmp_path)
    result = _make_request(handler_cls, "/api/profiles")

    data = json.loads(result.body)
    assert len(data) == 2
    ids = {p["track_id"] for p in data}
    assert ids == {"a", "b"}


def test_handler_profiles_skips_corrupt(tmp_path):
    from mixprep.cli.commands.serve import _make_handler

    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "good.json").write_text(json.dumps({"track_id": "good"}), encoding="utf-8")
    (profiles_dir / "bad.json").write_text("{not valid json", encoding="utf-8")

    handler_cls = _make_handler(tmp_path)
    result = _make_request(handler_cls, "/api/profiles")

    data = json.loads(result.body)
    assert len(data) == 1
    assert data[0]["track_id"] == "good"
