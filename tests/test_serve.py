"""
Serve command tests.

Covers:
- _make_handler: GET / returns viewer HTML
- _make_handler: GET /api/libraries returns list of library names
- _make_handler: GET /api/libraries returns empty list when no libraries
- _make_handler: GET /api/profiles/<lib> returns JSON list of profiles
- _make_handler: GET /api/profiles/<lib> returns empty list when profiles dir absent
- _make_handler: GET /api/profiles/<lib> skips corrupt JSON files
- _make_handler: GET /unknown returns 404
- _make_handler: path traversal in library name returns 400
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

    handler = handler_cls.__new__(handler_cls)
    handler.rfile = io.BytesIO(b"")
    handler.wfile = buf
    handler.path = path
    handler.headers = {}
    handler.request_version = "HTTP/1.1"
    handler.server = MagicMock()
    handler.connection = MagicMock()
    handler.client_address = ("127.0.0.1", 9999)

    responses = []
    headers = {}
    errors = []

    handler.send_response = lambda code, *a: responses.append(code)
    handler.send_header = lambda k, v: headers.__setitem__(k, v)
    handler.end_headers = lambda: None
    handler.send_error = lambda code, *a: errors.append(code)

    handler.do_GET()

    return MagicMock(
        responses=responses,
        headers=headers,
        errors=errors,
        body=buf.getvalue(),
    )


# ---------------------------------------------------------------------------
# GET / — viewer HTML
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


def test_handler_get_viewer_missing_returns_404(tmp_path):
    from mixprep.cli.commands.serve import _make_handler

    handler_cls = _make_handler(tmp_path)
    missing = tmp_path / "nonexistent.html"

    with patch("mixprep.cli.commands.serve._VIEWER_HTML", missing):
        result = _make_request(handler_cls, "/")

    assert 404 in result.errors


# ---------------------------------------------------------------------------
# GET /api/libraries
# ---------------------------------------------------------------------------


def test_handler_libraries_empty_when_no_dir(tmp_path):
    from mixprep.cli.commands.serve import _make_handler

    handler_cls = _make_handler(tmp_path)
    result = _make_request(handler_cls, "/api/libraries")

    assert 200 in result.responses
    assert result.headers.get("Content-Type") == "application/json; charset=utf-8"
    assert json.loads(result.body) == []


def test_handler_libraries_returns_names(tmp_path):
    from mixprep.cli.commands.serve import _make_handler

    libs = tmp_path / "libraries"
    (libs / "alpha" / "profiles").mkdir(parents=True)
    (libs / "beta" / "profiles").mkdir(parents=True)
    # dir without profiles/ — should be excluded
    (libs / "no_profiles").mkdir()

    handler_cls = _make_handler(tmp_path)
    result = _make_request(handler_cls, "/api/libraries")

    data = json.loads(result.body)
    assert set(data) == {"alpha", "beta"}


# ---------------------------------------------------------------------------
# GET /api/profiles/<lib>
# ---------------------------------------------------------------------------


def test_handler_profiles_empty_when_no_dir(tmp_path):
    from mixprep.cli.commands.serve import _make_handler

    handler_cls = _make_handler(tmp_path)
    result = _make_request(handler_cls, "/api/profiles/mylib")

    assert 200 in result.responses
    assert result.headers.get("Content-Type") == "application/json; charset=utf-8"
    assert json.loads(result.body) == []


def test_handler_profiles_returns_json(tmp_path):
    from mixprep.cli.commands.serve import _make_handler

    profiles_dir = tmp_path / "libraries" / "mylib" / "profiles"
    profiles_dir.mkdir(parents=True)
    (profiles_dir / "a.json").write_text(json.dumps({"track_id": "a", "bpm": 128.0}), encoding="utf-8")
    (profiles_dir / "b.json").write_text(json.dumps({"track_id": "b", "bpm": 132.0}), encoding="utf-8")

    handler_cls = _make_handler(tmp_path)
    result = _make_request(handler_cls, "/api/profiles/mylib")

    data = json.loads(result.body)
    assert len(data) == 2
    assert {p["track_id"] for p in data} == {"a", "b"}


def test_handler_profiles_skips_corrupt(tmp_path):
    from mixprep.cli.commands.serve import _make_handler

    profiles_dir = tmp_path / "libraries" / "mylib" / "profiles"
    profiles_dir.mkdir(parents=True)
    (profiles_dir / "good.json").write_text(json.dumps({"track_id": "good"}), encoding="utf-8")
    (profiles_dir / "bad.json").write_text("{not valid json", encoding="utf-8")

    handler_cls = _make_handler(tmp_path)
    result = _make_request(handler_cls, "/api/profiles/mylib")

    data = json.loads(result.body)
    assert len(data) == 1
    assert data[0]["track_id"] == "good"


# ---------------------------------------------------------------------------
# 404 / 400
# ---------------------------------------------------------------------------


def test_handler_get_unknown_returns_404(tmp_path):
    from mixprep.cli.commands.serve import _make_handler

    handler_cls = _make_handler(tmp_path)
    result = _make_request(handler_cls, "/not/a/route")
    assert 404 in result.errors


def test_handler_profiles_path_traversal_returns_400(tmp_path):
    from mixprep.cli.commands.serve import _make_handler

    handler_cls = _make_handler(tmp_path)
    result = _make_request(handler_cls, "/api/profiles/../secret")
    assert 400 in result.errors
