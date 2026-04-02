"""
Serve command — minimal HTTP server for the profile viewer.

Routes:
  GET /                        → viewer.html
  GET /index.html              → viewer.html
  GET /api/libraries           → list of available library names
  GET /api/profiles/<lib>      → list of track profiles for <lib>
  GET /api/audio/<lib>/<id>    → stream audio file for track_id
"""

from __future__ import annotations

import http.server
import json
import mimetypes
from pathlib import Path

from rich.console import Console

console = Console()

_VIEWER_HTML = Path(__file__).parent / "viewer.html"


def _make_handler(data_base: Path) -> type:
    """Return a handler class with data_base baked in."""

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                self._serve_file(_VIEWER_HTML, "text/html; charset=utf-8")
            elif self.path == "/api/libraries":
                self._serve_libraries()
            elif self.path.startswith("/api/profiles/"):
                lib = self.path[len("/api/profiles/"):]
                self._serve_profiles(lib)
            elif self.path.startswith("/api/audio/"):
                rest = self.path[len("/api/audio/"):]
                parts = rest.split("/", 1)
                if len(parts) == 2:
                    self._serve_audio(parts[0], parts[1])
                else:
                    self.send_error(400)
            else:
                self.send_error(404)

        def _serve_file(self, path: Path, content_type: str) -> None:
            if not path.exists():
                self.send_error(404)
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _serve_libraries(self) -> None:
            libs_root = data_base / "libraries"
            names: list[str] = []
            if libs_root.is_dir():
                for d in sorted(libs_root.iterdir()):
                    if d.is_dir() and (d / "profiles").is_dir():
                        names.append(d.name)
            self._send_json(names)

        def _serve_profiles(self, library: str) -> None:
            # Basic path traversal guard
            if not library or "/" in library or "\\" in library or ".." in library.split("/"):
                self.send_error(400)
                return
            profiles_dir = data_base / "libraries" / library / "profiles"
            items = []
            if profiles_dir.is_dir():
                for f in sorted(profiles_dir.glob("*.json")):
                    try:
                        items.append(json.loads(f.read_text(encoding="utf-8")))
                    except Exception:
                        pass
            self._send_json(items)

        def _serve_audio(self, library: str, track_id: str) -> None:
            if not library or "/" in library or "\\" in library or ".." in library:
                self.send_error(400)
                return
            if not track_id or "/" in track_id or "\\" in track_id or ".." in track_id:
                self.send_error(400)
                return
            profile_path = data_base / "libraries" / library / "profiles" / f"{track_id}.json"
            if not profile_path.exists():
                self.send_error(404)
                return
            try:
                profile = json.loads(profile_path.read_text(encoding="utf-8"))
                fp = profile.get("file_path")
                if not fp:
                    self.send_error(404)
                    return
                file_path = Path(fp)
            except Exception:
                self.send_error(500)
                return
            if not file_path.is_file():
                self.send_error(404)
                return
            content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
            size = file_path.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with file_path.open("rb") as f:
                while chunk := f.read(65536):
                    self.wfile.write(chunk)

        def _send_json(self, data: object) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: object) -> None:  # noqa: ARG002
            pass

    return _Handler


def run_serve(port: int) -> None:
    from mixprep.models.cache import _resolve_base, _xdg_data_home

    data_base = _resolve_base("MIXPREP_DATA_DIR", _xdg_data_home() / "mixprep" / "data")

    if not _VIEWER_HTML.exists():
        console.print(f"[red]Viewer not found:[/red] {_VIEWER_HTML}")
        raise SystemExit(1)

    handler = _make_handler(data_base)
    server = http.server.HTTPServer(("0.0.0.0", port), handler)

    console.print(f"MixPrep viewer at [link]http://127.0.0.1:{port}[/link]")
    console.print("Press [bold]Ctrl+C[/bold] to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/yellow]")
