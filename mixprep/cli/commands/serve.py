"""
Serve command — minimal HTTP server for the profile viewer.
"""

from __future__ import annotations

import http.server
import json
from pathlib import Path

from rich.console import Console

console = Console()

_VIEWER_HTML = Path(__file__).parent / "viewer.html"


def _make_handler(lib_dir: Path) -> type:
    """Return a handler class with library_dir baked in."""

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                self._serve_file(_VIEWER_HTML, "text/html; charset=utf-8")
            elif self.path == "/api/profiles":
                self._serve_profiles()
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

        def _serve_profiles(self) -> None:
            profiles_dir = lib_dir / "profiles"
            items = []
            if profiles_dir.is_dir():
                for f in sorted(profiles_dir.glob("*.json")):
                    try:
                        items.append(json.loads(f.read_text(encoding="utf-8")))
                    except Exception:
                        pass
            self._send_json(items)

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


def run_serve(library: str, port: int) -> None:
    from mixprep.models.cache import data_dir

    lib_dir = data_dir(library)
    if not lib_dir.is_dir():
        console.print(
            f"[red]Library {library!r} not found.[/red] "
            f"Run [bold]mixprep scan --library {library}[/bold] first."
        )
        raise SystemExit(1)

    if not _VIEWER_HTML.exists():
        console.print(f"[red]Viewer not found:[/red] {_VIEWER_HTML}")
        raise SystemExit(1)

    handler = _make_handler(lib_dir)
    server = http.server.HTTPServer(("0.0.0.0", port), handler)

    console.print(f"Serving [bold]{library}[/bold] at [link]http://127.0.0.1:{port}[/link]")
    console.print("Press [bold]Ctrl+C[/bold] to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/yellow]")
