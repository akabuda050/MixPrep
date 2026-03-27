from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from mixprep.models.cache import data_dir, ensure_data_dir
from mixprep.pipeline.scan import collect_audio_files, load_scan, scan_library, write_scan

console = Console()


def run_scan(directory: Path) -> None:
    if not directory.is_dir():
        console.print(f"[red]Not a directory:[/red] {directory}")
        raise SystemExit(1)

    dest = data_dir() / "scan.json"
    tmp = dest.with_suffix(".tmp")

    # Clean up any leftover .tmp from a previous interrupted scan
    if tmp.exists():
        tmp.unlink()

    existing = []
    if dest.exists():
        try:
            existing = load_scan(dest)
        except Exception as exc:
            console.print(f"[yellow]Warning: could not load existing scan: {exc}[/yellow]")

    ensure_data_dir()

    console.print(f"Scanning [bold]{directory}[/bold] ...")
    files = collect_audio_files(directory)

    if not files:
        console.print("[yellow]No audio files found.[/yellow]")
        return

    try:
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(" " * 40, total=len(files))

            def advance(path: Path) -> None:
                progress.update(task, advance=1, description=f"{path.name:<40.40}")

            entries = scan_library(directory, existing, files=files, progress_callback=advance)

        write_scan(entries, dest)
        console.print(f"[green]✓[/green] {len(entries)} tracks indexed → {dest}")

    except KeyboardInterrupt:
        if tmp.exists():
            tmp.unlink()
        console.print("\n[yellow]Scan interrupted. No changes written.[/yellow]")
        raise SystemExit(130)
