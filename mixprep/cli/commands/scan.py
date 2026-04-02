from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from mixprep.models.cache import data_dir, ensure_data_dir
from mixprep.pipeline.scan import collect_audio_files, prune_orphans, scan_library

console = Console()


def run_scan(directory: Path, library: str, prune: bool = False) -> None:
    if not directory.is_dir():
        console.print(f"[red]Not a directory:[/red] {directory}")
        raise SystemExit(1)

    ensure_data_dir(library)
    tracks_dir = data_dir(library) / "tracks"

    if prune:
        removed = prune_orphans(tracks_dir)
        if removed:
            console.print(f"[yellow]Pruned {removed} orphaned track(s).[/yellow]")

    console.print(f"Scanning [bold]{directory}[/bold] → library [bold]{library}[/bold] ...")
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

            entries = scan_library(directory, tracks_dir, files=files, progress_callback=advance)

        console.print(f"[green]✓[/green] {len(entries)} tracks indexed → {tracks_dir}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Scan interrupted. Partial results written.[/yellow]")
        raise SystemExit(130)
