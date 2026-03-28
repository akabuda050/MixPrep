from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from mixprep.models.cache import data_dir, ensure_data_dir
from mixprep.pipeline.scan import load_scan

console = Console()

_STAGES = ("ingest", "essentia", "classify")


def run_analyze(stage: str) -> None:
    if stage not in _STAGES:
        console.print(f"[red]Unknown stage:[/red] {stage!r}. Choose from: {', '.join(_STAGES)}")
        raise SystemExit(1)

    scan_path = data_dir() / "scan.json"
    if not scan_path.exists():
        console.print("[red]scan.json not found.[/red] Run [bold]mixprep scan[/bold] first.")
        raise SystemExit(1)

    try:
        tracks = load_scan(scan_path)
    except Exception as exc:
        console.print(f"[red]Failed to load scan.json:[/red] {exc}")
        raise SystemExit(1)

    if not tracks:
        console.print("[yellow]No tracks in scan.json.[/yellow]")
        return

    if stage == "ingest":
        _run_ingest(tracks)
    else:
        console.print(f"[yellow]Stage {stage!r} not yet implemented.[/yellow]")


def _run_ingest(tracks: list) -> None:
    from mixprep.pipeline.ingest import ingest_track, write_metadata

    ensure_data_dir()
    meta_dir = data_dir() / "metadata"

    console.print(f"Ingesting [bold]{len(tracks)}[/bold] tracks ...")

    failed = 0

    try:
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(" " * 40, total=len(tracks))

            for track in tracks:
                name = Path(track.file_path).name
                progress.update(task, advance=1, description=f"{name:<40.40}")
                try:
                    meta = ingest_track(track)
                    write_metadata(meta, meta_dir)
                except Exception as exc:
                    failed += 1
                    console.log(f"[yellow]Warning:[/yellow] {track.track_id}: {exc}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Ingest interrupted. Partial results written.[/yellow]")
        raise SystemExit(130)

    ok = len(tracks) - failed
    console.print(f"[green]✓[/green] {ok}/{len(tracks)} tracks ingested → {meta_dir}")
    if failed:
        console.print(f"[yellow]{failed} tracks failed (see warnings above)[/yellow]")
