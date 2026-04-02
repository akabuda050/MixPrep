from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from mixprep.models.cache import data_dir
from mixprep.pipeline.scan import load_tracks_dir

console = Console()

_STAGES = ("essentia", "profile")


def run_analyze(stage: str, library: str) -> None:
    if stage not in _STAGES:
        console.print(f"[red]Unknown stage:[/red] {stage!r}. Choose from: {', '.join(_STAGES)}")
        raise SystemExit(1)

    tracks_dir = data_dir(library) / "tracks"
    if not tracks_dir.is_dir():
        console.print(
            f"[red]No tracks found for library {library!r}.[/red] "
            f"Run [bold]mixprep scan --library {library}[/bold] first."
        )
        raise SystemExit(1)

    tracks = load_tracks_dir(tracks_dir)

    if not tracks:
        console.print("[yellow]No tracks in library.[/yellow]")
        return

    if stage == "essentia":
        _run_essentia(tracks, library)
    elif stage == "profile":
        _run_profile(tracks, library)


def _run_essentia(tracks: list, library: str) -> None:
    from mixprep.pipeline.essentia_runner import load_models, run_essentia, write_essentia

    essentia_dir = data_dir(library) / "essentia"
    essentia_dir.mkdir(parents=True, exist_ok=True)

    console.print("Loading Essentia models ...")
    try:
        load_models()
    except Exception as exc:
        console.print(f"[red]Failed to load models:[/red] {exc}")
        raise SystemExit(1)

    console.print(f"Running Essentia on [bold]{len(tracks)}[/bold] tracks ...")
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
                output = run_essentia(track)
                if output.flags.essentia_failed:
                    failed += 1
                write_essentia(output, essentia_dir)

    except KeyboardInterrupt:
        console.print("\n[yellow]Essentia interrupted. Partial results written.[/yellow]")
        raise SystemExit(130)

    ok = len(tracks) - failed
    console.print(f"[green]✓[/green] {ok}/{len(tracks)} tracks processed → {essentia_dir}")
    if failed:
        console.print(f"[yellow]{failed} tracks failed (essentia_failed=true in artifact)[/yellow]")


def _run_profile(tracks: list, library: str) -> None:
    from mixprep.pipeline.essentia_runner import load_essentia
    from mixprep.pipeline.profile import compute_profile, write_profile

    essentia_dir = data_dir(library) / "essentia"
    profiles_dir = data_dir(library) / "profiles"

    skipped = 0
    failed = 0
    processed = 0

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

            essentia_path = essentia_dir / f"{track.track_id}.json"
            if not essentia_path.exists():
                skipped += 1
                continue

            try:
                essentia = load_essentia(essentia_path)
            except Exception as exc:
                console.print(
                    f"[yellow]Skipping {track.track_id}: corrupt essentia artifact: {exc}[/yellow]"
                )
                failed += 1
                continue

            profile = compute_profile(track, essentia)
            write_profile(profile, profiles_dir)
            processed += 1

    console.print(f"[green]✓[/green] {processed} profiles written → {profiles_dir}")
    if skipped:
        console.print(
            f"[yellow]{skipped} tracks skipped "
            f"(no essentia artifact — run essentia stage first)[/yellow]"
        )
    if failed:
        console.print(f"[yellow]{failed} tracks failed (corrupt essentia artifact)[/yellow]")
