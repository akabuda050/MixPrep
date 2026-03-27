from __future__ import annotations

from rich.console import Console
from rich.table import Table

from mixprep.models.cache import model_path
from mixprep.models.downloader import (
    download_all,
    download_model,
    model_status,
)
from mixprep.models.registry import ModelRegistry

console = Console()
registry = ModelRegistry()


def run_download(model: str | None, force: bool) -> None:
    if model:
        entry = registry.get(model)
        if entry is None:
            console.print(f"[red]Unknown model:[/red] {model}")
            console.print(f"Available: {', '.join(registry.names())}")
            raise SystemExit(1)
        download_model(entry, force=force)
        console.print(f"[green]✓[/green] {model} ready")
    else:
        succeeded, failed = download_all(registry, force=force)
        for name, path in succeeded.items():
            console.print(f"[green]✓[/green] {name} → {path}")
        console.print(
            f"\n[green]Downloaded:[/green] {len(succeeded)}  [red]Failed:[/red] {len(failed)}"
        )
        if failed:
            console.print(
                "[yellow]Re-run [bold]mixprep models download[/bold]"
                " to retry failed models.[/yellow]"
            )
            raise SystemExit(1)


def run_status() -> None:
    rows = model_status(registry)
    table = Table(title="Model Status")
    table.add_column("Name", style="bold")
    table.add_column("File")
    table.add_column("Size (MB)")
    table.add_column("Description")

    for r in rows:
        file_col = "[green]✓[/green]" if r["present"] else "[red]✗[/red]"
        size = f"{r['size_mb']}" if r["size_mb"] is not None else "[dim]—[/dim]"
        table.add_row(r["name"], file_col, size, r["description"])

    console.print(table)


def run_clear(model: str | None, all: bool) -> None:
    if all:
        removed = 0
        for entry in registry.all():
            filename = entry.url.split("/")[-1]
            path = model_path(filename)
            if path.exists():
                path.unlink()
                console.print(f"[green]Removed[/green] {entry.name}")
                removed += 1
        console.print(f"[green]Done.[/green] {removed} model(s) removed.")
        return

    if not model:
        console.print("[red]Specify a model name or use --all[/red]")
        raise SystemExit(1)

    entry = registry.get(model)
    if entry is None:
        console.print(f"[red]Unknown model:[/red] {model}")
        console.print(f"Available: {', '.join(registry.names())}")
        raise SystemExit(1)
    filename = entry.url.split("/")[-1]
    path = model_path(filename)
    if path.exists():
        path.unlink()
        console.print(f"[green]Removed[/green] {path}")
    else:
        console.print(f"[yellow]Not present:[/yellow] {path}")
