from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import typer

app = typer.Typer(
    name="mixprep",
    help="DJ library intelligence and mix-set generation.",
    no_args_is_help=True,
)

models_app = typer.Typer(help="Manage pretrained models.")
app.add_typer(models_app, name="models")


@models_app.command("download")
def models_download(
    model: str = typer.Option(None, help="Specific model name, or all if omitted"),
    force: bool = typer.Option(False, help="Re-download even if already present"),
) -> None:
    """Download pretrained model weights."""
    from mixprep.cli.commands.models import run_download

    run_download(model=model, force=force)


@models_app.command("status")
def models_status() -> None:
    """Show download status of all models."""
    from mixprep.cli.commands.models import run_status

    run_status()


@models_app.command("clear")
def models_clear(
    model: str = typer.Argument(None, help="Model name to remove"),
    all: bool = typer.Option(False, "--all", help="Remove all downloaded models"),
) -> None:
    """Remove a downloaded model (or all with --all) from cache."""
    from mixprep.cli.commands.models import run_clear

    run_clear(model=model, all=all)


@app.command("analyze")
def analyze(
    stage: str = typer.Option(..., help="Pipeline stage: ingest | essentia | classify"),
) -> None:
    """Run an analysis stage against the scanned library."""
    from mixprep.cli.commands.analyze import run_analyze

    run_analyze(stage)


@app.command("scan")
def scan(
    directory: str = typer.Argument(..., help="Path to music library directory"),
) -> None:
    """Scan a music library directory and index all audio files."""
    from pathlib import Path

    from mixprep.cli.commands.scan import run_scan

    run_scan(Path(directory))


if __name__ == "__main__":
    app()
