"""Command-line interface skeleton for ICTC."""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(help="ICTC research prototype commands.", no_args_is_help=True)


@app.command("environment")
def show_environment(
    manifest: Path = typer.Option(
        Path("project/environment_manifest.json"),
        exists=True,
        readable=True,
        help="M00 environment manifest to display.",
    ),
) -> None:
    """Display the recorded M00 environment classification."""
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    typer.echo(json.dumps(payload["constraints"], indent=2))


@app.command("status")
def show_status(
    project_state: Path = typer.Option(
        Path("project/PROJECT_STATE.md"),
        exists=True,
        readable=True,
        help="Project state document to display.",
    ),
) -> None:
    """Display the current project-state document."""
    typer.echo(project_state.read_text(encoding="utf-8"))
